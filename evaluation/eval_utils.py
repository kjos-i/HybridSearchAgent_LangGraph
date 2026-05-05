"""Utility helpers for the hybrid search evaluation flow.

Deterministic metric functions live in eval_metrics.py — this module
keeps only the non-metric plumbing: data loading, prompt construction,
LangGraph message extraction, DeepEval context builders, the retrieval
preview formatter, generic math helpers, and the shared normalize_text
text utility (used both by metric snippet matching in eval_metrics and
anywhere else free-form text needs to be made comparable).
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from eval_models import EvalCase


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(value: str) -> str:
    """Lowercase, strip accents, and collapse punctuation for substring checks.

    Used both for keyword matching against the agent's answer and for
    snippet matching against retrieved chunk contents.  Lives here (not
    in eval_metrics) because nothing about it is metric-specific —
    it's a generic text comparator that any module needing case- and
    accent-insensitive substring matching can reuse.
    """
    if not value:
        return ""
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    ascii_value = ascii_value.lower()
    ascii_value = re.sub(r"[^a-z0-9.%\-\s]", " ", ascii_value)
    return re.sub(r"\s+", " ", ascii_value).strip()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cases(dataset_path: Path) -> list[EvalCase]:
    """Load and Pydantic-validate evaluation cases from a JSON file.

    Raises SystemExit (rather than letting Pydantic / JSON errors
    bubble up) on a missing file, invalid JSON, or schema-mismatched rows.
    Failing fast at startup is preferable to crashing mid-run after the
    judge LLM has already burned tokens on the first few cases.
    """
    if not dataset_path.exists():
        raise SystemExit(f"Dataset file not found: {dataset_path}")
    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        return [EvalCase.model_validate(item) for item in payload]
    except (json.JSONDecodeError, ValidationError) as exc:
        raise SystemExit(f"Invalid evaluation dataset '{dataset_path}': {exc}") from exc


# ---------------------------------------------------------------------------
# Prompt & text helpers
# ---------------------------------------------------------------------------

def make_prompt(case: EvalCase) -> str:
    """Build the full prompt string sent to the agent for an eval case.

    Starts with the case's question (whitespace-stripped) and appends an
    "Evaluation constraints" block when the case carries an
    answer_style directive or metadata_filters — the latter become
    a natural-language hint asking the agent to constrain its retrieval
    to matching documents.  Cases with neither set get just the question
    so the prompt stays minimal when no constraints apply.
    """
    prompt = case.question.strip()
    additions: list[str] = []

    if case.answer_style:
        additions.append(case.answer_style)

    if case.metadata_filters:
        filter_text = ", ".join(f"{key}={value}" for key, value in case.metadata_filters.items())
        additions.append(f"If you use retrieval filters, constrain the search to: {filter_text}.")

    if additions:
        prompt += "\n\nEvaluation constraints:\n- " + "\n- ".join(additions)

    return prompt


def extract_message_text(content: Any) -> str:
    """Extract plain text from a LangChain message content field.

    LangChain messages can carry the content in three shapes depending on
    the provider: a plain string, a list of dict blocks (each with a
    "text" or "content" key), or a list mixing strings and dicts.
    Handles all three and falls back to str() for anything else so
    the caller always receives a string and never a typeError.
    """
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()

    return str(content).strip()


def extract_agent_retrieval_results(messages: list[Any]) -> list[dict[str, Any]]:
    """Collect retrieval results from every ToolMessage in the message history.

    Each ToolMessage carries the JSON output of a tool call; the
    retrieval tool wraps its hits as {"results": [...]}.  This helper
    walks the message list, parses each tool message's content, and
    flattens every "results" array into a single list — preserving
    order so the agent's last-known retrieval ranking survives.  Tool
    messages with malformed JSON or no "results" key are silently
    skipped, since the harness still wants partial data when one tool
    call goes wrong.
    """
    results: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        content = msg.content
        if not content:
            continue
        try:
            tool_output = json.loads(content) if isinstance(content, str) else content
            if isinstance(tool_output, dict) and "results" in tool_output:
                results.extend(tool_output["results"])
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return results


# ---------------------------------------------------------------------------
# DeepEval context builders
# ---------------------------------------------------------------------------

def build_expected_output(case: EvalCase) -> str:
    """Build DeepEval's expected_output field from the case's answer points.

    Joins expected_answer_points with a space when present, otherwise
    returns a generic "answer from the corpus" sentinel.  The sentinel
    keeps DeepEval's contextual_recall metric well-defined for cases that
    intentionally have no gold answer (e.g. open-ended exploration cases
    that only assert retrieval quality).
    """
    if case.expected_answer_points:
        return " ".join(case.expected_answer_points)
    return "Provide a corpus-grounded answer only."


def build_gold_context(case: EvalCase) -> list[str]:
    """Build DeepEval's gold context list for a case.

    Prefers expected_answer_points because each point is a discrete
    claim the answer should cover — that shape works directly with
    contextual recall.  When no answer points exist, falls back to the
    generic expected-output sentence plus one "Relevant source: …"
    line per expected_sources entry, so cases that only have source
    ground truth still produce a non-empty context.
    """
    if case.expected_answer_points:
        return case.expected_answer_points

    fallback = [build_expected_output(case)]
    fallback.extend(f"Relevant source: {source}" for source in case.expected_sources)
    return fallback


def build_retrieval_context(results: list[dict[str, Any]]) -> list[str]:
    """Build DeepEval's retrieval_context from raw retrieval results.

    Each result becomes one annotated string with provenance metadata
    (source filename, chunk ID, backend, score) followed by the chunk
    text — newlines collapsed to spaces so each result stays one entry.
    An empty result list returns ["No context retrieved."] rather
    than [] because DeepEval treats empty contexts as a hard error.
    """
    if not results:
        return ["No context retrieved."]

    contexts: list[str] = []
    for item in results:
        source = Path((item.get("source") or "unknown")).name
        chunk_id = item.get("chunk_id", "?")
        backend = item.get("backend", "unknown")
        score = item.get("score")
        text = str(item.get("page_content", "")).strip().replace("\n", " ")
        contexts.append(
            f"[source={source} | chunk_id={chunk_id} | backend={backend} | score={score}] {text}"
        )
    return contexts


def preview_results(results: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    """Return a trimmed preview of retrieval results for the JSON report.

    Caps the result list at limit entries and truncates each chunk's
    page_content to 300 characters so the JSON report stays readable
    without dragging the full corpus along — the SQLite ledger keeps the
    full text in the agent's tool-message history if a deeper inspection
    is needed.
    """
    return [
        {
            "source": Path((item.get("source") or "unknown")).name,
            "chunk_id": item.get("chunk_id"),
            "backend": item.get("backend"),
            "score": item.get("score"),
            "snippet": str(item.get("page_content", ""))[:300],
        }
        for item in results[:limit]
    ]


# ---------------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------------

def safe_mean(values: list[float | int | None], decimals: int = 3) -> float:
    """Return the mean of non-None values, or 0.0 if the list is empty.

    decimals controls the rounding precision.  The default of 3 is
    only a fallback for callers that don't have a metric key on hand —
    metric-specific callers (ReportManager.build_summary,
    EvaluationEngine.evaluate_case) pass metric_decimals()[key]
    so storage precision flows from the registry, not from this default.
    """
    numeric = [float(value) for value in values if value is not None]
    return round(sum(numeric) / len(numeric), decimals) if numeric else 0.0
