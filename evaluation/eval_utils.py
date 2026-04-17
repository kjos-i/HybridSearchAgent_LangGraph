"""Utility helpers for the DeepEval-based hybrid search evaluation flow."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from eval_models import EvalCase


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cases(dataset_path: Path) -> list[EvalCase]:
    """Load and validate evaluation cases from a JSON file."""
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
    """Build the full prompt string sent to the agent for an eval case."""
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
    """Extract plain text from a LangChain message content field."""
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
    """Extract retrieval results from ToolMessage objects in the agent's message history."""
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
    """Join expected answer points into a single string for DeepEval's expected_output field."""
    if case.expected_answer_points:
        return " ".join(case.expected_answer_points)
    return "Provide a corpus-grounded answer only."


def build_gold_context(case: EvalCase) -> list[str]:
    """Build the gold-standard context list passed to DeepEval's context field."""
    if case.expected_answer_points:
        return case.expected_answer_points

    fallback = [build_expected_output(case)]
    fallback.extend(f"Relevant source: {source}" for source in case.expected_sources)
    return fallback


def build_retrieval_context(results: list[dict[str, Any]]) -> list[str]:
    """Format raw retrieval results into annotated strings for DeepEval's retrieval_context field."""
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
    """Return a trimmed preview of retrieval results for inclusion in the JSON report."""
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
# Retrieval quality metrics
# ---------------------------------------------------------------------------

def compute_source_hit_rate(case: EvalCase, results: list[dict[str, Any]]) -> float:
    """Fraction of expected sources that appear anywhere in the retrieved results."""
    if not case.expected_sources:
        return 1.0
    expected = {Path(source).name.lower() for source in case.expected_sources}
    actual = {Path((item.get("source") or "unknown")).name.lower() for item in results}
    return round(len(expected & actual) / max(len(expected), 1), 3)


def compute_mrr(case: EvalCase, results: list[dict[str, Any]]) -> float:
    """Mean Reciprocal Rank: 1 / rank of the first relevant result.

    Scores 1.0 when the top result is from an expected source, 0.5 when it is
    second, 0.33 when third, and 0.0 when no expected source appears at all.
    Returns 1.0 when no expected sources are defined (nothing to check).
    """
    if not case.expected_sources:
        return 1.0
    expected = {Path(source).name.lower() for source in case.expected_sources}
    for rank, item in enumerate(results, start=1):
        source = Path((item.get("source") or "unknown")).name.lower()
        if source in expected:
            return round(1.0 / rank, 3)
    return 0.0


def compute_precision_at_k(case: EvalCase, results: list[dict[str, Any]]) -> float:
    """Precision@k: fraction of retrieved results that come from an expected source.

    Returns 1.0 when no expected sources are defined (nothing to check).
    """
    if not case.expected_sources or not results:
        return 1.0
    expected = {Path(source).name.lower() for source in case.expected_sources}
    hits = sum(
        1 for item in results
        if Path((item.get("source") or "unknown")).name.lower() in expected
    )
    return round(hits / len(results), 3)


def compute_recall_at_k(case: EvalCase, results: list[dict[str, Any]]) -> float:
    """Recall@k: fraction of expected sources found in the top-k retrieved results.

    Returns 1.0 when no expected sources are defined.
    """
    if not case.expected_sources:
        return 1.0
    if not results:
        return 0.0
    expected = {Path(source).name.lower() for source in case.expected_sources}
    found = {Path((item.get("source") or "unknown")).name.lower() for item in results}
    return round(len(expected & found) / len(expected), 3)


def compute_ndcg_at_k(case: EvalCase, results: list[dict[str, Any]]) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Uses binary relevance: a result is relevant (1) if its source file matches
    one of the expected sources, irrelevant (0) otherwise.
    NDCG compares the actual ranking against the ideal ranking where all
    relevant results sit at the top.
    Returns 1.0 when no expected sources are defined.
    """
    if not case.expected_sources or not results:
        return 1.0 if not case.expected_sources else 0.0

    expected = {Path(source).name.lower() for source in case.expected_sources}

    relevance = [
        1.0 if Path((item.get("source") or "unknown")).name.lower() in expected else 0.0
        for item in results
    ]

    # DCG
    dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevance))

    # Ideal DCG — all relevant results pushed to the top
    ideal = sorted(relevance, reverse=True)
    idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal))

    return round(dcg / idcg, 3) if idcg > 0 else 0.0


def compute_backend_distribution(results: list[dict[str, Any]]) -> dict[str, int]:
    """Count retrieved results by search backend (fts, vector, hybrid).

    A non-empty mix of both fts and vector indicates the fusion is actively
    contributing both backends to the final result set.
    """
    distribution: dict[str, int] = {}
    for item in results:
        backend = str(item.get("backend") or "unknown")
        distribution[backend] = distribution.get(backend, 0) + 1
    return distribution


def compute_metadata_match_ratio(case: EvalCase, results: list[dict[str, Any]]) -> float:
    """Fraction of retrieved results that satisfy all metadata filters defined on the case."""
    if not case.metadata_filters:
        return 1.0
    if not results:
        return 0.0

    matches = sum(
        1 for result in results
        if all(
            str(result.get(key, (result.get("metadata") or {}).get(key))) == str(expected)
            for key, expected in case.metadata_filters.items()
        )
    )
    return round(matches / len(results), 3)


# ---------------------------------------------------------------------------
# Answer quality metrics
# ---------------------------------------------------------------------------

def normalize_text(value: str) -> str:
    """Lowercase, strip accents, and remove punctuation for keyword comparison."""
    if not value:
        return ""
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    ascii_value = ascii_value.lower()
    ascii_value = re.sub(r"[^a-z0-9.%\-\s]", " ", ascii_value)
    return re.sub(r"\s+", " ", ascii_value).strip()


def compute_keyword_checks(case: EvalCase, answer: str) -> dict[str, Any]:
    """Check required and disallowed keywords against the agent's answer."""
    if not answer:
        return {
            "required_keyword_hit_rate": 0.0 if case.required_keywords else 1.0,
            "disallowed_keyword_hits": 0,
        }

    normalized_answer = normalize_text(answer)

    required_hit_rate = (
        round(
            sum(1 for kw in case.required_keywords if normalize_text(kw) in normalized_answer)
            / len(case.required_keywords),
            3,
        )
        if case.required_keywords
        else 1.0
    )

    disallowed_hits = sum(
        1 for kw in case.disallowed_keywords if normalize_text(kw) in normalized_answer
    )

    return {
        "required_keyword_hit_rate": required_hit_rate,
        "disallowed_keyword_hits": disallowed_hits,
    }


# ---------------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------------

def safe_mean(values: list[float | int | None]) -> float:
    """Return the mean of non-None values, or 0.0 if the list is empty."""
    numeric = [float(value) for value in values if value is not None]
    return round(sum(numeric) / len(numeric), 3) if numeric else 0.0
