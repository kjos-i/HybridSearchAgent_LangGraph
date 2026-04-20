"""Utility helpers for the hybrid search evaluation flow.

Deterministic metric functions live in ``eval_metrics.py`` — this module
keeps only the non-metric plumbing: data loading, prompt construction,
LangGraph message extraction, DeepEval context builders, the retrieval
preview formatter, and generic math helpers.
"""

from __future__ import annotations

import json
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
# Math utilities
# ---------------------------------------------------------------------------

def safe_mean(values: list[float | int | None]) -> float:
    """Return the mean of non-None values, or 0.0 if the list is empty."""
    numeric = [float(value) for value in values if value is not None]
    return round(sum(numeric) / len(numeric), 3) if numeric else 0.0
