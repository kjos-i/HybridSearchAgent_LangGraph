"""Deterministic metric functions for the hybrid search evaluation flow.

Each ``compute_*`` function takes an ``EvalCase`` (or an answer string) plus
the retrieval results and returns a single score.  Two parallel families
are provided:

- **Source-level** (``compute_hit_at_k``, ``compute_mrr``,
  ``compute_precision_at_k``, ``compute_recall_at_k``,
  ``compute_ndcg_at_k``) — a result is *relevant* when its source filename
  matches one of ``case.expected_sources``.
- **Chunk-level** (``compute_chunk_*`` variants) — a result is *relevant*
  when a normalized snippet from ``case.expected_chunks`` appears as a
  substring of the result's ``page_content``.  Snippet matching is used
  instead of ``chunk_id`` because chunk IDs change silently whenever the
  chunking strategy is tuned, whereas a short representative snippet of
  text is far more stable across re-chunking.

The auxiliary ``compute_metadata_match_ratio`` and
``compute_backend_distribution`` helpers and the answer-side
``compute_keyword_checks`` live here too since they are all
deterministic metrics computed directly from retrieval / answer data.
"""

from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path
from typing import Any

from eval_models import EvalCase


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(value: str) -> str:
    """Lowercase, strip accents, and collapse punctuation for substring checks.

    Used both for keyword matching against the agent's answer and for
    snippet matching against retrieved chunk contents.
    """
    if not value:
        return ""
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    ascii_value = ascii_value.lower()
    ascii_value = re.sub(r"[^a-z0-9.%\-\s]", " ", ascii_value)
    return re.sub(r"\s+", " ", ascii_value).strip()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _expected_source_set(case: EvalCase) -> set[str]:
    """Lowercased filename set from ``case.expected_sources``."""
    return {Path(source).name.lower() for source in case.expected_sources}


def _result_source(item: dict[str, Any]) -> str:
    """Return the lowercased filename of a retrieval result."""
    return Path((item.get("source") or "unknown")).name.lower()


def _chunk_content_normalized(item: dict[str, Any]) -> str:
    """Return the normalized page_content of a retrieval result."""
    return normalize_text(str(item.get("page_content", "")))


def _chunk_relevance_flags(
    results: list[dict[str, Any]],
    expected_snippets_normalized: list[str],
) -> list[bool]:
    """Per-result binary relevance flags for chunk matching.

    A result is relevant when any of the expected snippets appears as a
    substring of its normalized ``page_content``.
    """
    return [
        any(snip in _chunk_content_normalized(item) for snip in expected_snippets_normalized)
        for item in results
    ]


def _ndcg_from_relevance(relevance: list[float]) -> float:
    """Normalized Discounted Cumulative Gain from a binary relevance vector."""
    dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevance))
    ideal = sorted(relevance, reverse=True)
    idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal))
    return round(dcg / idcg, 3) if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Source-level retrieval metrics (toggle_group="source")
# ---------------------------------------------------------------------------

def compute_hit_at_k(case: EvalCase, results: list[dict[str, Any]]) -> float:
    """Hit@k: 1.0 if at least one expected source appears in the retrieved results, else 0.0.

    Binary health check — complements recall@k, which reports the fraction of
    expected sources found. Returns 1.0 when no expected sources are defined.
    """
    if not case.expected_sources:
        return 1.0
    expected = _expected_source_set(case)
    actual = {_result_source(item) for item in results}
    return 1.0 if expected & actual else 0.0


def compute_mrr(case: EvalCase, results: list[dict[str, Any]]) -> float:
    """Mean Reciprocal Rank: 1 / rank of the first relevant result.

    Scores 1.0 when the top result is from an expected source, 0.5 when it is
    second, 0.33 when third, and 0.0 when no expected source appears at all.
    Returns 1.0 when no expected sources are defined (nothing to check).
    """
    if not case.expected_sources:
        return 1.0
    expected = _expected_source_set(case)
    for rank, item in enumerate(results, start=1):
        if _result_source(item) in expected:
            return round(1.0 / rank, 3)
    return 0.0


def compute_precision_at_k(case: EvalCase, results: list[dict[str, Any]]) -> float:
    """Precision@k: fraction of retrieved results that come from an expected source.

    Returns 1.0 when no expected sources are defined (nothing to check).
    """
    if not case.expected_sources or not results:
        return 1.0
    expected = _expected_source_set(case)
    hits = sum(1 for item in results if _result_source(item) in expected)
    return round(hits / len(results), 3)


def compute_recall_at_k(case: EvalCase, results: list[dict[str, Any]]) -> float:
    """Recall@k: fraction of expected sources found in the top-k retrieved results.

    Returns 1.0 when no expected sources are defined.
    """
    if not case.expected_sources:
        return 1.0
    if not results:
        return 0.0
    expected = _expected_source_set(case)
    found = {_result_source(item) for item in results}
    return round(len(expected & found) / len(expected), 3)


def compute_ndcg_at_k(case: EvalCase, results: list[dict[str, Any]]) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Uses binary relevance: a result is relevant (1) if its source file matches
    one of the expected sources, irrelevant (0) otherwise.
    Returns 1.0 when no expected sources are defined.
    """
    if not case.expected_sources:
        return 1.0
    if not results:
        return 0.0
    expected = _expected_source_set(case)
    relevance = [1.0 if _result_source(item) in expected else 0.0 for item in results]
    return _ndcg_from_relevance(relevance)


# ---------------------------------------------------------------------------
# Chunk-level retrieval metrics (toggle_group="chunk")
#
# A retrieved chunk is *relevant* when a normalized snippet from
# ``case.expected_chunks`` appears as a substring of the chunk's
# ``page_content``.  Returns 1.0 whenever ``case.expected_chunks`` is
# empty — mirroring the source-level fallback, so chunk metrics are
# effectively opt-in per test case.
# ---------------------------------------------------------------------------

def compute_chunk_hit_at_k(case: EvalCase, results: list[dict[str, Any]]) -> float:
    """Chunk Hit@k: 1.0 if any expected snippet is found in any retrieved chunk."""
    if not case.expected_chunks:
        return 1.0
    snippets = [normalize_text(s) for s in case.expected_chunks if s]
    if not snippets:
        return 1.0
    return 1.0 if any(_chunk_relevance_flags(results, snippets)) else 0.0


def compute_chunk_mrr(case: EvalCase, results: list[dict[str, Any]]) -> float:
    """Chunk MRR: 1 / rank of the first retrieved chunk matching an expected snippet."""
    if not case.expected_chunks:
        return 1.0
    snippets = [normalize_text(s) for s in case.expected_chunks if s]
    if not snippets:
        return 1.0
    for rank, is_relevant in enumerate(_chunk_relevance_flags(results, snippets), start=1):
        if is_relevant:
            return round(1.0 / rank, 3)
    return 0.0


def compute_chunk_precision_at_k(case: EvalCase, results: list[dict[str, Any]]) -> float:
    """Chunk Precision@k: fraction of retrieved chunks matching any expected snippet."""
    if not case.expected_chunks or not results:
        return 1.0
    snippets = [normalize_text(s) for s in case.expected_chunks if s]
    if not snippets:
        return 1.0
    flags = _chunk_relevance_flags(results, snippets)
    return round(sum(flags) / len(flags), 3)


def compute_chunk_recall_at_k(case: EvalCase, results: list[dict[str, Any]]) -> float:
    """Chunk Recall@k: fraction of expected snippets found in at least one retrieved chunk."""
    if not case.expected_chunks:
        return 1.0
    snippets = [normalize_text(s) for s in case.expected_chunks if s]
    if not snippets:
        return 1.0
    if not results:
        return 0.0
    contents = [_chunk_content_normalized(item) for item in results]
    covered = sum(1 for snip in snippets if any(snip in content for content in contents))
    return round(covered / len(snippets), 3)


def compute_chunk_ndcg_at_k(case: EvalCase, results: list[dict[str, Any]]) -> float:
    """Chunk NDCG@k: ranking quality using binary relevance at the chunk level."""
    if not case.expected_chunks:
        return 1.0
    snippets = [normalize_text(s) for s in case.expected_chunks if s]
    if not snippets:
        return 1.0
    if not results:
        return 0.0
    relevance = [1.0 if flag else 0.0 for flag in _chunk_relevance_flags(results, snippets)]
    return _ndcg_from_relevance(relevance)


# ---------------------------------------------------------------------------
# Auxiliary retrieval metrics (always-on — not toggleable)
# ---------------------------------------------------------------------------

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
# Answer-side metrics
# ---------------------------------------------------------------------------

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
