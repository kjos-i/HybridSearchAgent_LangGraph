"""Deterministic metric functions for the hybrid search evaluation flow.

Each compute_* function takes an EvalCase (or an answer string) plus
the retrieval results and returns a single score.  Two parallel families
are provided:

- **Source-level** (compute_hit_at_k, compute_mrr,
  compute_precision_at_k, compute_recall_at_k,
  compute_ndcg_at_k) — a result is *relevant* when its source filename
  matches one of case.expected_sources.
- **Chunk-level** (compute_chunk_* variants) — a result is *relevant*
  when a normalized snippet from case.expected_chunks appears as a
  substring of the result's page_content.  Snippet matching is used
  instead of chunk_id because chunk IDs change silently whenever the
  chunking strategy is tuned, whereas a short representative snippet of
  text is far more stable across re-chunking.

The auxiliary compute_metadata_match_ratio and
compute_backend_distribution helpers and the answer-side
compute_keyword_checks live here too since they are all
deterministic metrics computed directly from retrieval / answer data.

Module layout: public compute_* API at the top, then the batch
compute_all_chunk_metrics helper, then private metric helpers at the
bottom (_expected_source_set and friends) — top-down style so that
readers see the public surface before the internals.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from eval_metric_registry import metric_decimals
from eval_models import EvalCase
from eval_utils import normalize_text

# Storage-precision lookup, resolved once at import time so the per-call
# overhead is a single dict access.  Each round() site below pulls its
# decimals via this map so adding/renaming a metric in the registry flows
# through the compute functions without a parallel edit here.
_DECIMALS: dict[str, int] = metric_decimals()


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
            return round(1.0 / rank, _DECIMALS["mrr"])
    return 0.0


def compute_precision_at_k(case: EvalCase, results: list[dict[str, Any]]) -> float:
    """Precision@k: fraction of retrieved results that come from an expected source.

    Returns 1.0 when no expected sources are defined (nothing to check).
    """
    if not case.expected_sources or not results:
        return 1.0
    expected = _expected_source_set(case)
    hits = sum(1 for item in results if _result_source(item) in expected)
    return round(hits / len(results), _DECIMALS["precision_at_k"])


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
    return round(len(expected & found) / len(expected), _DECIMALS["recall_at_k"])


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
    return _ndcg_from_relevance(relevance, _DECIMALS["ndcg_at_k"])


# ---------------------------------------------------------------------------
# Chunk-level retrieval metrics (toggle_group="chunk")
#
# A retrieved chunk is *relevant* when a normalized snippet from
# case.expected_chunks appears as a substring of the chunk's
# page_content.  Returns 1.0 whenever case.expected_chunks is
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
            return round(1.0 / rank, _DECIMALS["chunk_mrr"])
    return 0.0


def compute_chunk_precision_at_k(case: EvalCase, results: list[dict[str, Any]]) -> float:
    """Chunk Precision@k: fraction of retrieved chunks matching any expected snippet."""
    if not case.expected_chunks or not results:
        return 1.0
    snippets = [normalize_text(s) for s in case.expected_chunks if s]
    if not snippets:
        return 1.0
    flags = _chunk_relevance_flags(results, snippets)
    return round(sum(flags) / len(flags), _DECIMALS["chunk_precision_at_k"])


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
    return round(covered / len(snippets), _DECIMALS["chunk_recall_at_k"])


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
    return _ndcg_from_relevance(relevance, _DECIMALS["chunk_ndcg_at_k"])


def compute_all_chunk_metrics(case: EvalCase, results: list[dict[str, Any]]) -> dict[str, float]:
    """Compute all five chunk-level metrics in a single pass.

    The five compute_chunk_* helpers each independently normalize
    snippets and recompute the per-result relevance flag list — fine for
    unit tests, but redundant when the engine wants every metric for the
    same case.  This batch variant prepares the snippets and flags once
    and reuses them across all five outputs, then the engine only has to
    look up each value.

    Returns a dict keyed by the registry metric keys
    (chunk_hit_at_k, chunk_mrr, chunk_precision_at_k,
    chunk_recall_at_k, chunk_ndcg_at_k).
    """
    # Mirror the per-function fallbacks: when there are no expected chunks
    # (or all snippets normalize to empty), every metric returns 1.0 so the
    # case isn't penalized for an opt-in metric it didn't opt into.
    if not case.expected_chunks:
        return {
            "chunk_hit_at_k":       1.0,
            "chunk_mrr":            1.0,
            "chunk_precision_at_k": 1.0,
            "chunk_recall_at_k":    1.0,
            "chunk_ndcg_at_k":      1.0,
        }
    snippets = [normalize_text(s) for s in case.expected_chunks if s]
    if not snippets:
        return {
            "chunk_hit_at_k":       1.0,
            "chunk_mrr":            1.0,
            "chunk_precision_at_k": 1.0,
            "chunk_recall_at_k":    1.0,
            "chunk_ndcg_at_k":      1.0,
        }
    if not results:
        # No retrieved results: precision is 1.0 (vacuously — nothing
        # irrelevant was returned), the rest are 0.0.  Matches the
        # individual functions.
        return {
            "chunk_hit_at_k":       0.0,
            "chunk_mrr":            0.0,
            "chunk_precision_at_k": 1.0,
            "chunk_recall_at_k":    0.0,
            "chunk_ndcg_at_k":      0.0,
        }

    flags = _chunk_relevance_flags(results, snippets)

    # hit_at_k: any flag set?
    hit = 1.0 if any(flags) else 0.0

    # mrr: 1 / rank of first relevant flag.
    mrr = 0.0
    for rank, flag in enumerate(flags, start=1):
        if flag:
            mrr = round(1.0 / rank, _DECIMALS["chunk_mrr"])
            break

    # precision@k: relevant fraction of retrieved.
    precision = round(sum(flags) / len(flags), _DECIMALS["chunk_precision_at_k"])

    # recall@k: snippet coverage requires checking each snippet against the
    # full content set (a flag tells you whether *some* snippet matched a
    # given result, not which one), so this stays a separate pass.
    contents = [_chunk_content_normalized(item) for item in results]
    covered = sum(1 for snip in snippets if any(snip in content for content in contents))
    recall = round(covered / len(snippets), _DECIMALS["chunk_recall_at_k"])

    # ndcg@k: binary relevance from the same flags.
    relevance = [1.0 if flag else 0.0 for flag in flags]
    ndcg = _ndcg_from_relevance(relevance, _DECIMALS["chunk_ndcg_at_k"])

    return {
        "chunk_hit_at_k":       hit,
        "chunk_mrr":            mrr,
        "chunk_precision_at_k": precision,
        "chunk_recall_at_k":    recall,
        "chunk_ndcg_at_k":      ndcg,
    }


# ---------------------------------------------------------------------------
# Auxiliary retrieval metrics (always-on — not toggleable)
# ---------------------------------------------------------------------------

def compute_backend_distribution(results: list[dict[str, Any]]) -> dict[str, int]:
    """Count retrieved results bucketed by search backend label.

    Returns a {backend_name: count} dict — diagnostic, not a score.
    A healthy hybrid run should show non-zero counts under both fts
    and one of the vector variants; a single-backend distribution
    suggests the fusion mechanism failed silently.  Results without a
    backend field (or with a falsy value) are bucketed under
    "unknown" so the bookkeeping never silently drops rows.
    """
    distribution: dict[str, int] = {}
    for item in results:
        backend = str(item.get("backend") or "unknown")
        distribution[backend] = distribution.get(backend, 0) + 1
    return distribution


def compute_metadata_match_ratio(case: EvalCase, results: list[dict[str, Any]]) -> float:
    """Fraction of retrieved results that satisfy every metadata filter on the case.

    Each filter is a {key: expected_value} pair on case.metadata_filters.
    A result matches when every filter key resolves to the expected value
    on the result — looked up first as a top-level key on the result dict,
    then under a nested "metadata" sub-dict so retrievers that flatten
    metadata and retrievers that nest it both score correctly.
    Comparison is string-based so numeric-vs-string mismatches don't fail
    spuriously.  Returns 1.0 when the case defines no filters (vacuous
    pass) and 0.0 when the case defines filters but no results came back.
    """
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
    return round(matches / len(results), _DECIMALS["metadata_match_ratio"])


# ---------------------------------------------------------------------------
# Answer-side metrics
# ---------------------------------------------------------------------------

def compute_keyword_checks(case: EvalCase, answer: str) -> dict[str, Any]:
    """Check required and disallowed keywords against the agent's answer.

    Returns a composite dict with two fields:

    - required_keyword_hit_rate: fraction of case.required_keywords
      that appear (as substrings, after normalization) in the answer.
      Vacuously 1.0 when the case has no required keywords.
    - disallowed_keyword_hits: count of case.disallowed_keywords
      that appear in the answer.  Target is 0 — the keyword gate flips
      the case to REVIEW on any non-zero count.

    Both keyword lists and the answer are run through normalize_text
    before substring matching, so case, accents, and most punctuation are
    ignored.  An empty answer scores 0.0 on the hit rate when keywords
    are required (a clear failure signal) and 1.0 otherwise.
    """
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
            _DECIMALS["required_keyword_hit_rate"],
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
# Internal helpers
# ---------------------------------------------------------------------------

def _expected_source_set(case: EvalCase) -> set[str]:
    """Lowercased filename set from case.expected_sources."""
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
    substring of its normalized page_content.
    """
    return [
        any(snip in _chunk_content_normalized(item) for snip in expected_snippets_normalized)
        for item in results
    ]


def _ndcg_from_relevance(relevance: list[float], decimals: int) -> float:
    """Normalized Discounted Cumulative Gain from a binary relevance vector.

    decimals is supplied by each caller from the registry — the source-
    level and chunk-level NDCG callers each look up their own metric's
    decimals so this shared helper stays metric-agnostic.
    """
    dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevance))
    ideal = sorted(relevance, reverse=True)
    idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal))
    return round(dcg / idcg, decimals) if idcg > 0 else 0.0
