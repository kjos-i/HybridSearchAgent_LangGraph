"""Unit tests for the deterministic compute_* functions in eval_metrics.

Covers source-level retrieval metrics (Hit@k, MRR, Precision@k,
Recall@k, NDCG@k), their chunk-level snippet-match analogues, the
single-pass batch helper compute_all_chunk_metrics (cross-checked
against the per-function helpers so the engine's shortcut and the
public per-metric API can't drift), and the auxiliary always-on metrics
(backend distribution, metadata match, keyword checks).  Each test pins
one branch of the metric — happy path, vacuous-pass fallback when the
case has no ground truth, and edge cases like empty result lists or
all-blank snippets.
"""

from __future__ import annotations

from eval_metrics import (
    compute_all_chunk_metrics,
    compute_backend_distribution,
    compute_chunk_hit_at_k,
    compute_chunk_mrr,
    compute_chunk_ndcg_at_k,
    compute_chunk_precision_at_k,
    compute_chunk_recall_at_k,
    compute_hit_at_k,
    compute_keyword_checks,
    compute_metadata_match_ratio,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
)
from eval_models import EvalCase


def _case(**overrides) -> EvalCase:
    """Build an EvalCase with sensible defaults; override any field with kwargs."""
    defaults = dict(id="t", question="q")
    defaults.update(overrides)
    return EvalCase(**defaults)


def _result(source: str, page_content: str = "", **extra) -> dict:
    """Build a retrieval result dict shaped like the ones the retriever returns."""
    return {"source": source, "page_content": page_content, **extra}


# ---------------------------------------------------------------------------
# Source-level retrieval metrics
# ---------------------------------------------------------------------------

class TestHitAtK:
    def test_no_expected_sources_passes_vacuously(self):
        case = _case()
        assert compute_hit_at_k(case, [_result("any.pdf")]) == 1.0

    def test_match_returns_one(self):
        case = _case(expected_sources=["a.pdf", "b.pdf"])
        assert compute_hit_at_k(case, [_result("a.pdf")]) == 1.0

    def test_no_match_returns_zero(self):
        case = _case(expected_sources=["a.pdf"])
        assert compute_hit_at_k(case, [_result("z.pdf")]) == 0.0

    def test_case_insensitive_filename(self):
        case = _case(expected_sources=["A.PDF"])
        assert compute_hit_at_k(case, [_result("a.pdf")]) == 1.0

    def test_path_strips_to_filename(self):
        # expected sources can include path; matched against basename only.
        case = _case(expected_sources=["docs/report/A.pdf"])
        assert compute_hit_at_k(case, [_result("other/path/A.pdf")]) == 1.0

    def test_empty_results(self):
        case = _case(expected_sources=["a.pdf"])
        assert compute_hit_at_k(case, []) == 0.0


class TestMRR:
    def test_first_result_relevant(self):
        case = _case(expected_sources=["a.pdf"])
        assert compute_mrr(case, [_result("a.pdf"), _result("b.pdf")]) == 1.0

    def test_second_result_relevant(self):
        case = _case(expected_sources=["a.pdf"])
        assert compute_mrr(case, [_result("z.pdf"), _result("a.pdf")]) == 0.5

    def test_third_result_relevant(self):
        case = _case(expected_sources=["a.pdf"])
        results = [_result("x.pdf"), _result("y.pdf"), _result("a.pdf")]
        # 1/3 rounded to registry precision (3 decimals) = 0.333.
        assert compute_mrr(case, results) == 0.333

    def test_no_match_returns_zero(self):
        case = _case(expected_sources=["a.pdf"])
        assert compute_mrr(case, [_result("z.pdf")]) == 0.0

    def test_no_expected_sources_passes_vacuously(self):
        case = _case()
        assert compute_mrr(case, [_result("any.pdf")]) == 1.0


class TestPrecisionAtK:
    def test_all_relevant(self):
        case = _case(expected_sources=["a.pdf", "b.pdf"])
        results = [_result("a.pdf"), _result("b.pdf")]
        assert compute_precision_at_k(case, results) == 1.0

    def test_half_relevant(self):
        case = _case(expected_sources=["a.pdf"])
        results = [_result("a.pdf"), _result("z.pdf")]
        assert compute_precision_at_k(case, results) == 0.5

    def test_none_relevant(self):
        case = _case(expected_sources=["a.pdf"])
        assert compute_precision_at_k(case, [_result("z.pdf")]) == 0.0

    def test_no_results(self):
        case = _case(expected_sources=["a.pdf"])
        assert compute_precision_at_k(case, []) == 1.0

    def test_no_expected_sources(self):
        assert compute_precision_at_k(_case(), [_result("any.pdf")]) == 1.0


class TestRecallAtK:
    def test_all_expected_found(self):
        case = _case(expected_sources=["a.pdf", "b.pdf"])
        results = [_result("a.pdf"), _result("b.pdf"), _result("c.pdf")]
        assert compute_recall_at_k(case, results) == 1.0

    def test_partial(self):
        case = _case(expected_sources=["a.pdf", "b.pdf"])
        assert compute_recall_at_k(case, [_result("a.pdf")]) == 0.5

    def test_duplicate_relevant_does_not_double_count(self):
        # Recall measures distinct expected files, not retrieved chunks.
        case = _case(expected_sources=["a.pdf", "b.pdf"])
        results = [_result("a.pdf"), _result("a.pdf"), _result("a.pdf")]
        assert compute_recall_at_k(case, results) == 0.5

    def test_no_results(self):
        case = _case(expected_sources=["a.pdf"])
        assert compute_recall_at_k(case, []) == 0.0

    def test_no_expected_sources(self):
        assert compute_recall_at_k(_case(), [_result("any.pdf")]) == 1.0


class TestNDCGAtK:
    def test_perfect_ranking(self):
        case = _case(expected_sources=["a.pdf", "b.pdf"])
        results = [_result("a.pdf"), _result("b.pdf")]
        assert compute_ndcg_at_k(case, results) == 1.0

    def test_irrelevant_in_middle_drops_score(self):
        case = _case(expected_sources=["a.pdf", "b.pdf"])
        results = [_result("a.pdf"), _result("z.pdf"), _result("b.pdf")]
        score = compute_ndcg_at_k(case, results)
        assert 0.0 < score < 1.0

    def test_no_match_returns_zero(self):
        case = _case(expected_sources=["a.pdf"])
        assert compute_ndcg_at_k(case, [_result("z.pdf")]) == 0.0

    def test_no_results(self):
        assert compute_ndcg_at_k(_case(expected_sources=["a.pdf"]), []) == 0.0

    def test_no_expected_sources(self):
        assert compute_ndcg_at_k(_case(), [_result("any.pdf")]) == 1.0


# ---------------------------------------------------------------------------
# Chunk-level retrieval metrics
# ---------------------------------------------------------------------------

class TestChunkHitAtK:
    def test_snippet_in_chunk_returns_one(self):
        case = _case(expected_chunks=["regulation 42"])
        results = [_result("a.pdf", page_content="See regulation 42 for details.")]
        assert compute_chunk_hit_at_k(case, results) == 1.0

    def test_snippet_missing_returns_zero(self):
        case = _case(expected_chunks=["regulation 42"])
        results = [_result("a.pdf", page_content="Unrelated content.")]
        assert compute_chunk_hit_at_k(case, results) == 0.0

    def test_normalization_handles_accents(self):
        case = _case(expected_chunks=["cafe"])
        results = [_result("a.pdf", page_content="The café opened today.")]
        assert compute_chunk_hit_at_k(case, results) == 1.0

    def test_no_expected_chunks_passes_vacuously(self):
        assert compute_chunk_hit_at_k(_case(), [_result("a.pdf", "content")]) == 1.0

    def test_empty_snippet_list_passes_vacuously(self):
        # Whitespace-only snippets get filtered out, leaving nothing to check.
        assert compute_chunk_hit_at_k(_case(expected_chunks=["", "  "]), [_result("a.pdf")]) == 1.0


class TestChunkMRR:
    def test_first_chunk_matches(self):
        case = _case(expected_chunks=["foo"])
        results = [_result("a.pdf", "foo bar"), _result("b.pdf", "baz")]
        assert compute_chunk_mrr(case, results) == 1.0

    def test_second_chunk_matches(self):
        case = _case(expected_chunks=["foo"])
        results = [_result("a.pdf", "baz"), _result("b.pdf", "foo bar")]
        assert compute_chunk_mrr(case, results) == 0.5

    def test_no_chunk_matches(self):
        case = _case(expected_chunks=["foo"])
        results = [_result("a.pdf", "baz")]
        assert compute_chunk_mrr(case, results) == 0.0

    def test_no_expected_chunks(self):
        assert compute_chunk_mrr(_case(), [_result("a.pdf", "anything")]) == 1.0


class TestChunkPrecisionAtK:
    def test_all_chunks_match(self):
        case = _case(expected_chunks=["foo"])
        results = [_result("a.pdf", "foo"), _result("b.pdf", "foo bar")]
        assert compute_chunk_precision_at_k(case, results) == 1.0

    def test_half_match(self):
        case = _case(expected_chunks=["foo"])
        results = [_result("a.pdf", "foo"), _result("b.pdf", "baz")]
        assert compute_chunk_precision_at_k(case, results) == 0.5

    def test_no_results(self):
        # Empty results => vacuously 1.0 (no precision to compute).
        case = _case(expected_chunks=["foo"])
        assert compute_chunk_precision_at_k(case, []) == 1.0


class TestChunkRecallAtK:
    def test_all_snippets_found(self):
        case = _case(expected_chunks=["foo", "bar"])
        results = [_result("a.pdf", "foo and bar")]
        assert compute_chunk_recall_at_k(case, results) == 1.0

    def test_partial_coverage(self):
        case = _case(expected_chunks=["foo", "bar"])
        results = [_result("a.pdf", "foo only")]
        assert compute_chunk_recall_at_k(case, results) == 0.5

    def test_distinct_snippets_in_same_chunk_count_separately(self):
        # Both snippets are covered by one chunk => recall 1.0 (snippets,
        # not chunks, are the unit of recall).
        case = _case(expected_chunks=["foo", "bar"])
        results = [_result("a.pdf", "foo and bar")]
        assert compute_chunk_recall_at_k(case, results) == 1.0

    def test_no_results(self):
        case = _case(expected_chunks=["foo"])
        assert compute_chunk_recall_at_k(case, []) == 0.0


class TestChunkNDCGAtK:
    def test_perfect_ranking(self):
        case = _case(expected_chunks=["foo"])
        results = [_result("a.pdf", "foo"), _result("b.pdf", "irrelevant")]
        assert compute_chunk_ndcg_at_k(case, results) == 1.0

    def test_no_match(self):
        case = _case(expected_chunks=["foo"])
        assert compute_chunk_ndcg_at_k(case, [_result("a.pdf", "baz")]) == 0.0

    def test_no_results(self):
        assert compute_chunk_ndcg_at_k(_case(expected_chunks=["foo"]), []) == 0.0


# ---------------------------------------------------------------------------
# Auxiliary metrics
# ---------------------------------------------------------------------------

class TestBackendDistribution:
    def test_counts_per_backend(self):
        results = [
            _result("a.pdf", backend="fts"),
            _result("b.pdf", backend="vector"),
            _result("c.pdf", backend="fts"),
        ]
        assert compute_backend_distribution(results) == {"fts": 2, "vector": 1}

    def test_unknown_backend_when_missing(self):
        # A result with no backend field counts under "unknown" so the
        # bookkeeping never silently drops rows.
        assert compute_backend_distribution([_result("a.pdf")]) == {"unknown": 1}

    def test_empty_results(self):
        assert compute_backend_distribution([]) == {}


class TestMetadataMatchRatio:
    def test_no_filters_passes_vacuously(self):
        case = _case()
        assert compute_metadata_match_ratio(case, [_result("a.pdf")]) == 1.0

    def test_all_match(self):
        case = _case(metadata_filters={"category": "policy"})
        results = [
            _result("a.pdf", category="policy"),
            _result("b.pdf", category="policy"),
        ]
        assert compute_metadata_match_ratio(case, results) == 1.0

    def test_partial_match(self):
        case = _case(metadata_filters={"category": "policy"})
        results = [
            _result("a.pdf", category="policy"),
            _result("b.pdf", category="other"),
        ]
        assert compute_metadata_match_ratio(case, results) == 0.5

    def test_filter_resolves_via_metadata_subdict(self):
        # Some retrievers nest fields under a metadata key — the matcher
        # falls back to that when the top-level key is missing.
        case = _case(metadata_filters={"category": "policy"})
        results = [_result("a.pdf", metadata={"category": "policy"})]
        assert compute_metadata_match_ratio(case, results) == 1.0

    def test_no_results(self):
        case = _case(metadata_filters={"category": "policy"})
        assert compute_metadata_match_ratio(case, []) == 0.0


# ---------------------------------------------------------------------------
# Keyword checks
# ---------------------------------------------------------------------------

class TestKeywordChecks:
    def test_all_required_hit(self):
        case = _case(required_keywords=["Acme", "product"])
        out = compute_keyword_checks(case, "Acme launched a product")
        assert out["required_keyword_hit_rate"] == 1.0
        assert out["disallowed_keyword_hits"] == 0

    def test_partial_required(self):
        case = _case(required_keywords=["Acme", "product"])
        out = compute_keyword_checks(case, "Acme launched")
        assert out["required_keyword_hit_rate"] == 0.5

    def test_no_required_keywords_passes_vacuously(self):
        out = compute_keyword_checks(_case(), "anything")
        assert out["required_keyword_hit_rate"] == 1.0

    def test_disallowed_count(self):
        case = _case(disallowed_keywords=["I cannot", "I don't have"])
        out = compute_keyword_checks(case, "I cannot answer that, I don't have access")
        assert out["disallowed_keyword_hits"] == 2

    def test_normalization_handles_accents(self):
        case = _case(required_keywords=["cafe"])
        out = compute_keyword_checks(case, "café opened")
        assert out["required_keyword_hit_rate"] == 1.0

    def test_empty_answer_with_required_keywords(self):
        # Empty answer can't satisfy any required keyword — rate is 0.
        case = _case(required_keywords=["Acme"])
        out = compute_keyword_checks(case, "")
        assert out["required_keyword_hit_rate"] == 0.0
        assert out["disallowed_keyword_hits"] == 0

    def test_empty_answer_no_required_keywords(self):
        # No required keywords => vacuously 1.0 even with empty answer.
        out = compute_keyword_checks(_case(), "")
        assert out["required_keyword_hit_rate"] == 1.0


# ---------------------------------------------------------------------------
# compute_all_chunk_metrics — single-pass batch helper
# ---------------------------------------------------------------------------

class TestComputeAllChunkMetrics:
    """The batch helper is what the engine actually calls. The individual
    compute_chunk_* functions are kept for unit-test ergonomics — these
    tests pin that the batch result matches what the individual helpers
    return for the same input, so the engine's single-pass shortcut can't
    silently drift from the per-metric semantics tested above.
    """

    EXPECTED_KEYS = {
        "chunk_hit_at_k", "chunk_mrr", "chunk_precision_at_k",
        "chunk_recall_at_k", "chunk_ndcg_at_k",
    }

    def test_returns_all_five_keys(self):
        case = _case(expected_chunks=["alpha"])
        out = compute_all_chunk_metrics(case, [_result("a.pdf", "alpha here")])
        assert set(out.keys()) == self.EXPECTED_KEYS

    def test_matches_individual_helpers(self):
        # The batch result must equal each individual compute_chunk_*
        # function for the same case + results, otherwise the engine's
        # shortcut and the public per-metric helpers disagree.
        case = _case(expected_chunks=["alpha", "gamma"])
        results = [
            _result("a.pdf", "alpha first"),
            _result("b.pdf", "irrelevant"),
            _result("c.pdf", "gamma here"),
        ]
        batch = compute_all_chunk_metrics(case, results)
        assert batch["chunk_hit_at_k"]       == compute_chunk_hit_at_k(case, results)
        assert batch["chunk_mrr"]            == compute_chunk_mrr(case, results)
        assert batch["chunk_precision_at_k"] == compute_chunk_precision_at_k(case, results)
        assert batch["chunk_recall_at_k"]    == compute_chunk_recall_at_k(case, results)
        assert batch["chunk_ndcg_at_k"]      == compute_chunk_ndcg_at_k(case, results)

    def test_no_expected_chunks_returns_all_ones(self):
        # Mirrors the per-function fallback: opt-in metric, vacuously perfect.
        out = compute_all_chunk_metrics(_case(), [_result("a.pdf", "anything")])
        assert all(value == 1.0 for value in out.values())

    def test_no_results(self):
        case = _case(expected_chunks=["alpha"])
        out = compute_all_chunk_metrics(case, [])
        assert out["chunk_hit_at_k"]       == 0.0
        assert out["chunk_mrr"]            == 0.0
        # Precision is vacuously 1.0 when nothing was retrieved (nothing
        # irrelevant came back). Matches the individual helper.
        assert out["chunk_precision_at_k"] == 1.0
        assert out["chunk_recall_at_k"]    == 0.0
        assert out["chunk_ndcg_at_k"]      == 0.0

    def test_all_blank_snippets(self):
        # If every snippet normalizes to empty, the batch falls back to
        # the same all-1.0 short-circuit as the per-function helpers.
        case = _case(expected_chunks=[""])
        out = compute_all_chunk_metrics(case, [_result("a.pdf", "anything")])
        assert all(value == 1.0 for value in out.values())
