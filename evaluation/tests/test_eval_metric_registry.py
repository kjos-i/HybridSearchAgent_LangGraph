"""Consistency tests for the metric registry.

These tests guard the invariants that other modules silently rely on
(every per-case key has a format spec, every composite has matching
SQL columns, etc.). A regression here would otherwise manifest as
missing dashboard cards or KeyErrors in CSV / SQLite serialisation.
"""

from __future__ import annotations

from eval_metric_registry import (
    METRICS,
    case_sql_columns,
    chunk_metric_keys,
    csv_fieldnames,
    judge_tokens_metric_keys,
    keys_in_toggle_group,
    llm_metric_keys,
    metric_decimals,
    metric_fmts,
    metric_labels,
    retrieval_metric_keys,
    run_sql_columns,
    summary_avg_pairs,
    tokens_metric_keys,
)


# ---------------------------------------------------------------------------
# Uniqueness invariants
# ---------------------------------------------------------------------------

def test_all_metric_keys_unique():
    keys = [m.key for m in METRICS]
    assert len(keys) == len(set(keys)), "Duplicate MetricDef.key"


def test_summary_avg_keys_are_unique():
    avg_keys = [m.summary_avg_key for m in METRICS if m.summary_avg_key]
    assert len(avg_keys) == len(set(avg_keys)), "Duplicate summary_avg_key"


def test_sql_columns_are_unique():
    cols = [name for name, _ in case_sql_columns()]
    assert len(cols) == len(set(cols)), "Duplicate case SQL column"


# ---------------------------------------------------------------------------
# Label / format coverage
# ---------------------------------------------------------------------------

def test_every_metric_has_label_and_fmt():
    labels = metric_labels()
    fmts = metric_fmts()
    for metric in METRICS:
        assert metric.key in labels, f"{metric.key} missing from metric_labels()"
        assert metric.key in fmts, f"{metric.key} missing from metric_fmts()"


def test_summary_avg_keys_have_fmt_label_and_decimals():
    fmts = metric_fmts()
    labels = metric_labels()
    decimals = metric_decimals()
    for metric in METRICS:
        if metric.summary_avg_key:
            assert metric.summary_avg_key in fmts
            assert metric.summary_avg_key in labels
            assert metric.summary_avg_key in decimals


def test_composite_sub_columns_have_fmt_and_decimals():
    fmts = metric_fmts()
    decimals = metric_decimals()
    for metric in METRICS:
        if metric.composite_sql_columns:
            for col, _ in metric.composite_sql_columns:
                assert col in fmts, f"composite sub-column {col!r} missing from metric_fmts()"
                assert col in decimals, f"composite sub-column {col!r} missing from metric_decimals()"


def test_summary_avg_label_falls_back_to_parent_label():
    labels = metric_labels()
    for metric in METRICS:
        if not metric.summary_avg_key:
            continue
        expected = metric.summary_avg_label or metric.label
        assert labels[metric.summary_avg_key] == expected


# ---------------------------------------------------------------------------
# Composite invariants
# ---------------------------------------------------------------------------

def test_composite_metrics_declare_columns():
    for metric in METRICS:
        if metric.composite:
            assert metric.composite_sql_columns, (
                f"{metric.key} is composite but has no composite_sql_columns"
            )


def test_composite_metrics_have_no_sql_column():
    # Composites are exploded into multiple sub-columns; the parent metric
    # itself doesn't get a column — case_sql_columns() relies on this.
    for metric in METRICS:
        if metric.composite:
            assert metric.sql_column is None


# ---------------------------------------------------------------------------
# Toggle-group invariants
# ---------------------------------------------------------------------------

def test_llm_metric_keys_all_in_judge_toggle_group():
    for key in llm_metric_keys():
        metric = next(m for m in METRICS if m.key == key)
        assert metric.toggle_group == "judge"


def test_retrieval_metric_keys_belong_to_retrieval_group():
    # retrieval_metric_keys() includes both source-toggle metrics
    # (hit_at_k, mrr, …) and the always-on auxiliary metadata_match_ratio
    # so the dashboard's source-retrieval radar can render them together.
    # The invariant is that every returned key is in the retrieval family
    # and not a composite (composites are exploded separately).
    for key in retrieval_metric_keys():
        metric = next(m for m in METRICS if m.key == key)
        assert metric.group == "retrieval"
        assert not metric.composite


def test_chunk_metric_keys_all_in_chunk_toggle_group():
    for key in chunk_metric_keys():
        metric = next(m for m in METRICS if m.key == key)
        assert metric.toggle_group == "chunk"


def test_keys_in_toggle_group_match_toggle_field():
    for group in ("judge", "source", "chunk"):
        expected = [m.key for m in METRICS if m.toggle_group == group]
        assert keys_in_toggle_group(group) == expected


def test_token_metrics_are_always_on():
    # Token-tracking metrics should never be toggleable — they're cost
    # diagnostics that always run regardless of which metric groups
    # the user enables.
    for key in tokens_metric_keys() + judge_tokens_metric_keys():
        metric = next(m for m in METRICS if m.key == key)
        assert metric.toggle_group is None


# ---------------------------------------------------------------------------
# CSV / SQL plumbing
# ---------------------------------------------------------------------------

def test_csv_fieldnames_are_unique():
    fields = csv_fieldnames()
    assert len(fields) == len(set(fields)), "Duplicate CSV fieldnames"


def test_csv_fieldnames_include_every_metric_column():
    fields = set(csv_fieldnames())
    for col, _ in case_sql_columns():
        assert col in fields, f"case column {col!r} missing from csv_fieldnames()"


def test_run_sql_columns_match_summary_pairs():
    declared = {col for col, _ in run_sql_columns()}
    pair_keys = {avg_key for avg_key, _ in summary_avg_pairs()}
    assert pair_keys == declared, (
        "Mismatch between summary_avg_pairs() and run_sql_columns(). "
        f"Pairs only: {pair_keys - declared}; columns only: {declared - pair_keys}"
    )


def test_summary_avg_pairs_source_keys_resolve():
    by_key = {m.key: m for m in METRICS}
    for _avg_key, src_key in summary_avg_pairs():
        assert src_key in by_key, f"summary_avg_pair source {src_key!r} is not a registered metric"


# ---------------------------------------------------------------------------
# Sanity checks on specific metrics
# ---------------------------------------------------------------------------

def test_pass_rate_format_is_percent():
    fmts = metric_fmts()
    assert fmts["pass_rate"].endswith("%"), (
        "pass_rate is conceptually a fraction — its display format "
        "should be a percent so 0.85 renders as 85%."
    )


def test_avg_judge_score_has_distinct_run_label():
    # Per-case avg_judge_score and run-level avg_judge_run_score
    # render side-by-side on the dashboard, so they need distinct labels.
    labels = metric_labels()
    assert labels["avg_judge_score"] != labels["avg_judge_run_score"]


def test_judge_metrics_use_higher_storage_precision():
    # Judge scores roll up into avg_judge_score; storing each at extra
    # precision (decimals=4) keeps the mean from compounding display rounding.
    for key in llm_metric_keys():
        metric = next(m for m in METRICS if m.key == key)
        assert metric.decimals >= 4, (
            f"Judge metric {key!r} should store at decimals>=4 so the "
            f"avg_judge_score rollup doesn't lose precision; got {metric.decimals}"
        )


def test_token_metrics_store_as_integers():
    # Tokens are integer counts — fmt 'd' and decimals 0 keep them that way.
    for key in tokens_metric_keys() + judge_tokens_metric_keys():
        metric = next(m for m in METRICS if m.key == key)
        assert metric.fmt == "d", f"{key!r} should display as integer (fmt='d')"
        assert metric.decimals == 0, f"{key!r} should have decimals=0"
        assert metric.sql_type == "INTEGER", f"{key!r} should be stored as INTEGER"
