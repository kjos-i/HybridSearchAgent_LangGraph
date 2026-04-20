"""Single source of truth for every metric in the evaluation harness.

Other modules import helpers from here instead of hardcoding metric names,
display labels, SQL columns, or CSV fieldnames. To rename, add, or remove a
metric, edit the ``METRICS`` list below — every consumer derives its lists
from this registry automatically.

Renaming a metric
------------------
Change the ``key``, ``label``, and/or ``fmt`` in the ``METRICS`` list.  These
files pick up the change automatically (no edits needed):

- eval_dashboard.py      — column lists, display labels, decimal formatting
- eval_report_manager.py — CSV fieldnames, summary averages, print output
- eval_sqlite.py         — schema generation, INSERT statements, auto-migration

These files still need a manual update:

- eval_engine.py         — the dict key in ``build_metrics()`` (LLM metrics)
                           and the result dict in ``evaluate_case()``
- eval_utils.py          — the compute function name / return key (deterministic
                           metrics)
- eval_engine.py         — verdict gate logic in ``compute_case_status()`` if
                           the renamed metric is one of the gate metrics

Adding a metric
---------------
1. Add a ``MetricDef`` entry to the ``METRICS`` list below.
2. Write the compute function in ``eval_utils.py`` (deterministic) or add
   the DeepEval class to ``build_metrics()`` in ``eval_engine.py`` (LLM).
3. Add the result to the return dict in ``evaluate_case()``
   (``eval_engine.py``).
4. For composite metrics, add the exploded sub-columns to
   ``_COMPOSITE_SQL_COLUMNS`` below and handle extraction in
   ``_extract_case_metric_values()`` in ``eval_sqlite.py``.

Everything else (dashboard, CSV, SQLite schema, summaries) updates
automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

MetricGroup = Literal["llm", "retrieval", "chunk", "keyword", "summary", "latency"]
ToggleGroup = Literal["judge", "source", "chunk"]


@dataclass(frozen=True, slots=True)
class MetricDef:
    """Definition of a single evaluation metric."""

    key: str
    """Internal dict key used in the JSON report, e.g. ``"faithfulness"``."""

    label: str
    """Human-readable display label, e.g. ``"Faithfulness"``."""

    group: MetricGroup
    """Display/family classification (used by the dashboard and CSV layout)."""

    sql_column: str | None = None
    """Column name in the ``eval_cases`` table.  ``None`` for composites that
    are exploded into multiple columns (backend_distribution, keyword_checks)."""

    sql_type: str = "REAL"
    """SQL column type (used when generating schema)."""

    fmt: str = ".3f"
    """Python format spec for display."""

    summary_avg_key: str | None = None
    """Key in the ``eval_runs`` table for the run-level average.
    ``None`` when no run-level average is stored."""

    composite: bool = False
    """``True`` for metrics whose value is a dict rather than a scalar
    (``backend_distribution``, ``keyword_checks``).  These are not stored as a
    single SQL column — each sub-field gets its own column."""

    toggle_group: ToggleGroup | None = None
    """Which user-facing toggle controls whether this metric is computed.
    ``None`` means always-on regardless of ``ENABLED_METRIC_GROUPS`` in
    ``eval_config.py``. Toggleable values are ``"judge"`` (DeepEval LLM
    metrics), ``"source"`` (source-level retrieval quality), and ``"chunk"``
    (chunk-level retrieval quality, reserved for a future feature)."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METRICS: list[MetricDef] = [
    # ── LLM-judged (DeepEval) — toggle_group="judge" ──────────────────────
    MetricDef(key="answer_relevancy",     label="Answer Relevancy",      group="llm", sql_column="answer_relevancy",     toggle_group="judge"),
    MetricDef(key="faithfulness",         label="Faithfulness",           group="llm", sql_column="faithfulness",         toggle_group="judge"),
    MetricDef(key="contextual_precision", label="Ctx Precision",         group="llm", sql_column="contextual_precision", toggle_group="judge"),
    MetricDef(key="contextual_recall",    label="Ctx Recall",            group="llm", sql_column="contextual_recall",    toggle_group="judge"),
    MetricDef(key="contextual_relevancy", label="Ctx Relevancy",         group="llm", sql_column="contextual_relevancy", toggle_group="judge"),
    MetricDef(key="hallucination",        label="Hallucination",          group="llm", sql_column="hallucination",        toggle_group="judge"),
    MetricDef(key="correctness_g_eval",   label="Correctness (GEval)",   group="llm", sql_column="correctness_g_eval",   toggle_group="judge"),

    # ── Deterministic retrieval — source-level (toggle_group="source") ──
    MetricDef(key="hit_at_k",             label="Hit@k",            group="retrieval", sql_column="hit_at_k",             summary_avg_key="avg_hit_at_k",           toggle_group="source"),
    MetricDef(key="mrr",                  label="MRR",             group="retrieval", sql_column="mrr",                  summary_avg_key="avg_mrr",                toggle_group="source"),
    MetricDef(key="precision_at_k",       label="Precision@k",     group="retrieval", sql_column="precision_at_k",       summary_avg_key="avg_precision_at_k",     toggle_group="source"),
    MetricDef(key="recall_at_k",          label="Recall@k",        group="retrieval", sql_column="recall_at_k",          summary_avg_key="avg_recall_at_k",        toggle_group="source"),
    MetricDef(key="ndcg_at_k",            label="NDCG@k",          group="retrieval", sql_column="ndcg_at_k",            summary_avg_key="avg_ndcg_at_k",          toggle_group="source"),
    # ── Deterministic retrieval — always-on auxiliary ────────────────────
    MetricDef(key="metadata_match_ratio", label="Metadata Match",  group="retrieval", sql_column="metadata_match_ratio", summary_avg_key="avg_metadata_match_ratio"),
    MetricDef(key="backend_distribution", label="Backend Distribution", group="retrieval", composite=True),

    # ── Deterministic retrieval — chunk-level (toggle_group="chunk") ─────
    MetricDef(key="chunk_hit_at_k",       label="Chunk Hit@k",        group="chunk", sql_column="chunk_hit_at_k",       summary_avg_key="avg_chunk_hit_at_k",       toggle_group="chunk"),
    MetricDef(key="chunk_mrr",            label="Chunk MRR",          group="chunk", sql_column="chunk_mrr",            summary_avg_key="avg_chunk_mrr",            toggle_group="chunk"),
    MetricDef(key="chunk_precision_at_k", label="Chunk Precision@k", group="chunk", sql_column="chunk_precision_at_k", summary_avg_key="avg_chunk_precision_at_k", toggle_group="chunk"),
    MetricDef(key="chunk_recall_at_k",    label="Chunk Recall@k",    group="chunk", sql_column="chunk_recall_at_k",    summary_avg_key="avg_chunk_recall_at_k",    toggle_group="chunk"),
    MetricDef(key="chunk_ndcg_at_k",      label="Chunk NDCG@k",      group="chunk", sql_column="chunk_ndcg_at_k",      summary_avg_key="avg_chunk_ndcg_at_k",      toggle_group="chunk"),

    # ── Deterministic keyword / answer ────────────────────────────────────
    MetricDef(key="keyword_checks",       label="Keyword Checks",       group="keyword", composite=True),
    MetricDef(key="avg_judge_score",      label="Avg Judge Score",       group="summary", sql_column="avg_judge_score", fmt=".1f"),

    # ── Latency ───────────────────────────────────────────────────────────
    MetricDef(key="latency_seconds",            label="Latency",            group="latency", sql_column="latency_seconds",            summary_avg_key="avg_latency_seconds",            fmt=".2f"),
    MetricDef(key="retrieval_latency_seconds",  label="Retrieval Latency",  group="latency", sql_column="retrieval_latency_seconds",  summary_avg_key="avg_retrieval_latency_seconds",  fmt=".2f"),
    MetricDef(key="llm_latency_seconds",        label="LLM Latency",        group="latency", sql_column="llm_latency_seconds",        summary_avg_key="avg_llm_latency_seconds",        fmt=".2f"),
]

# Build a fast lookup by key.
_BY_KEY: dict[str, MetricDef] = {m.key: m for m in METRICS}


# ---------------------------------------------------------------------------
# Helper functions — every consumer imports from here
# ---------------------------------------------------------------------------

def by_key(key: str) -> MetricDef:
    """Look up a single MetricDef by its key."""
    return _BY_KEY[key]


def by_group(group: MetricGroup) -> list[MetricDef]:
    """All MetricDefs in a given group."""
    return [m for m in METRICS if m.group == group]


def llm_metric_keys() -> list[str]:
    """Ordered keys for the LLM-judged metrics.

    Replaces the hardcoded ``METRIC_COLUMNS`` list in the dashboard.
    """
    return [m.key for m in METRICS if m.group == "llm"]


def retrieval_metric_keys() -> list[str]:
    """Ordered keys for the scalar retrieval metrics (excludes composites).

    Replaces the hardcoded ``RETRIEVAL_COLUMNS`` list in the dashboard.
    """
    return [m.key for m in METRICS if m.group == "retrieval" and not m.composite]


def chunk_metric_keys() -> list[str]:
    """Ordered keys for the scalar chunk-level retrieval metrics."""
    return [m.key for m in METRICS if m.group == "chunk" and not m.composite]


def enabled_metric_keys(enabled_groups: set[str]) -> set[str]:
    """Return the keys of every metric that should be computed for a run.

    A metric is included when either:
      - its ``toggle_group`` is ``None`` (always-on), or
      - its ``toggle_group`` is in ``enabled_groups``.
    """
    return {
        m.key for m in METRICS
        if m.toggle_group is None or m.toggle_group in enabled_groups
    }


def is_enabled(key: str, enabled_groups: set[str]) -> bool:
    """Check whether a single metric key is enabled for the given groups."""
    metric = _BY_KEY.get(key)
    if metric is None:
        return False
    return metric.toggle_group is None or metric.toggle_group in enabled_groups


def keys_in_toggle_group(toggle_group: str) -> list[str]:
    """Return all metric keys assigned to the given ``toggle_group``."""
    return [m.key for m in METRICS if m.toggle_group == toggle_group]


def metric_labels() -> dict[str, str]:
    """``key -> display label`` mapping.

    Also includes ``summary_avg_key -> label`` entries so that run-level
    columns (e.g. ``avg_hit_at_k``) resolve to human-readable names.
    Replaces the hardcoded ``METRIC_LABELS`` dict in the dashboard.
    """
    labels: dict[str, str] = {m.key: m.label for m in METRICS}
    for m in METRICS:
        # SQL column name -> label (e.g. avg_judge_score -> Avg Judge Score).
        if m.sql_column and m.sql_column != m.key:
            labels[m.sql_column] = m.label
        # Summary avg key -> label (e.g. avg_hit_at_k -> Hit@k).
        if m.summary_avg_key:
            labels[m.summary_avg_key] = m.label
    # Fixed structural columns displayed in tables.
    labels["case_id"] = "Case ID"
    labels["category"] = "Category"
    labels["status"] = "Status"
    labels["avg_case_score"] = "Avg Judge Score"
    labels["pass_rate"] = "Pass Rate"
    return labels


def metric_fmts() -> dict[str, str]:
    """``key -> Python format spec`` mapping, mirroring :func:`metric_labels`.

    Every metric's ``fmt`` is reachable under its ``key``, ``sql_column``, and
    ``summary_avg_key``. ``avg_case_score`` is aliased to ``avg_judge_score``
    so the per-run judge-score KPI picks up the same format.
    """
    fmts: dict[str, str] = {m.key: m.fmt for m in METRICS}
    for m in METRICS:
        if m.sql_column and m.sql_column != m.key:
            fmts[m.sql_column] = m.fmt
        if m.summary_avg_key:
            fmts[m.summary_avg_key] = m.fmt
    fmts["avg_case_score"] = _BY_KEY["avg_judge_score"].fmt
    return fmts


def metric_fmt(key: str, default: str = ".3f") -> str:
    """Return the display format for a single metric key (or alias)."""
    m = _BY_KEY.get(key)
    if m is not None:
        return m.fmt
    for metric in METRICS:
        if metric.sql_column == key or metric.summary_avg_key == key:
            return metric.fmt
    if key == "avg_case_score":
        return _BY_KEY["avg_judge_score"].fmt
    return default


# ── SQL helpers ───────────────────────────────────────────────────────────

# Composite metrics are exploded into these concrete SQL columns.
_COMPOSITE_SQL_COLUMNS: list[tuple[str, str]] = [
    # backend_distribution
    ("backend_fts",                "INTEGER"),
    ("backend_vector",             "INTEGER"),
    ("backend_hybrid",             "INTEGER"),
    # keyword_checks
    ("required_keyword_hit_rate",  "REAL"),
    ("disallowed_keyword_hits",    "INTEGER"),
]


def case_sql_columns() -> list[tuple[str, str]]:
    """Ordered ``(column_name, sql_type)`` pairs for the metric-related
    columns in ``eval_cases``.  Does NOT include the fixed preamble columns
    (``case_row_id``, ``run_id``, etc.) or trailing text columns."""
    cols: list[tuple[str, str]] = []
    for m in METRICS:
        if m.composite:
            cols.extend(
                (col, typ) for col, typ in _COMPOSITE_SQL_COLUMNS
                if (m.key == "backend_distribution" and col.startswith("backend_"))
                or (m.key == "keyword_checks" and col in ("required_keyword_hit_rate", "disallowed_keyword_hits"))
            )
        elif m.sql_column:
            cols.append((m.sql_column, m.sql_type))
    return cols


def run_sql_columns() -> list[tuple[str, str]]:
    """Ordered ``(column_name, sql_type)`` pairs for the average-metric
    columns in ``eval_runs``.  Does NOT include fixed preamble or trailing
    columns."""
    return [
        (m.summary_avg_key, "REAL")
        for m in METRICS
        if m.summary_avg_key
    ]


# ── Report / CSV helpers ─────────────────────────────────────────────────

def summary_avg_pairs() -> list[tuple[str, str]]:
    """``(summary_avg_key, source_key)`` pairs for building run summaries.

    Example: ``("avg_hit_at_k", "hit_at_k")``.
    """
    return [(m.summary_avg_key, m.key) for m in METRICS if m.summary_avg_key]


def csv_fieldnames() -> list[str]:
    """Ordered column names for the per-case CSV export."""
    preamble = ["id", "category", "status"]
    metric_cols = [col for col, _ in case_sql_columns()]
    trailing = ["error_count"]
    return preamble + metric_cols + trailing
