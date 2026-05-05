"""Single source of truth for every metric in the evaluation harness.

Other modules import helpers from here instead of hardcoding metric names,
display labels, SQL columns, or CSV fieldnames.  To rename, add, or remove
a metric, edit the METRICS list below — every consumer derives its
lists from this registry automatically.

Renaming a metric
------------------
Change the key and/or label in the METRICS list.  These files
pick up the change automatically (no edits needed):

- eval_dashboard.py      — column lists, display labels, formatting
- eval_report_manager.py — CSV fieldnames, summary averages, print output
- eval_sqlite.py         — schema generation, INSERT statements, auto-migration

These files still need a manual update:

- eval_engine.py     — the dict key in build_metrics() (LLM metrics)
                           and the result dict in evaluate_case()
- eval_metrics.py    — the compute function name / return key
                           (deterministic metrics)
- eval_engine.py     — verdict gate logic in _compute_case_status() if
                           the renamed metric is one of the gate metrics

Adding a metric
---------------
1. Add a MetricDef entry to the METRICS list below.
2. Write the compute function in eval_metrics.py (deterministic) or add
   the DeepEval class to build_metrics() in eval_engine.py (LLM).
3. Add the result to the return dict in evaluate_case()
   (eval_engine.py).
4. For composite metrics, set composite=True and pass
   composite_sql_columns=(...) on the MetricDef.  Handle the
   per-case extraction in _extract_case_metric_values() in
   eval_sqlite.py (and update _KNOWN_COMPOSITE_COLUMNS so the
   import-time drift check accepts the new column).

Everything else (dashboard, CSV, SQLite schema, summaries) updates
automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

MetricGroup = Literal[
    "llm", "retrieval", "chunk", "keyword", "summary",
    "tokens", "judge_tokens", "latency",
]
ToggleGroup = Literal["judge", "source", "chunk"]


# Central defaults — every consumer (dashboard helpers, format/round helpers
# in the engine and metrics modules) imports these so there is exactly one
# place to change the program-wide fallbacks.  Keeping them as module-level
# constants (rather than inline = ".2f" defaults on the dataclass field)
# means callers like _fmt_for and format_metric_value can reuse the
# same fallback without re-declaring it.
DEFAULT_FMT: str = ".2f"
DEFAULT_DECIMALS: int = 3


@dataclass(frozen=True, slots=True)
class MetricDef:
    """Definition of a single evaluation metric."""

    key: str
    """Internal dict key used in the JSON report, e.g. "faithfulness"."""

    label: str
    """Human-readable display label, e.g. "Faithfulness"."""

    group: MetricGroup
    """Display/family classification (used by the dashboard and CSV layout)."""

    sql_column: str | None = None
    """Column name in the eval_cases table.  None for composites that
    are exploded into multiple columns (backend_distribution, keyword_checks)."""

    sql_type: str = "REAL"
    """SQL column type (used when generating schema)."""

    fmt: str = DEFAULT_FMT
    """Python format spec for display."""

    decimals: int = DEFAULT_DECIMALS
    """Number of decimal places for round() when computing/storing the
    raw value (independent from fmt, which controls display).  Storage
    precision and display precision are split deliberately: storage
    typically keeps an extra decimal so that downstream aggregation
    (e.g. avg_judge_score) doesn't compound display rounding.  Pure
    integer metrics (token counts, backend distribution counts) set
    decimals=0."""

    summary_avg_key: str | None = None
    """Key in the eval_runs table for the run-level average.
    None when no run-level average is stored."""

    summary_avg_fmt: str | None = None
    """Format spec for the run-level summary_avg_key value when it must
    differ from fmt.  Used when the per-case value is an integer
    (fmt="d") but the run-level mean is a float — e.g. token counts
    use ".0f" so the run-level mean renders as a whole number without
    crashing on a float input.  None falls back to fmt."""

    summary_avg_label: str | None = None
    """Display label for summary_avg_key when it must differ from the
    parent metric's label.  None falls back to label (the default
    behavior every other metric uses, where per-case and run-level views
    share one display name).  Only set this when the run-level aggregate
    must be visually distinguished from the per-case value in the same
    dashboard view — e.g. avg_judge_score per-case vs.
    avg_judge_run_score (the cross-case mean of the same field)."""

    composite: bool = False
    """True for metrics whose value is a dict rather than a scalar
    (backend_distribution, keyword_checks).  These are not stored as a
    single SQL column — each sub-field gets its own column."""

    composite_sql_columns: tuple[tuple[str, str], ...] | None = None
    """For composite metrics, the (column_name, sql_type) pairs that the
    composite expands into when generating the eval_cases schema and CSV
    columns.  None for non-composites.  Stored as a tuple of tuples (not
    a list) because MetricDef is frozen — mutable defaults aren't allowed.
    """

    toggle_group: ToggleGroup | None = None
    """Which user-facing toggle controls whether this metric is computed.
    None means always-on regardless of ENABLED_METRIC_GROUPS in
    eval_config.py. Toggleable values are "judge" (DeepEval LLM
    metrics), "source" (source-level retrieval quality), and "chunk"
    (chunk-level retrieval quality, reserved for a future feature)."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METRICS: list[MetricDef] = [
    # ── Run-level structural metric (one value per run) ──────────────────
    # pass_rate lives in the fixed preamble of eval_runs rather than
    # in run_sql_columns() (no summary_avg_key), and is per-run
    # rather than per-case, so it has no sql_column here.  Registered
    # purely for label and fmt lookup — .0% makes 0.85 render
    # as 85% without callers hardcoding the spec.
    MetricDef(key="pass_rate", label="Pass Rate", group="summary", fmt=".0%"),

    # ── LLM-judged (DeepEval) — toggle_group="judge" ──────────────────────
    # decimals=4 on every LLM metric: scores are stored at higher
    # precision than fmt=".2f" displays so the run-level mean
    # (avg_judge_score) doesn't compound display rounding when
    # multiplied by 100.
    MetricDef(key="answer_relevancy",     label="Answer Relevancy",      group="llm", sql_column="answer_relevancy",     toggle_group="judge", decimals=4),
    MetricDef(key="faithfulness",         label="Faithfulness",           group="llm", sql_column="faithfulness",         toggle_group="judge", decimals=4),
    MetricDef(key="contextual_precision", label="Ctx Precision",         group="llm", sql_column="contextual_precision", toggle_group="judge", decimals=4),
    MetricDef(key="contextual_recall",    label="Ctx Recall",            group="llm", sql_column="contextual_recall",    toggle_group="judge", decimals=4),
    MetricDef(key="contextual_relevancy", label="Ctx Relevancy",         group="llm", sql_column="contextual_relevancy", toggle_group="judge", decimals=4),
    MetricDef(key="hallucination",        label="Hallucination",          group="llm", sql_column="hallucination",        toggle_group="judge", decimals=4),
    MetricDef(key="correctness_g_eval",   label="Correctness (GEval)",   group="llm", sql_column="correctness_g_eval",   toggle_group="judge", decimals=4),

    # ── Deterministic retrieval — source-level (toggle_group="source") ──
    MetricDef(key="hit_at_k",             label="Hit@k",            group="retrieval", sql_column="hit_at_k",             summary_avg_key="avg_hit_at_k",           toggle_group="source"),
    MetricDef(key="mrr",                  label="MRR",             group="retrieval", sql_column="mrr",                  summary_avg_key="avg_mrr",                toggle_group="source"),
    MetricDef(key="precision_at_k",       label="Precision@k",     group="retrieval", sql_column="precision_at_k",       summary_avg_key="avg_precision_at_k",     toggle_group="source"),
    MetricDef(key="recall_at_k",          label="Recall@k",        group="retrieval", sql_column="recall_at_k",          summary_avg_key="avg_recall_at_k",        toggle_group="source"),
    MetricDef(key="ndcg_at_k",            label="NDCG@k",          group="retrieval", sql_column="ndcg_at_k",            summary_avg_key="avg_ndcg_at_k",          toggle_group="source"),
    # ── Deterministic retrieval — always-on auxiliary ────────────────────
    MetricDef(key="metadata_match_ratio", label="Metadata Match",  group="retrieval", sql_column="metadata_match_ratio", summary_avg_key="avg_metadata_match_ratio"),
    MetricDef(
        key="backend_distribution",
        label="Backend Distribution",
        group="retrieval",
        composite=True,
        # backend_other is a catch-all so a future backend label
        # (anything that isn't fts / vector / hybrid) lands somewhere
        # visible instead of being silently dropped from the ledger.
        composite_sql_columns=(
            ("backend_fts",    "INTEGER"),
            ("backend_vector", "INTEGER"),
            ("backend_hybrid", "INTEGER"),
            ("backend_other",  "INTEGER"),
        ),
    ),

    # ── Deterministic retrieval — chunk-level (toggle_group="chunk") ─────
    MetricDef(key="chunk_hit_at_k",       label="Chunk Hit@k",        group="chunk", sql_column="chunk_hit_at_k",       summary_avg_key="avg_chunk_hit_at_k",       toggle_group="chunk"),
    MetricDef(key="chunk_mrr",            label="Chunk MRR",          group="chunk", sql_column="chunk_mrr",            summary_avg_key="avg_chunk_mrr",            toggle_group="chunk"),
    MetricDef(key="chunk_precision_at_k", label="Chunk Precision@k", group="chunk", sql_column="chunk_precision_at_k", summary_avg_key="avg_chunk_precision_at_k", toggle_group="chunk"),
    MetricDef(key="chunk_recall_at_k",    label="Chunk Recall@k",    group="chunk", sql_column="chunk_recall_at_k",    summary_avg_key="avg_chunk_recall_at_k",    toggle_group="chunk"),
    MetricDef(key="chunk_ndcg_at_k",      label="Chunk NDCG@k",      group="chunk", sql_column="chunk_ndcg_at_k",      summary_avg_key="avg_chunk_ndcg_at_k",      toggle_group="chunk"),

    # ── Deterministic keyword / answer ────────────────────────────────────
    MetricDef(
        key="keyword_checks",
        label="Keyword Checks",
        group="keyword",
        composite=True,
        composite_sql_columns=(
            ("required_keyword_hit_rate", "REAL"),
            ("disallowed_keyword_hits",   "INTEGER"),
        ),
    ),
    MetricDef(
        key="avg_judge_score",
        label="Avg Judge Score",
        group="summary",
        sql_column="avg_judge_score",
        summary_avg_key="avg_judge_run_score",
        summary_avg_label="Avg Judge Score (Run)",
        fmt=".2f",
        # 0–1 scale (mean of the DeepEval LLM-judge scores, all on 0–1).
        # Default decimals=3 matches the rest of the 0–1 metric family
        # — display .2f then rounds to 2 decimals at render time.
    ),

    # ── Agent token consumption — always-on, integer per-case / float mean ──
    MetricDef(key="agent_input_tokens",  label="Agent Input Tokens",  group="tokens", sql_column="agent_input_tokens",  sql_type="INTEGER", fmt="d", decimals=0, summary_avg_key="avg_agent_input_tokens",  summary_avg_fmt=".0f"),
    MetricDef(key="agent_output_tokens", label="Agent Output Tokens", group="tokens", sql_column="agent_output_tokens", sql_type="INTEGER", fmt="d", decimals=0, summary_avg_key="avg_agent_output_tokens", summary_avg_fmt=".0f"),
    MetricDef(key="agent_total_tokens",  label="Agent Total Tokens",  group="tokens", sql_column="agent_total_tokens",  sql_type="INTEGER", fmt="d", decimals=0, summary_avg_key="avg_agent_total_tokens",  summary_avg_fmt=".0f"),

    # ── Judge token consumption — None when judge group is disabled ───────
    MetricDef(key="judge_input_tokens",  label="Judge Input Tokens",  group="judge_tokens", sql_column="judge_input_tokens",  sql_type="INTEGER", fmt="d", decimals=0, summary_avg_key="avg_judge_input_tokens",  summary_avg_fmt=".0f"),
    MetricDef(key="judge_output_tokens", label="Judge Output Tokens", group="judge_tokens", sql_column="judge_output_tokens", sql_type="INTEGER", fmt="d", decimals=0, summary_avg_key="avg_judge_output_tokens", summary_avg_fmt=".0f"),
    MetricDef(key="judge_total_tokens",  label="Judge Total Tokens",  group="judge_tokens", sql_column="judge_total_tokens",  sql_type="INTEGER", fmt="d", decimals=0, summary_avg_key="avg_judge_total_tokens",  summary_avg_fmt=".0f"),

    # ── Latency ───────────────────────────────────────────────────────────
    # latency_seconds and retrieval_latency_seconds are read from
    # perf_counter() and stored at 2 dp.  llm_latency_seconds is a
    # derived subtraction kept at 3 dp so the floor at 0.0 doesn't strip
    # sub-10ms differences when retrieval ≈ total.
    MetricDef(key="latency_seconds",            label="Latency",            group="latency", sql_column="latency_seconds",            summary_avg_key="avg_latency_seconds",            fmt=".2f", decimals=2),
    MetricDef(key="retrieval_latency_seconds",  label="Retrieval Latency",  group="latency", sql_column="retrieval_latency_seconds",  summary_avg_key="avg_retrieval_latency_seconds",  fmt=".2f", decimals=2),
    MetricDef(key="llm_latency_seconds",        label="LLM Latency",        group="latency", sql_column="llm_latency_seconds",        summary_avg_key="avg_llm_latency_seconds",        fmt=".2f", decimals=3),
]

# ---------------------------------------------------------------------------
# Helper functions — every consumer imports from here
# ---------------------------------------------------------------------------

def llm_metric_keys() -> list[str]:
    """Return the ordered keys of every LLM-judged metric in registry order."""
    return [m.key for m in METRICS if m.group == "llm"]


def retrieval_metric_keys() -> list[str]:
    """Return the ordered keys of every scalar retrieval metric.

    Composite metrics (backend_distribution) are excluded because they
    don't fit a single column — callers that need their sub-columns iterate
    MetricDef.composite_sql_columns directly.
    """
    return [m.key for m in METRICS if m.group == "retrieval" and not m.composite]


def chunk_metric_keys() -> list[str]:
    """Ordered keys for the scalar chunk-level retrieval metrics."""
    return [m.key for m in METRICS if m.group == "chunk" and not m.composite]


def tokens_metric_keys() -> list[str]:
    """Ordered keys for the agent token consumption metrics."""
    return [m.key for m in METRICS if m.group == "tokens"]


def judge_tokens_metric_keys() -> list[str]:
    """Ordered keys for the judge token consumption metrics."""
    return [m.key for m in METRICS if m.group == "judge_tokens"]


def keys_in_toggle_group(toggle_group: str) -> list[str]:
    """Return all metric keys assigned to the given toggle_group."""
    return [m.key for m in METRICS if m.toggle_group == toggle_group]


def metric_labels() -> dict[str, str]:
    """Return the key -> display label mapping consumed by the dashboard.

    Includes three sets of entries so any column the dashboard might
    encounter resolves to a human-readable label:

    - Per-case keys (faithfulness -> "Faithfulness").
    - Summary keys (avg_hit_at_k -> "Hit@k"), preferring
      summary_avg_label when set so the run-level view can be
      visually distinguished from the per-case view.
    - Fixed structural columns (case_id, category, status)
      that don't fit MetricDef but still appear in tables.
    """
    labels: dict[str, str] = {m.key: m.label for m in METRICS}
    for m in METRICS:
        # SQL column name -> label (e.g. avg_judge_score -> Avg Judge Score).
        if m.sql_column and m.sql_column != m.key:
            labels[m.sql_column] = m.label
        # Summary avg key -> label (e.g. avg_hit_at_k -> Hit@k).  Prefer the
        # explicit summary_avg_label when the run-level view needs to be
        # visually distinguished from the per-case label; otherwise reuse
        # the parent label so per-case and run-level views share a name.
        if m.summary_avg_key:
            labels[m.summary_avg_key] = m.summary_avg_label or m.label
    # Fixed structural columns displayed in tables (no per-case / run-level
    # aggregation pattern, so they don't fit MetricDef).  pass_rate was
    # in this list before it earned its own MetricDef — its label now flows
    # through the m.key -> m.label loop above.
    labels["case_id"] = "Case ID"
    labels["category"] = "Category"
    labels["status"] = "Status"
    return labels


def metric_fmts() -> dict[str, str]:
    """key -> format spec mapping for display formatting.

    Mirrors metric_labels(): the per-case fmt is also written under
    each summary_avg_key and each composite sub-column name, so callers
    that resolve formats for run-level columns (e.g. avg_hit_at_k) or
    composite sub-fields (e.g. required_keyword_hit_rate) hit the
    registry-declared format instead of a fallback default.
    """
    fmts: dict[str, str] = {m.key: m.fmt for m in METRICS}
    for m in METRICS:
        # Prefer summary_avg_fmt so integer per-case metrics
        # (fmt="d") can render their float run-level mean as
        # .0f without crashing the formatter.
        if m.summary_avg_key:
            fmts[m.summary_avg_key] = m.summary_avg_fmt or m.fmt
        if m.composite_sql_columns:
            for col, _ in m.composite_sql_columns:
                fmts[col] = m.fmt
    return fmts


def metric_decimals() -> dict[str, int]:
    """key -> int mapping for round() precision.

    Mirrors metric_fmts(): every metric's decimals is also written
    under its summary_avg_key and each composite sub-column name, so
    callers in the engine, metric compute functions, and report builders
    can resolve storage precision dynamically without hardcoding the int
    literal at the call site.  Used to drive round() calls so that
    e.g. renaming a metric or tightening its precision flows through the
    whole pipeline from one registry edit.
    """
    decimals: dict[str, int] = {m.key: m.decimals for m in METRICS}
    for m in METRICS:
        if m.summary_avg_key:
            decimals[m.summary_avg_key] = m.decimals
        if m.composite_sql_columns:
            for col, _ in m.composite_sql_columns:
                decimals[col] = m.decimals
    return decimals


# ── SQL helpers ───────────────────────────────────────────────────────────


def case_sql_columns() -> list[tuple[str, str]]:
    """Ordered (column_name, sql_type) pairs for the metric-related
    columns in eval_cases.

    Composites are exploded into their concrete sub-columns via the
    composite_sql_columns field on each MetricDef — adding a new
    composite metric just ships its expansion alongside its definition,
    no special-casing here.  Does NOT include the fixed preamble columns
    (case_row_id, run_id, etc.) or trailing text columns.
    """
    columns: list[tuple[str, str]] = []
    for m in METRICS:
        if m.composite:
            if m.composite_sql_columns:
                columns.extend(m.composite_sql_columns)
        elif m.sql_column:
            columns.append((m.sql_column, m.sql_type))
    return columns


def run_sql_columns() -> list[tuple[str, str]]:
    """Ordered (column_name, sql_type) pairs for the average-metric
    columns in eval_runs.  Does NOT include fixed preamble or trailing
    columns."""
    return [
        (m.summary_avg_key, "REAL")
        for m in METRICS
        if m.summary_avg_key
    ]


# ── Report / CSV helpers ─────────────────────────────────────────────────

def summary_avg_pairs() -> list[tuple[str, str]]:
    """(summary_avg_key, source_key) pairs for building run summaries.

    Example: ("avg_hit_at_k", "hit_at_k").
    """
    return [(m.summary_avg_key, m.key) for m in METRICS if m.summary_avg_key]


def csv_fieldnames() -> list[str]:
    """Return the ordered column names for the per-case CSV export.

    Column order is fixed structural preamble (id, category,
    status) → registry-derived metric columns → trailing diagnostics
    (error_count).  case_sql_columns() returns (name, sql_type)
    pairs and CSVs are stringly typed, so the SQL type is ignored here —
    only the column name matters.
    """
    preamble = ["id", "category", "status"]
    metric_cols = [col for col, _ in case_sql_columns()]
    trailing = ["error_count"]
    return preamble + metric_cols + trailing
