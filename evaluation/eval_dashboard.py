"""Streamlit dashboard for browsing evaluation runs stored in eval_ledger.db.

Layout: four tabs.

- Run Summary — KPI cards (pass rate, avg judge score, latency, tokens),
  per-case table with conditional colours, and an answer-detail expander
  per case.
- Deep Analysis — radar charts comparing metric balance, score-distribution
  histograms, a Pearson correlation heatmap, latency, and token usage,
  all scoped to the selected run.
- Historical Trends — multi-series Altair line charts that read every
  eval_runs row to track how the run-level averages move over time.
- Metrics Guide — embeds info_metrics.md so the in-app reference is
  always identical to the source-of-truth doc.

Auto-refresh: the dashboard polls MAX(run_timestamp) every
POLL_SECONDS seconds and busts the heavier load_runs /
load_cases caches only when a genuinely new run has arrived.

Launch from the project root (or the evaluation folder):

    streamlit run evaluation/eval_dashboard.py
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import altair as alt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from eval_metric_registry import (
    DEFAULT_FMT,
    chunk_metric_keys,
    judge_tokens_metric_keys,
    llm_metric_keys,
    metric_fmts,
    metric_labels,
    retrieval_metric_keys,
    tokens_metric_keys,
)

POLL_SECONDS = 10  # How often to check if new data has arrived

DB_PATH = Path(__file__).resolve().parent / "eval_ledger.db"

METRIC_COLUMNS: list[str] = llm_metric_keys()
RETRIEVAL_COLUMNS: list[str] = retrieval_metric_keys()
CHUNK_COLUMNS: list[str] = chunk_metric_keys()
TOKEN_COLUMNS: list[str] = tokens_metric_keys()
JUDGE_TOKEN_COLUMNS: list[str] = judge_tokens_metric_keys()
METRIC_LABELS: dict[str, str] = metric_labels()
METRIC_FMTS: dict[str, str] = metric_fmts()


def _fmt_for(col_name: str, default: str = DEFAULT_FMT) -> str:
    """Resolve a column's display format from the registry, falling back to default.

    default defaults to the registry-wide DEFAULT_FMT so callers
    that look up a column not present in the registry still match
    MetricDef.fmt's declared default — there is one source of truth
    for the program-wide fallback.
    """
    return METRIC_FMTS.get(col_name, default)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* Tab styling */
button[data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.95rem;
}

/* Expander header */
div[data-testid="stExpander"] summary {
    font-weight: 500;
}

/* Subtle section dividers */
hr {
    border-top: 1px solid #dee2e6 !important;
    margin: 1.5rem 0 !important;
}

/* White text for HTML-rendered table headers and index */
table th {
    color: #ffffff !important;
    font-weight: normal !important;
}

/* Shrink the metric value font so 7 cards fit on one row, and allow
   long labels to wrap instead of being clipped. Every nested wrapper
   inside ``stMetricLabel`` is overridden because the inner elements
   default to nowrap/ellipsis. ``min-height`` reserves space for a
   second line so single- and two-line cards align vertically. */
div[data-testid="stMetricValue"] {
    font-size: 1.75rem;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] *,
[data-testid="stMetric"] label,
[data-testid="stMetric"] label * {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    overflow-wrap: break-word !important;
    word-break: break-word !important;
    line-height: 1.2 !important;
}
[data-testid="stMetricLabel"] {
    min-height: 2.4em !important;
    max-width: 12ch !important;
}
</style>
"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=POLL_SECONDS)
def _get_latest_timestamp() -> str | None:
    """Return the most recent run_timestamp from the ledger.

    Cached with a short TTL so the heavier load functions are only busted
    when a genuinely new run has arrived, not on every rerun.
    """
    if not DB_PATH.exists():
        return None
    try:
        with sqlite3.connect(DB_PATH) as conn:
            result = conn.execute("SELECT MAX(run_timestamp) FROM eval_runs").fetchone()
            return result[0] if result else None
    except sqlite3.Error:
        logger.warning("Failed to read latest timestamp from %s", DB_PATH, exc_info=True)
        return None


@st.cache_data(show_spinner=False)
def load_runs() -> pd.DataFrame:
    """Load all evaluation runs from the SQLite ledger.

    Cached indefinitely; call load_runs.clear() to force a fresh read
    when new data is detected by the auto-refresh poll.
    """
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM eval_runs ORDER BY run_timestamp DESC", conn)
    if not df.empty:
        df["run_timestamp"] = pd.to_datetime(df["run_timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_cases() -> pd.DataFrame:
    """Load all per-case results from the SQLite ledger.

    Cached indefinitely; call load_cases.clear() to force a fresh read
    when new data is detected by the auto-refresh poll.
    """
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM eval_cases ORDER BY run_timestamp DESC, case_id", conn)
    if not df.empty:
        df["run_timestamp"] = pd.to_datetime(df["run_timestamp"])
    return df


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT_DEFAULTS: dict[str, object] = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", size=16),
    margin=dict(l=40, r=20, t=30, b=40),
)


def _hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    """Convert a hex colour like #636EFA to an rgba(...) string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_radar_chart(df: pd.DataFrame, metric_cols: list[str], color: str = "#636EFA") -> go.Figure:
    """Build a radar (polar) chart showing average scores across metrics.

    Plotly's hovertemplate takes a single format spec, so when the
    plotted metrics span multiple registered fmts we pick the first
    metric's fmt as a representative — every radar in this dashboard
    plots metrics from the same family (all 0–1 retrieval scores), so
    they share a fmt and the representative pick is exact.
    """
    available = [c for c in metric_cols if c in df.columns]
    if not available:
        return go.Figure()

    means = df[available].mean()
    labels = [METRIC_LABELS.get(c, c) for c in available]
    values = means.tolist()

    # Close the polygon
    labels.append(labels[0])
    values.append(values[0])

    value_fmt = _fmt_for(available[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill="toself",
        fillcolor=_hex_to_rgba(color, alpha=0.12),
        line=dict(color=color, width=2),
        marker=dict(size=5),
        hovertemplate=f"%{{theta}}: %{{r:{value_fmt}}}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12),
        polar=dict(
            radialaxis=dict(range=[0, 1], tickvals=[0.25, 0.5, 0.75, 1.0], gridcolor="#e9ecef"),
            angularaxis=dict(gridcolor="#e9ecef"),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=320,
        margin=dict(l=110, r=110, t=70, b=70),
    )
    return fig


def build_correlation_heatmap(df: pd.DataFrame, columns: list[str]) -> go.Figure:
    """Build a correlation heatmap between the given numeric columns.

    Metrics with zero variance (identical scores across all cases) produce
    NaN correlations.  These cells show the constant value (e.g. "= 1.00")
    so the user sees that the metric was perfect, not missing.
    """
    available = [c for c in columns if c in df.columns]
    if len(available) < 2:
        return go.Figure()

    corr = df[available].corr()
    labels = [METRIC_LABELS.get(c, c) for c in available]

    # Identify zero-variance metrics and their constant values.
    constant_metrics: dict[str, float] = {}
    for col in available:
        if df[col].std() == 0:
            constant_metrics[col] = df[col].iloc[0]

    # Build custom text: show the constant value for NaN cells.  The
    # constant is a metric value, so its display format follows the
    # registry; the non-NaN branch shows a Pearson correlation
    # coefficient (statistic, not a metric) and stays at 2 dp by chart
    # convention.
    def _cell_text(row_key: str, col_key: str, val: float) -> str:
        if pd.isna(val):
            const_key = row_key if row_key in constant_metrics else col_key
            return f"= {constant_metrics[const_key]:{_fmt_for(const_key)}}"
        return f"{val:.2f}"

    text_matrix = [
        [_cell_text(available[r], available[c], corr.iloc[r, c]) for c in range(len(available))]
        for r in range(len(available))
    ]

    # Replace NaN with 0 so the colour scale renders a neutral fill instead of
    # transparent/black holes.
    corr_filled = corr.fillna(0)

    fig = go.Figure(data=go.Heatmap(
        z=corr_filled.values,
        x=labels,
        y=labels,
        text=text_matrix,
        texttemplate="%{text}",
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12),
        height=480,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
    )

    if constant_metrics:
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=100),
            annotations=[dict(
                text="= X means the metric scored X on every case (zero variance) — correlation cannot be calculated",
                xref="paper", yref="paper",
                x=0, y=-0.28,
                xanchor="left",
                showarrow=False,
                font=dict(size=11, color="#8fa8c8"),
            )],
        )

    return fig


def build_stacked_latency(df: pd.DataFrame) -> go.Figure | None:
    """Build a stacked bar of retrieval vs LLM latency for every case in df.

    Returns None when neither latency column exists, so the caller
    can fall back to an info banner instead of rendering an empty chart.
    Long case_id labels are rotated to -45° rather than relying
    on Plotly's auto-rotation, which only kicks in when labels overflow
    the axis.
    """
    latency_cols = ["retrieval_latency_seconds", "llm_latency_seconds"]
    available = [c for c in latency_cols if c in df.columns]
    if not available:
        return None

    latency_data = df[["case_id"] + available].copy()
    melted = latency_data.melt(id_vars="case_id", var_name="component", value_name="seconds")
    melted["component"] = melted["component"].map({
        "retrieval_latency_seconds": "Retrieval",
        "llm_latency_seconds": "LLM Generation",
    })
    fig = px.bar(
        melted, x="case_id", y="seconds", color="component", barmode="stack",
        color_discrete_map={"Retrieval": "#636EFA", "LLM Generation": "#EF553B"},
    )
    # tickangle=-45 keeps long case_id labels readable instead of relying
    # on Plotly's auto-rotation, which only kicks in when labels overflow.
    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        xaxis_title="",
        yaxis_title="Seconds",
        xaxis=dict(tickangle=-45),
        yaxis=dict(gridcolor="#f0f0f0"),
        legend_title="Component",
        height=370,
        bargap=0.1,
    )
    return fig


def build_token_bar(df: pd.DataFrame) -> go.Figure | None:
    """Stacked bar of agent input vs output tokens per case (NULLs are dropped).

    Judge tokens are evaluation-only — they don't affect production
    runtime cost — so they're intentionally excluded from this view to
    keep the visual focused on the cost the agent actually incurs in
    serving traffic.  Returns None when no token columns are present
    or when every row is NULL.
    """
    token_cols = [c for c in ("agent_input_tokens", "agent_output_tokens") if c in df.columns]
    if not token_cols:
        return None

    data = df[["case_id"] + token_cols].copy()
    melted = data.melt(id_vars="case_id", var_name="component", value_name="tokens")
    melted = melted.dropna(subset=["tokens"])
    if melted.empty:
        return None
    melted["component"] = melted["component"].map({
        "agent_input_tokens":  "Input",
        "agent_output_tokens": "Output",
    }).fillna(melted["component"])

    fig = px.bar(
        melted, x="case_id", y="tokens", color="component", barmode="stack",
        color_discrete_map={"Input": "#636EFA", "Output": "#00CC96"},
    )
    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        xaxis_title="",
        yaxis_title="Tokens",
        xaxis=dict(tickangle=-45),
        yaxis=dict(gridcolor="#f0f0f0"),
        legend_title="Component",
        height=370,
        bargap=0.1,
    )
    return fig


def _color_by_thresholds(val: object, thresholds: list[tuple[float, str]], *, higher_is_better: bool = True) -> str:
    """Return a CSS color string based on threshold bands.

    Parameters
    ----------
    thresholds
        [(cutoff, css), ...] ordered from best to worst.  Each entry means
        "if the value is >= cutoff (or <= cutoff when higher_is_better=False),
        use this CSS".
    """
    if not isinstance(val, (int, float)) or pd.isna(val):
        return "color: #6b7280"
    for cutoff, css in thresholds:
        if (higher_is_better and val >= cutoff) or (not higher_is_better and val <= cutoff):
            return css
    return thresholds[-1][1] if thresholds else ""


# Re-usable threshold bands
_SCORE_BANDS: list[tuple[float, str]] = [
    (0.8, "color: #4ade80; font-weight: 600"),
    (0.5, "color: #fb923c; font-weight: 600"),
    (0.0, "color: #f87171; font-weight: 600"),
]
_STATUS_COLORS: dict[str, str] = {
    "PASS": "color: #4ade80; font-weight: bold",
    "REVIEW": "color: #f87171; font-weight: bold",
}


def _find_column(df: pd.DataFrame, *candidates: str) -> str | None:
    """Return the first column name from *candidates* that exists in *df*."""
    return next((c for c in candidates if c in df.columns), None)


def style_case_dataframe(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Apply conditional color formatting to metric and status columns.

    Colors only — value formatting (decimal precision, suffixes) is driven
    via st.column_config.NumberColumn at the call site, because
    st.dataframe respects Styler's CSS but silently ignores
    Styler.format() for cell values. See build_case_column_config.
    Latency is intentionally not color-coded; what counts as "fast" varies
    too much by use case to bake fixed thresholds into the table.
    Works with both raw keys and label-renamed columns.
    """
    all_metric_keys = METRIC_COLUMNS + RETRIEVAL_COLUMNS + CHUNK_COLUMNS
    all_metric_labels = [METRIC_LABELS.get(k, k) for k in all_metric_keys]
    score_cols = [c for c in all_metric_keys + all_metric_labels if c in df.columns]

    styler = df.style

    # 0-1 score columns
    if score_cols:
        styler = styler.map(
            lambda v: _color_by_thresholds(v, _SCORE_BANDS),
            subset=score_cols,
        )

    # Status column
    if "status" in df.columns:
        styler = styler.map(lambda v: _STATUS_COLORS.get(v, ""), subset=["status"])

    # Avg judge score (0-1 scale, same bands as other 0-1 score columns)
    judge_col = _find_column(df, "avg_judge_score", METRIC_LABELS.get("avg_judge_score", ""))
    if judge_col:
        styler = styler.map(lambda v: _color_by_thresholds(v, _SCORE_BANDS), subset=[judge_col])

    styler = styler.set_table_styles([
        {"selector": "th", "props": [("font-size", "0.8rem")]},
        {"selector": "thead th", "props": [("border-bottom", "1px solid #4a5568")]},
        {"selector": "td", "props": [("font-size", "0.85rem")]},
    ])
    styler = styler.hide(axis="index")

    return styler


def build_case_column_config(df: pd.DataFrame) -> dict[str, Any]:
    """Build per-case-table column_config with registry-driven formats.

    st.dataframe ignores Styler.format(), so cell precision must be
    set via column_config.NumberColumn(format=...).  Each numeric
    column's printf format is derived from its MetricDef.fmt in the
    registry — adding/changing a metric there flows through here without
    edits.  Latency columns get the trailing s suffix that matches
    the registry-declared seconds unit.
    """
    numeric_keys = (
        METRIC_COLUMNS + RETRIEVAL_COLUMNS + CHUNK_COLUMNS
        + TOKEN_COLUMNS + JUDGE_TOKEN_COLUMNS
        + ["avg_judge_score", "latency_seconds", "retrieval_latency_seconds", "llm_latency_seconds"]
    )
    label_to_key = {METRIC_LABELS.get(k, k): k for k in numeric_keys}

    config: dict[str, Any] = {}
    for col in df.columns:
        key = label_to_key.get(col)
        if key is None:
            continue
        # Registry fmts are Python-style (".2f", "d"); NumberColumn uses
        # printf-style ("%.2f", "%d") — prepend "%" to convert.
        printf_fmt = f"%{_fmt_for(key)}"
        if key.endswith("latency_seconds"):
            printf_fmt += "s"
        config[col] = st.column_config.NumberColumn(format=printf_fmt)
    return config


NOT_EVALUATED_LABEL = "Not evaluated"


def format_metric_value(value: object, fmt: str = DEFAULT_FMT, suffix: str = "") -> str:
    """Format a metric value for display.

    Returns "Not evaluated" when the value is None / NaN — which
    is how the harness signals that a toggleable metric group (judge, source,
    chunk) was turned off in eval_config.ENABLED_METRIC_GROUPS for the
    run being displayed.
    """
    if value is None:
        return NOT_EVALUATED_LABEL
    if isinstance(value, float) and pd.isna(value):
        return NOT_EVALUATED_LABEL
    return f"{value:{fmt}}{suffix}"


def compute_delta(current: float | None, previous: float | None, fmt: str = DEFAULT_FMT) -> str | None:
    """Return a display-ready +0.05 / -0.10 delta string, or None.

    None is returned when either input is None or NaN — that's
    how st.metric knows to render the card without a delta arrow.
    The format spec mirrors the value's own format so the delta sits at
    the same precision (e.g. a .0% value gets a +5% delta).
    """
    if current is None or previous is None:
        return None
    if isinstance(current, float) and pd.isna(current):
        return None
    if isinstance(previous, float) and pd.isna(previous):
        return None
    diff = current - previous
    return f"{diff:+{fmt}}"


def _prev_val(prev_row: pd.Series | None, col: str) -> float | None:
    """Safely fetch prev_row[col] when prev_row is the first run.

    Returns None when no previous run exists yet so KPI cards on the
    first run render without an empty delta string.
    """
    if prev_row is None:
        return None
    return prev_row[col]


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Eval Dashboard — HybridSearchAgent",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------
latest_ts = _get_latest_timestamp()
if "last_ts" not in st.session_state:
    st.session_state.last_ts = latest_ts

if latest_ts != st.session_state.last_ts:
    st.session_state.last_ts = latest_ts
    load_runs.clear()
    load_cases.clear()

st_autorefresh(interval=POLL_SECONDS * 1000, key="data_poll")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
if not DB_PATH.exists():
    st.warning(f"Database not found at `{DB_PATH}`. Run an evaluation first.")
    st.stop()

runs_df = load_runs()
cases_df = load_cases()

if runs_df.empty:
    st.info("No evaluation runs recorded yet.")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Evaluation Ledger")
    st.caption(f"Database: `{DB_PATH.name}`")

    st.metric("Total runs", len(runs_df))

    run_labels = {
        row.run_id: f"Run {row.run_id}  \u2014  {row.run_timestamp:%Y-%m-%d %H:%M}"
        for row in runs_df.itertuples()
    }

    selected_run_id = st.selectbox(
        "Select run",
        options=list(run_labels.keys()),
        format_func=lambda rid: run_labels[rid],
    )

    # Run metadata
    run_row = runs_df[runs_df["run_id"] == selected_run_id].iloc[0]
    runs_sorted = runs_df.sort_values("run_id")
    prev_runs = runs_sorted[runs_sorted["run_id"] < selected_run_id]
    prev_row = prev_runs.iloc[-1] if not prev_runs.empty else None

    st.divider()
    st.caption("Run details")

    def _parse_gate_thresholds(row: Any) -> dict[str, float] | None:
        """Return the gate_thresholds dict for a run row, or None if missing/invalid."""
        if row is None or "gate_thresholds" not in row:
            return None
        raw = row["gate_thresholds"]
        if raw is None:
            return None
        if isinstance(raw, dict):
            return raw
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except (TypeError, ValueError):
            return None

    # gate_thresholds is still parsed — the drift banner and the Metrics Guide
    # Verdict Logic table both rely on it, even though the sidebar doesn't show
    # it (active thresholds are documented in the Metrics Guide tab instead).
    gate_thresholds = _parse_gate_thresholds(run_row)
    prev_gate_thresholds = _parse_gate_thresholds(prev_row) if prev_row is not None else None

    st.markdown(
        f"**Judge:** {run_row['judge_model']}  \n"
        f"**Cases:** {int(run_row['case_count'])}"
    )

    # Enabled metric groups: parsed from the run's persisted JSON column so
    # the sidebar reflects what was toggled on for *this* run, not whatever
    # eval_config.py currently holds. Useful when partial-toggle runs land
    # in the ledger and the dashboard shows "Not evaluated" for some metrics.
    def _parse_enabled_groups(row: Any) -> list[str] | None:
        """Return the enabled_groups list for a run row, or None if missing/invalid."""
        if row is None or "enabled_groups" not in row:
            return None
        raw = row["enabled_groups"]
        if raw is None:
            return None
        if isinstance(raw, list):
            return raw
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else None
        except (TypeError, ValueError):
            return None

    enabled_groups_list = _parse_enabled_groups(run_row)
    if enabled_groups_list is not None:
        groups_display = ", ".join(sorted(enabled_groups_list)) if enabled_groups_list else "(none)"
        st.markdown(f"**Enabled groups:** {groups_display}")

    # Threshold list: pulled from the run's persisted gate_thresholds so the
    # sidebar reflects the values active for the selected run, not whatever
    # eval_config.py happens to hold today. Same key formatting as the drift
    # warning banner so the two stay visually consistent.
    if gate_thresholds:
        threshold_lines = "\n".join(
            f"- **{key}:** {value}"
            for key, value in gate_thresholds.items()
        )
        st.markdown(f"**Thresholds:**\n{threshold_lines}")

run_cases = cases_df[cases_df["run_id"] == selected_run_id].copy()


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.title("Evaluation Dashboard")
st.caption("HybridSearchAgent \u2014 DeepEval and deterministic evaluation results")

# Threshold drift warning: gate thresholds should stay stable across runs so
# trend comparisons remain apples-to-apples. Flag any change since the previous run.
if (
    gate_thresholds is not None
    and prev_gate_thresholds is not None
    and gate_thresholds != prev_gate_thresholds
):
    drift = [
        f"**{key}**: {prev_gate_thresholds.get(key)} \u2192 {gate_thresholds.get(key)}"
        for key in sorted(set(gate_thresholds) | set(prev_gate_thresholds))
        if gate_thresholds.get(key) != prev_gate_thresholds.get(key)
    ]
    st.warning(
        "\u26a0\ufe0f Gate thresholds changed since the previous run \u2014 trend "
        "comparisons against earlier runs may not be apples-to-apples.\n\n"
        + "\n\n".join(f"- {line}" for line in drift)
    )


# ===================================================================
# Tabbed layout
# ===================================================================

tab_summary, tab_analysis, tab_trends, tab_guide = st.tabs([
    "Run Summary",
    "Deep Analysis",
    "Historical Trends",
    "Metrics Guide",
])


# ===================================================================
# Tab 1 — Run Summary
# ===================================================================

with tab_summary:
    st.caption("Summary of metrics for the selected run.")

    def _kpi_label(col_name: str, fallback: str) -> str:
        """Pull the display label from the registry and force it onto 2 lines.

        Every KPI card's label renders on exactly 2 lines so all cards in a row
        share the same height. Labels with a space break at the last space;
        single-word labels get a non-breaking space appended as a filler
        second line. Works together with white-space: pre-line in the
        custom CSS, which honors the embedded \\n.
        """
        raw = METRIC_LABELS.get(col_name, fallback)
        if " " in raw:
            idx = raw.rfind(" ")
            return raw[:idx] + "\n" + raw[idx + 1:]
        return raw + "\n\u00a0"

    # --- KPI row 1: core performance ---
    st.subheader("Average Performance")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    # pass_rate carries fmt .0% in the registry — applying it via
    # _fmt_for for both the value and the delta means there is no
    # need to multiply by 100 here just to feed a .1f delta.
    _pass_fmt = _fmt_for("pass_rate")
    _prev_pass = _prev_val(prev_row, "pass_rate")
    c1.metric(
        _kpi_label("pass_rate", "Pass Rate"),
        format_metric_value(run_row["pass_rate"], _pass_fmt),
        delta=compute_delta(run_row["pass_rate"], _prev_pass, fmt=_pass_fmt),
        delta_color="normal",
    )
    _case_fmt = _fmt_for("avg_judge_run_score")
    c2.metric(
        _kpi_label("avg_judge_run_score", "Avg Judge Score"),
        format_metric_value(run_row["avg_judge_run_score"], _case_fmt),
        delta=compute_delta(run_row["avg_judge_run_score"], _prev_val(prev_row, "avg_judge_run_score"), fmt=_case_fmt),
        delta_color="normal",
    )
    _lat_fmt = _fmt_for("avg_latency_seconds")
    c3.metric(
        _kpi_label("avg_latency_seconds", "Latency"),
        format_metric_value(run_row["avg_latency_seconds"], _lat_fmt, "s"),
        delta=compute_delta(run_row["avg_latency_seconds"], _prev_val(prev_row, "avg_latency_seconds"), fmt=_lat_fmt),
        delta_color="inverse",
    )
    _ret_lat_fmt = _fmt_for("avg_retrieval_latency_seconds")
    c4.metric(
        _kpi_label("avg_retrieval_latency_seconds", "Retrieval Latency"),
        format_metric_value(run_row["avg_retrieval_latency_seconds"], _ret_lat_fmt, "s"),
        delta=compute_delta(run_row["avg_retrieval_latency_seconds"], _prev_val(prev_row, "avg_retrieval_latency_seconds"), fmt=_ret_lat_fmt),
        delta_color="inverse",
    )
    _llm_lat_fmt = _fmt_for("avg_llm_latency_seconds")
    c5.metric(
        _kpi_label("avg_llm_latency_seconds", "LLM Latency"),
        format_metric_value(run_row["avg_llm_latency_seconds"], _llm_lat_fmt, "s"),
        delta=compute_delta(run_row["avg_llm_latency_seconds"], _prev_val(prev_row, "avg_llm_latency_seconds"), fmt=_llm_lat_fmt),
        delta_color="inverse",
    )
    _agent_tok_fmt = _fmt_for("avg_agent_total_tokens")
    c6.metric(
        _kpi_label("avg_agent_total_tokens", "Agent Tokens"),
        format_metric_value(run_row.get("avg_agent_total_tokens"), _agent_tok_fmt),
        delta=compute_delta(run_row.get("avg_agent_total_tokens"), _prev_val(prev_row, "avg_agent_total_tokens"), fmt=_agent_tok_fmt),
        delta_color="inverse",
    )
    _judge_tok_fmt = _fmt_for("avg_judge_total_tokens")
    c7.metric(
        _kpi_label("avg_judge_total_tokens", "Judge Tokens"),
        format_metric_value(run_row.get("avg_judge_total_tokens"), _judge_tok_fmt),
        delta=compute_delta(run_row.get("avg_judge_total_tokens"), _prev_val(prev_row, "avg_judge_total_tokens"), fmt=_judge_tok_fmt),
        delta_color="inverse",
    )

    # --- KPI row 2: judge metric averages ---
    # Per-judge-metric run averages live in the metric_averages JSON
    # column (not as flat avg_* columns), so they're parsed once here
    # and once for the previous run to feed deltas.
    def _parse_metric_averages(row: pd.Series | None) -> dict[str, float | None]:
        """Parse the metric_averages JSON column off a run row.

        Per-judge-metric averages are stored as a JSON blob (not as flat
        avg_* columns) because new judge metrics shouldn't require a
        schema migration.  Returns {} when the row is missing, the
        column is non-string, or the JSON is malformed — the dashboard
        renders missing-metric KPIs as "Not evaluated" automatically.
        """
        if row is None:
            return {}
        raw = row.get("metric_averages")
        try:
            return json.loads(raw) if isinstance(raw, str) else {}
        except json.JSONDecodeError:
            return {}

    judge_avgs      = _parse_metric_averages(run_row)
    prev_judge_avgs = _parse_metric_averages(prev_row)

    st.subheader("Average Judge Scores")
    j1, j2, j3, j4, j5, j6, j7 = st.columns(7)
    judge_kpis = [
        ("answer_relevancy",     j1),
        ("faithfulness",         j2),
        ("correctness_g_eval",   j3),
        ("contextual_precision", j4),
        ("contextual_recall",    j5),
        ("contextual_relevancy", j6),
        ("hallucination",        j7),
    ]
    for metric_key, col_widget in judge_kpis:
        _jfmt = _fmt_for(metric_key)
        col_widget.metric(
            _kpi_label(metric_key, METRIC_LABELS.get(metric_key, metric_key)),
            format_metric_value(judge_avgs.get(metric_key), _jfmt),
            delta=compute_delta(judge_avgs.get(metric_key), prev_judge_avgs.get(metric_key), fmt=_jfmt),
            delta_color="normal",
        )

    # --- KPI row 3: retrieval quality ---
    st.subheader("Average Source Retrieval Quality")
    r1, r2, r3, r4, r5, r6 = st.columns(6)
    retrieval_kpis = [
        ("avg_hit_at_k", r1),
        ("avg_metadata_match_ratio", r2),
        ("avg_mrr", r3),
        ("avg_precision_at_k", r4),
        ("avg_recall_at_k", r5),
        ("avg_ndcg_at_k", r6),
    ]
    for col_name, col_widget in retrieval_kpis:
        _rfmt = _fmt_for(col_name)
        col_widget.metric(
            _kpi_label(col_name, col_name),
            format_metric_value(run_row.get(col_name), _rfmt),
            delta=compute_delta(run_row.get(col_name), _prev_val(prev_row, col_name), fmt=_rfmt),
            delta_color="normal",
        )

    # --- KPI row 4: chunk retrieval quality ---
    st.subheader("Average Chunk Retrieval Quality")
    k1, k2, k3, k4, k5 = st.columns(5)
    chunk_kpis = [
        ("avg_chunk_hit_at_k", k1),
        ("avg_chunk_mrr", k2),
        ("avg_chunk_precision_at_k", k3),
        ("avg_chunk_recall_at_k", k4),
        ("avg_chunk_ndcg_at_k", k5),
    ]
    for col_name, col_widget in chunk_kpis:
        _cfmt = _fmt_for(col_name)
        col_widget.metric(
            _kpi_label(col_name, col_name),
            format_metric_value(run_row.get(col_name), _cfmt),
            delta=compute_delta(run_row.get(col_name), _prev_val(prev_row, col_name), fmt=_cfmt),
            delta_color="normal",
        )

    st.divider()

    # --- Per-case table ---
    if run_cases.empty:
        st.warning("No case-level data for this run.")
    else:
        st.subheader("Per-case Results")
        st.caption("**Color keys for the table below:**")
        st.markdown(
            "- :green[**green**] = good (score ≥ 0.8)\n"
            "- :orange[**orange**] = borderline (≥ 0.5 but below 0.8)\n"
            "- :red[**red**] = failing (< 0.5)\n"
            "- :gray[**gray**] = not evaluated"
        )

        # Token totals are surfaced in the Average Performance KPI row above;
        # the per-case view shows the input/output split for cost drill-down.
        token_split_cols = [c for c in TOKEN_COLUMNS + JUDGE_TOKEN_COLUMNS if not c.endswith("_total_tokens")]
        display_cols = [
            "case_id", "category", "status", "avg_judge_score",
            *METRIC_COLUMNS,
            *RETRIEVAL_COLUMNS,
            *CHUNK_COLUMNS,
            "latency_seconds",
            *token_split_cols,
        ]
        available_display = [c for c in display_cols if c in run_cases.columns]
        display_df = run_cases[available_display].reset_index(drop=True)
        display_df = display_df.rename(columns=METRIC_LABELS)
        styled_df = style_case_dataframe(display_df)
        table_height = min(len(run_cases) * 38 + 40, 600)
        _col1 = METRIC_LABELS.get("case_id", "case_id")
        col_config = build_case_column_config(display_df)
        col_config[_col1] = st.column_config.Column(pinned=True)
        st.dataframe(
            styled_df,
            height=table_height,
            use_container_width=True,
            column_config=col_config,
        )

        # --- Metric bar chart ---
        st.subheader("Metric Scores by Case")
        # All per-case score metrics that share the 0-1 y-axis. Tokens and
        # latency are excluded because their scales (thousands / seconds)
        # would dwarf the 0-1 scores on the same bar chart. avg_judge_score
        # is excluded because it's a run-level summary, not a per-case score.
        all_metric_keys = (
            METRIC_COLUMNS
            + RETRIEVAL_COLUMNS
            + CHUNK_COLUMNS
            + ["required_keyword_hit_rate"]
        )
        available_metrics = [c for c in all_metric_keys if c in run_cases.columns]
        if available_metrics:
            # Two side-by-side filters — empty default in both means "show
            # everything," and the placeholder text on each communicates
            # that nothing needs to be touched until the user wants to narrow.
            case_filter_col, metric_filter_col = st.columns(2)
            with case_filter_col:
                all_case_ids = run_cases["case_id"].tolist()
                selected_cases = st.multiselect(
                    "Filter cases",
                    options=all_case_ids,
                    default=[],
                    placeholder="All cases selected (default)",
                    key="metric_chart_case_filter_v2",
                )
            with metric_filter_col:
                selected_metric_keys = st.multiselect(
                    "Filter metrics",
                    options=available_metrics,
                    default=[],
                    format_func=lambda k: METRIC_LABELS.get(k, k),
                    placeholder="All metrics selected (default)",
                    key="metric_chart_metric_filter_v2",
                )

            metric_display = selected_metric_keys if selected_metric_keys else available_metrics
            chart_cases = (
                run_cases[run_cases["case_id"].isin(selected_cases)]
                if selected_cases
                else run_cases
            )
            if metric_display:
                metric_label_map = {m: METRIC_LABELS.get(m, m) for m in metric_display}
                metric_label_cols = list(metric_label_map.values())
                bar_df = (
                    chart_cases[["case_id"] + metric_display]
                    .rename(columns=metric_label_map)
                )
                # Long-form for Altair grouped bar chart.
                long_df = bar_df.melt(
                    id_vars="case_id",
                    value_vars=metric_label_cols,
                    var_name="Metric",
                    value_name="Score",
                )
                long_df["Score"] = pd.to_numeric(long_df["Score"], errors="coerce")
                chart = (
                    alt.Chart(long_df)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "case_id:N",
                            title=None,
                            axis=alt.Axis(labelAngle=-40),
                        ),
                        xOffset=alt.XOffset("Metric:N"),
                        y=alt.Y("Score:Q"),
                        color=alt.Color("Metric:N"),
                        tooltip=["case_id", "Metric", "Score"],
                    )
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Select at least one metric to display the chart.")

        st.divider()

        # --- Answer detail ---
        st.subheader("Answer Detail")
        for _, case_row in run_cases.iterrows():
            is_pass = case_row["status"] == "PASS"
            icon = "\u2705" if is_pass else "\u26a0\ufe0f"
            score_str = (
                f"{case_row['avg_judge_score']:{_fmt_for('avg_judge_score')}}"
                if pd.notna(case_row.get("avg_judge_score"))
                else NOT_EVALUATED_LABEL
            )

            with st.expander(f"{icon}  **{case_row['case_id']}**  \u2014  {case_row['status']}  ({score_str})"):
                detail_left, detail_right = st.columns([3, 1])
                with detail_left:
                    st.markdown(f"**Question:**")
                    with st.container(border=True):
                        st.markdown(case_row["question"])
                    st.markdown(f"**Expected answer:**")
                    with st.container(border=True):
                        st.markdown(case_row["expected_output"])
                    st.markdown(f"**Agent answer:**")
                    with st.container(border=True):
                        st.markdown(case_row["answer"])
                with detail_right:
                    st.markdown("**Quick stats**")
                    st.metric("Score", format_metric_value(case_row.get("avg_judge_score"), _fmt_for("avg_judge_score")))
                    st.metric("Latency", format_metric_value(case_row.get("latency_seconds"), _fmt_for("latency_seconds"), "s"))
                    hit = case_row.get("hit_at_k")
                    if hit is None or pd.isna(hit):
                        hit_display = NOT_EVALUATED_LABEL
                    else:
                        hit_display = "Yes" if hit == 1.0 else "No"
                    st.metric("Hit@k", hit_display)

                if case_row.get("errors") and case_row["errors"] != "[]":
                    st.error(f"Errors: {case_row['errors']}")


# ===================================================================
# Tab 2 — Deep Analysis
# ===================================================================

with tab_analysis:

    st.caption("Deep analysis of metrics and correlations for the selected run.")

    if run_cases.empty:
        st.warning("No case-level data for this run.")
    else:
        available_metrics = [c for c in METRIC_COLUMNS if c in run_cases.columns]
        available_retrieval = [c for c in RETRIEVAL_COLUMNS if c in run_cases.columns]
        available_chunk = [c for c in CHUNK_COLUMNS if c in run_cases.columns]
        # Hallucination is "lower = better"; excluded from radar + correlation
        # where it would mix with higher-is-better metrics on the same axis.
        radar_metrics = [c for c in available_metrics if c != "hallucination"]
        correlation_metrics = [c for c in available_metrics if c != "hallucination"]

        # --- Radar charts: 2 on first row, 1 on second row ---
        st.subheader("Metric Balance")
        col_radar_llm, col_radar_ret = st.columns(2)

        with col_radar_llm:
            st.markdown("**Judge Metrics**")
            st.caption("Average scores across all cases in selected run. Identifies weak spots in answer quality. Hallucination is shown separately below.")
            if radar_metrics:
                st.plotly_chart(build_radar_chart(run_cases, radar_metrics, color="#636EFA"), key="radar_llm", use_container_width=True)

        with col_radar_ret:
            st.markdown("**Source Retrieval Metrics**")
            st.caption("Average scores across all cases in selected run. Highlights weak spots in source retrieval.")
            if available_retrieval:
                st.plotly_chart(build_radar_chart(run_cases, available_retrieval, color="#00CC96"), key="radar_retrieval", use_container_width=True)

        # Second row: chunk radar on the left, empty column on the right so the
        # chart keeps the same width and padding as the row above.
        col_radar_chunk, _col_radar_spacer = st.columns(2)
        with col_radar_chunk:
            st.markdown("**Chunk Retrieval Metrics**")
            st.caption("Average scores acorss all cases in selected run. Highlights whether the *right passages* were retrieved, not just the right files.")
            if available_chunk:
                st.plotly_chart(build_radar_chart(run_cases, available_chunk, color="#AB63FA"), key="radar_chunk", use_container_width=True)
            else:
                st.info("No chunk metrics available \u2014 set `expected_chunks` in test cases and enable the `chunk` group.")

        # --- Hallucination (standalone, lower = better) ---
        if "hallucination" in run_cases.columns:
            hallu_mean = run_cases["hallucination"].dropna().mean()
            if pd.notna(hallu_mean):
                threshold_val = gate_thresholds.get("judge_threshold", 0.5) if gate_thresholds else 0.5
                if hallu_mean <= threshold_val / 2:
                    hallu_color, hallu_label = "#00CC96", "low"
                elif hallu_mean <= threshold_val:
                    hallu_color, hallu_label = "#F4B400", "moderate"
                else:
                    hallu_color, hallu_label = "#EF553B", "high"
                col_hallu, _col_hallu_spacer = st.columns([1, 2])
                # Both the mean and the threshold sit on the same 0-1 scale
                # as hallucination; using its registry fmt keeps the
                # card consistent with how the metric appears elsewhere.
                _hallu_fmt = _fmt_for("hallucination")
                with col_hallu:
                    st.markdown("**Hallucination** &nbsp; <span style='color:#888;font-size:0.85em'>(lower = better)</span>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div style='font-size:2.2em;font-weight:600;color:{hallu_color};line-height:1.1'>{hallu_mean:{_hallu_fmt}}</div>"
                        f"<div style='color:{hallu_color};font-size:0.9em'>{hallu_label} \u2014 threshold {threshold_val:{_hallu_fmt}}</div>",
                        unsafe_allow_html=True,
                    )

        st.divider()

        # --- Score distribution ---
        st.subheader("Score Distribution")
        st.caption("Select a metric to see how scores spread across cases.")
        all_available = available_metrics + available_retrieval + available_chunk
        if all_available:
            selected_dist_metric = st.selectbox(
                "Metric",
                options=all_available,
                format_func=lambda c: METRIC_LABELS.get(c, c),
                label_visibility="collapsed",
            )
            fig_dist = px.histogram(
                run_cases,
                x=selected_dist_metric,
                nbins=10,
                color_discrete_sequence=["#636EFA"],
            )
            fig_dist.update_layout(
                **PLOTLY_LAYOUT_DEFAULTS,
                xaxis_title=METRIC_LABELS.get(selected_dist_metric, selected_dist_metric),
                yaxis_title="Cases",
                xaxis=dict(range=[0, 1.05]),
                height=400,
            )
            st.plotly_chart(fig_dist, key="histogram", use_container_width=True)

        st.divider()

        # --- Metric correlations ---
        st.subheader("Metric Correlations")
        st.caption("Pearson correlation. Hallucination is excluded (lower = better, would invert the sign).")
        _CORR_EXCLUDE = {"hit_at_k", "chunk_hit_at_k"}
        all_numeric = [c for c in correlation_metrics + available_retrieval + available_chunk if c not in _CORR_EXCLUDE]
        if len(all_numeric) >= 2:
            st.plotly_chart(build_correlation_heatmap(run_cases, all_numeric), key="heatmap", use_container_width=True)

        st.divider()

        # --- Latency breakdown ---
        st.subheader("Latency Breakdown by Case")
        st.caption("Retrieval time and LLM generation time per case.")
        fig_latency = build_stacked_latency(run_cases)
        if fig_latency:
            st.plotly_chart(fig_latency, key="latency_bar", use_container_width=True)

        st.divider()

        # --- Token usage ---
        st.subheader("Token Usage by Case")
        st.caption("Agent input and output tokens per case.")
        fig_tokens = build_token_bar(run_cases)
        if fig_tokens:
            st.plotly_chart(fig_tokens, key="token_bar", use_container_width=True)
        else:
            st.info("No token usage recorded for this run.")


# ===================================================================
# Tab 3 — Historical Trends
# ===================================================================

with tab_trends:

    st.caption("Historical trends across all evaluation runs in the ledger.")

    if len(runs_df) < 2:
        st.info("Trends require at least 2 evaluation runs. Only 1 run recorded so far.")
    else:
        trend_df = runs_df.sort_values("run_timestamp")

        def _series(df: pd.DataFrame, value_col: str, label: str) -> pd.DataFrame:
            """Reshape one wide column into long form (run_timestamp,
            value, series) so multiple metrics can be concatenated
            and rendered on a single st.line_chart with color=series.
            """
            if df.empty or value_col not in df.columns:
                return pd.DataFrame(columns=["run_timestamp", "value", "series"])
            out = df[["run_timestamp", value_col]].copy()
            out = out.rename(columns={value_col: "value"})
            out["value"] = pd.to_numeric(out["value"], errors="coerce")
            out["series"] = label
            return out.dropna(subset=["value"])

        def _judge_metric_series(df: pd.DataFrame) -> pd.DataFrame:
            """Explode the per-run metric_averages JSON column into long
            form so each judge metric becomes its own coloured line.
            """
            rows: list[dict[str, object]] = []
            for r in df.itertuples():
                raw = getattr(r, "metric_averages", None)
                try:
                    avgs = json.loads(raw) if isinstance(raw, str) else {}
                except json.JSONDecodeError:
                    avgs = {}
                for key, val in avgs.items():
                    if val is None:
                        continue
                    rows.append({
                        "run_timestamp": r.run_timestamp,
                        "value": val,
                        "series": METRIC_LABELS.get(key, key),
                    })
            out = pd.DataFrame(rows, columns=["run_timestamp", "value", "series"])
            if not out.empty:
                out["value"] = pd.to_numeric(out["value"], errors="coerce")
                out = out.dropna(subset=["value"])
            return out

        def _line(df: pd.DataFrame, title: str, empty_msg: str, y_label: str) -> None:
            """Render a long-form (run_timestamp, value, series) DataFrame as a
            multi-series line chart. Uses Altair instead of st.line_chart
            so the legend can wrap onto multiple rows when a chart has many
            series (e.g. 7 judge metrics) — st.line_chart only supports a
            single overflowing row that clips entries.
            """
            st.markdown(f"**{title}**")
            if df.empty:
                st.info(empty_msg)
                return
            df = df.sort_values("run_timestamp")
            chart = (
                alt.Chart(df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("run_timestamp:T", title="Time"),
                    y=alt.Y("value:Q", title=y_label),
                    color=alt.Color(
                        "series:N",
                        title=None,
                        # orient="bottom" + columns=4 wraps the
                        # legend into rows of 4. Charts with ≤ 4 series
                        # render as a single row; the 7-judge chart wraps
                        # to 4 + 3.
                        legend=alt.Legend(orient="bottom", columns=4, labelLimit=200),
                    ),
                    tooltip=["run_timestamp:T", "series:N", "value:Q"],
                )
                .properties(height=350)
            )
            st.altair_chart(chart, use_container_width=True)

        # --- Chart 1: Pass Rate & Avg Judge Score (Run) ---
        chart1_df = pd.concat([
            _series(trend_df, "pass_rate",          METRIC_LABELS.get("pass_rate", "Pass Rate")),
            _series(trend_df, "avg_judge_run_score", METRIC_LABELS.get("avg_judge_run_score", "Avg Judge Score (Run)")),
        ], ignore_index=True)
        _line(chart1_df, "Pass Rate and Avg Judge Score Over Time", "No summary score history yet.", "Score")

        # --- Chart 2: Per-judge-metric averages from metric_averages JSON ---
        _line(_judge_metric_series(trend_df), "Judge Scores Over Time", "No judge score history yet.", "Score")

        # --- Chart 3: Source retrieval averages ---
        source_df = pd.concat([
            _series(trend_df, f"avg_{key}", METRIC_LABELS.get(key, key))
            for key in RETRIEVAL_COLUMNS
        ], ignore_index=True)
        _line(source_df, "Source Retrieval Quality Over Time", "No source retrieval history yet.", "Score")

        # --- Chart 4: Chunk retrieval averages ---
        chunk_df = pd.concat([
            _series(trend_df, f"avg_{key}", METRIC_LABELS.get(key, key))
            for key in CHUNK_COLUMNS
        ], ignore_index=True)
        _line(chunk_df, "Chunk Retrieval Quality Over Time", "No chunk retrieval history yet.", "Score")

        # --- Chart 5: Latency ---
        latency_label_map = {
            "avg_latency_seconds":           "Total",
            "avg_retrieval_latency_seconds": "Retrieval",
            "avg_llm_latency_seconds":       "LLM Generation",
        }
        latency_df = pd.concat([
            _series(trend_df, col, label) for col, label in latency_label_map.items()
        ], ignore_index=True)
        _line(latency_df, "Latency Over Time", "No latency history yet.", "Seconds")

        # --- Chart 6: Agent + Judge token usage ---
        token_label_map = {
            "avg_agent_input_tokens":  "Agent Input",
            "avg_agent_output_tokens": "Agent Output",
            "avg_judge_input_tokens":  "Judge Input",
            "avg_judge_output_tokens": "Judge Output",
        }
        token_df = pd.concat([
            _series(trend_df, col, label) for col, label in token_label_map.items()
        ], ignore_index=True)
        _line(token_df, "Agent and Judge Token Usage Over Time", "No token history yet.", "Tokens")


# ===================================================================
# Tab 4 — Metrics Guide
# ===================================================================

with tab_guide:
    # Render info_metrics.md directly so this tab never drifts from the
    # source-of-truth doc. Reading inside the tab body (not at module
    # level) means markdown edits show up on the next Streamlit rerun
    # without restarting the dashboard.
    _guide_path = Path(__file__).resolve().parent / "info_metrics.md"
    if _guide_path.exists():
        st.markdown(_guide_path.read_text(encoding="utf-8"), unsafe_allow_html=True)
    else:
        st.warning(
            f"Metrics guide not found at `{_guide_path}`. "
            "Expected `info_metrics.md` alongside `eval_dashboard.py`."
        )
