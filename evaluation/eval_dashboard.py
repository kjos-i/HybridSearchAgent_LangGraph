"""Streamlit dashboard for browsing evaluation results stored in eval_ledger.db.

Launch from the project root (or the evaluation folder):
    streamlit run evaluation/eval_dashboard.py
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from eval_metric_registry import (
    chunk_metric_keys,
    llm_metric_keys,
    metric_fmts,
    metric_labels,
    retrieval_metric_keys,
)

POLL_SECONDS = 10  # How often to check if new data has arrived

DB_PATH = Path(__file__).resolve().parent / "eval_ledger.db"

METRIC_COLUMNS: list[str] = llm_metric_keys()
RETRIEVAL_COLUMNS: list[str] = retrieval_metric_keys()
CHUNK_COLUMNS: list[str] = chunk_metric_keys()
METRIC_LABELS: dict[str, str] = metric_labels()
METRIC_FMTS: dict[str, str] = metric_fmts()


def _fmt_for(col_name: str, default: str = ".3f") -> str:
    """Resolve a column's display format from the registry, falling back to ``default``."""
    return METRIC_FMTS.get(col_name, default)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* Tighten metric card spacing */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e2a3a 0%, #2c3e50 100%);
    border: 1px solid #3a4f65;
    border-radius: 8px;
    padding: 12px 16px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.25);
}
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: #8fa8c8;
    line-height: 1.2 !important;
}
/* Labels are forced to 2 lines by an explicit "\n" inserted in the label
   string itself (see _kpi_label). pre-line preserves that newline.
   word-break:normal stops Streamlit from splitting a long token (e.g.
   "Precision@k") mid-word when the card is narrow, which would produce a
   third line. If a token genuinely doesn't fit, we let it overflow rather
   than breaking it. */
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricLabel"] > *,
div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
    white-space: pre-line !important;
    overflow: visible !important;
    text-overflow: clip !important;
    word-break: normal !important;
    overflow-wrap: normal !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700;
    color: #e8edf2;
}

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

    Cached indefinitely; call ``load_runs.clear()`` to force a fresh read
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

    Cached indefinitely; call ``load_cases.clear()`` to force a fresh read
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
    """Convert a hex colour like ``#636EFA`` to an ``rgba(...)`` string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_radar_chart(df: pd.DataFrame, metric_cols: list[str], color: str = "#636EFA") -> go.Figure:
    """Build a radar (polar) chart showing average scores across metrics."""
    available = [c for c in metric_cols if c in df.columns]
    if not available:
        return go.Figure()

    means = df[available].mean()
    labels = [METRIC_LABELS.get(c, c) for c in available]
    values = means.tolist()

    # Close the polygon
    labels.append(labels[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill="toself",
        fillcolor=_hex_to_rgba(color, alpha=0.12),
        line=dict(color=color, width=2),
        marker=dict(size=5),
        hovertemplate="%{theta}: %{r:.3f}<extra></extra>",
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

    # Build custom text: show the constant value for NaN cells.
    def _cell_text(row_key: str, col_key: str, val: float) -> str:
        if pd.isna(val):
            # At least one of row/col has zero variance — show its constant.
            const_key = row_key if row_key in constant_metrics else col_key
            return f"= {constant_metrics[const_key]:.2f}"
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


def build_grouped_bar(
    df: pd.DataFrame,
    id_col: str,
    metric_cols: list[str],
) -> go.Figure:
    """Build a grouped bar chart of metric scores per case."""
    bar_data = df.set_index(id_col)[metric_cols]
    melted = bar_data.reset_index().melt(id_vars=id_col, var_name="metric", value_name="score")
    melted["metric_label"] = melted["metric"].map(METRIC_LABELS)

    fig = px.bar(
        melted,
        x=id_col,
        y="score",
        color="metric_label",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        xaxis_title="",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.05], gridcolor="#f0f0f0"),
        legend_title="Metric",
        height=400,
    )
    return fig


def build_stacked_latency(df: pd.DataFrame) -> go.Figure | None:
    """Build a stacked bar chart of retrieval vs LLM latency per case."""
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
    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        xaxis_title="",
        yaxis_title="Seconds",
        yaxis=dict(gridcolor="#f0f0f0"),
        legend_title="Component",
        height=370,
    )
    return fig


def build_trend_line(
    df: pd.DataFrame,
    time_col: str,
    value_cols: list[str],
    y_label: str = "Score",
    y_range: list[float] | None = None,
) -> go.Figure:
    """Build a multi-series line chart for trend data."""
    melted = df[[time_col] + value_cols].melt(
        id_vars=time_col, var_name="metric", value_name="value",
    )
    melted["metric_label"] = melted["metric"].map(METRIC_LABELS).fillna(melted["metric"])

    fig = px.line(
        melted, x=time_col, y="value",
        color="metric_label", markers=True,
    )
    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        xaxis_title="",
        xaxis_tickangle=0,
        yaxis_title=y_label,
        yaxis=dict(range=y_range, gridcolor="#f0f0f0") if y_range else dict(gridcolor="#f0f0f0"),
        legend_title="",
        height=380,
    )
    return fig


def _color_by_thresholds(val: object, thresholds: list[tuple[float, str]], *, higher_is_better: bool = True) -> str:
    """Return a CSS color string based on threshold bands.

    Parameters
    ----------
    thresholds
        ``[(cutoff, css), ...]`` ordered from best to worst.  Each entry means
        "if the value is >= cutoff (or <= cutoff when ``higher_is_better=False``),
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
_JUDGE_BANDS: list[tuple[float, str]] = [
    (80, "color: #4ade80; font-weight: 600"),
    (50, "color: #fb923c; font-weight: 600"),
    (0,  "color: #f87171; font-weight: 600"),
]
_LATENCY_BANDS: list[tuple[float, str]] = [
    (5,  "color: #4ade80; font-weight: 600"),
    (10, "color: #fb923c; font-weight: 600"),
    (99999, "color: #f87171; font-weight: 600"),
]
_STATUS_COLORS: dict[str, str] = {
    "PASS": "color: #4ade80; font-weight: bold",
    "REVIEW": "color: #f87171; font-weight: bold",
}


def _find_column(df: pd.DataFrame, *candidates: str) -> str | None:
    """Return the first column name from *candidates* that exists in *df*."""
    return next((c for c in candidates if c in df.columns), None)


def style_case_dataframe(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Apply conditional color formatting to metric, status, and latency columns.

    Works with both raw keys and label-renamed columns.
    """
    all_metric_keys = METRIC_COLUMNS + RETRIEVAL_COLUMNS + CHUNK_COLUMNS
    all_metric_labels = [METRIC_LABELS.get(k, k) for k in all_metric_keys]
    # Map label columns back to their underlying metric keys so we can still
    # resolve the registry-declared fmt when the caller has label-renamed the
    # dataframe.
    _label_to_key = {METRIC_LABELS.get(k, k): k for k in all_metric_keys}
    score_cols = [c for c in all_metric_keys + all_metric_labels if c in df.columns]

    styler = df.style

    # 0-1 score columns
    if score_cols:
        styler = styler.map(
            lambda v: _color_by_thresholds(v, _SCORE_BANDS),
            subset=score_cols,
        )
        styler = styler.format(
            {c: f"{{:{_fmt_for(_label_to_key.get(c, c))}}}" for c in score_cols},
            na_rep=NOT_EVALUATED_LABEL,
        )

    # Status column
    if "status" in df.columns:
        styler = styler.map(lambda v: _STATUS_COLORS.get(v, ""), subset=["status"])

    # Avg judge score (0-100 scale)
    judge_col = _find_column(df, "avg_judge_score", METRIC_LABELS.get("avg_judge_score", ""))
    if judge_col:
        styler = styler.map(lambda v: _color_by_thresholds(v, _JUDGE_BANDS), subset=[judge_col])
        styler = styler.format({judge_col: f"{{:{_fmt_for('avg_judge_score')}}}"}, na_rep=NOT_EVALUATED_LABEL)

    # Latency (lower is better) — always-on, so NaN only on legacy rows.
    latency_col = _find_column(df, "latency_seconds", METRIC_LABELS.get("latency_seconds", ""))
    if latency_col:
        styler = styler.map(
            lambda v: _color_by_thresholds(v, _LATENCY_BANDS, higher_is_better=False),
            subset=[latency_col],
        )
        styler = styler.format({latency_col: f"{{:{_fmt_for('latency_seconds')}}}s"}, na_rep="")

    styler = styler.set_table_styles([
        {"selector": "th", "props": [("font-size", "0.8rem")]},
        {"selector": "thead th", "props": [("border-bottom", "1px solid #4a5568")]},
        {"selector": "td", "props": [("font-size", "0.85rem")]},
    ])
    styler = styler.hide(axis="index")

    return styler


NOT_EVALUATED_LABEL = "Not evaluated"


def format_metric_value(value: object, fmt: str = ".3f", suffix: str = "") -> str:
    """Format a metric value for display.

    Returns ``"Not evaluated"`` when the value is ``None`` / ``NaN`` — which
    is how the harness signals that a toggleable metric group (judge, source,
    chunk) was turned off in ``eval_config.ENABLED_METRIC_GROUPS`` for the
    run being displayed.
    """
    if value is None:
        return NOT_EVALUATED_LABEL
    if isinstance(value, float) and pd.isna(value):
        return NOT_EVALUATED_LABEL
    return f"{value:{fmt}}{suffix}"


def compute_delta(current: float | None, previous: float | None, fmt: str = ".1f") -> str | None:
    """Compute a display-ready delta string, or None when it can't be computed."""
    if current is None or previous is None:
        return None
    if isinstance(current, float) and pd.isna(current):
        return None
    if isinstance(previous, float) and pd.isna(previous):
        return None
    diff = current - previous
    return f"{diff:+{fmt}}"


def _prev_val(prev_row: pd.Series | None, col: str) -> float | None:
    """Safely extract a value from the previous run row."""
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
        f"**Cases:** {int(run_row['case_count'])}  \n"
        f"**Time:** {run_row['run_timestamp']:%Y-%m-%d %H:%M}"
    )

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
        second line. Works together with ``white-space: pre-line`` in the
        custom CSS, which honors the embedded ``\\n``.
        """
        raw = METRIC_LABELS.get(col_name, fallback)
        if " " in raw:
            idx = raw.rfind(" ")
            return raw[:idx] + "\n" + raw[idx + 1:]
        return raw + "\n\u00a0"

    # --- KPI row 1: core performance ---
    st.subheader("Average Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    _prev_pass = _prev_val(prev_row, "pass_rate")
    c1.metric(
        _kpi_label("pass_rate", "Pass Rate"),
        f"{run_row['pass_rate']:.0%}",
        delta=compute_delta(run_row["pass_rate"] * 100, _prev_pass * 100 if _prev_pass is not None else None),
        delta_color="normal",
    )
    _case_fmt = _fmt_for("avg_case_score")
    c2.metric(
        _kpi_label("avg_case_score", "Avg Judge Score"),
        format_metric_value(run_row["avg_case_score"], _case_fmt),
        delta=compute_delta(run_row["avg_case_score"], _prev_val(prev_row, "avg_case_score"), fmt=_case_fmt),
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

    # --- KPI row 2: retrieval quality ---
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

    # --- KPI row 3: chunk retrieval quality ---
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
        display_cols = [
            "case_id", "category", "status", "avg_judge_score",
            *METRIC_COLUMNS,
            *RETRIEVAL_COLUMNS,
            *CHUNK_COLUMNS,
            "latency_seconds",
        ]
        available_display = [c for c in display_cols if c in run_cases.columns]
        display_df = run_cases[available_display].reset_index(drop=True)
        display_df = display_df.rename(columns=METRIC_LABELS)
        styled_df = style_case_dataframe(display_df)
        table_height = min(len(run_cases) * 38 + 40, 600)
        _col1 = METRIC_LABELS.get("case_id", "case_id")
        st.dataframe(
            styled_df,
            height=table_height,
            use_container_width=True,
            column_config={
                _col1: st.column_config.Column(pinned=True),
            },
        )

        # --- Metric bar chart ---
        st.subheader("Metric Scores by Case")
        metric_display = [c for c in METRIC_COLUMNS if c in run_cases.columns]
        if metric_display:
            all_case_ids = run_cases["case_id"].tolist()
            selected_cases = st.multiselect(
                "Filter cases",
                options=all_case_ids,
                default=all_case_ids,
            )
            bar_df = run_cases[run_cases["case_id"].isin(selected_cases)] if selected_cases else run_cases
            st.plotly_chart(
                build_grouped_bar(bar_df, "case_id", metric_display),
                key="bar_summary",
                use_container_width=True,
            )

        st.divider()

        # --- Answer detail ---
        st.subheader("Answer Detail")
        for _, case_row in run_cases.iterrows():
            is_pass = case_row["status"] == "PASS"
            icon = "\u2705" if is_pass else "\u26a0\ufe0f"
            score_str = (
                f"{case_row['avg_judge_score']:{_fmt_for('avg_judge_score')}}/100"
                if pd.notna(case_row.get("avg_judge_score"))
                else NOT_EVALUATED_LABEL
            )

            with st.expander(f"{icon}  **{case_row['case_id']}**  \u2014  {case_row['status']}  ({score_str})"):
                detail_left, detail_right = st.columns([3, 1])
                with detail_left:
                    st.markdown(f"**Question**")
                    st.info(case_row["question"])
                    st.markdown(f"**Expected answer**")
                    st.success(case_row["expected_output"])
                    st.markdown(f"**Agent answer**")
                    if is_pass:
                        st.success(case_row["answer"])
                    else:
                        st.warning(case_row["answer"])
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
            st.markdown("**LLM Generation Metrics**")
            st.caption("Average scores across all cases \u2014 identifies weak spots in answer quality. Hallucination is shown separately below.")
            if radar_metrics:
                st.plotly_chart(build_radar_chart(run_cases, radar_metrics, color="#636EFA"), key="radar_llm", use_container_width=True)

        with col_radar_ret:
            st.markdown("**Source Retrieval Metrics**")
            st.caption("Source-file level \u2014 highlights weak spots in which documents the retriever picks.")
            if available_retrieval:
                st.plotly_chart(build_radar_chart(run_cases, available_retrieval, color="#00CC96"), key="radar_retrieval", use_container_width=True)

        # Second row: chunk radar on the left, empty column on the right so the
        # chart keeps the same width and padding as the row above.
        col_radar_chunk, _col_radar_spacer = st.columns(2)
        with col_radar_chunk:
            st.markdown("**Chunk Retrieval Metrics**")
            st.caption("Chunk level \u2014 highlights whether the *right passages* were retrieved, not just the right files.")
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
                col_hallu, _col_hallu_spacer = st.columns([1, 3])
                with col_hallu:
                    st.markdown("**Hallucination** &nbsp; <span style='color:#888;font-size:0.85em'>(lower = better)</span>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div style='font-size:2.2em;font-weight:600;color:{hallu_color};line-height:1.1'>{hallu_mean:.2f}</div>"
                        f"<div style='color:{hallu_color};font-size:0.9em'>{hallu_label} \u2014 threshold {threshold_val:.2f}</div>",
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
        st.caption("Pearson correlation \u2014 does retrieval quality drive answer quality? Hallucination is excluded (lower = better, would invert the sign).")
        _CORR_EXCLUDE = {"hit_at_k", "chunk_hit_at_k"}
        all_numeric = [c for c in correlation_metrics + available_retrieval + available_chunk if c not in _CORR_EXCLUDE]
        if len(all_numeric) >= 2:
            st.plotly_chart(build_correlation_heatmap(run_cases, all_numeric), key="heatmap", use_container_width=True)

        st.divider()

        # --- Latency breakdown ---
        st.subheader("Latency Breakdown by Case")
        st.caption("Stacked bar: retrieval time vs LLM generation time per case.")
        fig_latency = build_stacked_latency(run_cases)
        if fig_latency:
            st.plotly_chart(fig_latency, key="latency_bar", use_container_width=True)


# ===================================================================
# Tab 3 — Historical Trends
# ===================================================================

with tab_trends:

    st.caption("Historical trends and comparisons across all evaluation runs.")

    if len(runs_df) < 2:
        st.info("Trends require at least 2 evaluation runs. Only 1 run recorded so far.")
    else:
        trend_df = runs_df.sort_values("run_timestamp")

        # --- Summary scores ---
        st.subheader("Summary Scores Over Time")
        summary_trend_cols = [
            "avg_case_score", "pass_rate",
            "avg_hit_at_k", "avg_mrr", "avg_precision_at_k",
        ]
        available_summary = [c for c in summary_trend_cols if c in trend_df.columns]
        if available_summary:
            st.plotly_chart(
                build_trend_line(trend_df, "run_timestamp", available_summary),
                key="summary_trend",
                use_container_width=True,
            )

        # --- DeepEval metric averages ---
        st.subheader("DeepEval Metric Averages Over Time")
        metric_avg_records = []
        for row in trend_df.itertuples():
            try:
                avgs = json.loads(row.metric_averages) if isinstance(row.metric_averages, str) else {}
            except Exception:
                avgs = {}
            avgs["run_timestamp"] = row.run_timestamp
            metric_avg_records.append(avgs)

        if metric_avg_records:
            metric_trend_df = pd.DataFrame(metric_avg_records).set_index("run_timestamp")
            metric_trend_df = metric_trend_df.apply(pd.to_numeric, errors="coerce")
            metric_trend_df = metric_trend_df.dropna(how="all", axis=1)
            if not metric_trend_df.empty:
                available_metric_trends = list(metric_trend_df.columns)
                st.plotly_chart(
                    build_trend_line(
                        metric_trend_df.reset_index(), "run_timestamp",
                        available_metric_trends, y_range=[0, 1.05],
                    ),
                    key="metric_trend",
                    use_container_width=True,
                )

        # --- Latency trends ---
        st.subheader("Latency Over Time")
        latency_trend_cols = ["avg_latency_seconds", "avg_retrieval_latency_seconds", "avg_llm_latency_seconds"]
        available_latency_trend = [c for c in latency_trend_cols if c in trend_df.columns]
        if available_latency_trend:
            latency_melted = trend_df[["run_timestamp"] + available_latency_trend].melt(
                id_vars="run_timestamp", var_name="component", value_name="seconds",
            )
            latency_melted["component"] = latency_melted["component"].map({
                "avg_latency_seconds": "Total",
                "avg_retrieval_latency_seconds": "Retrieval",
                "avg_llm_latency_seconds": "LLM Generation",
            })
            fig_lat = px.line(
                latency_melted, x="run_timestamp", y="seconds",
                color="component", markers=True,
            )
            fig_lat.update_layout(
                **PLOTLY_LAYOUT_DEFAULTS,
                xaxis_title="",
                yaxis_title="Seconds",
                yaxis=dict(gridcolor="#f0f0f0"),
                legend_title="",
                height=380,
            )
            st.plotly_chart(fig_lat, key="latency_trend", use_container_width=True)

    # --- Case tracking ---
    st.divider()
    st.subheader("Track a Case Across Runs")
    st.caption("Select a case to see how its metrics evolve over evaluation runs.")
    case_ids = sorted(cases_df["case_id"].unique())
    if case_ids:
        selected_case = st.selectbox("Case", case_ids, label_visibility="collapsed")
        case_history = cases_df[cases_df["case_id"] == selected_case].sort_values("run_timestamp")

        if len(case_history) >= 2:
            st.markdown("**LLM Metrics**")
            metric_history_cols = [c for c in METRIC_COLUMNS if c in case_history.columns]
            if metric_history_cols:
                st.plotly_chart(
                    build_trend_line(case_history, "run_timestamp", metric_history_cols, y_range=[0, 1.05]),
                    key="case_metric_trend",
                    use_container_width=True,
                )

            st.markdown("**Source Retrieval Metrics**")
            retrieval_history_cols = [c for c in RETRIEVAL_COLUMNS if c in case_history.columns]
            if retrieval_history_cols:
                st.plotly_chart(
                    build_trend_line(case_history, "run_timestamp", retrieval_history_cols, y_range=[0, 1.05]),
                    key="case_retrieval_trend",
                    use_container_width=True,
                )

            st.markdown("**Chunk Retrieval Metrics**")
            chunk_history_cols = [c for c in CHUNK_COLUMNS if c in case_history.columns]
            if chunk_history_cols:
                st.plotly_chart(
                    build_trend_line(case_history, "run_timestamp", chunk_history_cols, y_range=[0, 1.05]),
                    key="case_chunk_trend",
                    use_container_width=True,
                )
            else:
                st.caption("No chunk metrics recorded for this case.")
        else:
            st.info("This case has only been evaluated in one run so far.")


# ===================================================================
# Tab 4 — Metrics Guide
# ===================================================================

with tab_guide:

    st.caption("Reference guide for the dashboard tabs, metrics, and verdict logic.")

    # --- Dashboard tabs overview ---
    st.subheader("Dashboard Tabs")

    st.markdown(
        "**Run Summary** — Top-level KPIs (pass rate, avg judge score, latency), "
        "retrieval quality averages, a colour-coded per-case results table, "
        "grouped bar charts of metric scores, and expandable answer details for each case."
    )
    st.markdown(
        "**Deep Analysis** — Radar charts comparing LLM generation vs retrieval metric "
        "balance, score distribution histograms, a correlation heatmap showing how "
        "metrics relate to each other, and a stacked latency breakdown per case."
    )
    st.markdown(
        "**Historical Trends** — Line charts tracking summary scores, DeepEval metric "
        "averages, and latency across all evaluation runs over time. Includes a "
        "per-case tracker to follow a single case across runs. This tab is especially "
        "important because the underlying models (both the agent LLM and the judge LLM) "
        "can change over time — provider updates, version bumps, or switching models entirely "
        "can silently shift evaluation scores. Tracking trends across runs lets you detect "
        "regressions early and distinguish genuine pipeline improvements from model-driven "
        "score changes."
    )

    st.divider()

    # --- Metrics at a glance ---
    st.subheader("Metrics at a Glance")

    st.markdown("**LLM-Judged (DeepEval)**")
    st.dataframe(
        pd.DataFrame([
            {"Metric": "Answer Relevancy", "Key": "answer_relevancy", "Description": "Is the answer on-topic for the question?"},
            {"Metric": "Faithfulness", "Key": "faithfulness", "Description": "Are all claims supported by the retrieved context?"},
            {"Metric": "Contextual Precision", "Key": "contextual_precision", "Description": "Are relevant chunks ranked above irrelevant ones?"},
            {"Metric": "Contextual Recall", "Key": "contextual_recall", "Description": "Does the context cover all needed information?"},
            {"Metric": "Contextual Relevancy", "Key": "contextual_relevancy", "Description": "What fraction of retrieved chunks are relevant?"},
            {"Metric": "Hallucination", "Key": "hallucination", "Description": "Does the answer contradict the context?"},
            {"Metric": "Grounded Correctness (GEval)", "Key": "correctness_g_eval", "Description": "Is the answer correct compared to the expected output?"},
        ]),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("**Deterministic \u2014 Source-level Retrieval** (relevance by filename)")
    st.dataframe(
        pd.DataFrame([
            {"Metric": "Hit@k", "Key": "hit_at_k", "Description": "Did at least one expected source appear in the top-k?"},
            {"Metric": "MRR", "Key": "mrr", "Description": "How early does the first relevant result appear?"},
            {"Metric": "Precision@k", "Key": "precision_at_k", "Description": "What fraction of retrieved results are relevant?"},
            {"Metric": "Recall@k", "Key": "recall_at_k", "Description": "What fraction of expected sources were retrieved?"},
            {"Metric": "NDCG@k", "Key": "ndcg_at_k", "Description": "How good is the overall ranking quality?"},
        ]),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("**Deterministic \u2014 Chunk-level Retrieval** (relevance by snippet substring match)")
    st.dataframe(
        pd.DataFrame([
            {"Metric": "Chunk Hit@k", "Key": "chunk_hit_at_k", "Description": "Did at least one expected snippet appear in any retrieved chunk?"},
            {"Metric": "Chunk MRR", "Key": "chunk_mrr", "Description": "How early does the first snippet-matching chunk appear?"},
            {"Metric": "Chunk Precision@k", "Key": "chunk_precision_at_k", "Description": "What fraction of retrieved chunks contain any expected snippet?"},
            {"Metric": "Chunk Recall@k", "Key": "chunk_recall_at_k", "Description": "What fraction of expected snippets were found in a retrieved chunk?"},
            {"Metric": "Chunk NDCG@k", "Key": "chunk_ndcg_at_k", "Description": "How good is the chunk ranking under snippet-level relevance?"},
        ]),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("**Deterministic \u2014 Other**")
    st.dataframe(
        pd.DataFrame([
            {"Metric": "Metadata Match Ratio", "Key": "metadata_match_ratio", "Description": "Do retrieved results satisfy the metadata filters?"},
            {"Metric": "Backend Distribution", "Key": "backend_distribution", "Description": "How are results split across search backends?"},
            {"Metric": "Required Keyword Hit Rate", "Key": "required_keyword_hit_rate", "Description": "Does the answer contain the required key terms?"},
            {"Metric": "Disallowed Keyword Hits", "Key": "disallowed_keyword_hits", "Description": "Does the answer avoid disallowed terms?"},
        ]),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("**Latency**")
    st.dataframe(
        pd.DataFrame([
            {"Metric": "Latency", "Key": "latency_seconds", "Description": "Total wall-clock time for the agent to answer a case."},
            {"Metric": "Retrieval Latency", "Key": "retrieval_latency_seconds", "Description": "Time spent in the direct retriever call."},
            {"Metric": "LLM Latency", "Key": "llm_latency_seconds", "Description": "Estimated LLM generation time (total \u2212 retrieval)."},
        ]),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("**Summary**")
    st.dataframe(
        pd.DataFrame([
            {"Metric": "Pass Rate", "Key": "pass_rate", "Description": "Fraction of cases with a PASS verdict (run-level)."},
            {"Metric": "Average Judge Score", "Key": "avg_judge_score", "Description": "Mean of all LLM-judged scores, expressed as 0\u2013100 (per case and run)."},
        ]),
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    # --- Score colour legend ---
    st.subheader("Score Colours")
    st.markdown(
        "The per-case results table uses coloured text to highlight score quality at a glance.\n\n"
        "**LLM-judged & retrieval metrics** (0\u20131 scale)\n\n"
        "| Colour | Range | Meaning |\n"
        "|--------|-------|---------|\n"
        "| :green[green] | \u2265 0.8 | Good — the metric is comfortably above the pass threshold |\n"
        "| :orange[orange] | \u2265 0.5 | Borderline — at or near the verdict gate threshold |\n"
        "| :red[red] | < 0.5 | Poor — below the pass threshold, likely triggers REVIEW |\n\n"
        "**Avg Judge Score** (0\u2013100 scale)\n\n"
        "Same logic scaled to 100: green \u2265 80, orange \u2265 50, red < 50.\n\n"
        "**Latency**\n\n"
        "| Colour | Range | Meaning |\n"
        "|--------|-------|---------|\n"
        "| :green[green] | \u2264 5 s | Fast |\n"
        "| :orange[orange] | \u2264 10 s | Moderate |\n"
        "| :red[red] | > 10 s | Slow — may indicate retrieval or LLM bottlenecks |\n\n"
        "**Status column:** PASS is green, REVIEW is red."
    )

    st.divider()

    # --- Detailed metric explanations ---
    st.subheader("LLM-Judged Metrics (DeepEval)")
    st.markdown(
        "These metrics use a judge LLM (configured via `JUDGE_MODEL` in `eval_config.py`) "
        "to score each test case. All return a score between 0 and 1, a reason, and a pass/fail "
        "result based on the configured threshold."
    )

    with st.expander("Answer Relevancy"):
        st.markdown(
            "**What it evaluates:** Whether the agent's answer is relevant to the user's question. "
            "It checks that the response actually addresses what was asked rather than providing "
            "tangential or off-topic information.\n\n"
            "**How it is calculated:** The judge LLM analyses the input question and the actual output. "
            "It generates synthetic questions that the actual output could answer, then measures the "
            "overlap between those synthetic questions and the original input. A high overlap means the "
            "answer is on-topic.\n\n"
            "**Why it matters:** A RAG system can retrieve perfect context but still produce an answer "
            "that drifts from the question. This is one of the two gate metrics for the PASS/REVIEW verdict."
        )

    with st.expander("Faithfulness"):
        st.markdown(
            "**What it evaluates:** Whether every claim in the agent's answer is supported by the "
            "retrieved context. A faithful answer makes no statements that go beyond what the context provides.\n\n"
            "**How it is calculated:** The judge LLM extracts individual claims from the actual output and "
            "checks each one against the retrieval context. The score is the fraction of claims that are "
            "fully supported. A score of 1.0 means every claim traces back to a retrieved chunk.\n\n"
            "**Why it matters:** Faithfulness is the core safeguard against hallucination in RAG. This is "
            "one of the two gate metrics for the PASS/REVIEW verdict."
        )

    with st.expander("Contextual Precision"):
        st.markdown(
            "**What it evaluates:** Whether the relevant chunks in the retrieval context are ranked higher "
            "than irrelevant ones. It measures ranking quality, not just presence.\n\n"
            "**How it is calculated:** The judge LLM classifies each node in the retrieval context as relevant "
            "or irrelevant based on the expected output. It then computes a weighted precision score that "
            "rewards relevant items appearing earlier in the list.\n\n"
            "**Why it matters:** In a hybrid search system, retrieval order matters. If relevant documents are "
            "buried below noise, the LLM may miss or deprioritise them."
        )

    with st.expander("Contextual Recall"):
        st.markdown(
            "**What it evaluates:** Whether the retrieval context contains all the information needed to "
            "produce the expected output. It checks for completeness of the retrieved evidence.\n\n"
            "**How it is calculated:** The judge LLM breaks the expected output into individual sentences or "
            "claims, then checks whether each one can be attributed to at least one node in the retrieval "
            "context. The score is the fraction of expected claims that are supported.\n\n"
            "**Why it matters:** A retrieval system might return chunks that are individually relevant but "
            "collectively miss key facts needed for a complete answer."
        )

    with st.expander("Contextual Relevancy"):
        st.markdown(
            "**What it evaluates:** Whether each chunk in the retrieval context is relevant to the input "
            "question. Unlike contextual precision (which focuses on ranking), this measures overall noise level.\n\n"
            "**How it is calculated:** The judge LLM evaluates each retrieval context node against the input "
            "question. The score is the fraction of retrieved nodes that are relevant. A score of 1.0 means "
            "every retrieved chunk was useful.\n\n"
            "**Why it matters:** Retrieving too many irrelevant chunks dilutes the LLM's attention and can "
            "lead to worse answers or higher latency."
        )

    with st.expander("Hallucination"):
        st.markdown(
            "**What it evaluates:** Whether the agent's answer contradicts the provided context. While "
            "faithfulness checks for unsupported claims, hallucination specifically detects factual contradictions.\n\n"
            "**How it is calculated:** The judge LLM compares the actual output against the context and "
            "determines whether any statements in the output contradict the context. The score represents "
            "the degree to which the output is free of contradictions (higher is better).\n\n"
            "**Why it matters:** A hallucinated answer is worse than an incomplete one \u2014 it actively misleads "
            "the user. This provides an additional safety net beyond faithfulness."
        )

    with st.expander("Grounded Correctness (GEval)"):
        st.markdown(
            "**What it evaluates:** Whether the agent's answer correctly addresses the user's question, "
            "covers the important facts from the expected output, and avoids unsupported claims.\n\n"
            "**How it is calculated:** GEval uses a chain-of-thought approach. The judge LLM receives a custom "
            "criteria string along with the input, actual output, and expected output. It generates evaluation "
            "steps, scores each step, and combines them into a final score (0\u20131).\n\n"
            "**Why it matters:** The other metrics evaluate individual dimensions, but none directly ask "
            "\"is this answer correct?\" GEval provides a holistic correctness check."
        )

    st.divider()

    # --- Deterministic retrieval metrics (source-level) ---
    st.subheader("Deterministic Retrieval Metrics (Source-level)")
    st.markdown(
        "Source-level retrieval metrics decide relevance at the filename level \u2014 any chunk whose "
        "source filename is in `expected_sources` counts as relevant. Defined in `eval_metrics.py` and "
        "computed from the direct retrieval results."
    )

    with st.expander("Hit@k"):
        st.markdown(
            "**What it evaluates:** Binary — did at least one expected source appear anywhere in the "
            "top-k retrieved results?\n\n"
            "**How it is calculated:** `1.0 if expected \u2229 retrieved else 0.0`. Source names are compared "
            "by filename (case-insensitive). Returns 1.0 if no expected sources are defined.\n\n"
            "**Why it matters:** The most basic retrieval health check \u2014 did the system find *anything* "
            "right? This is one of the two retrieval gate checks for the PASS/REVIEW verdict (must equal 1.0)."
        )

    with st.expander("Mean Reciprocal Rank (MRR)"):
        st.markdown(
            "**What it evaluates:** How early the first relevant result appears in the ranked retrieval list. "
            "Scores 1.0 if the first result is relevant, 0.5 if the second is, 0.33 if the third is, and so on.\n\n"
            "**How it is calculated:** `1 / rank` where rank is the position (1-indexed) of the first retrieved "
            "result whose source filename matches any expected source. Returns 0.0 if no match is found.\n\n"
            "**Why it matters:** In RAG, the top-ranked result often has the most influence on the LLM's answer. "
            "MRR tells you whether the retrieval pipeline is placing the most important document first."
        )

    with st.expander("Precision@k"):
        st.markdown(
            "**What it evaluates:** The fraction of all retrieved results (top k) that come from an expected source.\n\n"
            "**How it is calculated:** `(results from expected sources) / (total results)`. Source matching is "
            "by filename (case-insensitive). Returns 1.0 if no expected sources are defined.\n\n"
            "**Why it matters:** A low precision means the retrieval pipeline is returning a lot of noise alongside "
            "the relevant documents, wasting context window space."
        )

    with st.expander("Recall@k"):
        st.markdown(
            "**What it evaluates:** The fraction of expected source documents that appear in the top-k "
            "retrieved results. Measures retrieval completeness.\n\n"
            "**How it is calculated:** `|expected \u2229 retrieved| / |expected|`. Where hit@k tells you "
            "*whether* anything relevant was found, recall@k tells you *how much* of the expected set was found.\n\n"
            "**Why it matters:** Reported alongside precision@k for a complete precision\u2013recall picture. "
            "Together they reveal the trade-off between retrieving broadly and retrieving precisely."
        )

    with st.expander("NDCG@k"):
        st.markdown(
            "**What it evaluates:** The quality of the ranking compared to the ideal ranking where all "
            "relevant results are at the top. A ranking-aware metric that penalises relevant results appearing "
            "lower in the list.\n\n"
            "**How it is calculated:** Uses binary relevance (1 if source matches expected, 0 otherwise). "
            "DCG = sum of (relevance / log2(rank + 2)). NDCG = DCG / ideal DCG.\n\n"
            "**Why it matters:** MRR only looks at the first relevant result. NDCG evaluates the entire ranked "
            "list, making it the standard metric for evaluating ranked retrieval."
        )

    st.divider()

    # --- Chunk-level retrieval metrics ---
    st.subheader("Chunk-level Retrieval Metrics")
    st.markdown(
        "These mirror the source-level metrics but apply snippet-substring matching: a chunk is relevant "
        "when any snippet from the case's `expected_chunks` appears (normalized) in the chunk's `page_content`. "
        "Cases with an empty `expected_chunks` list score 1.0. Catches pipelines that find the right documents "
        "but rank the wrong chunk within them."
    )

    with st.expander("Chunk Hit@k"):
        st.markdown(
            "**What it evaluates:** Binary \u2014 did any retrieved chunk contain any expected snippet?\n\n"
            "**How it is calculated:** `1.0 if any(snippet in chunk.page_content for snippet in expected_chunks "
            "for chunk in results) else 0.0`. Both sides are normalized before comparison.\n\n"
            "**Why it matters:** Catches cases where the right *file* is retrieved but the actual "
            "answer-containing chunk is missed \u2014 a failure mode source-level Hit@k cannot detect."
        )

    with st.expander("Chunk MRR"):
        st.markdown(
            "**What it evaluates:** The reciprocal rank of the first retrieved chunk containing any expected "
            "snippet. Measures how early the right *passage* appears, not just the right file.\n\n"
            "**How it is calculated:** Iterate retrieved chunks in rank order; return `1 / rank` (1-indexed) "
            "for the first chunk whose normalized page_content contains any normalized expected snippet. "
            "Returns 0.0 if no chunk matches.\n\n"
            "**Why it matters:** Reveals whether the chunk ranker places the relevant passage near the top or "
            "buries it behind less relevant chunks from the same file."
        )

    with st.expander("Chunk Precision@k"):
        st.markdown(
            "**What it evaluates:** The fraction of retrieved chunks containing at least one expected snippet "
            "\u2014 how much of the context window is being spent on useful passages.\n\n"
            "**How it is calculated:** `(chunks matching any snippet) / (total retrieved chunks)`. Returns 1.0 "
            "if `expected_chunks` is empty.\n\n"
            "**Why it matters:** High source-level precision but low chunk-level precision means you are "
            "retrieving the right files but wasting slots on unhelpful sections within them."
        )

    with st.expander("Chunk Recall@k"):
        st.markdown(
            "**What it evaluates:** The fraction of expected snippets that appear in at least one retrieved "
            "chunk \u2014 coverage of the known-good passages.\n\n"
            "**How it is calculated:** `|{snippet : snippet \u2208 some retrieved chunk}| / |expected_chunks|`. "
            "Unlike source-level recall, this tracks distinct *passages*, so two expected passages from the "
            "same file count separately.\n\n"
            "**Why it matters:** A case can retrieve the right file but miss multiple required passages. "
            "Chunk recall surfaces that \u2014 source-level recall would read 1.0 despite the gap."
        )

    with st.expander("Chunk NDCG@k"):
        st.markdown(
            "**What it evaluates:** Ranking quality when each chunk gets a binary relevance label of 1 if it "
            "contains any expected snippet, else 0.\n\n"
            "**How it is calculated:** Build a binary relevance list by snippet-substring check per chunk. "
            "Apply DCG/IDCG: DCG = sum(rel_i / log2(rank_i + 2)); NDCG = DCG / IDCG. Returns 1.0 if "
            "`expected_chunks` is empty.\n\n"
            "**Why it matters:** The chunk-level counterpart to NDCG@k \u2014 evaluates whether relevant "
            "*passages* are ranked before irrelevant ones, independent of file-level grouping."
        )

    st.divider()

    # --- Deterministic — Other metrics ---
    st.subheader("Deterministic \u2014 Other Metrics")
    st.markdown(
        "Metrics that don't fit the source/chunk retrieval axis. Some inspect metadata or backend "
        "bookkeeping on the retrieval side; others check the agent's answer text against keyword lists. "
        "Defined in `eval_metrics.py`."
    )

    with st.expander("Metadata Match Ratio"):
        st.markdown(
            "**What it evaluates:** The fraction of retrieved results that satisfy all metadata filters defined "
            "on the eval case (e.g. `category=policy`).\n\n"
            "**How it is calculated:** For each retrieved result, check whether all key-value pairs in the case's "
            "metadata filters match the result's metadata. Score = matching / total.\n\n"
            "**Why it matters:** When a case specifies metadata filters, those are hard constraints. If results "
            "violate the filter, the filtering logic is broken. This is the second retrieval gate check "
            "(threshold \u2265 0.8)."
        )

    with st.expander("Backend Distribution"):
        st.markdown(
            "**What it evaluates:** Counts retrieved results by search backend (e.g. `fts`, `vector`). "
            "Not a score \u2014 a diagnostic distribution.\n\n"
            "**How it is calculated:** Groups retrieval results by the `backend` field. Returns a dict like "
            '`{"fts": 3, "vector": 2}`.\n\n'
            "**Why it matters:** The hybrid search agent fuses full-text and vector search. This tells you "
            "whether both backends are actively contributing. If all results come from one backend, the "
            "fusion mechanism may not be working as intended."
        )

    with st.expander("Required Keyword Hit Rate"):
        st.markdown(
            "**What it evaluates:** The fraction of required keywords (defined in the eval case) that "
            "appear in the agent's answer.\n\n"
            "**How it is calculated:** Keywords and answer are normalised (lowercased, accents stripped, "
            "punctuation removed). Score = keywords found / total required keywords.\n\n"
            "**Why it matters:** Some questions demand specific terms in the answer (e.g. a regulation "
            "number). This is one of the keyword gate checks (threshold \u2265 0.5)."
        )

    with st.expander("Disallowed Keyword Hits"):
        st.markdown(
            "**What it evaluates:** The count of disallowed keywords that appear in the agent's answer.\n\n"
            "**How it is calculated:** Same normalisation as required keywords. Counts how many disallowed "
            "keywords appear as substrings in the normalised answer.\n\n"
            "**Why it matters:** Some answers should avoid certain terms. Any non-zero count is a failure "
            "signal. This is the second keyword gate check (must equal 0)."
        )

    st.divider()

    # --- Latency metrics ---
    st.subheader("Latency Metrics")
    st.markdown(
        "Always-on wall-clock timings captured during each case run. No pass threshold \u2014 they exist to "
        "surface performance regressions alongside quality metrics."
    )

    with st.expander("Latency"):
        st.markdown(
            "**What it evaluates:** Total wall-clock time, in seconds, for the agent to produce an answer for "
            "a single case. Covers the full graph invocation from input to final answer.\n\n"
            "**How it is calculated:** Measured around the agent invocation with `time.perf_counter()` before "
            "and after; the delta is recorded.\n\n"
            "**Why it matters:** End-to-end latency is what a user experiences. Spikes often point to "
            "over-retrieval or slow tool calls, and the run-level average helps catch regressions after prompt "
            "or model changes."
        )

    with st.expander("Retrieval Latency"):
        st.markdown(
            "**What it evaluates:** Time spent inside the direct retriever call \u2014 the hybrid search step "
            "that fetches candidate chunks from the index.\n\n"
            "**How it is calculated:** The retriever is invoked directly (outside of the full graph) and timed "
            "with `time.perf_counter()`. Best-effort estimate.\n\n"
            "**Why it matters:** Isolating retrieval cost from LLM cost makes it obvious which side of the "
            "pipeline is slow. A spike here usually points to index issues, large `k`, or slow backend fusion."
        )

    with st.expander("LLM Latency"):
        st.markdown(
            "**What it evaluates:** Estimated time the generation step took, derived as "
            "`latency_seconds \u2212 retrieval_latency_seconds`.\n\n"
            "**How it is calculated:** Subtraction of the two timings, clamped to \u2265 0.\n\n"
            "**Why it matters:** When total latency grows, this tells you whether retrieval or the LLM call is "
            "responsible \u2014 which determines whether to tune the index or switch generation models."
        )

    st.divider()

    # --- Summary metrics ---
    st.subheader("Summary Metrics")

    with st.expander("Pass Rate"):
        st.markdown(
            "**What it evaluates:** The fraction of cases in a run whose final status is `PASS`. Run-level "
            "rollup \u2014 there is no per-case equivalent (each case has a boolean-like status instead).\n\n"
            "**How it is calculated:** `pass_count / case_count`, rounded to 3 decimals. See Verdict Logic "
            "below for how each case's status is determined.\n\n"
            "**Why it matters:** The headline health number for a run. Compresses the three verdict gates "
            "(metrics, retrieval, keywords) into a single score to track across commits or configuration changes."
        )

    with st.expander("Average Judge Score"):
        st.markdown(
            "**What it evaluates:** The mean of all DeepEval LLM-judged metric scores, expressed as a "
            "percentage (0\u2013100). Derived, not independently measured.\n\n"
            "**How it is calculated:** Collects all non-None scores from the 7 DeepEval metrics, computes "
            "their arithmetic mean, and multiplies by 100. Rounded to 1 decimal place. Returns `None` when the "
            "`judge` metric group is disabled. The run-level value is the mean of each case's per-case average.\n\n"
            "**Why it matters:** A single summary number for quick comparison across cases. Useful for "
            "spotting overall quality trends without inspecting each metric individually."
        )

    st.divider()

    # --- Verdict logic ---
    st.subheader("Verdict Logic")
    _gt = gate_thresholds or {}
    _judge_t = _gt.get("judge_threshold", 0.5)
    _meta_t = _gt.get("metadata_match_threshold", 0.8)
    _kw_t = _gt.get("required_keyword_threshold", 0.5)
    st.markdown(
        'The final status field ("PASS" or "REVIEW") is computed by applying three independent gates. '
        "All three must pass for a PASS verdict. Threshold values shown below reflect the selected run's "
        "configured gate thresholds."
    )
    st.dataframe(
        pd.DataFrame([
            {"Gate": "metrics_ok", "Condition": f"faithfulness \u2265 {_judge_t} AND answer_relevancy \u2265 {_judge_t}"},
            {"Gate": "retrieval_ok", "Condition": f"hit_at_k = 1.0 AND metadata_match_ratio \u2265 {_meta_t}"},
            {"Gate": "keywords_ok", "Condition": f"required_keyword_hit_rate \u2265 {_kw_t} AND disallowed_keyword_hits = 0"},
        ]),
        hide_index=True,
        use_container_width=True,
    )
