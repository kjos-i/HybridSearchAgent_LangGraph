"""Streamlit dashboard for browsing evaluation results stored in eval_ledger.db.

Launch from the project root (or the evaluation folder):
    streamlit run evaluation/eval_dashboard.py
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

POLL_SECONDS = 10  # How often to check if new data has arrived

DB_PATH = Path(__file__).resolve().parent / "eval_ledger.db"

METRIC_COLUMNS = [
    "answer_relevancy",
    "faithfulness",
    "contextual_precision",
    "contextual_recall",
    "contextual_relevancy",
    "hallucination",
    "correctness_g_eval",
]

RETRIEVAL_COLUMNS = [
    "source_hit_rate",
    "metadata_match_ratio",
    "mrr",
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
]

METRIC_LABELS = {
    "answer_relevancy": "Answer Relevancy",
    "faithfulness": "Faithfulness",
    "contextual_precision": "Ctx Precision",
    "contextual_recall": "Ctx Recall",
    "contextual_relevancy": "Ctx Relevancy",
    "hallucination": "Hallucination",
    "correctness_g_eval": "Correctness (GEval)",
    "source_hit_rate": "Source Hit Rate",
    "metadata_match_ratio": "Metadata Match",
    "mrr": "MRR",
    "precision_at_k": "Precision@k",
    "recall_at_k": "Recall@k",
    "ndcg_at_k": "NDCG@k",
    "avg_case_score": "Avg Case Score",
    "pass_rate": "Pass Rate",
    "avg_source_hit_rate": "Source Hit Rate",
    "avg_mrr": "MRR",
    "avg_precision_at_k": "Precision@k",
}

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
div[data-testid="stMetric"] label {
    font-size: 0.78rem !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: #8fa8c8;
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
    except Exception:
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

PLOTLY_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=40, r=20, t=30, b=40),
)


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
        fillcolor=f"rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0, 2, 4))}, 0.12)",
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
    """Build a correlation heatmap between the given numeric columns."""
    available = [c for c in columns if c in df.columns]
    if len(available) < 2:
        return go.Figure()

    corr = df[available].corr()
    labels = [METRIC_LABELS.get(c, c) for c in available]

    fig = px.imshow(
        corr.values,
        x=labels,
        y=labels,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        text_auto=".2f",
        aspect="auto",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12),
        height=480,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def build_grouped_bar(
    df: pd.DataFrame,
    id_col: str,
    metric_cols: list[str],
    threshold: float | None = 0.5,
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
    if threshold is not None:
        fig.add_hline(
            y=threshold, line_dash="dash", line_color="#adb5bd", opacity=0.6,
            annotation_text=f"threshold ({threshold})", annotation_position="top left",
            annotation_font_color="#6c757d", annotation_font_size=11,
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
        yaxis_title=y_label,
        yaxis=dict(range=y_range, gridcolor="#f0f0f0") if y_range else dict(gridcolor="#f0f0f0"),
        legend_title="",
        height=380,
    )
    return fig


def style_case_dataframe(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Apply conditional color formatting to metric and retrieval columns."""
    display_metrics = [c for c in METRIC_COLUMNS + RETRIEVAL_COLUMNS if c in df.columns]

    def color_scores(val: object) -> str:
        if not isinstance(val, (int, float)):
            return ""
        if val >= 0.8:
            return "background-color: #d4edda; color: #155724"
        if val >= 0.5:
            return "background-color: #fff3cd; color: #856404"
        return "background-color: #f8d7da; color: #721c24"

    def color_status(val: object) -> str:
        if val == "PASS":
            return "background-color: #d4edda; color: #155724; font-weight: bold"
        if val == "REVIEW":
            return "background-color: #f8d7da; color: #721c24; font-weight: bold"
        return ""

    styler = df.style
    styler = styler.map(color_scores, subset=[c for c in display_metrics if c in df.columns])
    if "status" in df.columns:
        styler = styler.map(color_status, subset=["status"])
    styler = styler.format({c: "{:.3f}" for c in display_metrics if c in df.columns})
    if "latency_seconds" in df.columns:
        styler = styler.format({"latency_seconds": "{:.2f}s"})
    if "avg_metric_score" in df.columns:
        styler = styler.format({"avg_metric_score": "{:.1f}"})

    return styler


def compute_delta(current: float, previous: float | None, fmt: str = ".1f") -> str | None:
    """Compute a display-ready delta string, or None if no previous run."""
    if previous is None:
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
    st.markdown(
        f"**Judge:** {run_row['judge_model']}  \n"
        f"**Threshold:** {run_row['threshold']}  \n"
        f"**Cases:** {int(run_row['case_count'])}  \n"
        f"**Time:** {run_row['run_timestamp']:%Y-%m-%d %H:%M}"
    )

run_cases = cases_df[cases_df["run_id"] == selected_run_id].copy()


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.title("Evaluation Dashboard")
st.caption("HybridSearchAgent \u2014 DeepEval RAG evaluation results")


# ===================================================================
# Tabbed layout
# ===================================================================

tab_summary, tab_analysis, tab_trends = st.tabs([
    "Run Summary",
    "Deep Analysis",
    "Historical Trends",
])


# ===================================================================
# Tab 1 — Run Summary
# ===================================================================

with tab_summary:
    st.caption("Summary of metrics for the selected run.")

    # --- KPI row 1: core performance ---
    st.subheader("Average Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    _prev_pass = _prev_val(prev_row, "pass_rate")
    c1.metric(
        "Pass rate",
        f"{run_row['pass_rate']:.0%}",
        delta=compute_delta(run_row["pass_rate"] * 100, _prev_pass * 100 if _prev_pass is not None else None),
        delta_color="normal",
    )
    c2.metric(
        "Avg score",
        f"{run_row['avg_case_score']:.1f}",
        delta=compute_delta(run_row["avg_case_score"], _prev_val(prev_row, "avg_case_score")),
        delta_color="normal",
    )
    c3.metric(
        "Avg latency",
        f"{run_row['avg_latency_seconds']:.1f}s",
        delta=compute_delta(run_row["avg_latency_seconds"], _prev_val(prev_row, "avg_latency_seconds"), fmt=".1f"),
        delta_color="inverse",
    )
    c4.metric(
        "Retrieval",
        f"{run_row['avg_retrieval_latency_seconds']:.2f}s",
        delta=compute_delta(run_row["avg_retrieval_latency_seconds"], _prev_val(prev_row, "avg_retrieval_latency_seconds"), fmt=".2f"),
        delta_color="inverse",
    )
    c5.metric(
        "LLM",
        f"{run_row['avg_llm_latency_seconds']:.1f}s",
        delta=compute_delta(run_row["avg_llm_latency_seconds"], _prev_val(prev_row, "avg_llm_latency_seconds"), fmt=".1f"),
        delta_color="inverse",
    )

    # --- KPI row 2: retrieval quality ---
    st.subheader("Average Retrieval Quality")
    r1, r2, r3 = st.columns(3)
    r4, r5, r6 = st.columns(3)
    retrieval_kpis = [
        ("Source Hit Rate", "avg_source_hit_rate", r1),
        ("Metadata Match", "avg_metadata_match_ratio", r2),
        ("MRR", "avg_mrr", r3),
        ("Precision@k", "avg_precision_at_k", r4),
        ("Recall@k", "avg_recall_at_k", r5),
        ("NDCG@k", "avg_ndcg_at_k", r6),
    ]
    for label, col_name, col_widget in retrieval_kpis:
        col_widget.metric(
            label,
            f"{run_row[col_name]:.3f}",
            delta=compute_delta(run_row[col_name], _prev_val(prev_row, col_name), fmt=".3f"),
            delta_color="normal",
        )

    st.divider()

    # --- Per-case table ---
    if run_cases.empty:
        st.warning("No case-level data for this run.")
    else:
        st.subheader("Per-case Results")
        if "category" in run_cases.columns:
            categories = sorted(run_cases["category"].dropna().unique())
            if categories:
                selected_categories = st.multiselect(
                    "Filter by category",
                    options=categories,
                )
                if selected_categories:
                    run_cases = run_cases[run_cases["category"].isin(selected_categories)]

        display_cols = [
            "case_id", "category", "status", "avg_metric_score",
            *METRIC_COLUMNS,
            *RETRIEVAL_COLUMNS,
            "latency_seconds",
        ]
        available_display = [c for c in display_cols if c in run_cases.columns]
        styled_df = style_case_dataframe(run_cases[available_display].set_index("case_id"))
        st.dataframe(styled_df, use_container_width=True, height=min(len(run_cases) * 38 + 40, 600))

        # --- Metric bar chart ---
        st.subheader("Metric Scores by Case")
        metric_display = [c for c in METRIC_COLUMNS if c in run_cases.columns]
        if metric_display:
            st.plotly_chart(
                build_grouped_bar(run_cases, "case_id", metric_display),
                key="bar_summary",
                use_container_width=True,
            )

        st.divider()

        # --- Answer detail ---
        st.subheader("Answer Detail")
        for _, case_row in run_cases.iterrows():
            is_pass = case_row["status"] == "PASS"
            icon = "\u2705" if is_pass else "\u26a0\ufe0f"
            score_str = f"{case_row['avg_metric_score']:.0f}/100" if pd.notna(case_row.get("avg_metric_score")) else ""

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
                    st.metric("Score", f"{case_row['avg_metric_score']:.0f}" if pd.notna(case_row.get("avg_metric_score")) else "N/A")
                    st.metric("Latency", f"{case_row['latency_seconds']:.1f}s" if pd.notna(case_row.get("latency_seconds")) else "N/A")
                    st.metric("Source hit", f"{case_row['source_hit_rate']:.0%}" if pd.notna(case_row.get("source_hit_rate")) else "N/A")

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

        # --- Radar charts side by side ---
        st.subheader("Metric Balance")
        col_radar_llm, col_radar_ret = st.columns(2)

        with col_radar_llm:
            st.markdown("**LLM Generation Metrics**")
            st.caption("Average scores across all cases \u2014 identifies weak spots in answer quality.")
            if available_metrics:
                st.plotly_chart(build_radar_chart(run_cases, available_metrics, color="#636EFA"), key="radar_llm", use_container_width=True)

        with col_radar_ret:
            st.markdown("**Retrieval Metrics**")
            st.caption("Average scores across all cases \u2014 identifies weak spots in search quality.")
            if available_retrieval:
                st.plotly_chart(build_radar_chart(run_cases, available_retrieval, color="#00CC96"), key="radar_retrieval", use_container_width=True)

        st.divider()

        # --- Score distribution ---
        st.subheader("Score Distribution")
        st.caption("Select a metric to see how scores spread across cases.")
        all_available = available_metrics + available_retrieval
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
        st.caption("Pearson correlation \u2014 does retrieval quality drive answer quality?")
        all_numeric = available_metrics + available_retrieval
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
            "avg_source_hit_rate", "avg_mrr", "avg_precision_at_k",
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
            col_case_llm, col_case_ret = st.columns(2)
            with col_case_llm:
                st.markdown("**LLM Metrics**")
                metric_history_cols = [c for c in METRIC_COLUMNS if c in case_history.columns]
                if metric_history_cols:
                    st.plotly_chart(
                        build_trend_line(case_history, "run_timestamp", metric_history_cols, y_range=[0, 1.05]),
                        key="case_metric_trend",
                        use_container_width=True,
                    )
            with col_case_ret:
                st.markdown("**Retrieval Metrics**")
                retrieval_history_cols = [c for c in RETRIEVAL_COLUMNS if c in case_history.columns]
                if retrieval_history_cols:
                    st.plotly_chart(
                        build_trend_line(case_history, "run_timestamp", retrieval_history_cols, y_range=[0, 1.05]),
                        key="case_retrieval_trend",
                        use_container_width=True,
                    )
        else:
            st.info("This case has only been evaluated in one run so far.")
