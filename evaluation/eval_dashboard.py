"""Streamlit dashboard for browsing evaluation results stored in eval_ledger.db.

Launch from the project root (or the evaluation folder):
    streamlit run evaluation/eval_dashboard.py
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def load_runs() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM eval_runs ORDER BY run_timestamp DESC", conn)
    if not df.empty:
        df["run_timestamp"] = pd.to_datetime(df["run_timestamp"])
    return df


@st.cache_data(ttl=30)
def load_cases() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM eval_cases ORDER BY run_timestamp DESC, case_id", conn)
    if not df.empty:
        df["run_timestamp"] = pd.to_datetime(df["run_timestamp"])
    return df


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Eval Ledger Dashboard", layout="wide")
st.title("Evaluation Ledger Dashboard")

if not DB_PATH.exists():
    st.warning(f"Database not found at `{DB_PATH}`. Run an evaluation first.")
    st.stop()

runs_df = load_runs()
cases_df = load_cases()

if runs_df.empty:
    st.info("No evaluation runs recorded yet.")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar — run selector
# ---------------------------------------------------------------------------

st.sidebar.header("Run selector")

run_labels = {
    row.run_id: f"Run {row.run_id}  —  {row.run_timestamp:%Y-%m-%d %H:%M}"
    for row in runs_df.itertuples()
}

selected_run_id = st.sidebar.selectbox(
    "Select a run to inspect",
    options=list(run_labels.keys()),
    format_func=lambda rid: run_labels[rid],
)


# ===================================================================
# Section 1 — Trends across all runs
# ===================================================================

st.header("Trends across runs")

if len(runs_df) >= 2:
    trend_df = runs_df.sort_values("run_timestamp")

    # --- Summary score trends ---
    st.subheader("Summary scores over time")
    summary_trend_cols = [
        "avg_case_score", "pass_rate",
        "avg_source_hit_rate", "avg_mrr", "avg_precision_at_k",
    ]
    available_summary = [c for c in summary_trend_cols if c in trend_df.columns]
    st.line_chart(trend_df.set_index("run_timestamp")[available_summary])

    # --- Metric-average trends ---
    st.subheader("DeepEval metric averages over time")
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
        if not metric_trend_df.dropna(how="all", axis=1).empty:
            st.line_chart(metric_trend_df.dropna(how="all", axis=1))

    # --- Latency trends ---
    st.subheader("Latency over time")
    latency_cols = ["avg_latency_seconds", "avg_retrieval_latency_seconds", "avg_llm_latency_seconds"]
    available_latency = [c for c in latency_cols if c in trend_df.columns]
    st.line_chart(trend_df.set_index("run_timestamp")[available_latency])
else:
    st.info("Trends require at least 2 runs. Only 1 run recorded so far.")


# ===================================================================
# Section 2 — Selected run detail
# ===================================================================

st.header("Selected run detail")

run_row = runs_df[runs_df["run_id"] == selected_run_id].iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Pass rate", f"{run_row['pass_rate']:.1%}")
col2.metric("Avg case score", f"{run_row['avg_case_score']:.1f}")
col3.metric("Cases", int(run_row["case_count"]))
col4.metric("Avg latency", f"{run_row['avg_latency_seconds']:.2f}s")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Source hit rate", f"{run_row['avg_source_hit_rate']:.3f}")
col6.metric("MRR", f"{run_row['avg_mrr']:.3f}")
col7.metric("Precision@k", f"{run_row['avg_precision_at_k']:.3f}")
col8.metric("Judge model", run_row["judge_model"])

# --- Per-case table ---
st.subheader("Per-case results")

run_cases = cases_df[cases_df["run_id"] == selected_run_id].copy()

if run_cases.empty:
    st.warning("No case-level data for this run.")
else:
    # Optional category filter
    if "category" in run_cases.columns:
        categories = sorted(run_cases["category"].dropna().unique())
        if categories:
            selected_categories = st.multiselect(
                "Filter by category",
                options=categories,
                default=categories,
            )
            run_cases = run_cases[run_cases["category"].isin(selected_categories)]

    display_cols = [
        "case_id", "category", "status", "avg_metric_score",
        *METRIC_COLUMNS,
        *RETRIEVAL_COLUMNS,
        "latency_seconds",
    ]
    available_display = [c for c in display_cols if c in run_cases.columns]
    st.dataframe(
        run_cases[available_display].set_index("case_id"),
        use_container_width=True,
    )

    # --- Per-case metric bar chart ---
    st.subheader("Metric scores by case")
    metric_display = [c for c in METRIC_COLUMNS if c in run_cases.columns]
    chart_df = run_cases.set_index("case_id")[metric_display]
    st.bar_chart(chart_df)

    # --- Case-over-time tracker ---
    if len(runs_df) >= 2:
        st.subheader("Track a case across runs")
        case_ids = sorted(cases_df["case_id"].unique())
        selected_case = st.selectbox("Select a case", case_ids)

        case_history = cases_df[cases_df["case_id"] == selected_case].sort_values("run_timestamp")
        if len(case_history) >= 2:
            metric_history = [c for c in METRIC_COLUMNS if c in case_history.columns]
            st.line_chart(case_history.set_index("run_timestamp")[metric_history])

            retrieval_history = [c for c in RETRIEVAL_COLUMNS if c in case_history.columns]
            st.line_chart(case_history.set_index("run_timestamp")[retrieval_history])
        else:
            st.info("This case has only been evaluated in one run so far.")

    # --- Expandable answer detail ---
    st.subheader("Answer detail")
    for _, case_row in run_cases.iterrows():
        with st.expander(f"{case_row['case_id']}  —  {case_row['status']}"):
            st.markdown(f"**Question:** {case_row['question']}")
            st.markdown(f"**Expected:** {case_row['expected_output']}")
            st.markdown(f"**Answer:**\n\n{case_row['answer']}")
            if case_row.get("errors") and case_row["errors"] != "[]":
                st.error(f"Errors: {case_row['errors']}")
