"""Streamlit dashboard for the Hybrid Search Agent.

Provides two interfaces:
  - **Chat** — conversational interface with the LangGraph agent.
  - **Search Explorer** — direct access to the HybridRetriever with tunable
    parameters and result visualisation.

Launch from the project root:
    cd HybridSearchAgent_LangGraph
    streamlit run dashboard.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import plotly.express as px
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# ---------------------------------------------------------------------------
# Project bootstrap — ensure imports and relative paths resolve correctly.
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from config import (  # noqa: E402
    FTS_MULTI_WEIGHTS,
    FTS_WEIGHT,
    MODEL_NAME,
    VECTOR_MMR_WEIGHT,
    VECTOR_SIMILARITY_WEIGHT,
)
from hybrid_search_agent import agent, retriever  # noqa: E402
from utils import fmt_score  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HybridSearchAgent",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — visual parity with the evaluation dashboard.
# ---------------------------------------------------------------------------

_CSS = """
<style>
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
button[data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.95rem;
}
div[data-testid="stExpander"] summary {
    font-weight: 500;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly defaults
# ---------------------------------------------------------------------------

_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", size=13),
    margin=dict(l=30, r=20, t=25, b=30),
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_thread_id" not in st.session_state:
    st.session_state.chat_thread_id = f"dashboard-{uuid4().hex[:8]}"
if "search_results" not in st.session_state:
    st.session_state.search_results = None

# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------


async def _invoke_agent(prompt: str, thread_id: str) -> dict:
    """Send a single message to the LangGraph agent and return the full state."""
    config = {"configurable": {"thread_id": thread_id}}
    return await agent.ainvoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=config,
    )


def _extract_response(result: dict) -> tuple[str, list[dict]]:
    """Extract the final answer and tool-call details from an agent result.

    Walks the message list to find the last AIMessage with text content
    (the answer) and collects all ToolMessage objects (retrieval calls).
    """
    messages = result.get("messages", [])

    # Last AIMessage with text content is the answer.
    answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                answer = content.strip()
                break
            if isinstance(content, list):
                text = "\n".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                ).strip()
                if text:
                    answer = text
                    break

    # Collect every ToolMessage (retrieval calls).
    tool_calls: list[dict] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        try:
            output = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
        except (json.JSONDecodeError, TypeError):
            output = msg.content
        tool_calls.append({"name": msg.name or "hybrid_search_tool", "output": output})

    return answer, tool_calls


def _render_tool_call(tc: dict) -> None:
    """Render a single tool call's results inside a Streamlit expander.

    If the output contains a results list (from the hybrid search tool),
    each result is displayed as a numbered line with source, backend, and score.
    Otherwise the raw JSON output is shown.
    """
    output = tc["output"]
    if isinstance(output, dict) and "results" in output:
        results = output["results"]
        st.caption(f"{len(results)} result(s) retrieved")
        for i, r in enumerate(results):
            source = Path(r.get("source") or "unknown").name
            score = r.get("score")
            backend = r.get("backend", "?")
            score_str = fmt_score(score)
            st.markdown(f"**{i + 1}.** `{source}` — {backend} (score: {score_str})")
    else:
        st.json(output)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def _export_chat_markdown(messages: list[dict]) -> str:
    """Format the chat history as a downloadable Markdown document.

    Includes user/agent messages, tool call summaries, and response latency.
    """
    lines = [f"# Chat Export — {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]
    for msg in messages:
        role = "You" if msg["role"] == "user" else "Agent"
        lines.append(f"## {role}")
        lines.append(msg["content"])
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                lines.append(f"\n**Tool call:** {tc['name']}")
                output = tc["output"]
                if isinstance(output, dict) and "results" in output:
                    for i, r in enumerate(output["results"]):
                        source = Path(r.get("source") or "unknown").name
                        score = r.get("score")
                        backend = r.get("backend", "?")
                        score_str = fmt_score(score)
                        lines.append(f"{i + 1}. `{source}` — {backend} (score: {score_str})")
        if msg.get("latency") is not None:
            lines.append(f"\n*Response time: {msg['latency']:.1f}s*")
        lines.append("")
    return "\n".join(lines)


def _export_search_markdown(data: dict) -> str:
    """Format search results as a downloadable Markdown document.

    Includes the query, search parameters, and each result with its
    metadata and full chunk text.
    """
    results = data["results"]
    params = data["params"]
    lines = [
        f"# Search Export — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"**Query:** {data['query']}  ",
        f"**Latency:** {data['latency']:.3f}s  ",
        f"**k:** {params['k']} | **Vector:** {params['vector_method']} | "
        f"**Multi FTS:** {params['multi_fts']} | "
        f"**Phrase:** {params['use_phrase']} | **Prefix:** {params['use_prefix']}",
        "",
    ]
    if params.get("metadata_filters"):
        filters = ", ".join(f"{k}={v}" for k, v in params["metadata_filters"].items())
        lines.append(f"**Filters:** {filters}")
        lines.append("")
    lines.append(f"## Results ({len(results)})")
    lines.append("")
    for i, r in enumerate(results):
        source = Path(r.get("source") or "unknown").name
        score = r.get("score")
        backend = r.get("backend", "unknown")
        score_str = fmt_score(score)
        chunk_id = r.get("chunk_id", "N/A")
        category = r.get("category") or "N/A"
        lines.append(f"### {i + 1}. {source}")
        lines.append(f"**Backend:** {backend} | **Score:** {score_str} | "
                      f"**Chunk:** {chunk_id} | **Category:** {category}")
        lines.append("")
        lines.append(r.get("page_content", ""))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("HybridSearchAgent")
    st.caption("FTS + Vector hybrid retrieval with LangGraph")

    st.divider()

    st.subheader("Agent")
    st.markdown(
        f"**Model:** {MODEL_NAME}  \n"
        f"**Thread:** `{st.session_state.chat_thread_id[:16]}\u2026`"
    )
    if st.button("New conversation", use_container_width=True):
        st.session_state.chat_messages = []
        st.session_state.chat_thread_id = f"dashboard-{uuid4().hex[:8]}"
        st.rerun()

    st.divider()

    st.subheader("Search Weights")
    st.caption(
        "Sliders mutate the shared retriever, so changes affect both the "
        "Chat tab (agent tool calls) and the Search Explorer tab."
    )
    sidebar_fts_w = st.slider("FTS", 0.0, 1.0, FTS_WEIGHT, 0.05, key="w_fts")
    sidebar_vec_sim_w = st.slider("Vector (similarity)", 0.0, 1.0, VECTOR_SIMILARITY_WEIGHT, 0.05, key="w_vec_sim")
    sidebar_vec_mmr_w = st.slider("Vector (MMR)", 0.0, 1.0, VECTOR_MMR_WEIGHT, 0.05, key="w_vec_mmr")

    retriever.fts_weight = sidebar_fts_w
    retriever.vector_similarity_weight = sidebar_vec_sim_w
    retriever.vector_mmr_weight = sidebar_vec_mmr_w

    st.divider()

    st.subheader("FTS Multi Mode Weights")
    sidebar_phrase_w = st.slider("Phrase", 0.0, 2.0, FTS_MULTI_WEIGHTS.get("phrase", 1.0), 0.1, key="w_phrase")
    sidebar_keyword_w = st.slider("Keyword", 0.0, 2.0, FTS_MULTI_WEIGHTS.get("keyword", 1.0), 0.1, key="w_keyword")
    sidebar_prefix_w = st.slider("Prefix", 0.0, 2.0, FTS_MULTI_WEIGHTS.get("prefix", 1.0), 0.1, key="w_prefix")

    fts_multi_weights_live = {
        "phrase": sidebar_phrase_w,
        "keyword": sidebar_keyword_w,
        "prefix": sidebar_prefix_w,
    }

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.title("Hybrid Search Agent")
st.caption("Chat with the agent or explore the retrieval pipeline directly")

tab_chat, tab_search = st.tabs(["Chat", "Search Explorer"])


# ===================================================================
# Tab 1 — Chat
# ===================================================================

with tab_chat:

    # ---- message history ----
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    with st.expander(f"Tool: {tc['name']}"):
                        _render_tool_call(tc)
            if msg.get("latency") is not None:
                st.caption(f"Response time: {msg['latency']:.1f}s")

    # ---- export ----
    if st.session_state.chat_messages:
        st.download_button(
            "Download conversation",
            data=_export_chat_markdown(st.session_state.chat_messages),
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )

    # ---- input ----
    if prompt := st.chat_input("Ask the agent\u2026"):
        # Show user message immediately.
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Invoke agent.
        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking\u2026"):
                started = time.perf_counter()
                try:
                    result = asyncio.run(
                        _invoke_agent(prompt, st.session_state.chat_thread_id),
                    )
                    latency = time.perf_counter() - started
                    answer, tool_calls = _extract_response(result)

                    st.markdown(answer)
                    for tc in tool_calls:
                        with st.expander(f"Tool: {tc['name']}"):
                            _render_tool_call(tc)
                    st.caption(f"Response time: {latency:.1f}s")

                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": answer,
                        "tool_calls": tool_calls,
                        "latency": latency,
                    })
                except Exception as exc:
                    error_text = f"Agent error: {type(exc).__name__}: {exc}"
                    st.error(error_text)
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": error_text,
                    })

        st.rerun()


# ===================================================================
# Tab 2 — Search Explorer
# ===================================================================

with tab_search:

    # ---- search controls ----
    # FTS mode lives outside the form so switching it triggers an immediate
    # rerun, allowing the form to show the correct widgets for each mode.
    col_mode_vec, col_mode_fts, col_mode_pad = st.columns([1, 1, 2])
    with col_mode_vec:
        vector_method = st.selectbox(
            "Vector method",
            options=["similarity", "mmr"],
            format_func=lambda v: {"similarity": "Similarity", "mmr": "MMR"}[v],
            key="vector_method",
        )
    with col_mode_fts:
        fts_mode = st.selectbox("FTS mode", ["Multi mode", "Single mode"], key="fts_mode")
        multi_fts = fts_mode == "Multi mode"

    with st.form("search_form"):
        query = st.text_input(
            "Search query",
            placeholder="Enter your search query\u2026",
        )

        col_k, col_flags = st.columns([1, 3])
        with col_k:
            k = st.slider("Results (k)", 1, 10, 5)
        with col_flags:
            if multi_fts:
                st.checkbox("Keyword", value=True, disabled=True, key="multi_kw")
                use_phrase = st.checkbox("Phrase", key="multi_phrase")
                use_prefix = st.checkbox("Prefix", key="multi_prefix")
            else:
                single_choice = st.radio(
                    "Search type",
                    options=["Keyword", "Phrase", "Prefix"],
                    key="single_radio",
                )
                use_phrase = single_choice == "Phrase"
                use_prefix = single_choice == "Prefix"

        _FILTER_KEYS = ["flt_category", "flt_language", "flt_filename", "flt_file_type", "flt_folder"]

        with st.expander("Metadata filters"):
            fc1, fc2 = st.columns(2)
            with fc1:
                filter_category = st.text_input("Category", placeholder="e.g. Norway", key="flt_category")
                filter_language = st.text_input("Language", placeholder="e.g. en", key="flt_language")
                filter_filename = st.text_input("Filename", placeholder="e.g. norway_facts.txt", key="flt_filename")
            with fc2:
                filter_file_type = st.text_input("File type", placeholder="e.g. .txt", key="flt_file_type")
                filter_folder = st.text_input("Folder", placeholder="e.g. documents", key="flt_folder")

        col_submit, col_clear = st.columns([3, 1])
        with col_submit:
            submitted = st.form_submit_button("Search", use_container_width=True, type="primary")
        with col_clear:
            cleared = st.form_submit_button("Clear filters", use_container_width=True)

    if cleared:
        for k in _FILTER_KEYS:
            st.session_state.pop(k, None)
        st.rerun()

    if submitted and query:
        metadata_filters: dict[str, str] = {}
        if filter_category:
            metadata_filters["category"] = filter_category
        if filter_language:
            metadata_filters["language"] = filter_language
        if filter_filename:
            metadata_filters["filename"] = filter_filename
        if filter_file_type:
            metadata_filters["file_type"] = filter_file_type
        if filter_folder:
            metadata_filters["folder"] = filter_folder

        with st.spinner("Searching\u2026"):
            started = time.perf_counter()
            try:
                raw_results = retriever.search(
                    query=query,
                    k=k,
                    vector_search_method=vector_method,
                    use_phrase=use_phrase,
                    use_prefix=use_prefix,
                    multi_fts=multi_fts,
                    fts_multi_weights=fts_multi_weights_live,
                    **metadata_filters,
                )
                search_latency = time.perf_counter() - started
                st.session_state.search_results = {
                    "query": query,
                    "results": [r.model_dump() for r in raw_results],
                    "latency": search_latency,
                    "params": {
                        "k": k,
                        "vector_method": vector_method,
                        "use_phrase": use_phrase,
                        "use_prefix": use_prefix,
                        "multi_fts": multi_fts,
                        "metadata_filters": metadata_filters,
                    },
                }
            except Exception as exc:
                st.error(f"Search error: {type(exc).__name__}: {exc}")
                st.session_state.search_results = None

    # ---- display results ----
    if st.session_state.search_results:
        data = st.session_state.search_results
        results = data["results"]

        st.divider()

        # -- summary metrics --
        backend_counts: dict[str, int] = {}
        for r in results:
            b = r.get("backend", "unknown")
            backend_counts[b] = backend_counts.get(b, 0) + 1
        unique_sources = {Path(r.get("source") or "unknown").name for r in results}

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Results", len(results))
        m2.metric("Latency", f"{data['latency']:.3f}s")
        m3.metric("Backends", len(backend_counts))
        m4.metric("Unique sources", len(unique_sources))

        # -- charts --
        col_pie, col_bar = st.columns(2)

        with col_pie:
            st.markdown("**Backend Distribution**")
            if backend_counts:
                fig_pie = px.pie(
                    values=list(backend_counts.values()),
                    names=list(backend_counts.keys()),
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_pie.update_layout(**_PLOTLY_LAYOUT, height=300)
                st.plotly_chart(fig_pie, use_container_width=True, key="pie_backend")

        with col_bar:
            st.markdown("**Score by Rank**")
            if results:
                score_df = pd.DataFrame([
                    {
                        "Rank": i + 1,
                        "Source": Path(r.get("source") or "unknown").name,
                        "Score": r.get("score") or 0,
                        "Backend": r.get("backend", "unknown"),
                    }
                    for i, r in enumerate(results)
                ])
                fig_bar = px.bar(
                    score_df,
                    x="Rank",
                    y="Score",
                    color="Backend",
                    hover_data=["Source"],
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_bar.update_layout(
                    **_PLOTLY_LAYOUT,
                    height=300,
                    yaxis=dict(gridcolor="#e9ecef"),
                )
                st.plotly_chart(fig_bar, use_container_width=True, key="bar_scores")
            st.caption("All scores normalized to 0\u20131 (higher = better).")

        # -- results table --
        st.markdown("**Ranked Results**")
        table_rows = []
        for i, r in enumerate(results):
            table_rows.append({
                "Rank": i + 1,
                "Source": Path(r.get("source") or "unknown").name,
                "Chunk": r.get("chunk_id"),
                "Backend": r.get("backend", "unknown"),
                "Score": r.get("score"),
                "Category": r.get("category") or "",
                "Preview": str(r.get("page_content", ""))[:200],
            })
        st.dataframe(
            pd.DataFrame(table_rows),
            use_container_width=True,
            hide_index=True,
        )

        # -- chunk details --
        st.markdown("**Chunk Details**")
        for i, r in enumerate(results):
            source = Path(r.get("source") or "unknown").name
            score = r.get("score")
            backend = r.get("backend", "unknown")
            score_str = fmt_score(score)

            with st.expander(f"#{i + 1} \u2014 {source} | {backend} | score: {score_str}"):
                st.markdown(r.get("page_content", ""))
                st.divider()
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.markdown(f"**Chunk ID:** {r.get('chunk_id', 'N/A')}")
                mc2.markdown(f"**Category:** {r.get('category') or 'N/A'}")
                mc3.markdown(f"**Language:** {r.get('language') or 'N/A'}")
                mc4.markdown(f"**File type:** {r.get('file_type') or 'N/A'}")

        # -- export --
        st.divider()
        st.download_button(
            "Download results",
            data=_export_search_markdown(data),
            file_name=f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )
