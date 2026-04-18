# Hybrid Search Agent

Hybrid retrieval agent for question-answering over a local document corpus. Built with LangGraph, OpenAI LLM-model, SQLite FTS5, Chroma vector store, and Streamlit.

The agent combines full-text search (BM25 via SQLite FTS5) and vector search (Chroma + OpenAI embeddings) into a single hybrid retrieval pipeline. When a user asks a question, the LangGraph agent calls the `hybrid_search_tool`, which queries both backends, normalizes and fuses scores, deduplicates results, and returns a ranked set of chunks. The agent then generates a grounded answer from the retrieved context. Metadata-aware filtering (category, language, file type, etc.) is supported across both backends. 

The hybrid retrieval agent's document ingestion and metadata pipeline (`chroma_fts_metadata_chunks.ipynb`) loads source documents, enriches metadata, splits into chunks, and indexes them into both the Chroma vector store and the SQLite FTS5 index. See [`info_chroma_fts_metadata_chunks.md`](info_chroma_fts_metadata_chunks.md) for a detailed walkthrough. 

The project also includes an evaluation harness that measures the agent's retrieval and answer quality across test cases using LLM-as-judge, deterministic retrieval, and keyword-based metrics. See [`evaluation/README.md`](evaluation/README.md) for full details.

The Streamlit dashboard provides both a conversational chat interface and a search explorer for direct retrieval with tunable parameters:

```bash
streamlit run dashboard.py
```

See [Run Instructions](#run-instructions) below for how to start the agent.

All scripts in this folder are Learning/Demo status.

## Use Cases

This agent is built for question-answering over a local document corpus using hybrid retrieval, and can be adapted for different document sets and domains by re-running the ingestion pipeline with new source files. The same retrieval architecture works for any scenario where combining keyword search and semantic search improves recall. For example internal knowledge bases, policy documents, technical documentation, research papers, or customer support content.

The documents indexed, the metadata fields used for filtering, and the search weights that control the FTS/vector balance are all configurable without touching the core agent logic. The retrieval pipeline and agent behavior are independently tunable — search weights and FTS modes via `config.py`, agent reasoning via `system_prompt_hybrid_search.txt`. For best results, keep the corpus focused on a coherent domain — mixing unrelated document sets dilutes retrieval precision and makes it harder for the agent to ground its answers.

### Where to make changes for a different corpus or domain

| What to change | Where | Effect |
|---|---|---|
| **Source documents** | `documents/` folder — add, remove, or replace `.txt`, `.pdf`, `.docx` files | Controls which content is available for retrieval after re-running ingestion |
| **Ingestion pipeline** | `chroma_fts_metadata_chunks.ipynb` — metadata enrichment, chunking strategy, embedding model. See [`info_chroma_fts_metadata_chunks.md`](info_chroma_fts_metadata_chunks.md) for a detailed walkthrough | Controls how documents are split, what metadata is attached to each chunk, and how embeddings are generated |
| **Metadata fields** | `pydantic_models.py` — `ChunkMetadata` and `HybridSearchArgs` schemas; `fts_search.py` — FTS index columns | Controls which metadata fields are stored, indexed, and available as search filters |
| **Agent behavior and reasoning** | `system_prompt_hybrid_search.txt` — edit the system prompt | Controls how the agent interprets questions, uses the search tool, and structures its answers |
| **Search weights and balance** | `config.py` — `FTS_WEIGHT`, `VECTOR_SIMILARITY_WEIGHT`, `VECTOR_MMR_WEIGHT`; also adjustable in real time via the dashboard sidebar | Controls the relative influence of FTS vs vector results in the fused ranking |
| **FTS search modes and weights** | `config.py` — `FTS_MULTI_WEIGHTS`; also adjustable via the dashboard sidebar | Controls how multi-mode FTS queries are weighted and combined |
| **Retrieval parameters per eval case** | `evaluation/eval_cases.json` — `retrieval` object on each case | Controls k, vector method, and FTS modes used during evaluation |
| **Dashboard layout** | `dashboard.py` | Adapt the UI tabs, charts, and sidebar controls to match your domain |

## Folder Contents

### Core Python Scripts

| File | Purpose | Notes | Status |
|---|---|---|---|
| `hybrid_search_agent.py` | Main interactive hybrid-search agent with streaming output and tool instrumentation | Loads `system_prompt_hybrid_search.txt`, uses OpenAI LLM-model, `OpenAIEmbeddings`, and `InMemorySaver` | Learning/Demo |
| `hybrid_search.py` | Hybrid retriever logic that fuses FTS and vector scores | Normalizes BM25/Chroma scores, supports similarity or MMR, and deduplicates results | Learning/Demo |
| `fts_search.py` | SQLite FTS5 wrapper for indexing and querying chunks | Supports keyword, phrase, prefix, and weighted multi-mode FTS search with metadata filtering | Learning/Demo |
| `pydantic_models.py` | Shared data models and tool schema | Defines `ChunkMetadata`, `HybridSearchArgs`, and `SearchResult` | Learning/Demo |
| `utils.py` | Utilities for debug printing, logger setup, and graph PNG generation | Includes Mermaid graph export helper and dual-handler logger configuration | Learning/Demo |
| `config.py` | Central configuration for search weights, FTS tuning, and debug flags | Search strategy hyperparameters (`FTS_WEIGHT`, `VECTOR_SIMILARITY_WEIGHT`, `VECTOR_MMR_WEIGHT`, `FTS_MAX_SCORE`, `FTS_MULTI_WEIGHTS`) and runtime toggles (`DEBUG`, `UPDATES`, `DRAW`, `DEBUG_PRINT`, `PRINT`) | Learning/Demo |
| `dashboard.py` | Streamlit dashboard with Chat and Search Explorer tabs | Conversational agent chat with tool-call inspection, direct hybrid search with tunable weights/modes/metadata filters, result visualisation (score table, pie chart, bar chart, chunk details), markdown export for both tabs | Learning/Demo |

### Evaluation Harness

| Path | Purpose | Status |
|---|---|---|
| `evaluation/` | Evaluation harness — runs the agent end-to-end, scores each case with LLM-as-judge (DeepEval), deterministic retrieval, and keyword-based metrics, persists results to a SQLite ledger, and provides a Streamlit dashboard. See [`evaluation/README.md`](evaluation/README.md) for full details | Learning/Demo |

### Notebook

| File | Purpose | Status |
|---|---|---|
| `chroma_fts_metadata_chunks.ipynb` | End-to-end ingestion notebook: load `.txt/.pdf/.docx` files, enrich metadata, chunk content, and ingest into Chroma + SQLite FTS | Learning/Demo |

### Data and Runtime Artifacts

| Path | Type | Purpose |
|---|---|---|
| `documents/` | Source documents | Norway-themed `.txt`, `.pdf`, and `.docx` files used for chunking and retrieval experiments |
| `documents_with_metadata_df.csv` | CSV dataset | Export of document/chunk content with standardized metadata from the notebook workflow |
| `fts.db` | SQLite database | FTS5 index for keyword/phrase/prefix retrieval |
| `chroma_db/chroma.sqlite3` | SQLite database | Chroma persistence backing file |
| `chroma_db/<collection-id>/` | Chroma segment directory | Collection segment data for embeddings and stored chunks |
| `system_prompt_hybrid_search.txt` | Prompt file | Active system prompt loaded by `hybrid_search_agent.py` |
| `dev_system_prompt_hybrid_search.txt` | Prompt draft | Developer-facing prompt notes / draft instructions for the hybrid-search assistant |
| `agent_graph.png` | Image | Optional graph visualization output generated when `DRAW=True` |
| `evaluation/` | Subfolder | Evaluation harness, SQLite ledger, Streamlit dashboard, and test cases |
| `evaluation_results/` | Report output | Timestamped JSON and CSV evaluation reports (generated by the harness) |
| `agent.log` | Log file | Runtime log file created by `setup_logger()` when the agent is run |
| `__pycache__/` | Python cache | Compiled bytecode artifacts |
| `.git/` | Git metadata | Repository internals |

## Architecture Overview

```
chroma_fts_metadata_chunks.ipynb  (ingestion)
    ├── documents/                (.txt, .pdf, .docx)
    │       ↓
    ├── chroma_db/                (vector store — OpenAI embeddings)
    └── fts.db                    (FTS5 index — BM25 keyword search)

hybrid_search_agent.py  (interactive agent)
    ├── config.py                 (weights, flags)
    ├── system_prompt_hybrid_search.txt
    └── LangGraph Agent (gpt-4o-mini)
            └── hybrid_search_tool
                    └── HybridRetriever (hybrid_search.py)
                            ├── FTSStore (fts_search.py)  → fts.db
                            └── Chroma vector store       → chroma_db/
                                    ↓
                            normalize → fuse → deduplicate → ranked results
                                    ↓
                            agent generates grounded answer

dashboard.py  (Streamlit UI)
    ├── Chat tab           → LangGraph agent (same as above)
    └── Search Explorer    → HybridRetriever directly (bypass agent)

evaluation/eval_deepeval.py  (evaluation harness)
    └── EvaluationEngine
            ├── direct retriever    → retrieval quality metrics
            └── full agent          → LLM-as-judge + keyword metrics
                    ↓
            eval_sqlite.py          → eval_ledger.db
                    ↓
            eval_dashboard.py       → Streamlit evaluation dashboard
```

**Flow per query:**

1. The user sends a question to the LangGraph agent (via CLI or dashboard Chat tab).
2. The agent calls `hybrid_search_tool` with the query and optional metadata filters.
3. `HybridRetriever` queries both backends in parallel — `FTSStore` (BM25 keyword search against `fts.db`) and Chroma (vector similarity or MMR against `chroma_db/`).
4. Raw scores are normalized into 0–1 range, weighted by the configured backend weights, fused, and deduplicated.
5. The ranked chunks are returned to the agent, which generates a grounded answer from the retrieved context.
6. The dashboard Search Explorer tab provides direct access to the `HybridRetriever`, bypassing the agent, for retrieval experimentation with tunable weights and modes.

## Key Retrieval Behavior

- FTS mode options:
  - **Single mode**: One of keyword, phrase, or prefix matching at a time
  - **Multi mode**: Keyword always runs as baseline; phrase and prefix are additive (none, one, or both)
- Vector mode options:
  - Similarity search
  - MMR (max marginal relevance) search
- Metadata-aware search inputs currently include fields such as:
  - `category`, `language`, `filename`, `file_type`, `folder`
  - plus lower-level stored fields like `source`, `doc_hash_id`, `chunk_id`, and character offsets
- Final results are deduplicated and ranked using normalized hybrid scoring.

## Configuration

All tuning is done in `config.py`:

### Search Strategy

| Setting | Default | Description |
|---|---|---|
| `FTS_WEIGHT` | `0.5` | Weight of FTS (BM25) results in the fused ranking |
| `VECTOR_SIMILARITY_WEIGHT` | `0.5` | Weight of vector similarity results in the fused ranking |
| `VECTOR_MMR_WEIGHT` | `0.5` | Weight of vector MMR results in the fused ranking |
| `FTS_MAX_SCORE` | `20.0` | Ceiling used to normalize raw BM25 scores into 0–1 range |
| `FTS_MULTI_WEIGHTS` | `{"phrase": 1.0, "keyword": 1.0, "prefix": 1.0}` | Per-mode weights for multi-mode FTS ranking (values > 1.0 boost, < 1.0 dampen) |

### Debug and Logging Flags

| Setting | Default | Description |
|---|---|---|
| `DEBUG` | `False` | Enable internal retriever / LangGraph debug logging |
| `UPDATES` | `False` | Show LangGraph node state updates during execution |
| `DRAW` | `False` | Export a PNG of the agent graph on startup |
| `DEBUG_PRINT` | `False` | Enable helper debug output from utilities/tools |
| `PRINT` | `False` | Control top-level console logging visibility |

## Requirements

All dependencies are pinned in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

Dependencies are organized into three groups:

- **Core agent/runtime**: `langgraph`, `langchain`, `langchain-core`, `langchain-openai`, `langchain-chroma`, `chromadb`, `pydantic`, `python-dotenv`
- **Dashboard**: `pandas`, `plotly`, `streamlit`
- **Notebook / ingestion pipeline**: `langchain-community`, `langchain-text-splitters`, `langdetect`, `numpy`, `matplotlib`, `scikit-learn`, `docx2txt`, `pypdf`

## Environment Variables

Required for OpenAI-based embeddings/model calls:

```env
OPENAI_API_KEY=...
```

## Run Instructions

From this folder (important because the script expects local `fts.db` and `./chroma_db` paths):

```bash
cd HybridSearchAgent_LangGraph
python hybrid_search_agent.py
```

Optional utility runs:

```bash
python -c "from fts_search import FTSStore; print('FTS OK')"
python -c "from hybrid_search import HybridRetriever; print('Hybrid module import OK')"
```

To rebuild/refresh ingestion artifacts, run notebook cells in `chroma_fts_metadata_chunks.ipynb`. See [`info_chroma_fts_metadata_chunks.md`](info_chroma_fts_metadata_chunks.md) for a detailed walkthrough of each pipeline stage.

### Launch Dashboard

```bash
cd HybridSearchAgent_LangGraph
streamlit run dashboard.py
```

Opens at [http://localhost:8501](http://localhost:8501) with two tabs:

- **Chat** — conversational interface to the LangGraph agent with message history, tool-call inspection (expandable retrieval results with source, backend, and score), response latency, and markdown export
- **Search Explorer** — direct hybrid search with configurable vector method (Similarity / MMR), FTS mode (Single / Multi), metadata filters (category, language, filename, file type, folder), and result visualisation including summary metrics, backend distribution pie chart, score-by-rank bar chart, ranked results table, expandable chunk details, and markdown export

The sidebar provides:
- **Search weights** — FTS, Vector (similarity), and Vector (MMR) weight sliders that update the retriever in real time
- **FTS multi-mode weights** — phrase, keyword, and prefix weight sliders for fine-tuning multi-mode FTS ranking
- **Agent controls** — current thread ID and a "New conversation" button to reset chat state

### Run Evaluation Harness

```bash
cd HybridSearchAgent_LangGraph
python evaluation/eval_deepeval.py
```

Launch the evaluation dashboard:

```bash
streamlit run evaluation/eval_dashboard.py
```

See [`evaluation/README.md`](evaluation/README.md) for full details on metrics, configuration, and test cases.

## Suggested Learning Path

1. Start with `pydantic_models.py` to understand the metadata and result schemas.
2. Read `fts_search.py` for FTS indexing, filtering, and query behavior.
3. Read `hybrid_search.py` for score normalization, fusion, and deduplication logic.
4. Inspect `system_prompt_hybrid_search.txt` to see how the agent is guided to use the tool.
5. Run `hybrid_search_agent.py` for end-to-end interactive usage.
6. Explore `chroma_fts_metadata_chunks.ipynb` to understand the ingestion pipeline details.
7. Launch `dashboard.py` to interact with the agent via chat and explore retrieval behavior with tunable weights and modes through the Search Explorer.
8. Read `evaluation/README.md` and run the evaluation harness to measure agent quality.

## Notes

- This repository is intentionally educational and experimentation-oriented.
- Existing DB artifacts (`fts.db`, `chroma_db/...`) are stateful; deleting them resets indexed state.
- The current agent uses `InMemorySaver`, so conversation state is not persisted across process restarts.
- If you change metadata fields, keep `pydantic_models.py`, `fts_search.py`, and the notebook ingestion flow in sync.
- The notebook currently has cells present but not executed in the current saved state.
- The dashboard (`dashboard.py`) uses `InMemorySaver` with a unique thread ID per session, so chat history resets when the Streamlit app restarts or when the "New conversation" button is clicked.