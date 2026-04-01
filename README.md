# HybridSearchAgent_LangGraph

Learning/demo project for building a hybrid retrieval agent with LangGraph over a local document corpus.

The folder demonstrates how to combine:

- Full-text search (SQLite FTS5)
- Vector search (Chroma + OpenAI embeddings)
- Metadata-aware filtering across document and chunk fields
- Tool-driven agent orchestration with LangGraph/LangChain
- Prompt-driven agent behavior via `system_prompt_hybrid_search.txt`

All scripts in this folder are Learning/Demo status.

## Folder Contents

### Core Python Scripts

| File | Purpose | Notes | Status |
|---|---|---|---|
| `hybrid_search_agent.py` | Main interactive hybrid-search agent with streaming output and tool instrumentation | Loads `system_prompt_hybrid_search.txt`, uses `ChatOpenAI(gpt-4o-mini)`, `OpenAIEmbeddings`, and `InMemorySaver` | Learning/Demo |
| `hybrid_search.py` | Hybrid retriever logic that fuses FTS and vector scores | Normalizes BM25/Chroma scores, supports similarity or MMR, and deduplicates results | Learning/Demo |
| `fts_search.py` | SQLite FTS5 wrapper for indexing and querying chunks | Supports keyword, phrase, prefix, and weighted multi-mode FTS search with metadata filtering | Learning/Demo |
| `pydantic_models.py` | Shared data models and tool schema | Defines `ChunkMetadata`, `HybridSearchArgs`, and `SearchResult` | Learning/Demo |
| `utils.py` | Utilities for debug printing, logger setup, and graph PNG generation | Includes Mermaid graph export helper and dual-handler logger configuration | Learning/Demo |
| `config.py` | Global flags for debug output, graph drawing, and update verbosity | Centralized runtime toggles (`DEBUG`, `UPDATES`, `DRAW`, `DEBUG_PRINT`, `PRINT`) | Learning/Demo |

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
| `agent.log` | Log file | Runtime log file created by `setup_logger()` when the agent is run |
| `__pycache__/` | Python cache | Compiled bytecode artifacts |
| `.git/` | Git metadata | Repository internals |

## Architecture Overview

1. Documents are loaded, chunked, and enriched with metadata (primarily in the notebook).
2. Chunks are stored in two retrieval backends:
   - `fts.db` via `FTSStore`
   - `chroma_db` via Chroma vector store (`document_collection_1`)
3. `HybridRetriever` queries both backends, normalizes scores, and fuses them into a final ranking.
4. `hybrid_search_tool` exposes hybrid retrieval to the LangGraph agent with optional metadata filters.
5. `hybrid_search_agent.py` runs an interactive streaming loop with in-memory checkpointing and tool latency reporting.

## Key Retrieval Behavior

- FTS mode options:
  - Keyword matching
  - Phrase matching
  - Prefix matching
  - Multi-mode fusion (weighted keyword + phrase + prefix)
- Vector mode options:
  - Similarity search
  - MMR (max marginal relevance) search
- Metadata-aware search inputs currently include fields such as:
  - `category`, `language`, `filename`, `file_type`, `folder`
  - plus lower-level stored fields like `source`, `doc_hash_id`, `chunk_id`, and character offsets
- Final results are deduplicated and ranked using normalized hybrid scoring.

## Configuration Flags

In `config.py`:

- `DEBUG`: Enable internal retriever / LangGraph debug logging
- `UPDATES`: Show LangGraph node state updates during execution
- `DRAW`: Export a PNG of the agent graph on startup
- `DEBUG_PRINT`: Enable helper debug output from utilities/tools
- `PRINT`: Control top-level console logging visibility

## Requirements

Typical dependencies used by this folder:

- `langgraph`
- `langchain`
- `langchain-core`
- `langchain-openai`
- `langchain-chroma`
- `langchain-community`
- `chromadb`
- `pydantic`
- `python-dotenv`
- `pandas`
- `numpy`
- `python-docx` or `docx2txt` support for DOCX loading
- `pypdf` support for PDF loading

Install example:

```bash
pip install langgraph langchain langchain-core langchain-openai langchain-chroma langchain-community chromadb pydantic python-dotenv pandas numpy docx2txt pypdf
```

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

To rebuild/refresh ingestion artifacts, run notebook cells in:

- `chroma_fts_metadata_chunks.ipynb`

## Suggested Learning Path

1. Start with `pydantic_models.py` to understand the metadata and result schemas.
2. Read `fts_search.py` for FTS indexing, filtering, and query behavior.
3. Read `hybrid_search.py` for score normalization, fusion, and deduplication logic.
4. Inspect `system_prompt_hybrid_search.txt` to see how the agent is guided to use the tool.
5. Run `hybrid_search_agent.py` for end-to-end interactive usage.
6. Explore `chroma_fts_metadata_chunks.ipynb` to understand the ingestion pipeline details.

## Notes

- This repository is intentionally educational and experimentation-oriented.
- Existing DB artifacts (`fts.db`, `chroma_db/...`) are stateful; deleting them resets indexed state.
- The current agent uses `InMemorySaver`, so conversation state is not persisted across process restarts.
- If you change metadata fields, keep `pydantic_models.py`, `fts_search.py`, and the notebook ingestion flow in sync.
- The notebook currently has cells present but not executed in the current saved state.