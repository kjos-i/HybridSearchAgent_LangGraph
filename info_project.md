# Project Structure

A short, top-down map of the Hybrid Search Agent project. For detail on
individual files, see [README.md](README.md).

```
HybridSearchAgent_LangGraph/
│
├── README.md                                ← Project overview, setup, run instructions
├── info_project.md                          ← This file — folder structure at a glance
├── info_chroma_fts_metadata_chunks.md       ← Ingestion notebook walkthrough
├── info_considerations.md                   ← Design trade-offs, gotchas, improvement areas
├── requirements.txt                         ← Pinned Python dependencies
├── system_prompt_hybrid_search.txt          ← Agent system prompt
│
├── hybrid_search_agent.py                   ← Entry — LangGraph agent + interactive REPL
├── hybrid_search.py                         ← HybridRetriever (FTS + vector score fusion)
├── fts_search.py                            ← SQLite FTS5 wrapper (FTSStore)
├── pydantic_models.py                       ← ChunkMetadata, HybridSearchArgs, SearchResult
├── utils.py                                 ← Logger, score formatter, graph PNG export
├── config.py                                ← All runtime tuning (model, weights, flags)
│
├── dashboard.py                             ← Streamlit dashboard — Chat + Search Explorer
│
├── chroma_fts_metadata_chunks.ipynb         ← Ingestion pipeline (chunking, embedding, indexing)
├── documents/                               ← Source corpus (.txt, .pdf, .docx)
├── documents_with_metadata_df.csv           ← Optional snapshot of enriched metadata
│
├── ⟨runtime artifacts, generated on first run⟩
│   ├── chroma_db/                           ← Chroma vector store (OpenAI embeddings)
│   ├── fts.db                               ← SQLite FTS5 keyword index
│   ├── agent.log                            ← Runtime log shared across modules
│   └── agent_graph.png                      ← Optional graph diagram (when DRAW=True)
│
└── evaluation/                              ← Self-contained evaluation harness
    │
    ├── README.md                            ← Harness overview, configuration, run guide
    ├── info_metrics.md                      ← Per-metric reference + verdict-gate logic
    │
    ├── eval_deepeval.py                     ← CLI entry — runs every case end-to-end
    ├── eval_config.py                       ← Harness tuning (judge, thresholds, concurrency)
    │
    ├── eval_models.py                       ← EvalCase, RetrievalSettings Pydantic schemas
    ├── eval_metric_registry.py              ← Single source of truth — every MetricDef
    ├── eval_metrics.py                      ← Pure compute_* functions (deterministic metrics)
    ├── eval_utils.py                        ← Loaders, text extraction, math helpers
    │
    ├── eval_engine.py                       ← Per-case runner, judge LLM wrapper
    ├── eval_report_manager.py               ← Per-run summary → JSON / CSV artifacts
    ├── eval_sqlite.py                       ← SQLite ledger (eval_runs, eval_cases tables)
    ├── eval_dashboard.py                    ← Streamlit dashboard for harness results
    │
    ├── eval_cases.json                      ← Test cases (questions, expected sources, gates)
    │
    ├── ⟨runtime artifacts, generated on first run⟩
    │   ├── eval_ledger.db                   ← SQLite ledger of every evaluation run
    │   └── evaluation_results/              ← Timestamped JSON + CSV reports per run
    │
    └── tests/                               ← pytest suite for the evaluation harness
        ├── conftest.py
        ├── test_eval_utils.py
        ├── test_eval_metrics.py
        └── test_eval_metric_registry.py
```

## Layout at a glance

The parent project and the `evaluation/` subproject share a small set of
recurring concerns, each housed in its own dedicated module on each
side.

| Concern | Parent | `evaluation/` |
|---|---|---|
| Entry point | `hybrid_search_agent.py` | `eval_deepeval.py` |
| Tuning | `config.py` | `eval_config.py` |
| Schemas | `pydantic_models.py` | `eval_models.py` |
| Persistence | `fts.db` + `chroma_db/` | `eval_ledger.db` |
| Dashboard | `dashboard.py` | `eval_dashboard.py` |
| Tests | *(none at parent level)* | `evaluation/tests/` |

## Things specific to this project

- **Ingestion is a notebook.** `chroma_fts_metadata_chunks.ipynb` builds
  the two retrieval indexes (`chroma_db/` and `fts.db`) and is run
  separately from the agent. Full walkthrough in
  [info_chroma_fts_metadata_chunks.md](info_chroma_fts_metadata_chunks.md).
- **Data lives in two stores.** Vector embeddings go to Chroma
  (`chroma_db/`), raw text + metadata go to SQLite FTS5 (`fts.db`).
  `HybridRetriever` queries both and fuses scores at runtime.
- **Single CLI entry point.** `hybrid_search_agent.py` is both the
  importable module (used by `dashboard.py` and the eval harness) and
  the interactive REPL (`python hybrid_search_agent.py`).

Runtime artifacts (`*.db`, `*.log`, `chroma_db/`, generated
`evaluation_results/`) are created on first run and are safe to delete
or `.gitignore`.
