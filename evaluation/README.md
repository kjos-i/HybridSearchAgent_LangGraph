# Evaluation Harness for HybridSearchAgent

Evaluation harness for the hybrid retrieval agent. Runs the real LangGraph agent end-to-end and scores each test case with three families of metrics: **LLM-as-judge** (via DeepEval), **deterministic retrieval quality**, and **keyword-based answer checks**. Results are persisted to a SQLite ledger with a Streamlit dashboard for browsing runs over time.

The Streamlit dashboard can be used to visualize evaluation metrics stored in the SQLite ledger at any time:

```bash
python -m streamlit run eval_dashboard.py --server.port 8501
```

See [Run Instructions](#run-instructions) below for how to run the evaluation.

For more in-depth coverage of every metric, see [`info_metrics.md`](info_metrics.md) (detailed metric definitions, formulas, and interpretation).

All scripts in this folder are Learning/Demo status.

## What It Does

1. Loads test cases from [`eval_cases.json`](eval_cases.json) (question, expected answers, expected sources, keywords, retrieval config).
2. For each case, runs both the **direct retriever** (ground-truth search quality) and the **full LangGraph agent** (end-to-end answer quality).
3. Evaluates the agent's answer against seven LLM-as-judge metrics (via DeepEval), source- and chunk-level deterministic retrieval quality metrics, and keyword-based answer checks.
4. Assigns a **PASS / REVIEW** verdict based on combined metric, retrieval, and keyword gates.
5. Writes timestamped JSON and CSV reports to `evaluation_results/`.
6. Persists every run and per-case result to `eval_ledger.db` (SQLite) for cross-run comparison.
7. Provides a Streamlit dashboard that reads from the SQLite ledger to visualize trends, per-case scores, and latency over time.

## Folder Contents

### Python Scripts

| File | Purpose |
|---|---|
| [`eval_deepeval.py`](eval_deepeval.py) | Entry point — wires agent, engine, report manager, and SQLite ledger together |
| [`eval_engine.py`](eval_engine.py) | `EvaluationEngine` class — runs retrieval, agent invocation, and DeepEval metrics for each case |
| [`eval_metrics.py`](eval_metrics.py) | Deterministic metric computations — source-level and chunk-level IR metrics, metadata match, backend distribution, keyword checks |
| [`eval_utils.py`](eval_utils.py) | Utility helpers — data loading, prompt building, context formatting, result previews, text normalization |
| [`eval_models.py`](eval_models.py) | Pydantic data models: `EvalCase` and `RetrievalSettings` |
| [`eval_config.py`](eval_config.py) | Configuration constants: judge model, threshold, paths, concurrency, enabled metric groups (**all defaults live here**) |
| [`eval_metric_registry.py`](eval_metric_registry.py) | Single source of truth for metric names, display labels, SQL columns, and CSV fieldnames |
| [`eval_report_manager.py`](eval_report_manager.py) | `ReportManager` class — builds summary dicts, writes JSON and CSV reports |
| [`eval_sqlite.py`](eval_sqlite.py) | `EvalLedger` class — SQLite persistence layer with `eval_runs` and `eval_cases` tables |
| [`eval_dashboard.py`](eval_dashboard.py) | Streamlit dashboard for browsing and comparing evaluation runs |
| `requirements.txt` | Python dependencies for the evaluation harness (generated with pipreqs) |

### Documentation

| File | Purpose |
|---|---|
| [`README.md`](README.md) | This file — high-level overview, configuration, run instructions |
| [`info_metrics.md`](info_metrics.md) | Per-metric reference: range, direction, storage, computation, pass condition, examples |

### Tests

| Path | Purpose |
|---|---|
| [`tests/`](tests/) | Pytest suite covering metric computations, registry invariants, util helpers, and drift assertions |
| [`tests/conftest.py`](tests/conftest.py) | Adds the evaluation folder to `sys.path` so test modules can import sibling files directly |
| [`tests/test_eval_metric_registry.py`](tests/test_eval_metric_registry.py) | Registry consistency: uniqueness, label/format/decimals coverage, composite well-formedness, toggle-group invariants, CSV/SQL plumbing |
| [`tests/test_eval_metrics.py`](tests/test_eval_metrics.py) | Per-metric computations — source/chunk retrieval metrics, backend distribution, metadata match, keyword checks, batch chunk helper |
| [`tests/test_eval_utils.py`](tests/test_eval_utils.py) | Utility helpers — `safe_mean`, `extract_message_text`, `extract_agent_retrieval_results`, `make_prompt`, DeepEval context builders, preview formatter, `normalize_text` |

Run with:

```bash
cd HybridSearchAgent_LangGraph/evaluation
python -m pytest tests/ -q
```

### Data and Runtime Artifacts

| Path | Type | Purpose |
|---|---|---|
| [`eval_cases.json`](eval_cases.json) | JSON dataset | Test cases with questions, expected answers, expected sources, keywords, and retrieval config |
| `eval_ledger.db` | SQLite database | Persisted evaluation runs and per-case results — created automatically on first run |
| `.deepeval/` | DeepEval cache | Internal cache directory created by the DeepEval library |

### Output Artifacts (generated in `evaluation_results/`)

| Pattern | Type | Purpose |
|---|---|---|
| `deepeval_report_<timestamp>.json` | JSON | Full evaluation report with summary, per-case metrics, retrieval previews, and errors |
| `deepeval_summary_<timestamp>.csv` | CSV | Flat per-case summary table for quick comparison |

## Architecture Overview

```
eval_deepeval.py  (entry point)
    ├── eval_config.py            (settings, enabled metric groups, thresholds)
    ├── eval_utils.py             (load cases, build prompts, format contexts, normalize_text)
    ├── eval_metrics.py           (deterministic source & chunk retrieval metrics)
    ├── eval_models.py            (EvalCase, RetrievalSettings)
    ├── eval_metric_registry.py   (metric names, labels, SQL columns, CSV fieldnames)
    └── eval_engine.py            (EvaluationEngine)
            ├── run_retrieval_case()      → direct retriever call
            ├── run_agent_case()          → full LangGraph agent invocation
            ├── build_metrics()           → DeepEval judge panel (registry-checked)
            ├── _build_test_case()        → LLMTestCase for DeepEval
            ├── _run_metrics()            → all judge metrics (concurrent)
            └── _compute_case_status()    → PASS / REVIEW verdict
                    ↓
            eval_report_manager.py        → JSON + CSV reports
            eval_sqlite.py                → eval_ledger.db
                    ↓
            eval_dashboard.py             → Streamlit dashboard
```

**Flow per case:**

1. `run_retrieval_case()` queries the retriever directly. Used for retrieval quality metrics (hit@k, MRR, Precision@k, Recall@k, NDCG@k, metadata match ratio).
2. `run_agent_case()` invokes the full LangGraph agent with a unique thread ID. Captures the final answer and the chunks the agent actually retrieved via tool calls.
3. The agent's actual tool-call results are used for DeepEval context (faithfulness, contextual precision, contextual recall). Falls back to direct retrieval results if the agent made no tool call.
4. All seven DeepEval metrics run concurrently via `asyncio.gather`.
5. A three-gate verdict (metrics, retrieval, keywords) determines PASS or REVIEW.
6. Results are saved to JSON, CSV, and the SQLite ledger.

## Metrics

For full per-metric details (range, direction, storage, formulas, examples), see [`info_metrics.md`](info_metrics.md). The tables below are a high-level index.

### DeepEval LLM-as-Judge Metrics (7)

| Metric | What It Measures |
|---|---|
| `answer_relevancy` | Is the answer relevant to the question? |
| `faithfulness` | Is the answer grounded in the retrieved context? |
| `contextual_precision` | Are the relevant chunks ranked higher than irrelevant ones? |
| `contextual_recall` | Does the retrieved context cover the expected answer points? |
| `contextual_relevancy` | Are the retrieved chunks relevant to the question? |
| `hallucination` | Does the answer contradict the context? (lower is better) |
| `correctness_g_eval` | Does the answer correctly cover the expected output without unsupported claims? (GEval) |

### Source-level Retrieval Quality Metrics

Relevance is decided at the filename level. Any chunk whose source filename is in `expected_sources` counts as relevant.

| Metric | What It Measures |
|---|---|
| `hit_at_k` | Binary — 1.0 if at least one expected source appears in the top-k, else 0.0 |
| `metadata_match_ratio` | Fraction of results that satisfy all metadata filters on the case |
| `mrr` | Mean Reciprocal Rank — 1 / rank of the first relevant result |
| `precision_at_k` | Fraction of retrieved results from an expected source |
| `recall_at_k` | Fraction of expected sources found in the top-k results |
| `ndcg_at_k` | Normalized Discounted Cumulative Gain — penalizes relevant results ranked lower |

### Chunk-level Retrieval Quality Metrics

Relevance is decided at the chunk-text level via snippet substring matching against `expected_chunks`. Snippets are preferred over chunk IDs because chunk IDs change whenever the chunking strategy is tuned, while short representative text remains stable across re-chunking.

| Metric | What It Measures |
|---|---|
| `chunk_hit_at_k` | Binary — 1.0 if at least one retrieved chunk contains any expected snippet, else 0.0 |
| `chunk_mrr` | 1 / rank of the first chunk that matches any expected snippet |
| `chunk_precision_at_k` | Fraction of retrieved chunks that contain at least one expected snippet |
| `chunk_recall_at_k` | Fraction of expected snippets that were found in at least one retrieved chunk |
| `chunk_ndcg_at_k` | NDCG computed from per-chunk binary relevance labels |

### Answer Quality Checks

| Check | What It Measures |
|---|---|
| `required_keyword_hit_rate` | Fraction of required keywords present in the answer |
| `disallowed_keyword_hits` | Count of disallowed keywords found in the answer (should be 0) |

### Backend Distribution

`backend_distribution` is a diagnostic counter exploded into four integer columns (`backend_fts`, `backend_vector`, `backend_hybrid`, `backend_other`). The `backend_other` column is a catch-all so a future backend label can never silently disappear from the ledger.

## PASS / REVIEW Verdict

A case passes all three gates to earn PASS:

| Gate | Condition |
|---|---|
| **Metrics** | `faithfulness ≥ JUDGE_THRESHOLD` and `answer_relevancy ≥ JUDGE_THRESHOLD` |
| **Retrieval** | `hit_at_k = 1.0` and `metadata_match_ratio ≥ METADATA_MATCH_THRESHOLD` |
| **Keywords** | `required_keyword_hit_rate ≥ REQUIRED_KEYWORD_THRESHOLD` and `disallowed_keyword_hits = 0` |

Any case with runtime errors is automatically set to REVIEW regardless of scores.
Binary gates (`hit_at_k == 1.0`, `disallowed_keyword_hits == 0`) are not
configurable by design.

### Gate thresholds — configure once

`JUDGE_THRESHOLD`, `METADATA_MATCH_THRESHOLD`, and `REQUIRED_KEYWORD_THRESHOLD` set the
PASS/REVIEW gates. Their values are defined in [`eval_config.py`](eval_config.py) and are intended to be
set once per project and left alone so pass-rate trends stay comparable
across runs. Active thresholds are persisted with every run in the SQLite
ledger, and the dashboard shows a warning banner if they drift between the
selected run and the prior run.

## Configuration

All tuning is done in [`eval_config.py`](eval_config.py). That file is the single source of truth for default values.

| Setting | Description |
|---|---|
| `JUDGE_MODEL` | OpenAI model used as the LLM judge for all DeepEval metrics |
| `JUDGE_THRESHOLD` | DeepEval metric threshold; also the judge gate for PASS/REVIEW |
| `METADATA_MATCH_THRESHOLD` | Retrieval gate: minimum `metadata_match_ratio` for PASS (configure once) |
| `REQUIRED_KEYWORD_THRESHOLD` | Keyword gate: minimum `required_keyword_hit_rate` for PASS (configure once) |
| `DATASET_PATH` | Path to the test case dataset |
| `OUTPUT_DIR` | Directory for JSON and CSV report output |
| `CONCURRENCY` | Max number of eval cases running concurrently |
| `MAX_CASES` | Set to an int to limit the number of cases evaluated; `None` runs the full dataset |
| `ENABLED_METRIC_GROUPS` | Which toggleable metric groups to compute (see [Metric groups](#metric-groups) below) |

### Judge model temperature

Judge temperature is intentionally not exposed in `eval_config.py`. When `JUDGE_MODEL` is passed as a string, DeepEval constructs its own `GPTModel` wrapper internally and forces deterministic scoring (see DeepEval's `openai_model.py`). Reasoning models (o1, o3, etc.) are auto-handled by DeepEval because the OpenAI API rejects temperature on them. Varying judge temperature across runs would defeat the point of tracking trends over time, so there is no per-run override in this harness.

### Metric groups

`ENABLED_METRIC_GROUPS` (defined in [`eval_config.py`](eval_config.py)) controls which families of metrics are computed each run:

| Group | Metrics | Notes |
|---|---|---|
| `judge` | All 7 DeepEval LLM-judged metrics (faithfulness, answer_relevancy, contextual_*, hallucination, correctness_g_eval) | Skip to avoid judge-model API costs |
| `source` | Source-level retrieval quality — `hit_at_k`, `mrr`, `precision_at_k`, `recall_at_k`, `ndcg_at_k` | Filename-based relevance |
| `chunk` | Chunk-level retrieval quality — `chunk_hit_at_k`, `chunk_mrr`, `chunk_precision_at_k`, `chunk_recall_at_k`, `chunk_ndcg_at_k` | Snippet-based relevance against `expected_chunks` |

Always-on regardless of this setting: `metadata_match_ratio`, `backend_distribution`,
`keyword_checks`, latency metrics, agent/judge token counts, and `avg_judge_score`.

Disabled metrics are stored as `NULL` in the CSV and render as
"Not evaluated" in the dashboard. The verdict gate for a disabled group is
skipped (e.g. if `"judge"` is off, `faithfulness`/`answer_relevancy` are not
checked for PASS/REVIEW). The selected run's enabled groups are displayed in
the dashboard sidebar.

**Partial runs are excluded from the SQLite ledger.** To keep trend charts
comparable across runs, `EvalLedger.save_run` only persists runs where all
three metric groups (`judge`, `source`, `chunk`) were enabled. Partial runs
are still saved to JSON and CSV for debugging, but the dashboard will not
show them. The CLI prints a `SQLite ledger : skipped — partial run …` notice
when this happens.

## Test Cases ([`eval_cases.json`](eval_cases.json))

Each case defines:

```json
{
  "id": "population_and_capital",
  "question": "What is Norway's approximate population, and what is its capital city?",
  "expected_answer_points": ["Norway has around 5.5 million people.", "The capital city is Oslo."],
  "expected_sources": ["norway_facts.txt", "norway_population.pdf"],
  "expected_chunks": ["around 5.5 million people", "capital city is Oslo"],
  "required_keywords": ["5.5 million", "Oslo"],
  "disallowed_keywords": [],
  "metadata_filters": {},
  "answer_style": "Answer concisely and include a short Evidence section.",
  "retrieval": {
    "k": 5,
    "vector_search_method": "similarity",
    "use_phrase": false,
    "use_prefix": false,
    "multi_fts": true
  },
  "category": "Factual",
  "notes": "Core factual lookup case."
}
```

- **`expected_answer_points`** — used as DeepEval's `expected_output` and `context` (gold standard).
- **`expected_sources`** — used to compute source-level hit@k, MRR, Precision@k, Recall@k, and NDCG@k.
- **`expected_chunks`** — representative text snippets used for chunk-level retrieval metrics (substring match after normalization). Optional — omit to skip chunk-level scoring for the case.
- **`required_keywords` / `disallowed_keywords`** — checked against the agent's answer for the keyword gate.
- **`metadata_filters`** — passed to the retriever and checked against results for `metadata_match_ratio`.
- **`retrieval`** — per-case search hyperparameters (k, vector method, FTS modes).

## Requirements

Dependencies beyond the base HybridSearchAgent requirements are listed in `requirements.txt`:

```bash
pip install -r evaluation/requirements.txt
```

## Environment Variables

Required for OpenAI-based judge calls and embeddings:

```env
OPENAI_API_KEY=...
```

`eval_deepeval.py` calls `load_dotenv()` at startup, which loads the project root `.env` automatically.

## Run Instructions

### Run Evaluation

From the project root:

```bash
cd HybridSearchAgent_LangGraph
python evaluation/eval_deepeval.py
```

Output:

- Timestamped JSON and CSV reports in `evaluation_results/`
- Results persisted to `evaluation/eval_ledger.db`
- Summary printed to stdout

### Streamlit Dashboard

From the project root:

```bash
cd HybridSearchAgent_LangGraph
streamlit run evaluation/eval_dashboard.py
```

Opens at [http://localhost:8501](http://localhost:8501). The dashboard has four tabs:

- **Run Summary** — top-level KPIs (pass rate, avg judge score, latency), retrieval quality averages, a colour-coded per-case results table, grouped bar charts of metric scores, and expandable answer details for each case
- **Deep Analysis** — radar charts comparing LLM generation vs retrieval metric balance, score distribution histograms, a correlation heatmap showing how metrics relate to each other, and a stacked latency breakdown per case
- **Historical Trends** — line charts tracking summary scores, DeepEval metric averages, retrieval quality, latency, and token usage across all runs over time
- **Metrics Guide** — embeds [`info_metrics.md`](info_metrics.md) directly so the in-app guide always matches the source-of-truth doc

### Run Tests

From the evaluation folder:

```bash
cd HybridSearchAgent_LangGraph/evaluation
python -m pytest tests/ -q
```

The test suite covers metric computations, registry invariants (uniqueness, label/format/decimals coverage, composite well-formedness), util helpers, and drift assertions. The schema and composite-extractor drift checks in [`eval_sqlite.py`](eval_sqlite.py) also fire at module import time, so importing the harness in any context (production or test) fails loudly when the registry and SQL templates fall out of sync.

### View Ledger from CLI

```python
from evaluation.eval_sqlite import EvalLedger
ledger = EvalLedger()
# Use standard sqlite3 queries against ledger.db_path
```

## Suggested Learning Path

1. Start with [`eval_config.py`](eval_config.py) to understand the tunable settings (this is where every default lives).
2. Read [`eval_models.py`](eval_models.py) for the `EvalCase` and `RetrievalSettings` schemas.
3. Read [`eval_cases.json`](eval_cases.json) to see how test cases are structured.
4. Read [`eval_metric_registry.py`](eval_metric_registry.py) to see how metric names, labels, and SQL columns are defined in one place.
5. Read [`eval_utils.py`](eval_utils.py) for prompt building, context formatting, and data loading.
6. Read [`eval_metrics.py`](eval_metrics.py) for deterministic source- and chunk-level retrieval metric implementations.
7. Read [`eval_engine.py`](eval_engine.py) to understand the full evaluation flow per case.
8. Read [`info_metrics.md`](info_metrics.md) for per-metric definitions, formulas, and pass conditions.
9. Run `python evaluation/eval_deepeval.py` for an end-to-end evaluation.
10. Open the Streamlit dashboard to explore results visually.

## Notes

- This harness is intentionally educational and experimentation-oriented.
- The agent uses a unique `thread_id` per case to prevent memory bleed between evaluations.
- DeepEval metrics run concurrently via `asyncio.gather` for faster execution.
- Deleting `eval_ledger.db` resets all persisted run history. The file is recreated automatically on the next run.
- The `evaluation_results/` directory accumulates reports across runs. Clean it manually if needed.
- The SQLite auto-migration is **add-only** — registering a new metric in [`eval_metric_registry.py`](eval_metric_registry.py) extends the schema on next startup. Renaming or removing a metric needs a fresh DB: prefer **save-aside** (`mv evaluation/eval_ledger.db evaluation/eval_ledger_<YYYYMMDD>.db`) so the historical rows stay browsable. Outright wipe (`rm evaluation/eval_ledger.db`) is fine if the old data is not worth keeping.
