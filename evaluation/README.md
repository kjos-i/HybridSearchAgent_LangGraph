# Evaluation Harness for HybridSearchAgent

Evaluation harness for the hybrid retrieval agent. Runs the real LangGraph agent end-to-end and scores each test case with three families of metrics: **LLM-as-judge** (via DeepEval), **deterministic retrieval quality**, and **keyword-based answer checks**. Results are persisted to a SQLite ledger with a Streamlit dashboard for browsing runs over time.

The Streamlit dashboard can be used to visualize evaluation metrics stored in the SQLite ledger at any time:

```bash
python -m streamlit run eval_dashboard.py --server.port 8501
```

See [Run Instructions](#run-instructions) below for how to run the evaluation.

For more in-depth coverage, see [`info_eval_engine.md`](info_eval_engine.md) (evaluation engine internals) and [`info_metrics.md`](info_metrics.md) (detailed metric definitions and interpretation).

All scripts in this folder are Learning/Demo status.

## What It Does

1. Loads test cases from `eval_cases.json` (question, expected answers, expected sources, keywords, retrieval config).
2. For each case, runs both the **direct retriever** (ground-truth search quality) and the **full LangGraph agent** (end-to-end answer quality).
3. Evaluates the agent's answer against seven LLM-as-judge metrics (via DeepEval), six deterministic retrieval quality metrics, and keyword-based answer checks.
4. Assigns a **PASS / REVIEW** verdict based on combined metric, retrieval, and keyword gates.
5. Writes timestamped JSON and CSV reports to `evaluation_results/`.
6. Persists every run and per-case result to `eval_ledger.db` (SQLite) for cross-run comparison.
7. Provides a Streamlit dashboard that reads from the SQLite ledger to visualize trends, per-case scores, and latency over time.

## Folder Contents

### Python Scripts

| File | Purpose |
|---|---|
| `eval_deepeval.py` | Entry point — wires agent, engine, report manager, and SQLite ledger together |
| `eval_engine.py` | `EvaluationEngine` class — runs retrieval, agent invocation, and DeepEval metrics for each case |
| `eval_utils.py` | Utility helpers — data loading, prompt building, retrieval metric functions, keyword checks |
| `eval_models.py` | Pydantic data models: `EvalCase` and `RetrievalSettings` |
| `eval_config.py` | Configuration constants: judge model, threshold, paths, concurrency |
| `eval_metric_registry.py` | Single source of truth for metric names, display labels, SQL columns, and CSV fieldnames |
| `eval_report_manager.py` | `ReportManager` class — builds summary dicts, writes JSON and CSV reports |
| `eval_sqlite.py` | `EvalLedger` class — SQLite persistence layer with `eval_runs` and `eval_cases` tables |
| `eval_dashboard.py` | Streamlit dashboard for browsing and comparing evaluation runs |
| `requirements.txt` | Python dependencies for the evaluation harness (generated with pipreqs) |

### Data and Runtime Artifacts

| Path | Type | Purpose |
|---|---|---|
| `eval_cases.json` | JSON dataset | Test cases with questions, expected answers, expected sources, keywords, and retrieval config |
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
    ├── eval_config.py            (settings)
    ├── eval_utils.py             (load cases, build prompts, compute retrieval metrics)
    ├── eval_models.py            (EvalCase, RetrievalSettings)
    ├── eval_metric_registry.py   (metric names, labels, SQL columns, CSV fieldnames)
    └── eval_engine.py            (EvaluationEngine)
            ├── run_retrieval_case()     → direct retriever call
            ├── run_agent_case()         → full LangGraph agent invocation
            ├── build_test_case()        → LLMTestCase for DeepEval
            ├── run_metrics()            → 7 DeepEval metrics (concurrent)
            └── compute_case_status()    → PASS / REVIEW verdict
                    ↓
            eval_report_manager.py       → JSON + CSV reports
            eval_sqlite.py               → eval_ledger.db
                    ↓
            eval_dashboard.py            → Streamlit dashboard
```

**Flow per case:**

1. `run_retrieval_case()` queries the retriever directly — used for retrieval quality metrics (source hit rate, MRR, Precision@k, Recall@k, NDCG@k, metadata match ratio).
2. `run_agent_case()` invokes the full LangGraph agent with a unique thread ID — captures the final answer and the chunks the agent actually retrieved via tool calls.
3. The agent's actual tool-call results are used for DeepEval context (faithfulness, contextual precision, contextual recall). Falls back to direct retrieval results if the agent made no tool call.
4. All seven DeepEval metrics run concurrently via `asyncio.gather`.
5. A three-gate verdict (metrics, retrieval, keywords) determines PASS or REVIEW.
6. Results are saved to JSON, CSV, and the SQLite ledger.

## Metrics

### DeepEval LLM-as-Judge Metrics (7)

| Metric | What It Measures |
|---|---|
| `answer_relevancy` | Is the answer relevant to the question? |
| `faithfulness` | Is the answer grounded in the retrieved context? |
| `contextual_precision` | Are the relevant chunks ranked higher than irrelevant ones? |
| `contextual_recall` | Does the retrieved context cover the expected answer points? |
| `contextual_relevancy` | Are the retrieved chunks relevant to the question? |
| `hallucination` | Does the answer contain unsupported claims? |
| `correctness_g_eval` | Does the answer correctly cover the expected output without unsupported claims? (GEval) |

### Retrieval Quality Metrics (6)

| Metric | What It Measures |
|---|---|
| `source_hit_rate` | Fraction of expected sources found in retrieved results |
| `metadata_match_ratio` | Fraction of results that satisfy all metadata filters on the case |
| `mrr` | Mean Reciprocal Rank — 1 / rank of the first relevant result |
| `precision_at_k` | Fraction of retrieved results from an expected source |
| `recall_at_k` | Fraction of expected sources found in the top-k results |
| `ndcg_at_k` | Normalized Discounted Cumulative Gain — penalizes relevant results ranked lower |

### Answer Quality Checks

| Check | What It Measures |
|---|---|
| `required_keyword_hit_rate` | Fraction of required keywords present in the answer |
| `disallowed_keyword_hits` | Count of disallowed keywords found in the answer (should be 0) |

## PASS / REVIEW Verdict

A case passes all three gates to earn PASS:

| Gate | Condition |
|---|---|
| **Metrics** | `faithfulness` ≥ 0.5 and `answer_relevancy` ≥ 0.5 |
| **Retrieval** | `source_hit_rate` ≥ 0.5 and `metadata_match_ratio` ≥ 0.8 |
| **Keywords** | `required_keyword_hit_rate` ≥ 0.5 and `disallowed_keyword_hits` = 0 |

Any case with runtime errors is automatically set to REVIEW regardless of scores.

## Configuration

All tuning is done in `eval_config.py`:

| Setting | Default | Description |
|---|---|---|
| `JUDGE_MODEL` | `"gpt-4o"` | OpenAI model used as the LLM judge for all DeepEval metrics |
| `THRESHOLD` | `0.5` | Score threshold passed to each DeepEval metric |
| `DATASET_PATH` | `evaluation/eval_cases.json` | Path to the test case dataset |
| `OUTPUT_DIR` | `evaluation_results/` | Directory for JSON and CSV report output |
| `CONCURRENCY` | `2` | Max number of eval cases running concurrently |
| `MAX_CASES` | `None` | Set to an int (e.g. `3`) to limit the number of cases evaluated |

## Test Cases (`eval_cases.json`)

Each case defines:

```json
{
  "id": "population_and_capital",
  "question": "What is Norway's approximate population, and what is its capital city?",
  "expected_answer_points": ["Norway has around 5.5 million people.", "The capital city is Oslo."],
  "expected_sources": ["norway_facts.txt", "norway_population.pdf"],
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
- **`expected_sources`** — used to compute source hit rate, MRR, Precision@k, Recall@k, and NDCG@k.
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

The `.env` file path is set in `eval_deepeval.py`:

```python
load_dotenv()
```

Update this path to match your local setup before running.

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

- **Run Summary** — top-level KPIs (pass rate, average score, latency), retrieval quality averages, a colour-coded per-case results table, grouped bar charts of metric scores, and expandable answer details for each case
- **Deep Analysis** — radar charts comparing LLM generation vs retrieval metric balance, score distribution histograms, a correlation heatmap showing how metrics relate to each other, and a stacked latency breakdown per case
- **Historical Trends** — line charts tracking summary scores, DeepEval metric averages, and latency across all runs over time, plus a per-case tracker to follow a single case across runs
- **Metrics Guide** — embedded reference guide with metric explanations, score colour legend, and verdict logic

### View Ledger from CLI

```python
from evaluation.eval_sqlite import EvalLedger
ledger = EvalLedger()
# Use standard sqlite3 queries against ledger.db_path
```

## Suggested Learning Path

1. Start with `eval_config.py` to understand the tunable settings.
2. Read `eval_models.py` for the `EvalCase` and `RetrievalSettings` schemas.
3. Read `eval_cases.json` to see how test cases are structured.
4. Read `eval_metric_registry.py` to see how metric names, labels, and SQL columns are defined in one place.
5. Read `eval_utils.py` for prompt building, retrieval metrics, and keyword checks.
6. Read `eval_engine.py` to understand the full evaluation flow per case.
7. Run `python evaluation/eval_deepeval.py` for an end-to-end evaluation.
8. Open the Streamlit dashboard to explore results visually.

## Notes

- This harness is intentionally educational and experimentation-oriented.
- The agent uses a unique `thread_id` per case to prevent memory bleed between evaluations.
- DeepEval metrics run concurrently via `asyncio.gather` for faster execution.
- Deleting `eval_ledger.db` resets all persisted run history. The file is recreated automatically on the next run.
- The `evaluation_results/` directory accumulates reports across runs. Clean it manually if needed.
- If you add new metrics or change the SQLite schema, delete `eval_ledger.db` so it is recreated with the new columns.
