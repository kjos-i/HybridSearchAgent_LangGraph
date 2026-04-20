# Evaluation Engine - Flow & Schema

## Overview

`eval_engine.py` contains the `EvaluationEngine` class, which orchestrates end-to-end evaluation of the hybrid search agent. It runs each eval case through **three stages** — retrieval, agent execution, and metric scoring — then combines all signals into a final verdict.

---

## Data Models

### EvalCase (input)

```
EvalCase
├── id: str                          # unique case identifier
├── question: str                    # the user question to evaluate
├── expected_answer_points: list[str]# gold-standard answer fragments
├── expected_sources: list[str]      # documents that should be retrieved
├── expected_chunks: list[str]       # representative snippets for chunk-level metrics
├── required_keywords: list[str]     # must appear in the answer
├── disallowed_keywords: list[str]   # must NOT appear in the answer
├── metadata_filters: dict           # e.g. {"category": "policy"}
├── answer_style: str                # instruction appended to the prompt
├── retrieval: RetrievalSettings     # search hyperparameters (see below)
├── category: str                    # grouping tag
└── notes: str                       # free-text annotation
```

### RetrievalSettings

```
RetrievalSettings
├── k: int (1-10, default 5)         # number of results to retrieve
├── vector_search_method: "similarity" | "mmr"
├── use_phrase: bool
├── use_prefix: bool
└── multi_fts: bool (default True)
```

---

## Execution Flow

Below is the full pipeline that runs for **each** `EvalCase` inside the `evaluate_case()` method.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EvaluationEngine                             │
│  Inputs: agent, retriever, judge_model, threshold                   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                    evaluate_cases(cases, concurrency=2)
                                │
              ┌─────────────────┴─────────────────┐
              │  asyncio.gather + Semaphore(N)     │
              │  (bounded concurrency per case)    │
              └─────────────────┬─────────────────┘
                                │
               FOR EACH CASE ── evaluate_case(case)
                                │
         ┌──────────────────────┼──────────────────────┐
         ▼                                             ▼
  ┌──────────────┐                            ┌────────────────┐
  │ STAGE 1a     │                            │ STAGE 1b       │
  │ Direct       │                            │ Agent          │
  │ Retrieval    │                            │ Execution      │
  │              │                            │                │
  │ retriever    │                            │ agent.ainvoke  │
  │  .search()   │                            │  (HumanMessage)│
  │              │                            │                │
  │ Returns:     │                            │ Returns:       │
  │ - results[]  │                            │ - answer (str) │
  │ - latency    │                            │ - agent_results│
  │ - error?     │                            │ - latency      │
  └──────┬───────┘                            │ - error?       │
         │                                    └───────┬────────┘
         │                                            │
         │         ┌──────────────────────────────────┘
         │         │
         ▼         ▼
  ┌────────────────────────────────────────┐
  │ CONTEXT SELECTION                      │
  │                                        │
  │ if agent returned retrieval results:   │
  │   use agent_retrieval_results          │
  │ else:                                  │
  │   fall back to direct retrieval_results│
  └───────────────────┬────────────────────┘
                      │
                      ▼
  ┌────────────────────────────────────────┐
  │ STAGE 2: Build LLMTestCase            │
  │                                        │
  │ LLMTestCase(                           │
  │   input       = prompt with constraints│
  │   actual_output   = agent answer       │
  │   expected_output = joined answer pts  │
  │   context         = gold context       │
  │   retrieval_context = formatted chunks │
  │   completion_time   = agent latency    │
  │ )                                      │
  └───────────────────┬────────────────────┘
                      │
                      ▼
  ┌────────────────────────────────────────┐
  │ STAGE 3: Run DeepEval Metrics          │
  │ (all 7 metrics run concurrently)       │
  │                                        │
  │ LLM-judged (via judge_model):          │
  │ ├── AnswerRelevancyMetric              │
  │ ├── FaithfulnessMetric                 │
  │ ├── ContextualPrecisionMetric          │
  │ ├── ContextualRecallMetric             │
  │ ├── ContextualRelevancyMetric          │
  │ ├── HallucinationMetric               │
  │ └── GEval ("Grounded Correctness")     │
  │                                        │
  │ Each metric returns:                   │
  │   { score, reason, passed, threshold } │
  └───────────────────┬────────────────────┘
                      │
                      ▼
  ┌────────────────────────────────────────┐
  │ STAGE 4: Compute Retrieval Quality     │
  │ (computed from direct retrieval_results│
  │  — not the agent's context)            │
  │                                        │
  │ Source-level (filename relevance):     │
  │ ├── hit_at_k                           │
  │ ├── MRR (Mean Reciprocal Rank)         │
  │ ├── Precision@k                        │
  │ ├── Recall@k                           │
  │ └── NDCG@k                             │
  │                                        │
  │ Chunk-level (snippet-substring match): │
  │ ├── chunk_hit_at_k                     │
  │ ├── chunk_mrr                          │
  │ ├── chunk_precision_at_k               │
  │ ├── chunk_recall_at_k                  │
  │ └── chunk_ndcg_at_k                    │
  │                                        │
  │ Always-on:                             │
  │ ├── metadata_match_ratio               │
  │ ├── backend_distribution               │
  │ └── keyword_checks                     │
  └───────────────────┬────────────────────┘
                      │
                      ▼
  ┌────────────────────────────────────────┐
  │ STAGE 5: Final Verdict                 │
  │                                        │
  │ compute_case_status() applies 3 gates: │
  │                                        │
  │ metrics_ok:                            │
  │   faithfulness >= 0.5 AND              │
  │   answer_relevancy >= 0.5              │
  │                                        │
  │ retrieval_ok:                           │
  │   hit_at_k == 1.0 AND                  │
  │   metadata_match_ratio >= 0.8          │
  │                                        │
  │ keywords_ok:                           │
  │   required_keyword_hit_rate >= 0.5 AND │
  │   disallowed_keyword_hits == 0         │
  │                                        │
  │ ALL 3 gates pass  →  status = "PASS"   │
  │ ANY gate fails    →  status = "REVIEW" │
  │ ANY error present →  status = "REVIEW" │
  └───────────────────┬────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │ CASE RESULT   │
              │ (dict)        │
              └───────────────┘
```

---

## Output Schema

Each case produces a result dict with this structure:

```
{
  "id":                      str,
  "question":                str,
  "category":                str,
  "notes":                   str,
  "prompt_used":             str,       # full prompt with constraints appended
  "expected_output":         str,       # joined expected_answer_points
  "answer":                  str,       # agent's actual response

  // Latency breakdown
  "latency_seconds":         float,     # total agent round-trip
  "retrieval_latency_seconds": float,   # direct retriever call
  "llm_latency_seconds":     float,     # latency - retrieval_latency (estimated)

  // Retrieval config & filters used
  "retrieval_config":        { k, vector_search_method, use_phrase, use_prefix, multi_fts },
  "metadata_filters":        { ... },
  "expected_sources":        [ str, ... ],
  "retrieval_preview":       [ { source, chunk_id, backend, score, snippet }, ... ],  // top 5

  // Retrieval quality scores (from direct retrieval, not agent context)
  // Source-level — relevance by filename (expected_sources)
  "hit_at_k":                float,     # 1.0 if any expected source in top-k, else 0.0
  "mrr":                     float,     # 1/rank of first relevant result
  "precision_at_k":          float,     # relevant results / total results
  "recall_at_k":             float,     # found sources / expected sources
  "ndcg_at_k":               float,     # normalized ranking quality
  // Chunk-level — relevance by snippet substring (expected_chunks)
  "chunk_hit_at_k":          float,     # 1.0 if any expected snippet matched any chunk
  "chunk_mrr":               float,     # 1/rank of first chunk containing a snippet
  "chunk_precision_at_k":    float,     # fraction of chunks matching any snippet
  "chunk_recall_at_k":       float,     # fraction of expected snippets covered
  "chunk_ndcg_at_k":         float,     # normalized ranking quality on chunk labels
  // Always-on
  "metadata_match_ratio":    float,     # fraction of results matching filters
  "backend_distribution":    { "fts": N, "vector": N, ... },

  // Answer quality
  "keyword_checks": {
    "required_keyword_hit_rate": float,
    "disallowed_keyword_hits":   int
  },
  "avg_judge_score":         float,     # mean of all DeepEval scores * 100

  // DeepEval LLM-judged metrics
  "metrics": {
    "answer_relevancy":      { score, reason, passed, threshold },
    "faithfulness":          { score, reason, passed, threshold },
    "contextual_precision":  { score, reason, passed, threshold },
    "contextual_recall":     { score, reason, passed, threshold },
    "contextual_relevancy":  { score, reason, passed, threshold },
    "hallucination":         { score, reason, passed, threshold },
    "correctness_g_eval":    { score, reason, passed, threshold }
  },

  "status":                  "PASS" | "REVIEW",
  "errors":                  [ str, ... ]
}
```

---

## Key Design Decisions

### Why two retrieval paths?

The engine runs retrieval **twice** per case:

1. **Direct retrieval** (`run_retrieval_case`) — calls the retriever directly. Used to compute ground-truth retrieval quality metrics (hit_at_k, MRR, Precision@k, etc.). This isolates search quality from the agent's LLM reasoning.

2. **Agent retrieval** (`run_agent_case`) — extracted from ToolMessage objects in the agent's message history. This is what the agent actually saw and reasoned over. Used as `retrieval_context` in the DeepEval LLMTestCase so that faithfulness, contextual precision, and recall reflect the agent's real context window.

If the agent made no tool calls, direct retrieval results are used as a fallback for DeepEval context.

### Concurrency model

`evaluate_cases()` uses `asyncio.gather` with a `Semaphore(concurrency)` to bound how many cases run in parallel (default: 2). Within each case, the 7 DeepEval metrics also run concurrently via `asyncio.gather`. Each case gets a unique `thread_id` to prevent memory bleed in the LangGraph agent's checkpointer.

### Metrics are rebuilt per call

`build_metrics()` creates fresh metric instances every time to ensure clean state across concurrent runs — DeepEval metric objects store results internally, so reusing them across cases would cause data contamination.

### Metric group toggles

The engine accepts an `enabled_groups` set (sourced from
`eval_config.ENABLED_METRIC_GROUPS`) that controls which metric families are
computed:

- `"judge"` — DeepEval LLM-judged metrics. When disabled, `build_metrics()`
  returns `{}` so the judge model is never called, `metrics` in the case
  result is empty, and the `metrics_ok` verdict gate is treated as satisfied.
- `"source"` — source-level retrieval quality (hit_at_k, mrr, precision_at_k,
  recall_at_k, ndcg_at_k). Relevance is decided by filename match against
  `expected_sources`. When disabled, these fields are stored as `None` and
  the `hit_at_k` sub-condition in the retrieval gate is skipped.
- `"chunk"` — chunk-level retrieval quality (chunk_hit_at_k, chunk_mrr,
  chunk_precision_at_k, chunk_recall_at_k, chunk_ndcg_at_k). Relevance is
  decided by normalized-substring match of any `expected_chunks` snippet
  against the chunk's `page_content`. When disabled, these fields are stored
  as `None`. Cases with an empty `expected_chunks` list score 1.0 (nothing
  to check) rather than `None` so they do not poison run-level averages.

Always-on: `metadata_match_ratio`, `backend_distribution`, `keyword_checks`,
latency, and `avg_judge_score` (derived from whatever judge metrics ran —
`None` when no judge metrics ran).

The enabled groups are persisted in the summary (`enabled_groups` key),
stored in the `eval_runs.enabled_groups` SQLite column as a JSON array, and
surfaced in the dashboard sidebar.
