# Evaluation Metrics Reference

This document describes every metric produced by the Hybrid Search Agent (HSA) evaluation harness. Metrics are grouped into two categories: **LLM-judged metrics** (scored by a judge model via DeepEval) and **deterministic metrics** (computed directly from retrieval results and the agent's answer).

The harness has three toggleable metric groups configured via `ENABLED_METRIC_GROUPS` in `eval_config.py`:

- **`judge`** — DeepEval LLM-judged metrics (faithfulness, answer_relevancy, contextual_*, hallucination, correctness_g_eval).
- **`source`** — source-level retrieval metrics (hit_at_k, mrr, precision_at_k, recall_at_k, ndcg_at_k).
- **`chunk`** — chunk-level retrieval metrics (snippet-substring match against `expected_chunks`).

Always-on regardless of toggles: `metadata_match_ratio`, `backend_distribution`, `keyword_checks`, latency, agent/judge tokens, `avg_judge_score`. Disabled metrics are stored as NULL in the ledger and CSV, the dashboard renders them as "Not evaluated". Verdict gates skip sub-conditions whose metric group is disabled.

For information about the verdict logic (which metrics actually flip a case from PASS to REVIEW) jump to [PASS/REVIEW Gates Verdict Logic](#verdict-logic) at the bottom of this document.

### Metrics at a Glance

**LLM-Judged (DeepEval)**

| Metric | Description |
|--------|-------------|
| [Answer Relevancy](#answer-relevancy) | Is the answer on-topic for the question? |
| [Faithfulness](#faithfulness) | Are all claims supported by the retrieved context? |
| [Contextual Precision](#contextual-precision) | Are relevant chunks ranked above irrelevant ones? |
| [Contextual Recall](#contextual-recall) | Does the context cover all needed information? |
| [Contextual Relevancy](#contextual-relevancy) | What fraction of retrieved chunks are relevant? |
| [Hallucination](#hallucination) | Does the answer contradict the context? |
| [Grounded Correctness (GEval)](#grounded-correctness-geval) | Is the answer correct compared to the expected output? |

**Deterministic — Source-level Retrieval** (relevance by filename)

| Metric | Description |
|--------|-------------|
| [Hit@k](#hitk) | Did at least one expected source appear in the top-k? |
| [MRR](#mean-reciprocal-rank-mrr) | How early does the first relevant result appear? |
| [Precision@k](#precisionk) | What fraction of retrieved results are relevant? |
| [Recall@k](#recallk) | What fraction of expected sources were retrieved? |
| [NDCG@k](#ndcgk-normalized-discounted-cumulative-gain) | How good is the overall ranking quality? |

**Deterministic — Chunk-level Retrieval** (relevance by snippet substring match)

| Metric | Description |
|--------|-------------|
| [Chunk Hit@k](#chunk-hitk) | Did at least one expected snippet appear in any retrieved chunk? |
| [Chunk MRR](#chunk-mrr) | How early does the first snippet-matching chunk appear? |
| [Chunk Precision@k](#chunk-precisionk) | What fraction of retrieved chunks contain any expected snippet? |
| [Chunk Recall@k](#chunk-recallk) | What fraction of expected snippets were found in a retrieved chunk? |
| [Chunk NDCG@k](#chunk-ndcgk) | How good is the chunk ranking under snippet-level relevance? |

**Deterministic — Other**

| Metric | Description |
|--------|-------------|
| [Metadata Match Ratio](#metadata-match-ratio) | Do retrieved results satisfy the metadata filters? |
| [Backend Distribution](#backend-distribution) | How are results split across search backends? |
| [Required Keyword Hit Rate](#required-keyword-hit-rate) | Does the answer contain the required key terms? |
| [Disallowed Keyword Hits](#disallowed-keyword-hits) | Does the answer avoid disallowed terms? |

**Latency**

| Metric | Description |
|--------|-------------|
| [Latency](#latency) | Total wall-clock time for the agent to answer a case. |
| [Retrieval Latency](#retrieval-latency) | Time spent in the direct retriever call. |
| [LLM Latency](#llm-latency) | Estimated LLM generation time (total − retrieval). |

**Summary**

| Metric | Description |
|--------|-------------|
| [Pass Rate](#pass-rate) | Fraction of cases with a PASS verdict (run-level). |
| [Avg Judge Score](#avg-judge-score) | Mean of all LLM-judged scores per case, then meaned across cases at run level. |

---

## LLM-Judged Metrics (DeepEval)

These metrics use a judge LLM (configured via `JUDGE_MODEL` in [`eval_config.py`](eval_config.py)) to score each test case. They are bound to a `_TokenTrackingJudgeLLM` wrapper so that token usage from the entire judge panel accumulates onto a single counter per case. Each metric returns `{score, reason, passed, threshold}`.

---

<a id="answer-relevancy"></a>
### Answer Relevancy

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | judge |
| Stored as | `eval_cases.answer_relevancy` |
| Computed by | DeepEval `AnswerRelevancyMetric` |
| Pass condition | `score >= JUDGE_THRESHOLD` (configurable in `eval_config.py`); also one half of the metrics gate for the case verdict |

**What it evaluates:** Whether the agent's answer is on-topic for the user's question, i.e. that the response actually addresses what was asked rather than providing tangential or off-topic information. It does not check whether the information is true.

**Why it is included:** A RAG system can retrieve perfect context but still produce an answer that drifts from the question. This metric catches cases where the LLM ignores the input or over-generalises. It is one of the two judge gates that flips a case to REVIEW.

**How it is calculated:** The judge LLM splits the actual output into individual statements and rates each for relevance to the input query. Score = relevant statements / total statements.

---

<a id="faithfulness"></a>
### Faithfulness

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | judge |
| Stored as | `eval_cases.faithfulness` |
| Computed by | DeepEval `FaithfulnessMetric` |
| Pass condition | `score >= JUDGE_THRESHOLD` (configurable in `eval_config.py`); also one half of the metrics gate for the case verdict |

**What it evaluates:** Whether every factual claim in the agent's answer is supported by the retrieved context. A faithful answer makes no statements that go beyond what the context provides.

**Why it is included:** The primary safeguard against hallucination in RAG. If the agent invents facts not present in the retrieved documents, the answer is unreliable even if it sounds correct. It is the second of the two judge gates that flips a case to REVIEW.

**How it is calculated:** The judge LLM extracts individual claims from the actual output and verifies each against the retrieval context. Score = supported claims / total claims. A score of 1.0 means every claim traces back to a retrieved chunk.

---

<a id="contextual-precision"></a>
### Contextual Precision

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | judge |
| Stored as | `eval_cases.contextual_precision` |
| Computed by | DeepEval `ContextualPrecisionMetric` |
| Pass condition | `score >= JUDGE_THRESHOLD` (judge metric line, not a verdict gate) |

**What it evaluates:** Whether the relevant chunks in the retrieval context are ranked higher than irrelevant ones. It measures ranking quality, not just presence.

**Why it is included:** In a hybrid search system, retrieval order matters. If relevant documents are buried below noise, the LLM may miss or deprioritise them. This metric surfaces ranking problems in the fusion pipeline that source-level Hit@k and Recall@k can't see.

**How it is calculated:** The judge LLM labels each node in the retrieval context as relevant or irrelevant against the input and expected output. DeepEval then computes a weighted cumulative precision: Precision is calculated at every rank where a relevant node is found, with a heavy penalty when relevant chunks sit deep in the list.

---

<a id="contextual-recall"></a>
### Contextual Recall

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | judge |
| Stored as | `eval_cases.contextual_recall` |
| Computed by | DeepEval `ContextualRecallMetric` |
| Pass condition | `score >= JUDGE_THRESHOLD` (judge metric line, not a verdict gate) |

**What it evaluates:** Whether the retrieval context contains all the information needed to produce the expected output. It checks completeness of the retrieved evidence.

**Why it is included:** A retrieval system might return chunks that are individually relevant but collectively miss key facts. Catches gaps in retrieval coverage that show up as the LLM "not knowing" something it should have been told.

**How it is calculated:** The judge LLM extracts each individual statement from the expected output and checks whether it can be attributed to any node in the retrieved context. Score = attributable statements / total statements in the expected output.

---

<a id="contextual-relevancy"></a>
### Contextual Relevancy

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | judge |
| Stored as | `eval_cases.contextual_relevancy` |
| Computed by | DeepEval `ContextualRelevancyMetric` |
| Pass condition | `score >= JUDGE_THRESHOLD` (judge metric line, not a verdict gate) |

**What it evaluates:** Whether each chunk in the retrieval context is relevant to the input question. Unlike contextual precision (which focuses on ranking), this measures the overall noise level in the context window.

**Why it is included:** Retrieving too many irrelevant chunks dilutes the LLM's attention and can produce worse answers and higher latency. Quantifies how much noise the retrieval pipeline is introducing independent of where in the ranking the noise appears.

**How it is calculated:** The judge LLM splits the retrieval context into individual statements and rates each for relevance to the input. Score = relevant statements / total statements in the context.

---

<a id="hallucination"></a>
### Hallucination

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | **lower is better** (target: 0.0; this is the only judge metric where high is bad) |
| Toggle group | judge |
| Stored as | `eval_cases.hallucination` |
| Computed by | DeepEval `HallucinationMetric` |
| Pass condition | `score <= JUDGE_THRESHOLD` (DeepEval inverts the comparison for this metric; not a verdict gate) |

**What it evaluates:** Whether the agent's answer contradicts the provided context. Faithfulness checks for *unsupported* claims, while hallucination specifically detects *contradictions* (claims that disagree with what the context says).

**Why it is included:** A hallucinated answer is worse than an incomplete one because it actively misleads the user. Provides a contradiction-focused safety net beyond faithfulness, which only verifies support.

**How it is calculated:** The judge LLM emits one verdict per context indicating whether the actual output contradicts that context. Score = `contradicting_verdicts / total_verdicts`, so 0.0 means no contradictions detected and 1.0 means every context was contradicted. DeepEval's `is_successful()` for this metric returns `score <= threshold`, which is the opposite of every other judge metric in the panel So, when comparing scores on the dashboard, treat the hallucination column as "smaller = better".

---

<a id="grounded-correctness-geval"></a>
### Grounded Correctness (GEval)

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | judge |
| Stored as | `eval_cases.correctness_g_eval` |
| Computed by | DeepEval `GEval` (custom criteria built in `eval_engine.build_metrics`) |
| Pass condition | `score >= JUDGE_THRESHOLD` (judge metric line, not a verdict gate) |

**What it evaluates:** Whether the agent's answer correctly addresses the user's question, covers the important facts from the expected output, and avoids unsupported claims. A holistic correctness check.

**Why it is included:** The other judge metrics evaluate individual dimensions (relevancy, faithfulness, context quality), but none directly ask "is this answer correct?". GEval fills that gap by comparing the actual answer against the expected answer points.

**How it is calculated:** A single GEval call with criteria *"Determine whether the actual output correctly answers the user's request, covers the important facts from the expected output, and avoids unsupported claims."*. The judge LLM uses chain-of-thought reasoning over `INPUT`, `ACTUAL_OUTPUT`, and `EXPECTED_OUTPUT`. It generates evaluation steps, scores each step, and combines them into a final 0–1 score.

---

## Deterministic Retrieval Metrics

These metrics are computed directly from the retrieval results without an LLM judge. They operate on the **direct retrieval** results (the retriever called bypassing the agent), not the agent's own retrieval context, so they isolate retrieval quality from generation behavior.

Retrieval quality is measured on two axes:

- **Source-level** — relevance is decided by filename match against `expected_sources`. Any chunk whose source filename is in the expected list counts as relevant, regardless of what text the chunk contains.
- **Chunk-level** — relevance is decided by substring match against `expected_chunks` (a list of short representative text snippets on the case). A retrieved chunk is relevant when any expected snippet appears inside the chunk's `page_content` after normalization. Snippets are preferred over chunk IDs because chunk IDs change whenever the chunking strategy is tuned, while short representative text remains stable across re-chunking.

Both axes produce the same five metrics (hit@k, MRR, precision@k, recall@k, NDCG@k). Chunk-level variants are prefixed with `chunk_`. Cases with no `expected_sources` or no `expected_chunks` score 1.0 for their axis (nothing to check), so they don't poison run-level averages.

---

<a id="hitk"></a>
### Hit@k

| Field | Value |
|-------|-------|
| Range | 0.0 or 1.0 |
| Direction | higher is better |
| Toggle group | source |
| Stored as | `eval_cases.hit_at_k`, `eval_runs.avg_hit_at_k` |
| Computed by | `compute_hit_at_k` in `eval_metrics.py` |
| Pass condition | `value == 1.0` (binary by design; one half of the retrieval gate) |

**What it evaluates:** A binary signal of whether at least one expected source appears anywhere in the top-k retrieved results. Relevance is decided at the source (filename) level, not the chunk level.

**Why it is included:** The most basic retrieval health check. Did the system find *anything* right? If no expected source appears at all, no amount of good ranking or generation can save the answer. Complements [recall@k](#recallk), which reports *how much* of the expected set was found.

**How it is calculated:** `1.0 if (expected_sources ∩ retrieved_sources) else 0.0`. Both sides are reduced to lowercase filename sets via `Path(source).name.lower()`, so duplicates collapse (five chunks from `A.pdf` count as one hit, not five). Returns 1.0 when no expected sources are defined (nothing to check).

**Worked example:** expected sources = `{A.pdf, B.pdf, C.pdf}`.

| Retrieved chunks | Unique retrieved filenames | Intersection | Score |
|------------------|----------------------------|--------------|-------|
| 3 chunks from `A.pdf`, `B.pdf`, `D.pdf` | `{A, B, D}` | `{A, B}` | 1.0 |
| 2 chunks from `A.pdf`, 1 from `B.pdf` | `{A, B}` | `{A, B}` | 1.0 |
| 10 chunks all from `A.pdf` | `{A}` | `{A}` | 1.0 |
| 3 chunks from `D.pdf`, `E.pdf`, `F.pdf` | `{D, E, F}` | `{}` | 0.0 |
| 3 chunks from `A.pdf`, `B.pdf`, `C.pdf` | `{A, B, C}` | `{A, B, C}` | 1.0 |

---

<a id="mean-reciprocal-rank-mrr"></a>
### Mean Reciprocal Rank (MRR)

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | source |
| Stored as | `eval_cases.mrr`, `eval_runs.avg_mrr` |
| Computed by | `compute_mrr` in `eval_metrics.py` |
| Pass condition | (not gated) |

**What it evaluates:** How early the first relevant result appears in the ranked retrieval list. 1.0 if the first result is relevant, 0.5 if the second is, 0.33 if the third is, and so on. Relevance is decided at the source (filename) level.

**Why it is included:** In RAG, the top-ranked result has the most influence on the LLM's generation. MRR tells you whether the retrieval pipeline is placing the most important document first or burying it.

**How it is calculated:** `1 / rank`, where `rank` is the 1-indexed position of the first retrieved result whose normalized filename is in `expected_sources`. Returns 0.0 if no expected source is found in the top-k, and 1.0 if no expected sources are defined for the case.

---

<a id="precisionk"></a>
### Precision@k

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | source |
| Stored as | `eval_cases.precision_at_k`, `eval_runs.avg_precision_at_k` |
| Computed by | `compute_precision_at_k` in `eval_metrics.py` |
| Pass condition | (not gated) |

**What it evaluates:** The fraction of the top-k results that come from an expected source, i.e. how much of the retrieval budget is spent on relevant documents. Relevance is decided at the source (filename) level.

**Why it is included:** A low precision means the retrieval pipeline is returning a lot of noise alongside relevant documents, wasting context-window space and risking confusing the LLM. Reported alongside [recall@k](#recallk) for the standard precision/recall trade-off picture.

**How it is calculated:** `(retrieved chunks whose filename ∈ expected_sources) / (total retrieved chunks)`. Filename comparison is case-insensitive. Returns 1.0 if no expected sources are defined or 0.0 if results are empty.

---

<a id="recallk"></a>
### Recall@k

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | source |
| Stored as | `eval_cases.recall_at_k`, `eval_runs.avg_recall_at_k` |
| Computed by | `compute_recall_at_k` in `eval_metrics.py` |
| Pass condition | (not gated) |

**What it evaluates:** The fraction of expected source documents that appear in the top-k retrieved results. Retrieval completeness, decided at the source (filename) level.

**Why it is included:** Where [hit@k](#hitk) tells you *whether* anything relevant was found (binary health check), recall@k tells you *how much* of the expected set was covered. Together with precision@k it reveals the trade-off between retrieving broadly and retrieving precisely.

**How it is calculated:** `|expected_sources ∩ retrieved_sources| / |expected_sources|`. Note this measures whether each expected *file* was retrieved, not whether specific expected passages were (for that, see [chunk_recall_at_k](#chunk-recallk)). Returns 1.0 if no expected sources are defined, 0.0 if results are empty.

---

<a id="ndcgk-normalized-discounted-cumulative-gain"></a>
### NDCG@k (Normalized Discounted Cumulative Gain)

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | source |
| Stored as | `eval_cases.ndcg_at_k`, `eval_runs.avg_ndcg_at_k` |
| Computed by | `compute_ndcg_at_k` in `eval_metrics.py` |
| Pass condition | (not gated) |

**What it evaluates:** Ranking quality compared to the ideal ranking where every relevant result is at the top. Penalises relevant results appearing lower in the list, on a logarithmic decay.

**Why it is included:** MRR only looks at the first relevant result. NDCG evaluates the entire ranked list, so it's sensitive to cases where multiple relevant documents exist but are scattered across positions. The standard metric for evaluating ranked retrieval in IR research.

**How it is calculated:** Each retrieved chunk gets a binary relevance label (1 if filename ∈ `expected_sources`, else 0). DCG = Σ rel_i / log2(rank_i + 2). The ideal DCG (IDCG) is computed by sorting the relevance labels in descending order. NDCG = DCG / IDCG. Returns 1.0 if no expected sources are defined, 0.0 if results are empty.

---

## Chunk-level Retrieval Metrics

These five metrics mirror the source-level ones but apply a finer-grained notion of relevance: A retrieved chunk is relevant when any snippet from the case's `expected_chunks` appears as a substring of the chunk's `page_content`. Both the snippets and the chunk text are normalized (lowercased, whitespace collapsed) before substring comparison, making the check tolerant to formatting differences.

Why a separate axis? A chunk from the right *file* can still miss the *passage* that actually answers the question. Chunk-level scores catch retrieval pipelines that find the right documents but rank the wrong chunk within them. They also degrade more gracefully when one expected file spans many chunks, source-level metrics only know "the file was retrieved." Cases with an empty `expected_chunks` list score 1.0 for every chunk metric.

---

<a id="chunk-hitk"></a>
### Chunk Hit@k

| Field | Value |
|-------|-------|
| Range | 0.0 or 1.0 |
| Direction | higher is better |
| Toggle group | chunk |
| Stored as | `eval_cases.chunk_hit_at_k`, `eval_runs.avg_chunk_hit_at_k` |
| Computed by | `compute_chunk_hit_at_k` in `eval_metrics.py` |
| Pass condition | (not gated) |

**What it evaluates:** A binary signal of whether *any* retrieved chunk contains *any* expected snippet.

**Why it is included:** The chunk-level analogue of source-level Hit@k. Confirms the retriever surfaced at least one chunk containing answer-bearing text, not just a chunk from the right file. Catches the case where the right document was retrieved but the wrong passage was selected within it.

**How it is calculated:** `1.0 if any(snippet in chunk.page_content for snippet in expected_chunks for chunk in results) else 0.0`. Both sides are normalized before comparison. Returns 1.0 if `expected_chunks` is empty, 0.0 if results are empty.

---

<a id="chunk-mrr"></a>
### Chunk MRR

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | chunk |
| Stored as | `eval_cases.chunk_mrr`, `eval_runs.avg_chunk_mrr` |
| Computed by | `compute_chunk_mrr` in `eval_metrics.py` |
| Pass condition | (not gated) |

**What it evaluates:** The reciprocal rank of the first retrieved chunk that contains any expected snippet. How early in the ranked list the right *passage* (not just the right file) appears.

**Why it is included:** Source-level MRR can show a "perfect" 1.0 when the top chunk is from the right file but doesn't contain the answer text. Chunk MRR is the stricter version that demands the answer-bearing passage actually surface near the top.

**How it is calculated:** Iterate retrieved chunks in rank order; return `1 / rank` (1-indexed) for the first chunk whose normalized `page_content` contains any normalized expected snippet. Returns 0.0 if no chunk matches, 1.0 if `expected_chunks` is empty.

---

<a id="chunk-precisionk"></a>
### Chunk Precision@k

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | chunk |
| Stored as | `eval_cases.chunk_precision_at_k`, `eval_runs.avg_chunk_precision_at_k` |
| Computed by | `compute_chunk_precision_at_k` in `eval_metrics.py` |
| Pass condition | (not gated) |

**What it evaluates:** The fraction of retrieved chunks that contain at least one expected snippet. How much of the context window is being spent on actually useful passages, not just chunks from useful files.

**Why it is included:** A retriever can hit perfect source-level precision while flooding the context with non-answer-bearing chunks from the right files. This metric exposes that failure mode.

**How it is calculated:** `(chunks containing any snippet) / (total retrieved chunks)`. Returns 1.0 if `expected_chunks` is empty, 0.0 if results are empty.

---

<a id="chunk-recallk"></a>
### Chunk Recall@k

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | chunk |
| Stored as | `eval_cases.chunk_recall_at_k`, `eval_runs.avg_chunk_recall_at_k` |
| Computed by | `compute_chunk_recall_at_k` in `eval_metrics.py` |
| Pass condition | (not gated) |

**What it evaluates:** The fraction of expected snippets that appear in at least one retrieved chunk. Coverage of the known-good passages.

**Why it is included:** Unlike source-level recall, this tracks distinct *passages* rather than distinct files, so two expected passages from the same file each contribute one unit of recall. This catches the case where the retriever consistently grabs only one expected passage per document and misses the others.

**How it is calculated:** `|{snippet : snippet ∈ some retrieved chunk}| / |expected_chunks|`. Returns 1.0 if `expected_chunks` is empty, 0.0 if results are empty.

---

<a id="chunk-ndcgk"></a>
### Chunk NDCG@k

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | chunk |
| Stored as | `eval_cases.chunk_ndcg_at_k`, `eval_runs.avg_chunk_ndcg_at_k` |
| Computed by | `compute_chunk_ndcg_at_k` in `eval_metrics.py` |
| Pass condition | (not gated) |

**What it evaluates:** Ranking quality when each retrieved chunk is labelled 1 if it contains any expected snippet, else 0. The chunk-level analogue of source-level NDCG.

**Why it is included:** Combines the snippet-aware relevance signal with NDCG's logarithmic position decay. Penalises pipelines that surface answer-bearing chunks but rank them below chunks that just happen to share a filename.

**How it is calculated:** Build a binary relevance list by checking each retrieved chunk for any snippet match. DCG = Σ rel_i / log2(rank_i + 2); IDCG sorts the relevance list in descending order; NDCG = DCG / IDCG. Returns 1.0 if `expected_chunks` is empty, 0.0 if results are empty.

---

## Deterministic — Other Metrics

These metrics don't fit the source/chunk retrieval axis. Some inspect metadata or backend bookkeeping on the retrieval side; others inspect the agent's answer text against keyword lists. All run regardless of toggle settings.

---

<a id="metadata-match-ratio"></a>
### Metadata Match Ratio

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | always-on (retrieval group) |
| Stored as | `eval_cases.metadata_match_ratio`, `eval_runs.avg_metadata_match_ratio` |
| Computed by | `compute_metadata_match_ratio` in `eval_metrics.py` |
| Pass condition | `value >= METADATA_MATCH_THRESHOLD` (configurable in `eval_config.py`); one half of the retrieval gate |

**What it evaluates:** The fraction of retrieved results that satisfy all metadata filters defined on the eval case (e.g. `category=policy`).

**Why it is included:** When a case specifies metadata filters, those filters are a hard constraint on the search backend. Results that violate the filter mean the filtering logic is broken or being ignored, a functional regression that doesn't show up in any pure-content metric.

**How it is calculated:** For each retrieved result, check whether all key-value pairs in `case.metadata_filters` match the result's metadata (string comparison). Score = matching results / total results. Returns 1.0 if no metadata filters are defined, 0.0 if results are empty.

---

<a id="backend-distribution"></a>
### Backend Distribution

| Field | Value |
|-------|-------|
| Range | dict of `{backend: count}` (each count integer ≥ 0) |
| Direction | informational (no fixed direction) |
| Toggle group | always-on (retrieval group) |
| Stored as | exploded into `eval_cases.backend_fts`, `backend_vector`, `backend_hybrid`, `backend_other` |
| Computed by | `compute_backend_distribution` in `eval_metrics.py` |
| Pass condition | (not gated; diagnostic) |

**What it evaluates:** How retrieved results are split across the search backends (`fts`, `vector` and its variants, `hybrid`). Not a score but a diagnostic distribution.

**Why it is included:** The hybrid search agent fuses full-text and vector search. This distribution tells you whether both backends are actively contributing to the result set. If every result comes from one backend, the fusion mechanism may not be working as intended (or one backend is silently failing).

**How it is calculated:** Iterates over retrieval results and groups them by their `backend` field. The composite metric is exploded at storage time into four integer columns: `backend_fts` (count where backend == "fts"), `backend_vector` (sum of "vector", "vector_similarity", "vector_mmr"), `backend_hybrid` (count where backend == "hybrid"), and `backend_other` (sum of any unrecognized backend label, a catch-all so a future backend can never silently disappear from the ledger).

---

<a id="required-keyword-hit-rate"></a>
### Required Keyword Hit Rate

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | always-on (keyword composite) |
| Stored as | `eval_cases.required_keyword_hit_rate` (sub-field of the `keyword_checks` composite) |
| Computed by | `compute_keyword_checks` in `eval_metrics.py` |
| Pass condition | `value >= REQUIRED_KEYWORD_THRESHOLD` (configurable in `eval_config.py`); one half of the keyword gate |

**What it evaluates:** The fraction of the case's `required_keywords` that appear in the agent's answer.

**Why it is included:** Some questions demand specific terms in the answer (e.g. a regulation number, a product name). This metric enforces that the answer contains the expected key terms. It catches the case where every judge metric passes but the answer paraphrases away from a contractually required term.

**How it is calculated:** Each keyword and the answer are normalized (lowercased, accents stripped, punctuation removed) before comparison. Score = (keywords found in answer) / (total required keywords). Returns 1.0 if no required keywords are defined.

---

<a id="disallowed-keyword-hits"></a>
### Disallowed Keyword Hits

| Field | Value |
|-------|-------|
| Range | integer ≥ 0 |
| Direction | lower is better (target: 0) |
| Toggle group | always-on (keyword composite) |
| Stored as | `eval_cases.disallowed_keyword_hits` (sub-field of the `keyword_checks` composite) |
| Computed by | `compute_keyword_checks` in `eval_metrics.py` |
| Pass condition | `value == 0` (binary by design; one half of the keyword gate) |

**What it evaluates:** The count of disallowed keywords that appear in the agent's answer.

**Why it is included:** Some answers should avoid certain terms (e.g. a refusal phrase, a deprecated product name, a competitor mention). Any non-zero count is a failure signal. Binary by design so even one hit fails the gate.

**How it is calculated:** Same normalization as required keywords. The metric is the count of distinct disallowed keywords that match as substrings in the normalized answer. Returns 0 if no disallowed keywords are defined.

---

## Latency Metrics

These metrics are always-on wall-clock timings captured by `evaluate_case` during each case run. They have no pass thresholds, they exist to surface performance regressions alongside quality metrics.

---

<a id="latency"></a>
### Latency

| Field | Value |
|-------|-------|
| Range | float seconds ≥ 0 |
| Direction | lower is better |
| Toggle group | always-on (latency group) |
| Stored as | `eval_cases.latency_seconds`, `eval_runs.avg_latency_seconds` |
| Computed by | `time.perf_counter()` around the agent invocation in `evaluate_case` |
| Pass condition | (not gated) |

**What it evaluates:** Total wall-clock time, in seconds, for the agent to produce an answer for a single case. Covers the full graph invocation from input to final answer.

**Why it is included:** End-to-end latency is what a user experiences. Tracking it per case lets you spot slow queries (e.g. broad questions that over-retrieve) and catch run-to-run regressions after prompt or model changes.

**How it is calculated:** `round(time.perf_counter() - started, _DECIMALS["latency_seconds"])` around the `agent.ainvoke` call. Stored precision is derived from the metric's registry `decimals` so the storage format updates in lockstep with the registry.

---

<a id="retrieval-latency"></a>
### Retrieval Latency

| Field | Value |
|-------|-------|
| Range | float seconds ≥ 0 |
| Direction | lower is better |
| Toggle group | always-on (latency group) |
| Stored as | `eval_cases.retrieval_latency_seconds`, `eval_runs.avg_retrieval_latency_seconds` |
| Computed by | `time.perf_counter()` around the direct retriever call in `evaluate_case` |
| Pass condition | (not gated) |

**What it evaluates:** Time spent inside the retriever call, the hybrid search step that fetches candidate chunks from the index, called directly (bypassing the agent).

**Why it is included:** Isolating retrieval cost from LLM cost makes it obvious which side of the pipeline is slow. A spike here usually points to index issues, large `k`, or slow backend fusion, rather than a slow generation step.

**How it is calculated:** `time.perf_counter()` around the retriever's `search` call, rounded to the registry-declared precision. Best-effort estimate, since the agent's internal retrieval may have slightly different overhead than this direct call.

---

<a id="llm-latency"></a>
### LLM Latency

| Field | Value |
|-------|-------|
| Range | float seconds ≥ 0 (clamped to ≥ 0) |
| Direction | lower is better |
| Toggle group | always-on (latency group) |
| Stored as | `eval_cases.llm_latency_seconds`, `eval_runs.avg_llm_latency_seconds` |
| Computed by | `latency_seconds − retrieval_latency_seconds` in `evaluate_case` |
| Pass condition | (not gated) |

**What it evaluates:** Estimated time the generation step took, derived as `latency_seconds − retrieval_latency_seconds`.

**Why it is included:** Exposes the generation side of the budget. When total latency grows, this split tells you whether retrieval or the LLM is responsible, which determines whether to tune the index, change `k`, or switch generation models.

**How it is calculated:** Subtraction of the two timings, clamped to `>= 0`. Slightly noisy because retrieval is timed on a separate call rather than instrumented inside the agent, so sub-10ms differences can be lost.

---

## Summary Metrics

These are run-level aggregates surfaced at the top of each report and in the `eval_runs` row. They let you compare runs over time without drilling into per-case details.

---

<a id="pass-rate"></a>
### Pass Rate

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | always (run-level summary) |
| Stored as | `eval_runs.pass_rate` |
| Computed by | `build_summary` in `eval_report_manager.py` |
| Pass condition | (not a gate; this is the result of all gates) |

**What it evaluates:** The fraction of cases in a run whose final `status` is `"PASS"` (i.e., every active gate succeeded). Cases that failed any gate end with status `"REVIEW"`.

**Why it is included:** The headline number for the question "is this run good?". Trend over time tells you whether agent quality is regressing or improving across commits and configuration changes.

**How it is calculated:** `pass_count / max(case_count, 1)`, rounded to the registry-declared precision. `pass_count` is the number of results with `status == "PASS"`. See [Verdict Logic](#verdict-logic) for how each case's status is determined.

---

<a id="avg-judge-score"></a>
### Avg Judge Score

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | always (run-level summary, only meaningful when judge group is enabled) |
| Stored as | `eval_cases.avg_judge_score` (per case), `eval_runs.avg_judge_run_score` (run level) |
| Computed by | `evaluate_case` in `eval_engine.py` (per case) and `build_summary` in `eval_report_manager.py` (run level) |
| Pass condition | (not gated) |

**What it evaluates:** Per case: the mean of all DeepEval LLM-judged scores for that case (on the 0–1 scale the judges already use). Run level (`avg_judge_run_score`): The mean of those per-case values across the run.

**Why it is included:** Provides a single judge-quality number per case and per run, useful as a sanity check alongside pass rate. A high pass rate with a low judge score might mean gates are too lenient, while a low pass rate with a high judge score might mean format/keyword/retrieval is breaking even though content quality is fine. Stored as `None` when the `judge` group is disabled (no scores to average).

**How it is calculated:** Per case: Collect each non-None `score` from the 7 DeepEval metrics into a list, then `safe_mean(scores, decimals)`, `None` when no judge produced a score. Run level: `safe_mean` of those per-case values, rounded to the registry-declared precision (NULL cases excluded).

---

<a id="verdict-logic"></a>
## PASS/REVIEW Gates Verdict Logic

The final `status` field (`"PASS"` or `"REVIEW"`) is computed by `_compute_case_status()` in [`eval_engine.py`](eval_engine.py). It applies three independent gates:

| Gate | Condition |
|------|-----------|
| **metrics_ok** | `faithfulness >= JUDGE_THRESHOLD` AND `answer_relevancy >= JUDGE_THRESHOLD` |
| **retrieval_ok** | `hit_at_k == 1.0` AND `metadata_match_ratio >= METADATA_MATCH_THRESHOLD` |
| **keywords_ok** | `required_keyword_hit_rate >= REQUIRED_KEYWORD_THRESHOLD` AND `disallowed_keyword_hits == 0` |

All three gates must pass for `status == "PASS"`. Any gate failing or any runtime error results in `status == "REVIEW"`. The `errors` field on the result records every failure that fired (not just the first), so you can triage from the dashboard without re-running the case.

### Gate thresholds

Threshold values are defined in [`eval_config.py`](eval_config.py). That file is the single source of truth so this document doesn't drift when you tune them. `JUDGE_THRESHOLD` sets both the judge gate and each DeepEval metric's pass/fail line (same value). `METADATA_MATCH_THRESHOLD` and `REQUIRED_KEYWORD_THRESHOLD` are dedicated knobs for the retrieval and keyword gates. Binary-by-design conditions (`hit_at_k == 1.0` and `disallowed_keyword_hits == 0`) are intentionally not configurable.

| Threshold | Applies to |
|-----------|-----------|
| `JUDGE_THRESHOLD` | faithfulness, answer_relevancy (judge gate) |
| `METADATA_MATCH_THRESHOLD` | metadata_match_ratio (retrieval gate) |
| `REQUIRED_KEYWORD_THRESHOLD` | required_keyword_hit_rate (keyword gate) |

**Configure once per project.** Thresholds should not change between runs once you start collecting trend data. Lowering a threshold after a regression will "fix" the pass rate without fixing the underlying issue. The active thresholds are persisted with each run (`eval_runs.gate_thresholds` as JSON) and the dashboard displays a warning banner if they change across runs.

### Gate behaviour when metric groups are disabled

Each gate's sub-conditions depend on metric groups that can be toggled off via `ENABLED_METRIC_GROUPS` in `eval_config.py`. The gate logic handles disabled groups as follows:

- **metrics_ok** — if `"judge"` is disabled, no DeepEval scores exist and the gate auto-passes (no judge signal to fail on).
- **retrieval_ok** — if `"source"` is disabled, the `hit_at_k` sub-condition is skipped and the gate reduces to `metadata_match_ratio >= METADATA_MATCH_THRESHOLD`. The `"chunk"` toggle does not participate in the gate.
- **keywords_ok** — always evaluated, keyword checks are always-on regardless of toggles.
