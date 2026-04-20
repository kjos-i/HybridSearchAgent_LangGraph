# Evaluation Metrics Reference

This document describes every metric produced by the evaluation harness. Metrics are grouped into two categories: **LLM-judged metrics** (scored by a judge model via DeepEval) and **deterministic metrics** (computed directly from retrieval results and the agent's answer).

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
| [Average Judge Score](#average-judge-score) | Mean of all LLM-judged scores, expressed as 0–100 (per case and run). |

---

## LLM-Judged Metrics (DeepEval)

These metrics use a judge LLM (configured via `JUDGE_MODEL` in `eval_config.py`, default `gpt-5.4-mini`) to score each test case. All are imported from the `deepeval` library and take two common arguments:

- `threshold` (float) — minimum score to pass (configured globally in `eval_config.py`, default `0.5`)
- `model` (str) — the judge model identifier

Each metric returns `{ score, reason, passed, threshold }`.

---

### Answer Relevancy

| Field | Value |
|-------|-------|
| **Key in report** | `answer_relevancy` |
| **Import** | `from deepeval.metrics import AnswerRelevancyMetric` |
| **Arguments** | `threshold`, `model` |

**What it evaluates:** Whether the agent's answer is relevant to the user's question. It checks that the response actually addresses what was asked rather than providing tangential or off-topic information.

**Why it is included:** A RAG system can retrieve perfect context but still produce an answer that drifts from the question. This metric catches cases where the LLM ignores the question or over-generalises. It is also one of the two gate metrics for the final PASS/REVIEW verdict.

**How it is calculated:** The judge LLM analyses the input question and the actual output. It generates synthetic questions that the actual output could answer, then measures the overlap between those synthetic questions and the original input. A high overlap means the answer is on-topic.

---

### Faithfulness

| Field | Value |
|-------|-------|
| **Key in report** | `faithfulness` |
| **Import** | `from deepeval.metrics import FaithfulnessMetric` |
| **Arguments** | `threshold`, `model` |

**What it evaluates:** Whether every claim in the agent's answer is supported by the retrieved context. A faithful answer makes no statements that go beyond what the context provides.

**Why it is included:** Faithfulness is the core safeguard against hallucination in RAG. If the agent invents facts not present in the retrieved documents, the answer is unreliable — even if it sounds correct. This metric is one of the two gate metrics for the final PASS/REVIEW verdict.

**How it is calculated:** The judge LLM extracts individual claims from the actual output and checks each one against the retrieval context. The score is the fraction of claims that are fully supported. A score of 1.0 means every claim traces back to a retrieved chunk.

---

### Contextual Precision

| Field | Value |
|-------|-------|
| **Key in report** | `contextual_precision` |
| **Import** | `from deepeval.metrics import ContextualPrecisionMetric` |
| **Arguments** | `threshold`, `model` |

**What it evaluates:** Whether the relevant chunks in the retrieval context are ranked higher than irrelevant ones. It measures ranking quality, not just presence.

**Why it is included:** In a hybrid search system, retrieval order matters. If relevant documents are buried below noise, the LLM may miss or deprioritise them. This metric surfaces ranking problems in the retrieval pipeline.

**How it is calculated:** The judge LLM classifies each node in the retrieval context as relevant or irrelevant based on the expected output. It then computes a weighted precision score that rewards relevant items appearing earlier in the list. Higher scores mean better ranking.

---

### Contextual Recall

| Field | Value |
|-------|-------|
| **Key in report** | `contextual_recall` |
| **Import** | `from deepeval.metrics import ContextualRecallMetric` |
| **Arguments** | `threshold`, `model` |

**What it evaluates:** Whether the retrieval context contains all the information needed to produce the expected output. It checks for completeness of the retrieved evidence.

**Why it is included:** A retrieval system might return chunks that are individually relevant but collectively miss key facts needed for a complete answer. This metric identifies gaps in retrieval coverage.

**How it is calculated:** The judge LLM breaks the expected output into individual sentences or claims, then checks whether each one can be attributed to at least one node in the retrieval context. The score is the fraction of expected claims that are supported.

---

### Contextual Relevancy

| Field | Value |
|-------|-------|
| **Key in report** | `contextual_relevancy` |
| **Import** | `from deepeval.metrics import ContextualRelevancyMetric` |
| **Arguments** | `threshold`, `model` |

**What it evaluates:** Whether each chunk in the retrieval context is relevant to the input question. Unlike contextual precision (which focuses on ranking), this measures overall noise level.

**Why it is included:** Retrieving too many irrelevant chunks dilutes the LLM's attention and can lead to worse answers or higher latency. This metric quantifies how much noise the retrieval pipeline is introducing.

**How it is calculated:** The judge LLM evaluates each retrieval context node against the input question and determines whether it is relevant. The score is the fraction of retrieved nodes that are relevant. A score of 1.0 means every retrieved chunk was useful.

---

### Hallucination

| Field | Value |
|-------|-------|
| **Key in report** | `hallucination` |
| **Import** | `from deepeval.metrics import HallucinationMetric` |
| **Arguments** | `threshold`, `model` |

**What it evaluates:** Whether the agent's answer contradicts the provided context. While faithfulness checks for unsupported claims, hallucination specifically detects factual contradictions.

**Why it is included:** A hallucinated answer is worse than an incomplete one — it actively misleads the user. This metric provides an additional safety net beyond faithfulness by focusing on contradictions rather than just missing support.

**How it is calculated:** The judge LLM compares the actual output against the context and determines whether any statements in the output contradict the context. The score represents the degree to which the output is free of contradictions (higher is better, i.e. less hallucination).

---

### Grounded Correctness (GEval)

| Field | Value |
|-------|-------|
| **Key in report** | `correctness_g_eval` |
| **Import** | `from deepeval.metrics import GEval` |
| **Arguments** | `name` ("Grounded Correctness"), `criteria` (custom string), `evaluation_params` ([INPUT, ACTUAL_OUTPUT, EXPECTED_OUTPUT]), `threshold`, `model` |

**What it evaluates:** Whether the agent's answer correctly addresses the user's question, covers the important facts from the expected output, and avoids unsupported claims. This is a custom criteria-based evaluation.

**Why it is included:** The other metrics evaluate individual dimensions (relevancy, faithfulness, context quality), but none directly ask "is this answer correct?" GEval provides a holistic correctness check by comparing the actual answer against the expected answer points.

**How it is calculated:** GEval uses a chain-of-thought approach. The judge LLM receives the custom criteria string along with the input, actual output, and expected output. It generates evaluation steps, scores each step, and combines them into a final score (0-1). The criteria used here are: *"Determine whether the actual output correctly answers the user's request, covers the important facts from the expected output, and avoids unsupported claims."*

---

## Deterministic Retrieval Metrics

These metrics are computed directly from the retrieval results without an LLM judge. They are defined in `eval_metrics.py` and operate on the **direct retrieval** results (not the agent's context).

Retrieval quality is measured on two axes:

- **Source-level** — relevance is decided by filename match against `expected_sources`. Any chunk whose source filename is in the expected list counts as relevant, regardless of what text the chunk contains.
- **Chunk-level** — relevance is decided by substring match against `expected_chunks` (a list of short representative text snippets on the case). A retrieved chunk is relevant when any expected snippet appears inside the chunk's `page_content` after normalization. Snippets are preferred over chunk IDs because chunk IDs change whenever the chunking strategy is tuned, while short representative text remains stable across re-chunking.

Both axes produce the same five metrics (hit@k, MRR, precision@k, recall@k, NDCG@k); chunk-level variants are prefixed with `chunk_`. Cases with no `expected_sources` or no `expected_chunks` score 1.0 for their axis (nothing to check) so they do not poison run-level averages.

---

### Hit@k

| Field | Value |
|-------|-------|
| **Key in report** | `hit_at_k` |
| **Defined in** | `eval_metrics.compute_hit_at_k()` |

**What it evaluates:** Relevance is decided at the source (filename) level, not the chunk level. The metric is binary: 1.0 if *at least one* expected source appears anywhere in the top-k retrieved results, 0.0 otherwise. It is the standard IR Hit@k (also called Success@k) metric.

**Why it is included:** The most basic retrieval health check — did the system find *anything* right? If no expected source appears at all, no amount of good ranking or generation can save the answer. This metric is one of the two retrieval gate checks for the PASS/REVIEW verdict (must equal 1.0). It complements [recall@k](#recallk), which reports *how much* of the expected set was found.

**How it is calculated:** `1.0 if (expected_sources ∩ retrieved_sources) else 0.0`. Source names are compared by filename (case-insensitive). Returns 1.0 if no expected sources are defined (nothing to check).

- Both the expected and retrieved sides are reduced to **sets of filenames** (`Path(source).name.lower()`), so duplicates collapse. If 5 retrieved chunks all come from `A.pdf`, that counts as **one** hit, not five.
- `expected_sources` on an `EvalCase` is a list of **filenames**, not specific chunks. Two different expected passages from the same PDF collapse into one entry.
- Because the metric is binary, retrieving *more* expected files does not raise the score further — one is enough. Use [recall@k](#recallk) if you need coverage.

**Worked example:** expected sources = `{A.pdf, B.pdf, C.pdf}` (3 files).

| Retrieved chunks | Unique retrieved filenames | Intersection | Score |
|------------------|----------------------------|--------------|-------|
| 3 chunks from `A.pdf`, `B.pdf`, `D.pdf` | `{A, B, D}` | `{A, B}` | 1.0 |
| 2 chunks from `A.pdf`, 1 from `B.pdf` | `{A, B}` | `{A, B}` | 1.0 |
| 10 chunks all from `A.pdf` | `{A}` | `{A}` | 1.0 |
| 3 chunks from `D.pdf`, `E.pdf`, `F.pdf` | `{D, E, F}` | `{}` | 0.0 |
| 3 chunks from `A.pdf`, `B.pdf`, `C.pdf` | `{A, B, C}` | `{A, B, C}` | 1.0 |

---

### Mean Reciprocal Rank (MRR)

| Field | Value |
|-------|-------|
| **Key in report** | `mrr` |
| **Defined in** | `eval_metrics.compute_mrr()` |

**What it evaluates:** Relevance is decided at the source (filename) level, not the chunk level. The metric measures how early the first relevant result appears in the ranked retrieval list. Scores 1.0 if the first result is relevant, 0.5 if the second is, 0.33 if the third is, and so on.

**Why it is included:** In RAG, the top-ranked result often has the most influence on the LLM's answer. MRR tells you whether the retrieval pipeline is placing the most important document first or burying it.

**How relevance is decided:** There is no LLM judge involved. Each eval case declares a list of `expected_sources` (filenames that should be retrieved for that question). Both the expected side and each retrieved item are normalized via `Path(source).name.lower()`, then compared as sets. The first retrieved result whose normalized filename is in the expected set is "the first relevant result." A retrieved chunk counts as relevant if its source filename is listed in `expected_sources` — any chunk from an expected file qualifies, even one that does not contain the answer passage.

**How it is calculated:** `1 / rank` where rank is the position (1-indexed) of the first retrieved result whose source filename matches any expected source. Returns 0.0 if no expected source is found, and 1.0 if no expected sources are defined.

Each eval case declares a list of `expected_sources` (filenames that should be retrieved for that question). Both the expected side and each retrieved item are normalized via `Path(source).name.lower()`, then compared as sets. The first retrieved result whose normalized filename is in the expected set is "the first relevant result."

---

### Precision@k

| Field | Value |
|-------|-------|
| **Key in report** | `precision_at_k` |
| **Defined in** | `eval_metrics.compute_precision_at_k()` |

**What it evaluates:** Relevance is decided at the source (filename) level, not the chunk level. The metric measures the fraction of all retrieved results (top k) that come from an expected source, i.e. how much of the retrieval budget is spent on relevant documents.

**Why it is included:** A low precision means the retrieval pipeline is returning a lot of noise alongside the relevant documents. This wastes context window space and can confuse the LLM.

**How it is calculated:** `(number of retrieved results from expected sources) / (total retrieved results)`. A retrieved chunk counts toward precision if its source filename (lowercased) is in `expected_sources`, regardless of whether the chunk's text actually contains the answer. Source matching is by filename (case-insensitive). Returns 1.0 if no expected sources are defined or if results are empty.

---

### Recall@k

| Field | Value |
|-------|-------|
| **Key in report** | `recall_at_k` |
| **Defined in** | `eval_metrics.compute_recall_at_k()` |

**What it evaluates:** Relevance is decided at the source (filename) level, not the chunk level. The metric measures the fraction of expected source documents that appear in the top-k retrieved results — i.e. retrieval completeness.

**Why it is included:** Where [hit@k](#hitk) tells you *whether* anything relevant was found (binary health check), recall@k tells you *how much* of the expected set was covered. It is reported alongside precision@k for a complete precision-recall picture — together they reveal the trade-off between retrieving broadly and retrieving precisely.

**How it is calculated:** `|expected_sources ∩ retrieved_sources| / |expected_sources|`. Recall measures whether each expected *file* was retrieved, not whether specific expected passages or chunks were. Returns 1.0 if no expected sources are defined, 0.0 if results are empty.

---

### NDCG@k (Normalized Discounted Cumulative Gain)

| Field | Value |
|-------|-------|
| **Key in report** | `ndcg_at_k` |
| **Defined in** | `eval_metrics.compute_ndcg_at_k()` |

**What it evaluates:** Relevance is decided at the source (filename) level, not the chunk level. The metric measures the quality of the ranking compared to the ideal ranking where all relevant results are at the top. It is a ranking-aware metric that penalises relevant results appearing lower in the list.

**Why it is included:** MRR only looks at the first relevant result. NDCG evaluates the entire ranked list, making it sensitive to cases where multiple relevant documents exist but are scattered across different positions. It is the standard metric for evaluating ranked retrieval in information retrieval research.

**How it is calculated:** Each retrieved chunk gets a binary relevance label of 1 if its filename is in `expected_sources`, else 0 — the chunk's actual content is not inspected. DCG = sum of (relevance_i / log2(rank_i + 2)). The ideal DCG (IDCG) is computed by sorting all relevance labels in descending order. NDCG = DCG / IDCG. Returns 1.0 if no expected sources are defined, 0.0 if results are empty.

---

## Chunk-level Retrieval Metrics

These five metrics mirror the source-level ones but apply a finer-grained notion of relevance: a retrieved chunk is relevant when any snippet from the case's `expected_chunks` appears as a substring of the chunk's `page_content`. Both the snippets and the chunk text are normalized (lowercased, whitespace collapsed) before substring comparison, making the check tolerant to formatting differences. Cases with an empty `expected_chunks` list score 1.0 for every chunk metric.

Why a separate axis? A chunk from the right *file* can still miss the *passage* that actually answers the question. Chunk-level scores catch retrieval pipelines that find the right documents but rank the wrong chunk within them. They also degrade more gracefully when one expected file spans many chunks — the source-level metrics only know "the file was retrieved."

---

### Chunk Hit@k

| Field | Value |
|-------|-------|
| **Key in report** | `chunk_hit_at_k` |
| **Defined in** | `eval_metrics.compute_chunk_hit_at_k()` |

**What it evaluates:** Binary signal of whether *any* retrieved chunk contains *any* expected snippet.

**How it is calculated:** `1.0 if any(snippet in chunk.page_content for snippet in expected_chunks for chunk in results) else 0.0`. Both sides are normalized before comparison. Returns 1.0 if `expected_chunks` is empty, 0.0 if results are empty.

---

### Chunk MRR

| Field | Value |
|-------|-------|
| **Key in report** | `chunk_mrr` |
| **Defined in** | `eval_metrics.compute_chunk_mrr()` |

**What it evaluates:** The reciprocal rank of the first retrieved chunk that contains any expected snippet. Measures how early in the ranked list the right *passage* appears, not just the right file.

**How it is calculated:** Iterate retrieved chunks in rank order; return `1 / rank` (1-indexed) for the first chunk whose normalized `page_content` contains any normalized expected snippet. Returns 0.0 if no chunk matches, 1.0 if `expected_chunks` is empty.

---

### Chunk Precision@k

| Field | Value |
|-------|-------|
| **Key in report** | `chunk_precision_at_k` |
| **Defined in** | `eval_metrics.compute_chunk_precision_at_k()` |

**What it evaluates:** The fraction of retrieved chunks that contain at least one expected snippet — i.e. how much of the context window is being spent on actually useful passages.

**How it is calculated:** `(number of chunks matching any snippet) / (total retrieved chunks)`. Returns 1.0 if `expected_chunks` is empty, 0.0 if results are empty.

---

### Chunk Recall@k

| Field | Value |
|-------|-------|
| **Key in report** | `chunk_recall_at_k` |
| **Defined in** | `eval_metrics.compute_chunk_recall_at_k()` |

**What it evaluates:** The fraction of expected snippets that appear in at least one retrieved chunk — i.e. coverage of the known-good passages.

**How it is calculated:** `|{snippet : snippet ∈ some retrieved chunk}| / |expected_chunks|`. Unlike source-level recall, this tracks distinct *passages* rather than distinct files — so two expected passages from the same file still contribute two units of recall. Returns 1.0 if `expected_chunks` is empty, 0.0 if results are empty.

---

### Chunk NDCG@k

| Field | Value |
|-------|-------|
| **Key in report** | `chunk_ndcg_at_k` |
| **Defined in** | `eval_metrics.compute_chunk_ndcg_at_k()` |

**What it evaluates:** Ranking quality when each retrieved chunk is labelled 1 if it contains any expected snippet, else 0.

**How it is calculated:** Build a binary relevance list by checking each retrieved chunk for any snippet match. Apply the standard DCG/IDCG formula: DCG = sum(rel_i / log2(rank_i + 2)); IDCG sorts the relevance list in descending order; NDCG = DCG / IDCG. Returns 1.0 if `expected_chunks` is empty, 0.0 if results are empty.

---

## Deterministic — Other Metrics

These metrics don't fit the source/chunk retrieval axis. Some inspect metadata or backend bookkeeping on the retrieval side; others inspect the agent's answer text against keyword lists. All are defined in `eval_metrics.py`.

---

### Metadata Match Ratio

| Field | Value |
|-------|-------|
| **Key in report** | `metadata_match_ratio` |
| **Defined in** | `eval_metrics.compute_metadata_match_ratio()` |

**What it evaluates:** The fraction of retrieved results that satisfy all metadata filters defined on the eval case (e.g. `category=policy`).

**Why it is included:** When a case specifies metadata filters, those filters represent a hard constraint. If the retrieval pipeline returns results that violate the filter, it means the filtering logic is broken or the filter is being ignored. This metric is the second retrieval gate check for the PASS/REVIEW verdict (threshold >= 0.8).

**How it is calculated:** For each retrieved result, check whether all key-value pairs in `case.metadata_filters` match the result's metadata (string comparison). The score is `matching_results / total_results`. Returns 1.0 if no metadata filters are defined, 0.0 if results are empty.

---

### Backend Distribution

| Field | Value |
|-------|-------|
| **Key in report** | `backend_distribution` |
| **Defined in** | `eval_metrics.compute_backend_distribution()` |

**What it evaluates:** Counts retrieved results by search backend (e.g. `fts`, `vector`). Not a score — a diagnostic distribution.

**Why it is included:** The hybrid search agent fuses full-text search and vector search. This distribution tells you whether both backends are actively contributing to the final result set. If all results come from one backend, the fusion mechanism may not be working as intended.

**How it is calculated:** Iterates over retrieval results and groups them by the `backend` field. Returns a dict like `{"fts": 3, "vector": 2}`.

---

### Required Keyword Hit Rate

| Field | Value |
|-------|-------|
| **Key in report** | `keyword_checks.required_keyword_hit_rate` |
| **Defined in** | `eval_metrics.compute_keyword_checks()` |

**What it evaluates:** The fraction of required keywords (defined in the eval case) that appear in the agent's answer.

**Why it is included:** Some questions demand specific terms in the answer (e.g. a regulation number, a product name). This metric enforces that the answer contains the expected key terms. It is one of the keyword gate checks for the PASS/REVIEW verdict (threshold >= 0.5).

**How it is calculated:** Each keyword and the answer are normalized (lowercased, accents stripped, punctuation removed). The score is `(keywords found in answer) / (total required keywords)`. Returns 1.0 if no required keywords are defined.

---

### Disallowed Keyword Hits

| Field | Value |
|-------|-------|
| **Key in report** | `keyword_checks.disallowed_keyword_hits` |
| **Defined in** | `eval_metrics.compute_keyword_checks()` |

**What it evaluates:** The count of disallowed keywords that appear in the agent's answer.

**Why it is included:** Some answers should avoid certain terms (e.g. mentioning a competitor, using a deprecated term). Any non-zero count is a failure signal. It is the second keyword gate check for the PASS/REVIEW verdict (must equal 0).

**How it is calculated:** Same normalization as required keywords. Counts how many disallowed keywords appear as substrings in the normalized answer. Returns 0 if no disallowed keywords are defined.

---

## Latency Metrics

These metrics are always-on timings captured by `eval_engine.evaluate_case()` during each case run. They are wall-clock measurements, not judged scores, and they have no pass threshold — they exist to surface performance regressions alongside quality metrics.

---

### Latency

| Field | Value |
|-------|-------|
| **Key in report** | `latency_seconds` |
| **Defined in** | `eval_engine.evaluate_case()` |
| **Registry group** | `latency` |

**What it evaluates:** Total wall-clock time, in seconds, for the agent to produce an answer for a single case. Covers the full graph invocation from input to final answer.

**Why it is included:** End-to-end latency is what a user experiences. Tracking it per case lets you spot slow queries (e.g. broad questions that over-retrieve) and catch run-to-run regressions after prompt or model changes. The run-level `avg_latency_seconds` in the ledger is the mean across cases.

**How it is calculated:** Measured around the agent invocation: `time.perf_counter()` is sampled before and after the graph run, and the delta is recorded. No threshold is applied.

---

### Retrieval Latency

| Field | Value |
|-------|-------|
| **Key in report** | `retrieval_latency_seconds` |
| **Defined in** | `eval_engine.evaluate_case()` |
| **Registry group** | `latency` |

**What it evaluates:** Time spent inside the retriever call — the hybrid search step that fetches candidate chunks from the index.

**Why it is included:** Isolating retrieval cost from LLM cost makes it obvious which side of the pipeline is slow. A spike here usually points to index issues, large `k`, or slow backend fusion, rather than a slow judge or generation step.

**How it is calculated:** The retriever is invoked directly (outside of the full graph) and timed with `time.perf_counter()`. This is a best-effort estimate since the graph's internal retrieval may have slightly different overhead.

---

### LLM Latency

| Field | Value |
|-------|-------|
| **Key in report** | `llm_latency_seconds` |
| **Defined in** | `eval_engine.evaluate_case()` |
| **Registry group** | `latency` |

**What it evaluates:** Estimated time the generation step took, as a derived number: `latency_seconds − retrieval_latency_seconds`.

**Why it is included:** Exposes the generation side of the budget. When total latency grows, this split tells you whether retrieval or the LLM call is responsible, which determines whether to tune the index, change `k`, or switch generation models.

**How it is calculated:** Subtraction of the two timings. Can be slightly noisy since retrieval is timed on a separate call; clamped to `>= 0`.

---

## Summary Metric

### Pass Rate

| Field | Value |
|-------|-------|
| **Key in report** | `pass_rate` (run-level only) |
| **Defined in** | `eval_report_manager.build_summary()` |
| **Registry group** | `summary` |

**What it evaluates:** The fraction of cases in a run whose final `status` is `"PASS"`. It is a run-level rollup; there is no per-case equivalent (each case has a boolean-like `status` instead).

**Why it is included:** Pass rate is the headline health number for a run. It compresses the three verdict gates (metrics, retrieval, keywords) into a single score you can track across commits or configuration changes.

**How it is calculated:** `pass_count / case_count`, where `pass_count` is the number of cases with `status == "PASS"`. Rounded to 3 decimals. See [Verdict Logic](#verdict-logic) for how each case's status is determined.

---

### Average Judge Score

| Field | Value |
|-------|-------|
| **Key in report** | `avg_judge_score` |
| **Defined in** | `eval_engine.evaluate_case()` |
| **Registry group** | `summary` |

**What it evaluates:** The mean of all DeepEval LLM-judged metric scores for a single case, expressed as a percentage (0–100). It is a *derived* metric — not independently measured — which is why it lives in its own summary category rather than under the LLM-judged or deterministic sections.

**Why it is included:** Provides a single summary number for quick comparison across cases. Useful for spotting overall quality trends without inspecting each metric individually.

**How it is calculated:** Collects all non-None scores from the 7 DeepEval metrics, computes their arithmetic mean, and multiplies by 100. Rounded to 1 decimal place. Returns `None` when the `judge` metric group is disabled (no judge scores to average). The run-level `avg_case_score` in the ledger is the mean of each case's `avg_judge_score`, skipping cases where it is `None`.

---

## Verdict Logic

The final `status` field ("PASS" or "REVIEW") is computed by `compute_case_status()` in `eval_engine.py`. It applies three independent gates:

| Gate | Condition |
|------|-----------|
| **metrics_ok** | `faithfulness >= JUDGE_THRESHOLD` AND `answer_relevancy >= JUDGE_THRESHOLD` |
| **retrieval_ok** | `hit_at_k == 1.0` AND `metadata_match_ratio >= METADATA_MATCH_THRESHOLD` |
| **keywords_ok** | `required_keyword_hit_rate >= REQUIRED_KEYWORD_THRESHOLD` AND `disallowed_keyword_hits == 0` |

All three gates must pass for `status = "PASS"`. Any gate failing or any runtime error results in `status = "REVIEW"`.

### Gate thresholds

Thresholds are defined in `eval_config.py`. `JUDGE_THRESHOLD` sets both the judge gate and the DeepEval metric pass/fail line (same value), and `METADATA_MATCH_THRESHOLD` and `REQUIRED_KEYWORD_THRESHOLD` are dedicated knobs for the retrieval and keyword gates. Binary-by-design conditions (`hit_at_k == 1.0` and `disallowed_keyword_hits == 0`) are intentionally not configurable.

**Configure once per project.** Thresholds should not change between runs once you start collecting trend data — lowering a threshold after a regression will "fix" the pass rate without fixing the underlying issue. The active thresholds are persisted with each run (`eval_runs.gate_thresholds` as JSON) and the dashboard displays a warning banner if they change across runs.

Defaults:

| Threshold | Default | Applies to |
|-----------|---------|-----------|
| `JUDGE_THRESHOLD` | `0.5` | faithfulness, answer_relevancy (judge gate) |
| `METADATA_MATCH_THRESHOLD` | `0.8` | metadata_match_ratio (retrieval gate) |
| `REQUIRED_KEYWORD_THRESHOLD` | `0.5` | required_keyword_hit_rate (keyword gate) |

### Gate behaviour when metric groups are disabled

Each gate's sub-conditions depend on metric groups that can be toggled off via `ENABLED_METRIC_GROUPS` in `eval_config.py`. The gate logic handles disabled groups as follows:

- **metrics_ok** — if `"judge"` is disabled, no DeepEval scores exist and the gate auto-passes (no judge signal to fail on).
- **retrieval_ok** — if `"source"` is disabled, the `hit_at_k` sub-condition is skipped and the gate reduces to `metadata_match_ratio >= METADATA_MATCH_THRESHOLD`. The `"chunk"` toggle does not participate in the gate.
- **keywords_ok** — always evaluated; keyword checks are always-on regardless of toggles.
