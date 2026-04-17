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

**Deterministic — Retrieval**

| Metric | Description |
|--------|-------------|
| [Source Hit Rate](#source-hit-rate) | Were the expected source documents retrieved? |
| [MRR](#mean-reciprocal-rank-mrr) | How early does the first relevant result appear? |
| [Precision@k](#precisionk) | What fraction of retrieved results are relevant? |
| [Recall@k](#recallk) | What fraction of expected sources were retrieved? |
| [NDCG@k](#ndcgk-normalized-discounted-cumulative-gain) | How good is the overall ranking quality? |
| [Metadata Match Ratio](#metadata-match-ratio) | Do retrieved results satisfy the metadata filters? |
| [Backend Distribution](#backend-distribution) | How are results split across search backends? |

**Deterministic — Answer**

| Metric | Description |
|--------|-------------|
| [Required Keyword Hit Rate](#required-keyword-hit-rate) | Does the answer contain the required key terms? |
| [Disallowed Keyword Hits](#disallowed-keyword-hits) | Does the answer avoid disallowed terms? |
| [Average Metric Score](#average-metric-score) | Mean of all LLM-judged scores (summary number). |

---

## LLM-Judged Metrics (DeepEval)

These metrics use a judge LLM (configured via `JUDGE_MODEL` in `eval_config.py`, default `gpt-4o`) to score each test case. All are imported from the `deepeval` library and take two common arguments:

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

These metrics are computed directly from the retrieval results without an LLM judge. They are defined in `eval_utils.py` and operate on the **direct retrieval** results (not the agent's context).

---

### Source Hit Rate

| Field | Value |
|-------|-------|
| **Key in report** | `source_hit_rate` |
| **Defined in** | `eval_utils.compute_source_hit_rate()` |

**What it evaluates:** The fraction of expected source documents that appear anywhere in the retrieved results, regardless of rank.

**Why it is included:** The most basic retrieval check — did the system find the right documents at all? If expected sources are missing entirely, no amount of good ranking can save the answer. This metric is one of the two retrieval gate checks for the PASS/REVIEW verdict (threshold >= 0.5).

**How it is calculated:** `|expected_sources ∩ retrieved_sources| / |expected_sources|`. Source names are compared by filename (case-insensitive). Returns 1.0 if no expected sources are defined.

---

### Mean Reciprocal Rank (MRR)

| Field | Value |
|-------|-------|
| **Key in report** | `mrr` |
| **Defined in** | `eval_utils.compute_mrr()` |

**What it evaluates:** How early the first relevant result appears in the ranked retrieval list. Scores 1.0 if the first result is relevant, 0.5 if the second is, 0.33 if the third is, and so on.

**Why it is included:** In RAG, the top-ranked result often has the most influence on the LLM's answer. MRR tells you whether the retrieval pipeline is placing the most important document first or burying it.

**How it is calculated:** `1 / rank` where rank is the position (1-indexed) of the first retrieved result whose source filename matches any expected source. Returns 0.0 if no expected source is found, and 1.0 if no expected sources are defined.

---

### Precision@k

| Field | Value |
|-------|-------|
| **Key in report** | `precision_at_k` |
| **Defined in** | `eval_utils.compute_precision_at_k()` |

**What it evaluates:** The fraction of all retrieved results (top k) that come from an expected source. Measures how much of the retrieval budget is spent on relevant documents.

**Why it is included:** A low precision means the retrieval pipeline is returning a lot of noise alongside the relevant documents. This wastes context window space and can confuse the LLM.

**How it is calculated:** `(number of retrieved results from expected sources) / (total retrieved results)`. Source matching is by filename (case-insensitive). Returns 1.0 if no expected sources are defined or if results are empty.

---

### Recall@k

| Field | Value |
|-------|-------|
| **Key in report** | `recall_at_k` |
| **Defined in** | `eval_utils.compute_recall_at_k()` |

**What it evaluates:** The fraction of expected source documents that appear in the top-k retrieved results. Measures retrieval completeness.

**Why it is included:** While source_hit_rate checks the same concept, recall@k is the standard IR metric name and is reported alongside precision@k for a complete precision-recall picture. Together they reveal the trade-off between retrieving broadly and retrieving precisely.

**How it is calculated:** `|expected_sources ∩ retrieved_sources| / |expected_sources|`. Functionally equivalent to source_hit_rate in this implementation (both use filename matching). Returns 1.0 if no expected sources are defined, 0.0 if results are empty.

---

### NDCG@k (Normalized Discounted Cumulative Gain)

| Field | Value |
|-------|-------|
| **Key in report** | `ndcg_at_k` |
| **Defined in** | `eval_utils.compute_ndcg_at_k()` |

**What it evaluates:** The quality of the ranking compared to the ideal ranking where all relevant results are at the top. A ranking-aware metric that penalises relevant results appearing lower in the list.

**Why it is included:** MRR only looks at the first relevant result. NDCG evaluates the entire ranked list, making it sensitive to cases where multiple relevant documents exist but are scattered across different positions. It is the standard metric for evaluating ranked retrieval in information retrieval research.

**How it is calculated:** Uses binary relevance (1 if source matches an expected source, 0 otherwise). DCG = sum of (relevance_i / log2(rank_i + 2)). The ideal DCG (IDCG) is computed by sorting all relevance labels in descending order. NDCG = DCG / IDCG. Returns 1.0 if no expected sources are defined, 0.0 if results are empty.

---

### Metadata Match Ratio

| Field | Value |
|-------|-------|
| **Key in report** | `metadata_match_ratio` |
| **Defined in** | `eval_utils.compute_metadata_match_ratio()` |

**What it evaluates:** The fraction of retrieved results that satisfy all metadata filters defined on the eval case (e.g. `category=policy`).

**Why it is included:** When a case specifies metadata filters, those filters represent a hard constraint. If the retrieval pipeline returns results that violate the filter, it means the filtering logic is broken or the filter is being ignored. This metric is the second retrieval gate check for the PASS/REVIEW verdict (threshold >= 0.8).

**How it is calculated:** For each retrieved result, check whether all key-value pairs in `case.metadata_filters` match the result's metadata (string comparison). The score is `matching_results / total_results`. Returns 1.0 if no metadata filters are defined, 0.0 if results are empty.

---

### Backend Distribution

| Field | Value |
|-------|-------|
| **Key in report** | `backend_distribution` |
| **Defined in** | `eval_utils.compute_backend_distribution()` |

**What it evaluates:** Counts retrieved results by search backend (e.g. `fts`, `vector`). Not a score — a diagnostic distribution.

**Why it is included:** The hybrid search agent fuses full-text search and vector search. This distribution tells you whether both backends are actively contributing to the final result set. If all results come from one backend, the fusion mechanism may not be working as intended.

**How it is calculated:** Iterates over retrieval results and groups them by the `backend` field. Returns a dict like `{"fts": 3, "vector": 2}`.

---

## Deterministic Answer Metrics

These metrics are computed from the agent's answer text and the eval case's keyword lists. Defined in `eval_utils.py`.

---

### Required Keyword Hit Rate

| Field | Value |
|-------|-------|
| **Key in report** | `keyword_checks.required_keyword_hit_rate` |
| **Defined in** | `eval_utils.compute_keyword_checks()` |

**What it evaluates:** The fraction of required keywords (defined in the eval case) that appear in the agent's answer.

**Why it is included:** Some questions demand specific terms in the answer (e.g. a regulation number, a product name). This metric enforces that the answer contains the expected key terms. It is one of the keyword gate checks for the PASS/REVIEW verdict (threshold >= 0.5).

**How it is calculated:** Each keyword and the answer are normalized (lowercased, accents stripped, punctuation removed). The score is `(keywords found in answer) / (total required keywords)`. Returns 1.0 if no required keywords are defined.

---

### Disallowed Keyword Hits

| Field | Value |
|-------|-------|
| **Key in report** | `keyword_checks.disallowed_keyword_hits` |
| **Defined in** | `eval_utils.compute_keyword_checks()` |

**What it evaluates:** The count of disallowed keywords that appear in the agent's answer.

**Why it is included:** Some answers should avoid certain terms (e.g. mentioning a competitor, using a deprecated term). Any non-zero count is a failure signal. It is the second keyword gate check for the PASS/REVIEW verdict (must equal 0).

**How it is calculated:** Same normalization as required keywords. Counts how many disallowed keywords appear as substrings in the normalized answer. Returns 0 if no disallowed keywords are defined.

---

### Average Metric Score

| Field | Value |
|-------|-------|
| **Key in report** | `average_metric_score` |
| **Defined in** | `eval_engine.evaluate_case()` |

**What it evaluates:** The mean of all DeepEval LLM-judged metric scores, expressed as a percentage (0-100).

**Why it is included:** Provides a single summary number for quick comparison across cases. Useful for spotting overall quality trends without inspecting each metric individually.

**How it is calculated:** Collects all non-None scores from the 7 DeepEval metrics, computes their arithmetic mean, and multiplies by 100. Rounded to 1 decimal place.

---

## Verdict Logic

The final `status` field ("PASS" or "REVIEW") is computed by `compute_case_status()` in `eval_engine.py`. It applies three independent gates:

| Gate | Condition |
|------|-----------|
| **metrics_ok** | `faithfulness >= 0.5` AND `answer_relevancy >= 0.5` |
| **retrieval_ok** | `source_hit_rate >= 0.5` AND `metadata_match_ratio >= 0.8` |
| **keywords_ok** | `required_keyword_hit_rate >= 0.5` AND `disallowed_keyword_hits == 0` |

All three gates must pass for `status = "PASS"`. Any gate failing or any runtime error results in `status = "REVIEW"`.
