"""Evaluation engine for the DeepEval-based hybrid search harness.

Defines EvaluationEngine, which orchestrates one full eval case:
direct retrieval (for ground-truth IR metrics), agent invocation (for
end-to-end answer quality and the chunks the agent actually used),
DeepEval LLM-judge metrics, and the deterministic source/chunk/keyword
metrics — all glued together by evaluate_case.

Module layout: public EvaluationEngine class first, then the private
_TokenTrackingJudgeLLM wrapper used by the judge panel, then the
module-level _sum_message_tokens helper at the bottom.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from eval_metric_registry import llm_metric_keys, metric_decimals
from eval_metrics import (
    compute_all_chunk_metrics,
    compute_backend_distribution,
    compute_hit_at_k,
    compute_keyword_checks,
    compute_metadata_match_ratio,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
)
from eval_models import EvalCase
from eval_utils import (
    build_expected_output,
    build_gold_context,
    build_retrieval_context,
    extract_agent_retrieval_results,
    extract_message_text,
    make_prompt,
    preview_results,
    safe_mean,
)

# Storage-precision lookup, resolved once at import time.  Every round()
# below pulls its decimals from this dict so the registry stays the single
# place to change a metric's precision.
_DECIMALS: dict[str, int] = metric_decimals()

try:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
        GEval,
        HallucinationMetric,
    )
    from deepeval.models.base_model import DeepEvalBaseLLM
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
except ImportError as exc:
    raise SystemExit(
        "This script requires the DeepEval stack. Install it with: pip install deepeval"
    ) from exc


class EvaluationEngine:
    """Runs retrieval, agent execution, and DeepEval metrics per eval case.

    One instance is constructed per evaluation run.  Holds the agent and
    retriever handles, the judge configuration, and the active gate
    thresholds — passed in by the caller so the engine doesn't reach
    into eval_config directly.  enabled_groups controls which
    toggleable metric families are computed; metrics whose group is
    disabled return None and the verdict gates skip them.
    """

    def __init__(
        self,
        *,
        agent: Any,
        retriever: Any,
        judge_model: str,
        threshold: float,
        metadata_match_threshold: float = 0.8,
        required_keyword_threshold: float = 0.5,
        enabled_groups: set[str] | None = None,
    ) -> None:
        """Capture the agent, retriever, judge config, and gate thresholds.

        enabled_groups defaults to all three toggleable groups when
        None so an engine constructed without explicit groups runs
        the full metric panel (matching the dashboard's ledger filter,
        which only persists fully-enabled runs).
        """
        self.agent = agent
        self.retriever = retriever
        self.judge_model = judge_model
        self.threshold = threshold
        self.metadata_match_threshold = metadata_match_threshold
        self.required_keyword_threshold = required_keyword_threshold
        self.enabled_groups: set[str] = set(enabled_groups) if enabled_groups is not None else {"judge", "source", "chunk"}

    def gate_thresholds(self) -> dict[str, float]:
        """Return the verdict gate thresholds in effect for this engine.

        Serialized into eval_runs.gate_thresholds for every run so
        the dashboard can flag drift between consecutive runs (changing
        a threshold mid-stream invalidates trend comparisons; the warning
        banner makes that visible without forcing a manual audit).
        """
        return {
            "judge_threshold": self.threshold,
            "metadata_match_threshold": self.metadata_match_threshold,
            "required_keyword_threshold": self.required_keyword_threshold,
        }

    def build_metrics(self, judge_llm: "_TokenTrackingJudgeLLM") -> dict[str, Any]:
        """Build the DeepEval judging panel bound to a token-tracking LLM.

        Returns a dict mapping metric name → DeepEval metric object.  A
        fresh panel is created per call to keep state clean across
        concurrent runs.  Returns an empty dict when the "judge"
        toggle group is disabled, which causes _run_metrics() to
        return no LLM-judged scores and zero judge tokens.

        Every metric is bound to the same judge_llm instance so that
        token usage from the entire panel accumulates onto a single
        wrapper; the caller reads totals off the wrapper after the
        panel finishes.
        """
        if "judge" not in self.enabled_groups:
            return {}

        metrics: dict[str, Any] = {
            "answer_relevancy": AnswerRelevancyMetric(threshold=self.threshold, model=judge_llm),
            "faithfulness": FaithfulnessMetric(threshold=self.threshold, model=judge_llm),
            "contextual_precision": ContextualPrecisionMetric(threshold=self.threshold, model=judge_llm),
            "contextual_recall": ContextualRecallMetric(threshold=self.threshold, model=judge_llm),
            "contextual_relevancy": ContextualRelevancyMetric(threshold=self.threshold, model=judge_llm),
            "hallucination": HallucinationMetric(threshold=self.threshold, model=judge_llm),
            "correctness_g_eval": GEval(
                name="Grounded Correctness",
                criteria=(
                    "Determine whether the actual output correctly answers the user's request, "
                    "covers the important facts from the expected output, and avoids unsupported claims."
                ),
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
                threshold=self.threshold,
                model=judge_llm,
            ),
        }

        # Hard guard against engine/registry drift: if the engine's panel
        # ever falls out of sync with llm_metric_keys(), every consumer
        # downstream (CSV, SQL, dashboard) silently emits NULLs for the
        # missing keys.  Use RuntimeError (not assert) so the check
        # survives python -O.
        registry_keys = set(llm_metric_keys())
        engine_keys = set(metrics.keys())
        if engine_keys != registry_keys:
            raise RuntimeError(
                f"Registry/engine metric key mismatch: {engine_keys ^ registry_keys}"
            )

        return metrics

    async def run_agent_case(
        self, case: EvalCase
    ) -> tuple[str, list[dict[str, Any]], float, int | None, int | None, str | None]:
        """Invoke the LangGraph agent for a single eval case.

        A unique thread_id is generated per case to prevent memory bleed between
        cases. Uses perf_counter for high-resolution latency measurement.
        Token counts are summed across every AIMessage in the result
        via LangChain's standardised usage_metadata (populated by
        ChatOpenAI and most modern providers).  Returns (None, None)
        for tokens when no message exposes the field, so that "couldn't
        measure" stays distinct from "actually zero" in the dashboard.

        Returns (answer, agent_retrieval_results, latency_seconds,
        agent_input_tokens, agent_output_tokens, error_or_None).
        """
        started = time.perf_counter()
        try:
            config = {"configurable": {"thread_id": f"deepeval-{case.id}-{uuid4().hex}"}}
            payload = {"messages": [HumanMessage(content=make_prompt(case))]}
            result = await self.agent.ainvoke(payload, config=config)
            answer = extract_message_text(result["messages"][-1].content)
            agent_retrieval_results = extract_agent_retrieval_results(result["messages"])
            input_tokens, output_tokens = _sum_message_tokens(result["messages"])
            return (
                answer, agent_retrieval_results,
                round(time.perf_counter() - started, _DECIMALS["latency_seconds"]),
                input_tokens, output_tokens, None,
            )
        except Exception as exc:
            return (
                "", [], round(time.perf_counter() - started, _DECIMALS["latency_seconds"]),
                None, None,
                f"Agent invocation failed: {type(exc).__name__}: {exc}",
            )

    def run_retrieval_case(self, case: EvalCase) -> tuple[list[dict[str, Any]], float, str | None]:
        """Run the retriever directly, bypassing the agent.

        Isolates retrieval quality from LLM reasoning — if the agent gives a wrong
        answer, this tells you whether search or generation was the failure point.
        Returns (results, retrieval_latency_seconds, error_or_None).
        """
        started = time.perf_counter()
        try:
            results = self.retriever.search(
                query=case.question,
                k=case.retrieval.k,
                vector_search_method=case.retrieval.vector_search_method,
                use_phrase=case.retrieval.use_phrase,
                use_prefix=case.retrieval.use_prefix,
                multi_fts=case.retrieval.multi_fts,
                **case.metadata_filters,
            )
            return (
                [result.model_dump() for result in results],
                round(time.perf_counter() - started, _DECIMALS["retrieval_latency_seconds"]),
                None,
            )
        except Exception as exc:
            return (
                [],
                round(time.perf_counter() - started, _DECIMALS["retrieval_latency_seconds"]),
                f"Retrieval failed: {type(exc).__name__}: {exc}",
            )

    def _build_test_case(
        self,
        case: EvalCase,
        answer: str,
        latency: float,
        retrieval_results: list[dict[str, Any]],
    ) -> LLMTestCase:
        """Package case data and results into a DeepEval LLMTestCase.

        Pulls the four DeepEval fields from registry-aware builders in
        eval_utils so the per-metric input shapes (input prompt,
        actual output, gold context, retrieval context) stay consistent
        with how the deterministic metrics are computed against the
        same case.  actual_output falls back to the empty string
        when the agent failed, so DeepEval's a_measure doesn't crash
        on None and the case can still earn a structured failure.
        """
        return LLMTestCase(
            input=make_prompt(case),
            actual_output=answer or "",
            expected_output=build_expected_output(case),
            context=build_gold_context(case),
            retrieval_context=build_retrieval_context(retrieval_results),
            completion_time=latency,
        )

    async def _run_metric(self, metric_name: str, metric: Any, test_case: LLMTestCase) -> tuple[str, dict[str, Any]]:
        """Run a single DeepEval metric and return its (name, details) pair.

        The details dict carries score (rounded to the registry-declared
        precision), reason (the judge's explanation), passed (the
        metric's own is_successful() verdict), and threshold (the
        pass/fail line bound on the metric).  Wrapped in a try/except so a
        single judge failure (network, schema, rate-limit) doesn't break the
        asyncio.gather over the whole panel — the failed metric returns
        a structured failure dict and the rest of the panel still produces
        scores.
        """
        try:
            await metric.a_measure(test_case)
            score = getattr(metric, "score", None)
            passed = metric.is_successful() if hasattr(metric, "is_successful") else bool(
                score and score >= getattr(metric, "threshold", 0.5)
            )
            return metric_name, {
                # metric_name is guaranteed in _DECIMALS because
                # build_metrics raises RuntimeError if the engine's
                # panel falls out of sync with llm_metric_keys().  Direct
                # subscript keeps the storage precision purely registry-driven.
                "score": round(float(score), _DECIMALS[metric_name]) if score is not None else None,
                "reason": getattr(metric, "reason", ""),
                "passed": bool(passed),
                "threshold": getattr(metric, "threshold", None),
            }
        except Exception as exc:
            return metric_name, {
                "score": None,
                "reason": f"Metric failed: {type(exc).__name__}: {exc}",
                "passed": False,
                "threshold": getattr(metric, "threshold", None),
            }

    async def _run_metrics(
        self, test_case: LLMTestCase
    ) -> tuple[dict[str, Any], int | None, int | None]:
        """Run the judge panel concurrently and report aggregate token usage.

        Returns (metric_results, judge_input_tokens, judge_output_tokens).
        Token counts are summed across every judge LLM call made for
        this case via the shared _TokenTrackingJudgeLLM instance.
        Returns (None, None) for token counts when the judge group
        is disabled (no panel runs, no LLM calls happen at all).
        """
        if "judge" not in self.enabled_groups:
            return {}, None, None

        judge_llm = _TokenTrackingJudgeLLM(self.judge_model)
        metrics = self.build_metrics(judge_llm)
        pairs = await asyncio.gather(
            *(self._run_metric(metric_name, metric, test_case) for metric_name, metric in metrics.items())
        )
        return dict(pairs), judge_llm.input_tokens, judge_llm.output_tokens

    def _compute_case_status(
        self,
        metric_results: dict[str, Any],
        hit_at_k: float | None,
        metadata_match_ratio: float,
        keyword_checks: dict[str, Any],
    ) -> str:
        """Return 'PASS' or 'REVIEW' based on three combined gates.

        Gate thresholds come from self.threshold (judge),
        self.metadata_match_threshold, and self.required_keyword_threshold.
        See eval_config.py for the configure-once contract.

        - metrics_ok: faithfulness and answer_relevancy >= self.threshold
          (skipped when the "judge" toggle group is disabled).
        - retrieval_ok: hit_at_k == 1.0 (skipped when "source" is disabled)
          and metadata_match_ratio >= self.metadata_match_threshold.
        - keywords_ok: required_keyword_hit_rate >= self.required_keyword_threshold
          and no disallowed keywords present.
        """
        if "judge" in self.enabled_groups:
            faithfulness = metric_results.get("faithfulness", {}).get("score")
            answer_relevancy = metric_results.get("answer_relevancy", {}).get("score")
            required_scores = [score for score in [faithfulness, answer_relevancy] if score is not None]
            metrics_ok = bool(required_scores) and all(score >= self.threshold for score in required_scores)
        else:
            metrics_ok = True

        retrieval_ok = metadata_match_ratio >= self.metadata_match_threshold
        if "source" in self.enabled_groups:
            retrieval_ok = retrieval_ok and hit_at_k == 1.0

        keywords_ok = (
            keyword_checks.get("required_keyword_hit_rate", 1.0) >= self.required_keyword_threshold
            and keyword_checks.get("disallowed_keyword_hits", 0) == 0
        )
        return "PASS" if metrics_ok and retrieval_ok and keywords_ok else "REVIEW"

    async def evaluate_case(self, case: EvalCase) -> dict[str, Any]:
        """Run a single eval case end-to-end and return the per-case result dict.

        Pipeline:

        1. Direct retrieval — drives the deterministic source/chunk
           retrieval metrics (ground-truth search quality, isolated from
           the agent's reasoning).
        2. Agent invocation — captures the final answer, the chunks the
           agent actually retrieved via tool calls, and token usage.
        3. DeepEval judge panel — runs against the agent's *actual*
           tool-call results when present (so faithfulness and contextual
           precision/recall reflect what the agent saw), falling back to
           direct retrieval results when the agent made no tool call.
        4. Always-on metrics (metadata match, backend distribution,
           keyword checks) plus the toggleable source/chunk metric
           families when their groups are enabled.
        5. Three-gate verdict via _compute_case_status; any case with
           runtime errors is forced to REVIEW regardless of scores.

        The returned dict's keys match the columns expected downstream
        by eval_report_manager (CSV / JSON) and eval_sqlite
        (ledger), so adding a metric here means adding it to the
        registry too — no plumbing changes elsewhere.
        """
        errors: list[str] = []

        # Direct retrieval: used for hit_at_k, MRR, Precision@k, Recall@k,
        # NDCG@k, and metadata_match_ratio as ground-truth search quality checks.
        retrieval_results, retrieval_latency, retrieval_error = self.run_retrieval_case(case)
        if retrieval_error:
            errors.append(retrieval_error)

        # Agent run: captures both the final answer and the chunks the agent actually retrieved.
        (
            answer, agent_retrieval_results, latency,
            agent_input_tokens, agent_output_tokens, agent_error,
        ) = await self.run_agent_case(case)
        if agent_error:
            errors.append(agent_error)

        agent_total_tokens = (
            agent_input_tokens + agent_output_tokens
            if agent_input_tokens is not None and agent_output_tokens is not None
            else None
        )

        # Use the agent's actual tool-call results for DeepEval context so that faithfulness,
        # contextual precision, and recall are measured against what the agent truly used.
        # Fall back to direct retrieval results if the agent made no tool call.
        deepeval_context_results = agent_retrieval_results if agent_retrieval_results else retrieval_results

        test_case = self._build_test_case(case, answer, latency, deepeval_context_results)
        metric_results, judge_input_tokens, judge_output_tokens = await self._run_metrics(test_case)
        judge_total_tokens = (
            judge_input_tokens + judge_output_tokens
            if judge_input_tokens is not None and judge_output_tokens is not None
            else None
        )

        # Always-on metrics.
        metadata_match_ratio = compute_metadata_match_ratio(case, retrieval_results)
        backend_distribution = compute_backend_distribution(retrieval_results)
        keyword_checks = compute_keyword_checks(case, answer)

        # Source-level retrieval metrics — only computed when toggle group is enabled.
        source_on = "source" in self.enabled_groups
        hit_at_k = compute_hit_at_k(case, retrieval_results) if source_on else None
        mrr = compute_mrr(case, retrieval_results) if source_on else None
        precision_at_k = compute_precision_at_k(case, retrieval_results) if source_on else None
        recall_at_k = compute_recall_at_k(case, retrieval_results) if source_on else None
        ndcg_at_k = compute_ndcg_at_k(case, retrieval_results) if source_on else None

        # Chunk-level retrieval metrics — only computed when toggle group is enabled.
        # compute_all_chunk_metrics is a single-pass variant of the five
        # compute_chunk_* helpers: it normalizes the expected snippets and
        # builds the per-result relevance flag list once, then derives every
        # metric from that shared work — five times less repetition than
        # calling the helpers individually.
        chunk_on = "chunk" in self.enabled_groups
        chunk_metrics = compute_all_chunk_metrics(case, retrieval_results) if chunk_on else {}
        chunk_hit_at_k       = chunk_metrics.get("chunk_hit_at_k")
        chunk_mrr            = chunk_metrics.get("chunk_mrr")
        chunk_precision_at_k = chunk_metrics.get("chunk_precision_at_k")
        chunk_recall_at_k    = chunk_metrics.get("chunk_recall_at_k")
        chunk_ndcg_at_k      = chunk_metrics.get("chunk_ndcg_at_k")

        # avg_judge_score: None when the judge panel didn't run (empty metric_results).
        judge_scores = [
            details.get("score")
            for details in metric_results.values()
            if details.get("score") is not None
        ]
        avg_judge_score = (
            round(safe_mean(judge_scores), _DECIMALS["avg_judge_score"])
            if judge_scores
            else None
        )

        status = self._compute_case_status(metric_results, hit_at_k, metadata_match_ratio, keyword_checks)
        if errors:
            status = "REVIEW"

        return {
            "id": case.id,
            "question": case.question,
            "category": case.category,
            "notes": case.notes,
            "prompt_used": make_prompt(case),
            "expected_output": build_expected_output(case),
            "answer": answer,
            "latency_seconds": latency,
            "retrieval_latency_seconds": retrieval_latency,
            "llm_latency_seconds": round(max(latency - retrieval_latency, 0.0), _DECIMALS["llm_latency_seconds"]),
            "retrieval_config": case.retrieval.model_dump(),
            "metadata_filters": case.metadata_filters,
            "expected_sources": case.expected_sources,
            "retrieval_preview": preview_results(retrieval_results),
            "hit_at_k": hit_at_k,
            "metadata_match_ratio": metadata_match_ratio,
            "mrr": mrr,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "ndcg_at_k": ndcg_at_k,
            "chunk_hit_at_k": chunk_hit_at_k,
            "chunk_mrr": chunk_mrr,
            "chunk_precision_at_k": chunk_precision_at_k,
            "chunk_recall_at_k": chunk_recall_at_k,
            "chunk_ndcg_at_k": chunk_ndcg_at_k,
            "backend_distribution": backend_distribution,
            "keyword_checks": keyword_checks,
            "avg_judge_score": avg_judge_score,
            "agent_input_tokens":  agent_input_tokens,
            "agent_output_tokens": agent_output_tokens,
            "agent_total_tokens":  agent_total_tokens,
            "judge_input_tokens":  judge_input_tokens,
            "judge_output_tokens": judge_output_tokens,
            "judge_total_tokens":  judge_total_tokens,
            "metrics": metric_results,
            "status": status,
            "errors": errors,
        }

    async def evaluate_cases(self, cases: list[EvalCase], concurrency: int = 2) -> list[dict[str, Any]]:
        """Run every case concurrently, bounded by an asyncio.Semaphore.

        concurrency is clamped to a minimum of 1 so a misconfigured
        0 doesn't deadlock on a zero-permit semaphore.  Returns the
        per-case dicts in original case order (asyncio.gather
        preserves input order regardless of completion order).
        """
        semaphore = asyncio.Semaphore(max(concurrency, 1))

        async def bounded_eval(case: EvalCase) -> dict[str, Any]:
            async with semaphore:
                return await self.evaluate_case(case)

        return await asyncio.gather(*(bounded_eval(case) for case in cases))


# ---------------------------------------------------------------------------
# DeepEval LLM wrapper that tracks judge token usage
# ---------------------------------------------------------------------------

class _TokenTrackingJudgeLLM(DeepEvalBaseLLM):
    """DeepEval-compatible LLM that wraps ChatOpenAI and counts tokens.

    One instance is created per evaluate_case invocation and shared
    across every metric in the judge panel; after the panel runs, the
    engine reads input_tokens / output_tokens directly off the
    wrapper.

    ChatOpenAI is used internally (rather than the openai SDK directly)
    to match the agent's LLM stack and to get standardised
    usage_metadata extraction. Schema-based generation is supported
    via with_structured_output(..., include_raw=True) so the
    wrapping AIMessage stays available for token accounting even when
    DeepEval asks for a Pydantic-typed response.

    The lock guards counter updates in a_generate because all judge
    metrics in a case run concurrently inside a single
    asyncio.gather — without it, interleaved increments could
    corrupt totals on bursty runs. generate (sync) does NOT acquire
    the lock: asyncio.Lock cannot be used outside a running event
    loop, and DeepEval only invokes the sync path single-threadedly.
    """

    def __init__(self, model: str) -> None:
        """Wrap a fresh ChatOpenAI client and zero the token counters."""
        self._model_name = model
        self._client     = ChatOpenAI(model=model)
        self.input_tokens  = 0
        self.output_tokens = 0
        self._lock = asyncio.Lock()

    def load_model(self) -> ChatOpenAI:
        """Return the wrapped LangChain client (DeepEval interface contract)."""
        return self._client

    def get_model_name(self) -> str:
        """Return the model name string (DeepEval interface contract)."""
        return self._model_name

    def generate(self, prompt: str, schema: Any = None) -> Any:
        """Synchronous generation path, used by DeepEval's sync test runs.

        When schema is set, generation goes through
        with_structured_output(..., include_raw=True) so the raw
        AIMessage stays available for token accounting alongside the
        parsed Pydantic value.  Token counter updates skip the lock
        because DeepEval's sync code path is single-threaded — see the
        class docstring for the lock policy.
        """
        if schema is not None:
            structured = self._client.with_structured_output(schema, include_raw=True)
            result = structured.invoke(prompt)
            self._record(result.get("raw"))
            return result["parsed"]
        response = self._client.invoke(prompt)
        self._record(response)
        return response.content

    async def a_generate(self, prompt: str, schema: Any = None) -> Any:
        """Async generation path used by the concurrent judge panel.

        Mirrors generate but acquires self._lock around counter
        updates because every metric in the panel runs concurrently
        inside one asyncio.gather — without the lock, interleaved
        increments could corrupt totals on bursty runs.
        """
        if schema is not None:
            structured = self._client.with_structured_output(schema, include_raw=True)
            result = await structured.ainvoke(prompt)
            async with self._lock:
                self._record(result.get("raw"))
            return result["parsed"]
        response = await self._client.ainvoke(prompt)
        async with self._lock:
            self._record(response)
        return response.content

    def _record(self, message: Any) -> None:
        """Add this call's token usage to the running totals.

        Returns early when usage_metadata is absent — true for
        non-OpenAI providers and for some LangChain corner cases where
        the metadata isn't propagated.
        """
        usage = getattr(message, "usage_metadata", None)
        if usage:
            self.input_tokens  += int(usage.get("input_tokens", 0)  or 0)
            self.output_tokens += int(usage.get("output_tokens", 0) or 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sum_message_tokens(messages: list[Any]) -> tuple[int | None, int | None]:
    """Sum input/output token counts across every AIMessage in messages.

    Reads LangChain's standardised usage_metadata attribute,
    populated by ChatOpenAI and most modern providers (returns
    {"input_tokens": int, "output_tokens": int, "total_tokens":
    int}). Returns (None, None) when no AIMessage carries the
    field — typically because a non-OpenAI provider doesn't expose it.
    Returning None instead of (0, 0) keeps "couldn't measure"
    distinct from "actually zero" so the dashboard renders a clear
    NOT_EVALUATED rather than a misleading zero.
    """
    saw_usage    = False
    input_total  = 0
    output_total = 0
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        usage = getattr(message, "usage_metadata", None)
        if not usage:
            continue
        saw_usage     = True
        input_total  += int(usage.get("input_tokens", 0) or 0)
        output_total += int(usage.get("output_tokens", 0) or 0)
    if not saw_usage:
        return None, None
    return input_total, output_total
