"""Evaluation engine for the DeepEval-based hybrid search harness."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage

from eval_metric_registry import llm_metric_keys
from eval_models import EvalCase
from eval_utils import (
    build_expected_output,
    build_gold_context,
    build_retrieval_context,
    compute_backend_distribution,
    compute_keyword_checks,
    compute_metadata_match_ratio,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_source_hit_rate,
    extract_agent_retrieval_results,
    extract_message_text,
    make_prompt,
    preview_results,
    safe_mean,
)

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
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
except ImportError as exc:
    raise SystemExit(
        "This script requires the DeepEval stack. Install it with: pip install deepeval"
    ) from exc


class EvaluationEngine:
    """Runs retrieval, agent execution, and DeepEval metrics for evaluation cases."""

    def __init__(
        self,
        *,
        agent: Any,
        retriever: Any,
        judge_model: str,
        threshold: float,
    ) -> None:
        self.agent = agent
        self.retriever = retriever
        self.judge_model = judge_model
        self.threshold = threshold

    def build_metrics(self) -> dict[str, Any]:
        """Build the DeepEval judging panel.

        Returns a dict mapping metric name → DeepEval metric object. A fresh
        instance is created per call to ensure clean state across concurrent runs.
        """
        metrics: dict[str, Any] = {
            "answer_relevancy": AnswerRelevancyMetric(threshold=self.threshold, model=self.judge_model),
            "faithfulness": FaithfulnessMetric(threshold=self.threshold, model=self.judge_model),
            "contextual_precision": ContextualPrecisionMetric(threshold=self.threshold, model=self.judge_model),
            "contextual_recall": ContextualRecallMetric(threshold=self.threshold, model=self.judge_model),
            "contextual_relevancy": ContextualRelevancyMetric(threshold=self.threshold, model=self.judge_model),
            "hallucination": HallucinationMetric(threshold=self.threshold, model=self.judge_model),
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
                model=self.judge_model,
            ),
        }

        assert set(metrics.keys()) == set(llm_metric_keys()), (
            f"Registry/engine metric key mismatch: {set(metrics.keys()) ^ set(llm_metric_keys())}"
        )

        return metrics

    async def run_agent_case(self, case: EvalCase) -> tuple[str, list[dict[str, Any]], float, str | None]:
        """Invoke the LangGraph agent for a single eval case.

        A unique thread_id is generated per case to prevent memory bleed between
        cases. Uses perf_counter for high-resolution latency measurement.
        Returns (answer, agent_retrieval_results, latency_seconds, error_or_None).
        """
        started = time.perf_counter()
        try:
            config = {"configurable": {"thread_id": f"deepeval-{case.id}-{uuid4().hex}"}}
            payload = {"messages": [HumanMessage(content=make_prompt(case))]}
            result = await self.agent.ainvoke(payload, config=config)
            answer = extract_message_text(result["messages"][-1].content)
            agent_retrieval_results = extract_agent_retrieval_results(result["messages"])
            return answer, agent_retrieval_results, round(time.perf_counter() - started, 3), None
        except Exception as exc:
            return "", [], round(time.perf_counter() - started, 3), f"Agent invocation failed: {type(exc).__name__}: {exc}"

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
            return [result.model_dump() for result in results], round(time.perf_counter() - started, 3), None
        except Exception as exc:
            return [], round(time.perf_counter() - started, 3), f"Retrieval failed: {type(exc).__name__}: {exc}"

    def build_test_case(
        self,
        case: EvalCase,
        answer: str,
        latency: float,
        retrieval_results: list[dict[str, Any]],
    ) -> LLMTestCase:
        """Package case data and results into the LLMTestCase object DeepEval expects."""
        return LLMTestCase(
            input=make_prompt(case),
            actual_output=answer or "",
            expected_output=build_expected_output(case),
            context=build_gold_context(case),
            retrieval_context=build_retrieval_context(retrieval_results),
            completion_time=latency,
        )

    async def run_metric(self, metric_name: str, metric: Any, test_case: LLMTestCase) -> tuple[str, dict[str, Any]]:
        """Run a single DeepEval metric and return a (name, details) pair."""
        try:
            await metric.a_measure(test_case)
            score = getattr(metric, "score", None)
            passed = metric.is_successful() if hasattr(metric, "is_successful") else bool(
                score and score >= getattr(metric, "threshold", 0.5)
            )
            return metric_name, {
                "score": round(float(score), 4) if score is not None else None,
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

    async def run_metrics(self, test_case: LLMTestCase) -> dict[str, Any]:
        """Run all metrics concurrently and return a combined results dict."""
        metrics = self.build_metrics()
        pairs = await asyncio.gather(
            *(self.run_metric(metric_name, metric, test_case) for metric_name, metric in metrics.items())
        )
        return dict(pairs)

    @staticmethod
    def compute_case_status(
        metric_results: dict[str, Any],
        source_hit_rate: float,
        metadata_match_ratio: float,
        keyword_checks: dict[str, Any],
    ) -> str:
        """Return 'PASS' or 'REVIEW' based on three combined gates.

        - metrics_ok: faithfulness and answer_relevancy >= 0.5
        - retrieval_ok: source_hit_rate >= 0.5 and metadata_match_ratio >= 0.8
        - keywords_ok: required_keyword_hit_rate >= 0.5 and no disallowed keywords present
        """
        faithfulness = metric_results.get("faithfulness", {}).get("score")
        answer_relevancy = metric_results.get("answer_relevancy", {}).get("score")

        required_scores = [score for score in [faithfulness, answer_relevancy] if score is not None]

        metrics_ok = bool(required_scores) and all(score >= 0.5 for score in required_scores)
        retrieval_ok = source_hit_rate >= 0.5 and metadata_match_ratio >= 0.8
        keywords_ok = (
            keyword_checks.get("required_keyword_hit_rate", 1.0) >= 0.5
            and keyword_checks.get("disallowed_keyword_hits", 0) == 0
        )
        return "PASS" if metrics_ok and retrieval_ok and keywords_ok else "REVIEW"

    async def evaluate_case(self, case: EvalCase) -> dict[str, Any]:
        """Run a single eval case end-to-end: retrieval → agent → metrics → verdict."""
        errors: list[str] = []

        # Direct retrieval: used for source_hit_rate, MRR, Precision@k, Recall@k,
        # NDCG@k, and metadata_match_ratio as ground-truth search quality checks.
        retrieval_results, retrieval_latency, retrieval_error = self.run_retrieval_case(case)
        if retrieval_error:
            errors.append(retrieval_error)

        # Agent run: captures both the final answer and the chunks the agent actually retrieved.
        answer, agent_retrieval_results, latency, agent_error = await self.run_agent_case(case)
        if agent_error:
            errors.append(agent_error)

        # Use the agent's actual tool-call results for DeepEval context so that faithfulness,
        # contextual precision, and recall are measured against what the agent truly used.
        # Fall back to direct retrieval results if the agent made no tool call.
        deepeval_context_results = agent_retrieval_results if agent_retrieval_results else retrieval_results

        test_case = self.build_test_case(case, answer, latency, deepeval_context_results)
        metric_results = await self.run_metrics(test_case)

        source_hit_rate = compute_source_hit_rate(case, retrieval_results)
        metadata_match_ratio = compute_metadata_match_ratio(case, retrieval_results)
        mrr = compute_mrr(case, retrieval_results)
        precision_at_k = compute_precision_at_k(case, retrieval_results)
        recall_at_k = compute_recall_at_k(case, retrieval_results)
        ndcg_at_k = compute_ndcg_at_k(case, retrieval_results)
        backend_distribution = compute_backend_distribution(retrieval_results)
        keyword_checks = compute_keyword_checks(case, answer)
        avg_judge_score = round(
            safe_mean([
                details.get("score")
                for details in metric_results.values()
                if details.get("score") is not None
            ]) * 100,
            1,
        )

        status = self.compute_case_status(metric_results, source_hit_rate, metadata_match_ratio, keyword_checks)
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
            "llm_latency_seconds": round(max(latency - retrieval_latency, 0.0), 3),
            "retrieval_config": case.retrieval.model_dump(),
            "metadata_filters": case.metadata_filters,
            "expected_sources": case.expected_sources,
            "retrieval_preview": preview_results(retrieval_results),
            "source_hit_rate": source_hit_rate,
            "metadata_match_ratio": metadata_match_ratio,
            "mrr": mrr,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "ndcg_at_k": ndcg_at_k,
            "backend_distribution": backend_distribution,
            "keyword_checks": keyword_checks,
            "avg_judge_score": avg_judge_score,
            "metrics": metric_results,
            "status": status,
            "errors": errors,
        }

    async def evaluate_cases(self, cases: list[EvalCase], concurrency: int = 2) -> list[dict[str, Any]]:
        """Run all eval cases concurrently, bounded by the concurrency limit."""
        semaphore = asyncio.Semaphore(max(concurrency, 1))

        async def bounded_eval(case: EvalCase) -> dict[str, Any]:
            async with semaphore:
                return await self.evaluate_case(case)

        return await asyncio.gather(*(bounded_eval(case) for case in cases))
