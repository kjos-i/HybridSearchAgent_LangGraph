"""Report generation and persistence for the DeepEval-based hybrid search evaluation flow."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from eval_utils import safe_mean


class ReportManager:
    """Handles evaluation summaries and persisted reports."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)

    def build_summary(self, results: list[dict[str, Any]], judge_model: str) -> dict[str, Any]:
        """Aggregate per-case results into a top-level summary dict."""
        metric_names = sorted({name for item in results for name in item["metrics"].keys()})
        metric_averages = {}

        for name in metric_names:
            scores = [
                item["metrics"][name]["score"]
                for item in results
                if item["metrics"].get(name, {}).get("score") is not None
            ]
            metric_averages[name] = safe_mean(scores)

        pass_count = sum(1 for item in results if item["status"] == "PASS")

        return {
            "case_count": len(results),
            "pass_count": pass_count,
            "pass_rate": round(pass_count / max(len(results), 1), 3),
            "avg_case_score": safe_mean([item["average_metric_score"] for item in results]),
            "avg_source_hit_rate": safe_mean([item["source_hit_rate"] for item in results]),
            "avg_metadata_match_ratio": safe_mean([item["metadata_match_ratio"] for item in results]),
            "avg_mrr": safe_mean([item["mrr"] for item in results]),
            "avg_precision_at_k": safe_mean([item["precision_at_k"] for item in results]),
            "avg_recall_at_k": safe_mean([item["recall_at_k"] for item in results]),
            "avg_ndcg_at_k": safe_mean([item["ndcg_at_k"] for item in results]),
            "avg_latency_seconds": safe_mean([item["latency_seconds"] for item in results]),
            "avg_retrieval_latency_seconds": safe_mean([item["retrieval_latency_seconds"] for item in results]),
            "avg_llm_latency_seconds": safe_mean([item["llm_latency_seconds"] for item in results]),
            "judge_model": judge_model,
            "metric_averages": metric_averages,
        }

    def save_report(self, report: dict[str, Any]) -> tuple[Path, Path]:
        """Write timestamped JSON and CSV reports to the output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"deepeval_report_{timestamp}.json"
        csv_path = self.output_dir / f"deepeval_summary_{timestamp}.csv"

        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "id",
                    "category",
                    "status",
                    "avg_metric_score",
                    "source_hit_rate",
                    "metadata_match_ratio",
                    "mrr",
                    "precision_at_k",
                    "recall_at_k",
                    "ndcg_at_k",
                    "backend_fts",
                    "backend_vector",
                    "backend_hybrid",
                    "required_keyword_hit_rate",
                    "disallowed_keyword_hits",
                    "answer_relevancy",
                    "faithfulness",
                    "contextual_precision",
                    "contextual_recall",
                    "contextual_relevancy",
                    "hallucination",
                    "correctness_g_eval",
                    "latency_seconds",
                    "retrieval_latency_seconds",
                    "llm_latency_seconds",
                    "error_count",
                ],
            )
            writer.writeheader()

            for item in report["results"]:
                metrics = item.get("metrics", {})
                kw = item.get("keyword_checks", {})
                bd = item.get("backend_distribution", {})
                writer.writerow(
                    {
                        "id": item["id"],
                        "category": item.get("category", ""),
                        "status": item["status"],
                        "avg_metric_score": item["average_metric_score"],
                        "source_hit_rate": item["source_hit_rate"],
                        "metadata_match_ratio": item["metadata_match_ratio"],
                        "mrr": item.get("mrr"),
                        "precision_at_k": item.get("precision_at_k"),
                        "recall_at_k": item.get("recall_at_k"),
                        "ndcg_at_k": item.get("ndcg_at_k"),
                        "backend_fts": bd.get("fts", 0),
                        "backend_vector": bd.get("vector", 0),
                        "backend_hybrid": bd.get("hybrid", 0),
                        "required_keyword_hit_rate": kw.get("required_keyword_hit_rate"),
                        "disallowed_keyword_hits": kw.get("disallowed_keyword_hits"),
                        "answer_relevancy": metrics.get("answer_relevancy", {}).get("score"),
                        "faithfulness": metrics.get("faithfulness", {}).get("score"),
                        "contextual_precision": metrics.get("contextual_precision", {}).get("score"),
                        "contextual_recall": metrics.get("contextual_recall", {}).get("score"),
                        "contextual_relevancy": metrics.get("contextual_relevancy", {}).get("score"),
                        "hallucination": metrics.get("hallucination", {}).get("score"),
                        "correctness_g_eval": metrics.get("correctness_g_eval", {}).get("score"),
                        "latency_seconds": item["latency_seconds"],
                        "retrieval_latency_seconds": item.get("retrieval_latency_seconds"),
                        "llm_latency_seconds": item.get("llm_latency_seconds"),
                        "error_count": len(item.get("errors", [])),
                    }
                )

        return json_path, csv_path

    @staticmethod
    def print_summary(summary: dict[str, Any], json_path: Path, csv_path: Path) -> None:
        """Print a formatted summary of the evaluation run to stdout."""
        print("\n" + "=" * 80)
        print("DEEPEVAL SUMMARY FOR HYBRID SEARCH AGENT")
        print("=" * 80)
        print(f"Cases evaluated      : {summary['case_count']}")
        print(f"Pass rate            : {summary['pass_count']}/{summary['case_count']} ({summary['pass_rate']:.1%})")
        print(f"Avg case score       : {summary['avg_case_score']}/100")
        print(f"Avg source hit rate  : {summary['avg_source_hit_rate']}")
        print(f"Avg metadata match   : {summary['avg_metadata_match_ratio']}")
        print(f"Avg MRR              : {summary['avg_mrr']}")
        print(f"Avg Precision@k      : {summary['avg_precision_at_k']}")
        print(f"Avg Recall@k         : {summary['avg_recall_at_k']}")
        print(f"Avg NDCG@k           : {summary['avg_ndcg_at_k']}")
        print(f"Avg latency          : {summary['avg_latency_seconds']}s")
        print(f"  Retrieval          : {summary['avg_retrieval_latency_seconds']}s")
        print(f"  LLM                : {summary['avg_llm_latency_seconds']}s")
        print(f"Judge model          : {summary['judge_model']}")
        print("Metric averages      :")
        for name, score in summary["metric_averages"].items():
            print(f"  - {name}: {score}")
        print(f"JSON report          : {json_path}")
        print(f"CSV summary          : {csv_path}")
