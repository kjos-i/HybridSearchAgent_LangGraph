"""Report generation and persistence for the DeepEval-based hybrid search evaluation flow."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from eval_metric_registry import (
    csv_fieldnames,
    llm_metric_keys,
    summary_avg_pairs,
)
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

        summary: dict[str, Any] = {
            "case_count": len(results),
            "pass_count": pass_count,
            "pass_rate": round(pass_count / max(len(results), 1), 3),
            "avg_case_score": safe_mean([item["avg_judge_score"] for item in results]),
        }

        # Per-metric run-level averages — driven by the registry.
        for avg_key, src_key in summary_avg_pairs():
            summary[avg_key] = safe_mean([item[src_key] for item in results])

        summary["judge_model"] = judge_model
        summary["metric_averages"] = metric_averages

        return summary

    def save_report(self, report: dict[str, Any]) -> tuple[Path, Path]:
        """Write timestamped JSON and CSV reports to the output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"deepeval_report_{timestamp}.json"
        csv_path = self.output_dir / f"deepeval_summary_{timestamp}.csv"

        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        fieldnames = csv_fieldnames()
        llm_keys = llm_metric_keys()

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            for item in report["results"]:
                metrics = item.get("metrics", {})
                kw = item.get("keyword_checks", {})
                bd = item.get("backend_distribution", {})

                row: dict[str, Any] = {
                    "id": item["id"],
                    "category": item.get("category", ""),
                    "status": item["status"],
                    # Scalar metrics — directly from the result dict.
                    "avg_judge_score": item.get("avg_judge_score"),
                    "source_hit_rate": item.get("source_hit_rate"),
                    "metadata_match_ratio": item.get("metadata_match_ratio"),
                    "mrr": item.get("mrr"),
                    "precision_at_k": item.get("precision_at_k"),
                    "recall_at_k": item.get("recall_at_k"),
                    "ndcg_at_k": item.get("ndcg_at_k"),
                    # Composite: backend_distribution
                    "backend_fts": bd.get("fts", 0),
                    "backend_vector": bd.get("vector", 0),
                    "backend_hybrid": bd.get("hybrid", 0),
                    # Composite: keyword_checks
                    "required_keyword_hit_rate": kw.get("required_keyword_hit_rate"),
                    "disallowed_keyword_hits": kw.get("disallowed_keyword_hits"),
                    # Latency
                    "latency_seconds": item.get("latency_seconds"),
                    "retrieval_latency_seconds": item.get("retrieval_latency_seconds"),
                    "llm_latency_seconds": item.get("llm_latency_seconds"),
                    # Errors
                    "error_count": len(item.get("errors", [])),
                }

                # LLM-judged scores — driven by the registry.
                for key in llm_keys:
                    row[key] = metrics.get(key, {}).get("score")

                writer.writerow(row)

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

        for avg_key, _ in summary_avg_pairs():
            if avg_key in summary:
                label = avg_key.replace("avg_", "Avg ").replace("_", " ").title()
                print(f"{label:21s}: {summary[avg_key]}")

        print(f"Judge model          : {summary['judge_model']}")
        print("Metric averages      :")
        for name, score in summary["metric_averages"].items():
            print(f"  - {name}: {score}")
        print(f"JSON report          : {json_path}")
        print(f"CSV summary          : {csv_path}")
