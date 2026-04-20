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

    def build_summary(
        self,
        results: list[dict[str, Any]],
        judge_model: str,
        enabled_groups: set[str] | None = None,
    ) -> dict[str, Any]:
        """Aggregate per-case results into a top-level summary dict.

        ``enabled_groups`` is stored on the summary so downstream consumers
        (ledger, dashboard) know which metric groups were toggled on for this
        run. Disabled metrics show up as ``None`` averages.
        """
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

        # avg_case_score: skip cases whose judge panel didn't run (score is None).
        case_scores = [item["avg_judge_score"] for item in results if item.get("avg_judge_score") is not None]

        summary: dict[str, Any] = {
            "case_count": len(results),
            "pass_count": pass_count,
            "pass_rate": round(pass_count / max(len(results), 1), 3),
            "avg_case_score": safe_mean(case_scores) if case_scores else None,
        }

        # Per-metric run-level averages — driven by the registry. None entries
        # are filtered before averaging so disabled metrics don't poison the mean.
        for avg_key, src_key in summary_avg_pairs():
            scores = [item[src_key] for item in results if item.get(src_key) is not None]
            summary[avg_key] = safe_mean(scores) if scores else None

        summary["judge_model"] = judge_model
        summary["enabled_groups"] = sorted(enabled_groups) if enabled_groups is not None else None
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
                    "hit_at_k": item.get("hit_at_k"),
                    "metadata_match_ratio": item.get("metadata_match_ratio"),
                    "mrr": item.get("mrr"),
                    "precision_at_k": item.get("precision_at_k"),
                    "recall_at_k": item.get("recall_at_k"),
                    "ndcg_at_k": item.get("ndcg_at_k"),
                    # Chunk-level retrieval metrics
                    "chunk_hit_at_k": item.get("chunk_hit_at_k"),
                    "chunk_mrr": item.get("chunk_mrr"),
                    "chunk_precision_at_k": item.get("chunk_precision_at_k"),
                    "chunk_recall_at_k": item.get("chunk_recall_at_k"),
                    "chunk_ndcg_at_k": item.get("chunk_ndcg_at_k"),
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
        not_evaluated = "Not evaluated"

        def _fmt(value: Any, suffix: str = "") -> str:
            return not_evaluated if value is None else f"{value}{suffix}"

        print("\n" + "=" * 80)
        print("DEEPEVAL SUMMARY FOR HYBRID SEARCH AGENT")
        print("=" * 80)
        print(f"Cases evaluated      : {summary['case_count']}")
        print(f"Pass rate            : {summary['pass_count']}/{summary['case_count']} ({summary['pass_rate']:.1%})")
        print(f"Avg judge score      : {_fmt(summary.get('avg_case_score'), '/100')}")

        enabled = summary.get("enabled_groups")
        if enabled is not None:
            print(f"Enabled metric groups: {', '.join(enabled) if enabled else '(none)'}")

        for avg_key, _ in summary_avg_pairs():
            if avg_key in summary:
                label = avg_key.replace("avg_", "Avg ").replace("_", " ").title()
                print(f"{label:21s}: {_fmt(summary[avg_key])}")

        print(f"Judge model          : {summary['judge_model']}")
        print("Metric averages      :")
        metric_averages = summary.get("metric_averages") or {}
        if metric_averages:
            for name, score in metric_averages.items():
                print(f"  - {name}: {_fmt(score)}")
        else:
            print(f"  {not_evaluated}")
        print(f"JSON report          : {json_path}")
        print(f"CSV summary          : {csv_path}")
