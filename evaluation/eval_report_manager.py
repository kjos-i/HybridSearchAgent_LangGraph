"""Report generation and on-disk persistence for the evaluation flow.

Defines ReportManager, which builds the run-level summary dict from
per-case results, writes timestamped JSON and CSV artifacts to the
evaluation_results/ directory, and prints a tabular summary to
stdout.  Every column name, label, and storage precision is sourced from
eval_metric_registry so adding or renaming a metric there flows
through every output format here without parallel edits.

Module layout: public ReportManager class first, then the private
_build_csv_row helper at the bottom — top-down style.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from eval_metric_registry import (
    csv_fieldnames,
    llm_metric_keys,
    metric_decimals,
    metric_fmts,
    metric_labels,
    summary_avg_pairs,
)
from eval_utils import safe_mean

# Storage-precision and display-format lookups, resolved once at import time
# so the per-metric round() precision and printed format stay registry-
# driven without re-computing the dicts on every report build.
_DECIMALS: dict[str, int] = metric_decimals()
_FMTS:     dict[str, str] = metric_fmts()


class ReportManager:
    """Build run summaries and write JSON / CSV artifacts for one eval run.

    Stateless apart from the output directory — one instance can build
    multiple reports across runs, and the same instance can be reused
    by callers that drive the harness from a notebook.
    """

    def __init__(self, output_dir: Path) -> None:
        """Capture the directory that save_report will write into.

        The directory is not created here; save_report calls
        mkdir(parents=True, exist_ok=True) lazily so a ReportManager
        constructed in a dry-run path doesn't leave an empty folder behind.
        """
        self.output_dir = Path(output_dir)

    def build_summary(
        self,
        results: list[dict[str, Any]],
        judge_model: str,
        enabled_groups: set[str] | None = None,
    ) -> dict[str, Any]:
        """Aggregate per-case results into a single run-level summary dict.

        Computes the run-level pass rate, every per-metric average pulled
        from the registry's summary_avg_pairs, and a bag of judge
        metric averages keyed by name (the JSON metric_averages blob
        the dashboard reads for its judge KPI cards).  Each average
        filters None scores so cases where the metric didn't run
        (judge group disabled, fixture-only metric in live mode) don't
        poison the mean.  enabled_groups is recorded on the summary
        so downstream consumers (ledger, dashboard) can render disabled
        metrics as "Not evaluated" instead of misleading zeros.
        """
        metric_names = sorted({name for item in results for name in item["metrics"].keys()})
        metric_averages = {}

        for name in metric_names:
            scores = [
                item["metrics"][name]["score"]
                for item in results
                if item["metrics"].get(name, {}).get("score") is not None
            ]
            # Per-metric storage precision flows from the registry — falls
            # back to safe_mean's default when a name lookup misses
            # (shouldn't happen for registered metrics, but keeps this
            # safe if a stray key arrives in item["metrics"]).
            metric_averages[name] = safe_mean(scores, decimals=_DECIMALS.get(name, 3))

        pass_count = sum(1 for item in results if item["status"] == "PASS")

        summary: dict[str, Any] = {
            "case_count": len(results),
            "pass_count": pass_count,
            "pass_rate": round(pass_count / max(len(results), 1), _DECIMALS["pass_rate"]),
        }

        # Per-metric run-level averages — driven by the registry.  None
        # entries are filtered before averaging so cases where the source
        # metric didn't run (e.g. judge panel disabled, fixture-only) don't
        # poison the mean.  avg_judge_run_score is included here via the
        # avg_judge_score MetricDef's summary_avg_key — no special-
        # casing needed at this level.
        for avg_key, src_key in summary_avg_pairs():
            scores = [item[src_key] for item in results if item.get(src_key) is not None]
            summary[avg_key] = (
                safe_mean(scores, decimals=_DECIMALS.get(avg_key, 3))
                if scores
                else None
            )

        summary["judge_model"] = judge_model
        summary["enabled_groups"] = sorted(enabled_groups) if enabled_groups is not None else None
        summary["metric_averages"] = metric_averages

        return summary

    def save_report(self, report: dict[str, Any]) -> tuple[Path, Path]:
        """Write timestamped JSON and CSV artifacts; return both paths.

        The JSON is the full structured report (summary + per-case
        results, errors, retrieval previews); the CSV is a flat per-case
        table with one row per case and registry-derived columns, useful
        for spreadsheet-driven inspection.  Filenames are timestamped
        (deepeval_report_<YYYYmmdd_HHMMSS>.json and
        deepeval_summary_<...>.csv) so successive runs accumulate
        side-by-side instead of overwriting one another.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"deepeval_report_{timestamp}.json"
        csv_path = self.output_dir / f"deepeval_summary_{timestamp}.csv"

        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        fieldnames = csv_fieldnames()
        llm_keys = set(llm_metric_keys())

        # extrasaction="ignore" lets _build_csv_row return extra
        # keys without raising — only fields listed in fieldnames
        # are written, in that order.
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for item in report.get("results", []):
                writer.writerow(_build_csv_row(item, fieldnames, llm_keys))

        return json_path, csv_path

    @staticmethod
    def print_summary(summary: dict[str, Any], json_path: Path, csv_path: Path) -> None:
        """Print a tabular run summary to stdout for the CLI invocation.

        Mirrors what the dashboard shows (pass rate, enabled groups,
        judge model, registry-derived per-metric averages) so a one-shot
        CLI run gives the user the same headline numbers without
        spinning up Streamlit.  None averages render as
        "Not evaluated" to keep the meaning explicit.
        """
        not_evaluated = "Not evaluated"

        def _fmt(value: Any, suffix: str = "") -> str:
            return not_evaluated if value is None else f"{value}{suffix}"

        print("\n" + "=" * 80)
        print("DEEPEVAL SUMMARY FOR HYBRID SEARCH AGENT")
        print("=" * 80)
        print(f"Cases evaluated      : {summary['case_count']}")
        _pass_fmt = _FMTS.get("pass_rate", "")
        print(
            f"Pass rate            : {summary['pass_count']}/{summary['case_count']} "
            f"({summary['pass_rate']:{_pass_fmt}})"
        )

        enabled = summary.get("enabled_groups")
        if enabled is not None:
            print(f"Enabled metric groups: {', '.join(enabled) if enabled else '(none)'}")

        # Run-level averages — labels come from the registry so every
        # consumer (dashboard, CSV, print) shows the same display name.
        labels = metric_labels()
        for avg_key, _ in summary_avg_pairs():
            if avg_key in summary:
                label = labels.get(avg_key, avg_key)
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Backend keys that roll up into the backend_vector CSV column — kept
# in sync with eval_sqlite._VECTOR_BACKEND_KEYS so SQL and CSV agree on
# how to bucket vector-method labels (vector_similarity / vector_mmr).
_VECTOR_BACKEND_KEYS: frozenset[str] = frozenset({"vector", "vector_similarity", "vector_mmr"})
_KNOWN_BACKEND_KEYS: frozenset[str] = frozenset({"fts", "hybrid"}) | _VECTOR_BACKEND_KEYS


def _build_csv_row(
    item: dict[str, Any],
    fieldnames: list[str],
    llm_keys: set[str],
) -> dict[str, Any]:
    """Flatten one per-case result dict into a single CSV row.

    Driving the row from the registry-derived fieldnames (instead
    of a hardcoded literal) means adding a metric in
    eval_metric_registry flows into the CSV with no edits here.

    Four field shapes are special-cased because they don't live as
    flat scalars on the result dict:

    - LLM judge scores nest under metrics[name]["score"].
    - backend_distribution sub-fields (the backend_* columns)
      live under backend_distribution[<key without backend_ prefix>].
      backend_other is a catch-all for any unrecognized backend label
      so a future backend can never silently disappear from the CSV.
    - keyword_checks sub-fields live under keyword_checks[name].
    - error_count is derived from the length of the errors list.

    Everything else (token counts, latency, scalar retrieval metrics,
    avg_judge_score, structural id/category/status) is a top-level
    scalar pulled via item.get(field).
    """
    metrics = item.get("metrics", {})
    bd = item.get("backend_distribution", {})
    kw = item.get("keyword_checks", {})
    row: dict[str, Any] = {}
    for field in fieldnames:
        if field in llm_keys:
            row[field] = metrics.get(field, {}).get("score")
        elif field == "backend_fts":
            row[field] = bd.get("fts", 0)
        elif field == "backend_vector":
            row[field] = sum(bd.get(key, 0) for key in _VECTOR_BACKEND_KEYS)
        elif field == "backend_hybrid":
            row[field] = bd.get("hybrid", 0)
        elif field == "backend_other":
            row[field] = sum(count for key, count in bd.items() if key not in _KNOWN_BACKEND_KEYS)
        elif field in ("required_keyword_hit_rate", "disallowed_keyword_hits"):
            row[field] = kw.get(field)
        elif field == "error_count":
            row[field] = len(item.get("errors", []))
        elif field == "id":
            row[field] = item["id"]
        else:
            row[field] = item.get(field)
    return row
