"""
DeepEval-based evaluation harness for `hybrid_search_agent.py`.

Features:
- Runs the real LangGraph hybrid-search agent.
- Evaluates retrieved context and final answers with DeepEval RAG metrics.
- Saves timestamped JSON and CSV reports under `evaluation_results/`.

Requirements:
    pip install deepeval

Usage (run from the project root):
    python evaluation/eval_deepeval.py

Configuration:
    Edit eval_config.py to change the judge model, threshold, dataset path, etc.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

from eval_config import (
    BASE_DIR,
    CONCURRENCY,
    DATASET_PATH,
    JUDGE_MODEL,
    MAX_CASES,
    OUTPUT_DIR,
    THRESHOLD,
)
from eval_engine import EvaluationEngine
from eval_report_manager import ReportManager
from eval_sqlite import EvalLedger
from eval_utils import load_cases

# hybrid_search_agent.py and its dependencies live in the project root (BASE_DIR).
# sys.path gets the project root so the import resolves, and chdir ensures that
# relative paths inside hybrid_search_agent.py (e.g. ./chroma_db) also resolve correctly.
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)
from hybrid_search_agent import agent, retriever  # noqa: E402


async def main() -> None:
    """Load config, run all eval cases, and write the JSON and CSV reports."""
    load_dotenv("C:/Users/kjosi/dotenv/.env")

    cases = load_cases(DATASET_PATH)
    if MAX_CASES:
        cases = cases[:MAX_CASES]

    print(f"\nDeepEval is running ({len(cases)} case(s), judge={JUDGE_MODEL}), please wait...")

    eval_engine = EvaluationEngine(
        agent=agent,
        retriever=retriever,
        judge_model=JUDGE_MODEL,
        threshold=THRESHOLD,
    )
    report_manager = ReportManager(OUTPUT_DIR)

    results = await eval_engine.evaluate_cases(cases, concurrency=CONCURRENCY)
    summary = report_manager.build_summary(results, judge_model=JUDGE_MODEL)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(DATASET_PATH),
        "judge_model": JUDGE_MODEL,
        "threshold": THRESHOLD,
        "summary": summary,
        "results": results,
    }

    json_path, csv_path = report_manager.save_report(report)

    ledger = EvalLedger()
    run_id = ledger.save_run(report)
    print(f"SQLite ledger        : {ledger.db_path}  (run_id={run_id})")

    report_manager.print_summary(summary, json_path, csv_path)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    asyncio.run(main())
