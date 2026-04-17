"""SQLite ledger for persisting evaluation results across runs.

Creates and manages `eval_ledger.db` in the evaluation folder with two tables:
  - eval_runs:    one row per evaluation run (summary-level metrics).
  - eval_cases:   one row per case per run (per-case metrics and details).

Usage:
    from eval_sqlite import EvalLedger
    ledger = EvalLedger()
    ledger.save_run(report)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "eval_ledger.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp       TEXT    NOT NULL,
    dataset_path        TEXT,
    judge_model         TEXT,
    threshold           REAL,
    case_count          INTEGER,
    pass_count          INTEGER,
    pass_rate           REAL,
    avg_case_score      REAL,
    avg_source_hit_rate REAL,
    avg_metadata_match_ratio REAL,
    avg_mrr             REAL,
    avg_precision_at_k  REAL,
    avg_recall_at_k     REAL,
    avg_ndcg_at_k       REAL,
    avg_latency_seconds REAL,
    avg_retrieval_latency_seconds REAL,
    avg_llm_latency_seconds       REAL,
    metric_averages     TEXT
);
"""

_CREATE_CASES_TABLE = """
CREATE TABLE IF NOT EXISTS eval_cases (
    case_row_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                      INTEGER NOT NULL REFERENCES eval_runs(run_id),
    run_timestamp               TEXT    NOT NULL,
    case_id                     TEXT    NOT NULL,
    category                    TEXT    DEFAULT '',
    question                    TEXT,
    status                      TEXT,
    avg_metric_score            REAL,
    source_hit_rate             REAL,
    metadata_match_ratio        REAL,
    mrr                         REAL,
    precision_at_k              REAL,
    recall_at_k                 REAL,
    ndcg_at_k                   REAL,
    backend_fts                 INTEGER,
    backend_vector              INTEGER,
    backend_hybrid              INTEGER,
    required_keyword_hit_rate   REAL,
    disallowed_keyword_hits     INTEGER,
    answer_relevancy            REAL,
    faithfulness                REAL,
    contextual_precision        REAL,
    contextual_recall           REAL,
    contextual_relevancy        REAL,
    hallucination               REAL,
    correctness_g_eval          REAL,
    latency_seconds             REAL,
    retrieval_latency_seconds   REAL,
    llm_latency_seconds         REAL,
    error_count                 INTEGER,
    answer                      TEXT,
    expected_output             TEXT,
    retrieval_config            TEXT,
    errors                      TEXT
);
"""


# ---------------------------------------------------------------------------
# Ledger class
# ---------------------------------------------------------------------------

class EvalLedger:
    """Thin wrapper around a SQLite database that stores evaluation runs and per-case results."""

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_tables(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_RUNS_TABLE)
            conn.execute(_CREATE_CASES_TABLE)
            # Migrate: add category column to existing databases.
            try:
                conn.execute("ALTER TABLE eval_cases ADD COLUMN category TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass  # Column already exists

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_run(self, report: dict[str, Any]) -> int:
        """Persist a full evaluation report (summary + per-case results).

        Parameters
        ----------
        report : dict
            The same report dict that is written to the JSON file, containing
            keys ``generated_at``, ``summary``, ``results``, etc.

        Returns
        -------
        int
            The ``run_id`` assigned by SQLite for this evaluation run.
        """
        summary = report.get("summary", {})

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO eval_runs (
                    run_timestamp, 
                    dataset_path, 
                    judge_model,
                    threshold,
                    case_count, 
                    pass_count, 
                    pass_rate, 
                    avg_case_score,
                    avg_source_hit_rate, 
                    avg_metadata_match_ratio,
                    avg_mrr, 
                    avg_precision_at_k, 
                    avg_recall_at_k, 
                    avg_ndcg_at_k,
                    avg_latency_seconds, 
                    avg_retrieval_latency_seconds, 
                    avg_llm_latency_seconds,
                    metric_averages
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.get("generated_at"),
                    report.get("dataset_path"),
                    report.get("judge_model"),
                    report.get("threshold"),
                    summary.get("case_count"),
                    summary.get("pass_count"),
                    summary.get("pass_rate"),
                    summary.get("avg_case_score"),
                    summary.get("avg_source_hit_rate"),
                    summary.get("avg_metadata_match_ratio"),
                    summary.get("avg_mrr"),
                    summary.get("avg_precision_at_k"),
                    summary.get("avg_recall_at_k"),
                    summary.get("avg_ndcg_at_k"),
                    summary.get("avg_latency_seconds"),
                    summary.get("avg_retrieval_latency_seconds"),
                    summary.get("avg_llm_latency_seconds"),
                    json.dumps(summary.get("metric_averages", {})),
                ),
            )
            run_id: int = cursor.lastrowid  # type: ignore[assignment]

            for item in report.get("results", []):
                metrics = item.get("metrics", {})
                kw = item.get("keyword_checks", {})
                bd = item.get("backend_distribution", {})

                conn.execute(
                    """
                    INSERT INTO eval_cases (
                        run_id,
                        run_timestamp,
                        case_id,
                        category,
                        question,
                        status,
                        avg_metric_score,
                        source_hit_rate, 
                        metadata_match_ratio, 
                        mrr, 
                        precision_at_k, 
                        recall_at_k, 
                        ndcg_at_k,
                        backend_fts, 
                        backend_vector, 
                        backend_hybrid,
                        required_keyword_hit_rate, 
                        disallowed_keyword_hits,
                        answer_relevancy, 
                        faithfulness,
                        contextual_precision, 
                        contextual_recall, 
                        contextual_relevancy,
                        hallucination, 
                        correctness_g_eval,
                        latency_seconds, 
                        retrieval_latency_seconds, 
                        llm_latency_seconds,
                        error_count, 
                        answer, 
                        expected_output, 
                        retrieval_config, 
                        errors
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        report.get("generated_at"),
                        item.get("id"),
                        item.get("category", ""),
                        item.get("question"),
                        item.get("status"),
                        item.get("average_metric_score"),
                        item.get("source_hit_rate"),
                        item.get("metadata_match_ratio"),
                        item.get("mrr"),
                        item.get("precision_at_k"),
                        item.get("recall_at_k"),
                        item.get("ndcg_at_k"),
                        bd.get("fts", 0),
                        bd.get("vector", 0) + bd.get("vector_similarity", 0) + bd.get("vector_mmr", 0),
                        bd.get("hybrid", 0),
                        kw.get("required_keyword_hit_rate"),
                        kw.get("disallowed_keyword_hits"),
                        metrics.get("answer_relevancy", {}).get("score"),
                        metrics.get("faithfulness", {}).get("score"),
                        metrics.get("contextual_precision", {}).get("score"),
                        metrics.get("contextual_recall", {}).get("score"),
                        metrics.get("contextual_relevancy", {}).get("score"),
                        metrics.get("hallucination", {}).get("score"),
                        metrics.get("correctness_g_eval", {}).get("score"),
                        item.get("latency_seconds"),
                        item.get("retrieval_latency_seconds"),
                        item.get("llm_latency_seconds"),
                        len(item.get("errors", [])),
                        item.get("answer"),
                        item.get("expected_output"),
                        json.dumps(item.get("retrieval_config", {})),
                        json.dumps(item.get("errors", [])),
                    ),
                )

        return run_id
