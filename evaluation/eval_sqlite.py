"""SQLite ledger for persisting evaluation results across runs.

Creates and manages ``eval_ledger.db`` in the evaluation folder with two tables:
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

from eval_metric_registry import case_sql_columns, llm_metric_keys, run_sql_columns

DB_PATH = Path(__file__).resolve().parent / "eval_ledger.db"

# ---------------------------------------------------------------------------
# Schema — generated from the metric registry
# ---------------------------------------------------------------------------

_RUNS_FIXED_PREFIX = """\
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp       TEXT    NOT NULL,
    dataset_path        TEXT,
    judge_model         TEXT,
    threshold           REAL,
    case_count          INTEGER,
    pass_count          INTEGER,
    pass_rate           REAL,
    avg_case_score      REAL,"""

_RUNS_FIXED_SUFFIX = """\
    enabled_groups      TEXT,
    gate_thresholds     TEXT,
    metric_averages     TEXT
);"""

_CASES_FIXED_PREFIX = """\
CREATE TABLE IF NOT EXISTS eval_cases (
    case_row_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                      INTEGER NOT NULL REFERENCES eval_runs(run_id),
    run_timestamp               TEXT    NOT NULL,
    case_id                     TEXT    NOT NULL,
    category                    TEXT    DEFAULT '',
    question                    TEXT,
    status                      TEXT,"""

_CASES_FIXED_SUFFIX = """\
    error_count                 INTEGER,
    answer                      TEXT,
    expected_output             TEXT,
    retrieval_config            TEXT,
    errors                      TEXT
);"""


def _build_create_sql(prefix: str, columns: list[tuple[str, str]], suffix: str) -> str:
    col_lines = "\n".join(f"    {name:<36s}{typ}," for name, typ in columns)
    return f"{prefix}\n{col_lines}\n{suffix}"


_CREATE_RUNS_TABLE = _build_create_sql(_RUNS_FIXED_PREFIX, run_sql_columns(), _RUNS_FIXED_SUFFIX)
_CREATE_CASES_TABLE = _build_create_sql(_CASES_FIXED_PREFIX, case_sql_columns(), _CASES_FIXED_SUFFIX)


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
            # Auto-migrate: ensure every registry-defined column exists.
            for col, typ in case_sql_columns():
                try:
                    conn.execute(f"ALTER TABLE eval_cases ADD COLUMN {col} {typ}")
                except sqlite3.OperationalError:
                    pass  # Column already exists
            for col, typ in run_sql_columns():
                try:
                    conn.execute(f"ALTER TABLE eval_runs ADD COLUMN {col} {typ}")
                except sqlite3.OperationalError:
                    pass
            # Legacy migration: category column on older databases.
            try:
                conn.execute("ALTER TABLE eval_cases ADD COLUMN category TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            # Migration: enabled_groups column on older databases.
            try:
                conn.execute("ALTER TABLE eval_runs ADD COLUMN enabled_groups TEXT")
            except sqlite3.OperationalError:
                pass
            # Migration: gate_thresholds column on older databases.
            try:
                conn.execute("ALTER TABLE eval_runs ADD COLUMN gate_thresholds TEXT")
            except sqlite3.OperationalError:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # Runs are only persisted when every toggleable metric group ran, so trend
    # charts in the dashboard compare like-with-like. Partial runs still land
    # in the JSON/CSV artifacts for debugging.
    FULL_METRIC_GROUPS: frozenset[str] = frozenset({"judge", "source", "chunk"})

    def save_run(self, report: dict[str, Any]) -> int | None:
        """Persist a full evaluation report (summary + per-case results).

        Only runs where every toggleable metric group was enabled are persisted.
        Partial runs are skipped so that ledger-backed trend charts stay
        comparable across runs. JSON/CSV reports still capture every run.

        Parameters
        ----------
        report : dict
            The same report dict that is written to the JSON file, containing
            keys ``generated_at``, ``summary``, ``results``, etc.

        Returns
        -------
        int or None
            The ``run_id`` assigned by SQLite for this evaluation run, or
            ``None`` if the run was skipped because not all metric groups
            were enabled.
        """
        summary = report.get("summary", {})

        enabled_groups_raw = summary.get("enabled_groups") or []
        if set(enabled_groups_raw) != self.FULL_METRIC_GROUPS:
            return None

        # ── Build run-level INSERT dynamically ────────────────────────
        run_fixed_cols = [
            "run_timestamp", "dataset_path", "judge_model", "threshold",
            "case_count", "pass_count", "pass_rate", "avg_case_score",
        ]
        run_avg_cols = [col for col, _ in run_sql_columns()]
        run_trailing_cols = ["enabled_groups", "gate_thresholds", "metric_averages"]

        run_cols = run_fixed_cols + run_avg_cols + run_trailing_cols
        run_placeholders = ", ".join("?" for _ in run_cols)
        run_col_str = ", ".join(run_cols)

        enabled_groups_value: str | None = (
            json.dumps(summary.get("enabled_groups"))
            if summary.get("enabled_groups") is not None
            else None
        )

        gate_thresholds_value: str | None = (
            json.dumps(report.get("gate_thresholds"))
            if report.get("gate_thresholds") is not None
            else None
        )

        run_values = (
            report.get("generated_at"),
            report.get("dataset_path"),
            report.get("judge_model"),
            report.get("threshold"),
            summary.get("case_count"),
            summary.get("pass_count"),
            summary.get("pass_rate"),
            summary.get("avg_case_score"),
            *[summary.get(col) for col in run_avg_cols],
            enabled_groups_value,
            gate_thresholds_value,
            json.dumps(summary.get("metric_averages", {})),
        )

        with self._connect() as conn:
            cursor = conn.execute(
                f"INSERT INTO eval_runs ({run_col_str}) VALUES ({run_placeholders})",
                run_values,
            )
            run_id: int = cursor.lastrowid  # type: ignore[assignment]

            # ── Build case-level INSERT dynamically ───────────────────
            case_fixed_cols = [
                "run_id", "run_timestamp", "case_id", "category",
                "question", "status",
            ]
            case_metric_cols = [col for col, _ in case_sql_columns()]
            case_trailing_cols = [
                "error_count", "answer", "expected_output",
                "retrieval_config", "errors",
            ]
            case_cols = case_fixed_cols + case_metric_cols + case_trailing_cols
            case_placeholders = ", ".join("?" for _ in case_cols)
            case_col_str = ", ".join(case_cols)

            llm_keys = llm_metric_keys()

            for item in report.get("results", []):
                metrics = item.get("metrics", {})
                kw = item.get("keyword_checks", {})
                bd = item.get("backend_distribution", {})

                metric_values = _extract_case_metric_values(item, metrics, kw, bd, llm_keys)

                case_values = (
                    run_id,
                    report.get("generated_at"),
                    item.get("id"),
                    item.get("category", ""),
                    item.get("question"),
                    item.get("status"),
                    *metric_values,
                    len(item.get("errors", [])),
                    item.get("answer"),
                    item.get("expected_output"),
                    json.dumps(item.get("retrieval_config", {})),
                    json.dumps(item.get("errors", [])),
                )

                conn.execute(
                    f"INSERT INTO eval_cases ({case_col_str}) VALUES ({case_placeholders})",
                    case_values,
                )

        return run_id


def _extract_case_metric_values(
    item: dict[str, Any],
    metrics: dict[str, Any],
    kw: dict[str, Any],
    bd: dict[str, Any],
    llm_keys: list[str],
) -> tuple[Any, ...]:
    """Extract metric values in the same order as ``case_sql_columns()``.

    Handles scalar metrics (read directly from ``item``), composite metrics
    (exploded from ``bd`` / ``kw``), and LLM-judged metrics (nested under
    ``metrics[key]["score"]``).
    """
    values: list[Any] = []
    for col, _ in case_sql_columns():
        if col in llm_keys:
            values.append(metrics.get(col, {}).get("score"))
        elif col == "backend_fts":
            values.append(bd.get("fts", 0))
        elif col == "backend_vector":
            values.append(
                bd.get("vector", 0) + bd.get("vector_similarity", 0) + bd.get("vector_mmr", 0)
            )
        elif col == "backend_hybrid":
            values.append(bd.get("hybrid", 0))
        elif col == "required_keyword_hit_rate":
            values.append(kw.get("required_keyword_hit_rate"))
        elif col == "disallowed_keyword_hits":
            values.append(kw.get("disallowed_keyword_hits"))
        elif col == "avg_judge_score":
            values.append(item.get("avg_judge_score"))
        else:
            values.append(item.get(col))
    return tuple(values)
