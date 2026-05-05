"""SQLite ledger for persisting evaluation results across runs.

Creates and manages eval_ledger.db in the evaluation folder with two tables:
  - eval_runs:    one row per evaluation run (summary-level metrics).
  - eval_cases:   one row per case per run (per-case metrics and details).

Schema is generated automatically from the metric registry, so adding a metric
to eval_metric_registry.py automatically extends both tables on next startup
via the auto-migration block in _ensure_tables().

Schema policy
-------------
The auto-migration is **add-only**. _try_add_column will append new
columns when a metric is added to the registry, but it does not rename
or drop columns when one is removed or renamed. By design — this harness
treats the ledger as disposable trend data, not a system of record.

When a registry change renames or removes a metric column, the
recommended fix is to **save the existing ledger aside** under a
dated name so the historical rows stay browsable, and let the next
run create a fresh DB.  Outright wiping (rm) also works if the
history is not worth keeping.  The alternative (compatibility shims
for old column names, manual ALTER TABLE scripts) would only
complicate the schema for trend data the harness already treats as
disposable.  Save-aside (preferred)::

    mv evaluation/eval_ledger.db evaluation/eval_ledger_<YYYYMMDD>.db

Wipe (only if archived rows are not worth keeping)::

    rm evaluation/eval_ledger.db

A future "Possible improvement" in info_considerations.md sketches a
multi-ledger dashboard browser that would expose archived
eval_ledger_*.db files in a sidebar dropdown, which is what makes
the save-aside path strictly better than rm.

Module layout: imports → migration helper → SQL templates → EvalLedger
class → all the private internals (column lists, drift assertions, insert
helpers, value extractor) at the bottom. 
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from eval_metric_registry import (
    METRICS,
    case_sql_columns,
    llm_metric_keys,
    run_sql_columns,
)

DB_PATH = Path(__file__).resolve().parent / "eval_ledger.db"


# ---------------------------------------------------------------------------
# Migration helper
# ---------------------------------------------------------------------------

def _try_add_column(conn: sqlite3.Connection, table: str, col_def: str) -> None:
    """ALTER TABLE ... ADD COLUMN, swallowing only the 'already exists' error.

    SQLite reports duplicate columns as OperationalError with a message
    starting "duplicate column name". Any other OperationalError
    (malformed SQL, locked DB, disk full, permission issue) is re-raised so
    a real schema problem can't silently corrupt later writes. The traceback
    reaches stderr via Python's default exception handler.
    """
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
    except sqlite3.OperationalError as exc:
        if "duplicate column name" in str(exc).lower():
            return
        raise


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
    pass_rate           REAL,"""

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
    """Format a CREATE TABLE statement with metric columns aligned at character 37.

    If your column name is mrr (3 letters), this adds 33 spaces after it so
    the data type (e.g. REAL) starts exactly at the 37th character.
    """
    column_lines = "\n".join(f"    {name:<36s}{datatype}," for name, datatype in columns)
    return f"{prefix}\n{column_lines}\n{suffix}"


_CREATE_RUNS_TABLE = _build_create_sql(_RUNS_FIXED_PREFIX, run_sql_columns(), _RUNS_FIXED_SUFFIX)
_CREATE_CASES_TABLE = _build_create_sql(_CASES_FIXED_PREFIX, case_sql_columns(), _CASES_FIXED_SUFFIX)


# ---------------------------------------------------------------------------
# Ledger class
# ---------------------------------------------------------------------------

class EvalLedger:
    """Thin SQLite wrapper for run-level and per-case evaluation history.

    Owns one eval_ledger.db file with two tables (eval_runs and
    eval_cases).  save_run is the only write entry point, it
    inserts one eval_runs row plus N eval_cases rows per
    invocation.  Reads happen directly in the dashboard layer.
    """

    # Runs are only persisted when every toggleable metric group ran, so trend
    # charts in the dashboard compare like-with-like. Partial runs still land
    # in the JSON/CSV artifacts for debugging.
    FULL_METRIC_GROUPS: frozenset[str] = frozenset({"judge", "source", "chunk"})

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        """Open or create the ledger at db_path and ensure the schema is current.

        Calls _ensure_tables so any newly-registered metric
        column is added via _try_add_column before the run writes.
        """
        self.db_path = Path(db_path)
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        """Open a fresh sqlite3 connection (one per with block)."""
        return sqlite3.connect(self.db_path)

    def _ensure_tables(self) -> None:
        """Create the tables if absent and add any newly registered metric columns.

        The CREATE TABLE statements always run with IF NOT EXISTS, so
        first-time setup just creates the tables. The _try_add_column
        loops only matter when a metric was added to eval_metric_registry
        after the ledger DB was created, they extend an existing ledger to
        include the new columns without requiring a wipe.
        """
        with self._connect() as conn:
            conn.execute(_CREATE_RUNS_TABLE)
            conn.execute(_CREATE_CASES_TABLE)
            for col, typ in case_sql_columns():
                _try_add_column(conn, "eval_cases", f"{col} {typ}")
            for col, typ in run_sql_columns():
                _try_add_column(conn, "eval_runs", f"{col} {typ}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_run(self, report: dict[str, Any]) -> int | None:
        """Persist a full evaluation report (summary + per-case results).

        Only runs where every toggleable metric group was enabled.
        Partial runs are skipped so that ledger-backed trend charts stay
        comparable across runs. JSON/CSV reports still capture every run.

        Parameters
        ----------
        report : dict
            The same report dict that is written to the JSON file, containing
            keys generated_at, summary, results, etc.

        Returns
        -------
        int or None
            The run_id assigned by SQLite for this evaluation run, or
            None if the run was skipped because not all metric groups
            were enabled.
        """
        summary = report.get("summary", {})
        enabled_groups_raw = set(summary.get("enabled_groups") or [])
        if enabled_groups_raw != self.FULL_METRIC_GROUPS:
            return None

        run_avg_cols = [col for col, _ in run_sql_columns()]
        run_values = (
            report.get("generated_at"),
            report.get("dataset_path"),
            report.get("judge_model"),
            report.get("threshold"),
            summary.get("case_count"),
            summary.get("pass_count"),
            summary.get("pass_rate"),
            *[summary.get(col) for col in run_avg_cols],
            json.dumps(summary.get("enabled_groups")) if summary.get("enabled_groups") is not None else None,
            json.dumps(report.get("gate_thresholds")) if report.get("gate_thresholds") is not None else None,
            json.dumps(summary.get("metric_averages", {})),
        )

        with self._connect() as conn:
            run_id = _insert_run_row(conn, run_values)
            _insert_case_rows(
                conn,
                run_id=run_id,
                run_timestamp=report.get("generated_at"),
                results=report.get("results", []),
            )

        return run_id


# ---------------------------------------------------------------------------
# Fixed-column lists. Must align with the SQL prefix / suffix templates
# above. _assert_fixed_cols_align_with_sql (further down) verifies the
# alignment at import time so any drift fails loudly.
# ---------------------------------------------------------------------------

_RUN_FIXED_COLS = [
    "run_timestamp", "dataset_path", "judge_model", "threshold",
    "case_count", "pass_count", "pass_rate",
]
_RUN_TRAILING_COLS = ["enabled_groups", "gate_thresholds", "metric_averages"]

_CASE_FIXED_COLS = [
    "run_id", "run_timestamp", "case_id", "category",
    "question", "status",
]
_CASE_TRAILING_COLS = [
    "error_count", "answer", "expected_output",
    "retrieval_config", "errors",
]


# ---------------------------------------------------------------------------
# Drift assertions (run at import time)
# ---------------------------------------------------------------------------

def _columns_in_sql_block(sql: str) -> list[str]:
    """Extract column names declared in a CREATE TABLE SQL block.

    Used at import time to verify that the hardcoded _RUN_FIXED_COLS
    / _CASE_FIXED_COLS / *_TRAILING_COLS lists match the column
    names embedded in the SQL templates above. Without this check, a
    schema edit that touched only the SQL string would silently misalign
    every INSERT by one column.
    """
    columns: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip().rstrip(",")
        if not stripped or stripped.startswith(("CREATE", ")", "--")):
            continue
        token = stripped.split(None, 1)[0]
        # Skip table-level constraints (PRIMARY KEY, FOREIGN KEY, etc.).
        if token.isupper():
            continue
        columns.append(token)
    return columns


def _assert_fixed_cols_align_with_sql() -> None:
    """Fail loudly at import time if the SQL templates and column lists drift.

    The INSERT placeholders are built from _RUN_FIXED_COLS /
    _CASE_FIXED_COLS / *_TRAILING_COLS, while the schema itself is
    built from the SQL prefix/suffix string literals.  An edit that
    touches one but not the other would silently misalign every INSERT
    by one column, producing very confusing data corruption.  This check
    catches that at import time with a clear diff in the error message.
    """
    runs_declared = _columns_in_sql_block(_RUNS_FIXED_PREFIX) + _columns_in_sql_block(_RUNS_FIXED_SUFFIX)
    expected_runs = ["run_id"] + _RUN_FIXED_COLS + _RUN_TRAILING_COLS
    if runs_declared != expected_runs:
        raise RuntimeError(
            "eval_runs SQL prefix/suffix and _RUN_FIXED_COLS/_RUN_TRAILING_COLS drifted.\n"
            f"  in SQL:   {runs_declared}\n  in lists: {expected_runs}"
        )

    cases_declared = _columns_in_sql_block(_CASES_FIXED_PREFIX) + _columns_in_sql_block(_CASES_FIXED_SUFFIX)
    expected_cases = ["case_row_id"] + _CASE_FIXED_COLS + _CASE_TRAILING_COLS
    if cases_declared != expected_cases:
        raise RuntimeError(
            "eval_cases SQL prefix/suffix and _CASE_FIXED_COLS/_CASE_TRAILING_COLS drifted.\n"
            f"  in SQL:   {cases_declared}\n  in lists: {expected_cases}"
        )


_assert_fixed_cols_align_with_sql()


# _extract_case_metric_values and _build_csv_row (in
# eval_report_manager) special-case each composite sub-column by name.
# If the registry renames or adds a composite sub-column without updating
# those extractors, the new column writes NULL silently. The check below
# fails at import time when the registry's composite_sql_columns and the
# extractor's known set fall out of sync.
_KNOWN_COMPOSITE_COLUMNS: frozenset[str] = frozenset({
    "backend_fts", "backend_vector", "backend_hybrid", "backend_other",
    "required_keyword_hit_rate", "disallowed_keyword_hits",
})


def _assert_composite_extractors_match_registry() -> None:
    """Fail loudly at import time if the registry's composite columns drift
    from the extractors in this module."""
    registry_composite_cols: set[str] = set()
    for metric in METRICS:
        if metric.composite_sql_columns:
            for col, _ in metric.composite_sql_columns:
                registry_composite_cols.add(col)

    missing_in_extractor = registry_composite_cols - _KNOWN_COMPOSITE_COLUMNS
    extra_in_extractor = _KNOWN_COMPOSITE_COLUMNS - registry_composite_cols
    if missing_in_extractor or extra_in_extractor:
        raise RuntimeError(
            "Composite column drift between registry and eval_sqlite extractor.\n"
            f"  registry has, extractor doesn't: {sorted(missing_in_extractor)}\n"
            f"  extractor has, registry doesn't: {sorted(extra_in_extractor)}"
        )


_assert_composite_extractors_match_registry()


# ---------------------------------------------------------------------------
# INSERT helpers
# ---------------------------------------------------------------------------

def _run_columns() -> list[str]:
    """Ordered column list for an eval_runs INSERT."""
    return _RUN_FIXED_COLS + [col for col, _ in run_sql_columns()] + _RUN_TRAILING_COLS


def _case_columns() -> list[str]:
    """Ordered column list for an eval_cases INSERT."""
    return _CASE_FIXED_COLS + [col for col, _ in case_sql_columns()] + _CASE_TRAILING_COLS


def _insert_run_row(conn: sqlite3.Connection, run_values: tuple[Any, ...]) -> int:
    """Insert one eval_runs row and return the autoincremented run_id.

    run_values must be ordered to match _run_columns().  The
    returned run_id is used as the foreign key for the case rows
    written by _insert_case_rows.
    """
    cols = _run_columns()
    placeholders = ", ".join("?" for _ in cols)
    cursor = conn.execute(
        f"INSERT INTO eval_runs ({', '.join(cols)}) VALUES ({placeholders})",
        run_values,
    )
    return cursor.lastrowid  # type: ignore[return-value]


def _insert_case_rows(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    run_timestamp: str | None,
    results: list[dict[str, Any]],
) -> None:
    """Insert one eval_cases row per result, all linked to run_id.

    Composite metric values (backend_* and keyword_checks
    sub-fields) and LLM judge scores are extracted via
    _extract_case_metric_values so the column order matches
    case_sql_columns() exactly.
    """
    cols = _case_columns()
    placeholders = ", ".join("?" for _ in cols)
    llm_keys = llm_metric_keys()

    for item in results:
        metrics = item.get("metrics", {})
        kw = item.get("keyword_checks", {})
        bd = item.get("backend_distribution", {})
        errors = item.get("errors", [])

        metric_values = _extract_case_metric_values(item, metrics, kw, bd, llm_keys)

        case_values = (
            run_id,
            run_timestamp,
            item.get("id"),
            item.get("category", ""),
            item.get("question"),
            item.get("status"),
            *metric_values,
            len(errors),
            item.get("answer"),
            item.get("expected_output"),
            json.dumps(item.get("retrieval_config", {})),
            json.dumps(errors),
        )
        conn.execute(
            f"INSERT INTO eval_cases ({', '.join(cols)}) VALUES ({placeholders})",
            case_values,
        )


# ---------------------------------------------------------------------------
# Composite metric value extractor
# ---------------------------------------------------------------------------

# Backend keys that roll up into the backend_vector SQL column.  The
# retriever labels its backend by vector_search_method (similarity or
# mmr), so both labels collapse onto a single column.  Anything outside
# this set lands in backend_other rather than being silently dropped.
_VECTOR_BACKEND_KEYS: frozenset[str] = frozenset({"vector", "vector_similarity", "vector_mmr"})
_KNOWN_BACKEND_KEYS: frozenset[str] = frozenset({"fts", "hybrid"}) | _VECTOR_BACKEND_KEYS


def _extract_case_metric_values(
    item: dict[str, Any],
    metrics: dict[str, Any],
    kw: dict[str, Any],
    bd: dict[str, Any],
    llm_keys: list[str],
) -> tuple[Any, ...]:
    """Extract metric values in the same order as case_sql_columns().

    Handles scalar metrics (read directly from item), composite metrics
    (exploded from bd / kw), and LLM-judged metrics (nested under
    metrics[key]["score"]).  Unknown backends are summed into
    backend_other so a new backend label can never silently disappear
    from the ledger.
    """
    values: list[Any] = []
    for col, _ in case_sql_columns():
        if col in llm_keys:
            values.append(metrics.get(col, {}).get("score"))
        elif col == "backend_fts":
            values.append(bd.get("fts", 0))
        elif col == "backend_vector":
            values.append(sum(bd.get(key, 0) for key in _VECTOR_BACKEND_KEYS))
        elif col == "backend_hybrid":
            values.append(bd.get("hybrid", 0))
        elif col == "backend_other":
            values.append(sum(count for key, count in bd.items() if key not in _KNOWN_BACKEND_KEYS))
        elif col == "required_keyword_hit_rate":
            values.append(kw.get("required_keyword_hit_rate"))
        elif col == "disallowed_keyword_hits":
            values.append(kw.get("disallowed_keyword_hits"))
        elif col == "avg_judge_score":
            values.append(item.get("avg_judge_score"))
        else:
            values.append(item.get(col))
    return tuple(values)
