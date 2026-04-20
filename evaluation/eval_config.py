"""Configuration for the DeepEval evaluation harness.

Edit the values below to control how the evaluation run behaves.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Paths ---
DATASET_PATH = Path(__file__).resolve().parent / "eval_cases.json"
OUTPUT_DIR = BASE_DIR / "evaluation" / "evaluation_results"

# --- Judge settings ---
JUDGE_MODEL = "gpt-5.4-mini"
JUDGE_THRESHOLD = 0.5

# --- Verdict gate thresholds (configure once per project) ---
# Do NOT change these between runs once you've started collecting trend data.
# Lowering a threshold after a regression will "fix" the pass rate without
# fixing the underlying issue. If you genuinely need stricter/looser gates,
# reset the ledger (or at least tag it as a new regime) so trend charts
# compare like-with-like. The active thresholds are persisted with each run
# and the dashboard warns if they change across runs.
#
# Binary-by-design gates are NOT configurable:
#   - hit_at_k must equal 1.0 (must retrieve at least one expected source)
#   - disallowed_keyword_hits must equal 0 (zero tolerance)
#
# The judge gate (faithfulness, answer_relevancy) reuses JUDGE_THRESHOLD above,
# the same value that sets each DeepEval metric's pass/fail line.
METADATA_MATCH_THRESHOLD = 0.8
REQUIRED_KEYWORD_THRESHOLD = 0.5

# --- Run settings ---
CONCURRENCY = 2
MAX_CASES = 3   # Set to an int (e.g. 3) to limit the number of cases evaluated

# --- Metric groups ---
# Controls which toggleable metric groups are computed for each evaluation run.
#   "judge"  — DeepEval LLM-judged metrics (faithfulness, answer_relevancy, contextual_*, etc.)
#   "source" — source-level retrieval metrics (hit_at_k, mrr, precision_at_k, recall_at_k, ndcg_at_k)
#   "chunk"  — chunk-level retrieval metrics (snippet-substring match against expected_chunks)
#
# Always-on regardless of this setting:
#   metadata_match_ratio, backend_distribution, keyword_checks, latency, avg_judge_score.
#
# Disabled metrics are stored as NULL in the ledger and CSV; the dashboard renders them as
# "Not evaluated". Verdict gates skip sub-conditions whose metric group is disabled.
ENABLED_METRIC_GROUPS: set[str] = {"source", "chunk", "judge"}

# Typo guard: fails fast at import time if ENABLED_METRIC_GROUPS contains an unknown
# group name (e.g. "sauce" instead of "source"). Python's set type accepts any string,
# so without this check a misspelled group would silently disable the intended metrics
# and produce NULL columns in the ledger with no warning.
_VALID_METRIC_GROUPS: set[str] = {"judge", "source", "chunk"}
_invalid_groups = ENABLED_METRIC_GROUPS - _VALID_METRIC_GROUPS
if _invalid_groups:
    raise ValueError(
        f"Invalid entries in ENABLED_METRIC_GROUPS: {sorted(_invalid_groups)}. "
        f"Valid values are: {sorted(_VALID_METRIC_GROUPS)}."
    )
