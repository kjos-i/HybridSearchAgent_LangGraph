"""Configuration for the DeepEval evaluation harness.

Edit the values below to control how the evaluation run behaves.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Paths ---
DATASET_PATH = Path(__file__).resolve().parent / "eval_cases.json"
OUTPUT_DIR = BASE_DIR / "evaluation_results"

# --- Judge settings ---
JUDGE_MODEL = "gpt-4o"
THRESHOLD = 0.5

# --- Run settings ---
CONCURRENCY = 2
MAX_CASES = None   # Set to an int (e.g. 3) to limit the number of cases evaluated
