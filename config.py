"""
Global Configuration Settings.

This module acts as a central 'control panel' for the agent's behavior,
including search strategy weights, debugging logs, UI feedback, and
graph visualization.
"""

# ---------------------------------------------------------------------------
# Search strategy hyperparameters
# ---------------------------------------------------------------------------

# Relative weight of each retrieval backend in the fused ranking.
FTS_WEIGHT = 0.5
VECTOR_SIMILARITY_WEIGHT = 0.5
VECTOR_MMR_WEIGHT = 0.5

# Ceiling used to normalize raw SQLite BM25 scores into 0-1 range.
FTS_MAX_SCORE = 20.0

# Per-mode weights for multi-mode FTS (keyword always runs; phrase and prefix
# are additive).  Values > 1.0 boost that mode; < 1.0 dampen it.
FTS_MULTI_WEIGHTS: dict[str, float] = {"phrase": 1.0, "keyword": 1.0, "prefix": 1.0}

# ---------------------------------------------------------------------------
# Debug and logging flags
# ---------------------------------------------------------------------------

# Enable verbose LangGraph system logs
DEBUG = False

# Show real-time node updates during agent execution
UPDATES = False

# Generate and save a PNG of the agent's logic graph on startup
DRAW = False

# Enable the custom 'debug_print' utility for internal tool logic
DEBUG_PRINT = False

# Control overall console output/logging visibility
PRINT = False
