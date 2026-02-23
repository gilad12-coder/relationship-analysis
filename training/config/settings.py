"""Tuneable pipeline settings.

For fixed constants that do not change between experiments, see
``training.config.constants``.

Sections
--------
  1.  Dataset      — file path, sheet, column names
  2.  Labels       — classification targets
  3.  Splitting    — shared test sizing + per-mode train/val config
  4.  DSPy / GEPA  — model lists and optimizer settings
  5.  Output       — directories and filenames
  6.  Logging
"""

import os

# ── 1. Dataset ──────────────────────────────────────────────────────────────
DATASET_PATH = "data/relationships.csv"
DATASET_SHEET = "Sheet1"  # ignored for CSV; only .xlsx and .csv are supported
TEXT_COLUMN = "text"

# ── 2. Labels ───────────────────────────────────────────────────────────────
LABELS = ["professional", "romantic", "family", "friendship", "unknown", "irrelevant"]

# ── 3. Splitting ────────────────────────────────────────────────────────────
RANDOM_STATE = 42

# Supported values: 0.90, 0.95, 0.99  (must be a key in constants.Z_SCORES).
CONFIDENCE_LEVEL = 0.95
MAX_F1_CI_WIDTH = 0.2
F1_CONSERVATIVE_ESTIMATE = 0.5

MIN_HOLDOUT_SIZE = 0.10
MAX_HOLDOUT_SIZE = 0.30

# Fraction of the total dataset reserved for val in each mode.
DSPY_VAL_SIZE = 0.10

# ── 4. DSPy / GEPA ─────────────────────────────────────────────────────────
# Optional custom base URL for the LM provider (e.g. a proxy or local endpoint).
# Read from LM_BASE_URL env var; empty or unset means use the provider's default.
LM_BASE_URL = os.environ.get("LM_BASE_URL", None)

# Optional API key for the LM provider.
# Read from LM_API_KEY env var; empty or unset means use the provider's default.
LM_API_KEY = os.environ.get("LM_API_KEY", None)

# LM sampling parameters.
# Recommended values for gpt-oss-120b: temperature=1.0, max_tokens=16000.
LM_TEMPERATURE = 1.0
LM_MAX_TOKENS = 16000

# Reasoning effort for models that support it (e.g. gpt-oss-120b, o-series).
# Options: "low", "medium", "high". Set to None to disable.
LM_REASONING_EFFORT = "high"

# Must be DSPy-compatible model identifier strings.
GENERATION_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "openai/gpt-4.1-mini",
]

# Should generally be strong reasoning models.
REFLECTION_MODELS = [
    "openai/gpt-4o",
    "openai/o4-mini",
    "openai/gpt-4.1",
]

# "light" = fewer iterations; "medium" = balanced; "heavy" = maximum iterations.
GEPA_AUTO = "heavy"
GEPA_TRACK_STATS = True

# ── 5. Output ───────────────────────────────────────────────────────────────
RESULTS_DIR = "training/results"
GRID_SEARCH_CSV = "grid_search_results.csv"
EVAL_CSV = "evaluation.csv"
SUMMARY_MD = "summary.md"
PREDICTIONS_SUBDIR = "predictions"
PROGRAMS_SUBDIR = "programs"
PROGRAMS_MANIFEST = "manifest.json"
SPLITS_SUBDIR = "splits"

# ── 6. Logging ──────────────────────────────────────────────────────────────
# Options: "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
LOG_LEVEL_CONSOLE = "INFO"
LOG_LEVEL_FILE = "DEBUG"
LOG_FILE = "pipeline.log"
LOG_ROTATION = "50 MB"
