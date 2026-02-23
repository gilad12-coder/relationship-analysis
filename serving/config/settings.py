"""Serving-specific settings.

Only the subset of configuration needed to load and run optimised programs.
"""

import os

from training.config.settings import LABELS as TRAINING_LABELS

# Keep serving labels aligned with training outputs.
LABELS = list(TRAINING_LABELS)
POSITIVE_CLASS = "true"
RESULTS_DIR = "training/results"
PROGRAMS_SUBDIR = "programs"
PROGRAMS_MANIFEST = "manifest.json"

# Optional custom base URL for the LM provider.
# Read from LM_BASE_URL env var; empty or unset means use the provider's default.
LM_BASE_URL = os.environ.get("LM_BASE_URL", None)

# Optional API key for the LM provider.
# Read from LM_API_KEY env var; empty or unset means use the provider's default.
LM_API_KEY = os.environ.get("LM_API_KEY", None)

# LM sampling parameters.
LM_TEMPERATURE = 1.0
LM_MAX_TOKENS = 16000

# Reasoning effort for models that support it.
# Options: "low", "medium", "high". Set to None to disable.
LM_REASONING_EFFORT = "high"
