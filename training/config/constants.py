"""Project-wide constants that are fixed by design.

These values are not tuneable between experiments. For settings that may
change per run, see ``training.config.settings``.
"""

# Canonical set of values treated as boolean True when reading label columns.
# All other values are treated as False.
LABEL_TRUE_VALUES: frozenset = frozenset(
    {True, "true", "True", "TRUE", "yes", "YES", "1", 1}
)

# The string the DSPy model produces for the positive class.
POSITIVE_CLASS = "true"

# sklearn ``zero_division`` parameter. 0.0 surfaces zero-denominator failures
# rather than silently ignoring them.
METRIC_ZERO_DIVISION = 0.0

# Mapping of confidence level to z-score for F1 CI computation.
Z_SCORES: dict[float, float] = {
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576,
}

# Fallback z-score when the requested confidence level is not in Z_SCORES.
DEFAULT_Z_SCORE = 1.960

# Supported dataset file extensions for loading.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".xlsx", ".csv"})
