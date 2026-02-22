"""
Dataset Split Pipeline
======================

Produces per-label splits for both DSPy/GEPA prompt optimisation and
transformer fine-tuning in a single run. The test set is locked first from
label statistics alone — before any mode-specific decisions — so it is
identical across both split types.

DSPy splits
-----------
The trainset is maximised because GEPA samples trajectories from it for
reflection. A small val set (config.DSPY_VAL_SIZE) is held out for Pareto
frontier tracking during optimisation.

Transformer splits
------------------
The pipeline returns only the locked ``trainval`` pool for transformer work.
How to split that pool later (or whether to use all of it) is left to the
downstream transformer training stage.

Return structure of run()
-------------------------
A ``Splits`` object with one attribute per label. Access via dot notation::

    splits.romantic.test               # shared test DataFrame
    splits.romantic.dspy.train         # GEPA training DataFrame
    splits.romantic.dspy.val           # GEPA validation DataFrame
    splits.romantic.transformer.trainval  # full transformer pool

Key access also works for loops: ``splits["romantic"]``.
"""

import math
from dataclasses import dataclass

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from training.config import (
    CONFIDENCE_LEVEL,
    DEFAULT_Z_SCORE,
    DSPY_VAL_SIZE,
    F1_CONSERVATIVE_ESTIMATE,
    LABELS,
    MAX_F1_CI_WIDTH,
    MAX_TEST_SIZE,
    MIN_TEST_SIZE,
    RANDOM_STATE,
    TEXT_COLUMN,
    Z_SCORES,
)


@dataclass
class DspySplit:
    train: pd.DataFrame
    val: pd.DataFrame


@dataclass
class TransformerSplit:
    trainval: pd.DataFrame


@dataclass
class LabelSplit:
    test: pd.DataFrame
    dspy: DspySplit
    transformer: TransformerSplit


@dataclass
class Splits:
    """Container for per-label splits, accessible by attribute and key.

    The keys are driven by config.LABELS, so this remains valid if labels are
    added, removed, or renamed.
    """

    by_label: dict[str, LabelSplit]

    def __post_init__(self) -> None:
        """Validates that the split container matches configured labels.

        Args:
            None.

        Returns:
            None.
        """
        expected = set(LABELS)
        actual = set(self.by_label.keys())
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            raise ValueError(f"Missing split entries for labels: {missing}")
        if extra:
            raise ValueError(f"Unexpected split entries for labels: {extra}")

    def __getitem__(self, label: str) -> LabelSplit:
        """Returns the split bundle for a label.

        Args:
            label: Label name from config.LABELS.

        Returns:
            LabelSplit for the requested label.
        """
        return self.by_label[label]

    def __getattr__(self, name: str) -> LabelSplit:
        """Provides attribute-style access to label splits.

        Args:
            name: Label name looked up as an attribute.

        Returns:
            LabelSplit for the requested label.
        """
        try:
            return self.by_label[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def keys(self):
        """Returns available label names.

        Args:
            None.

        Returns:
            Dict-keys view over configured label names.
        """
        return self.by_label.keys()

    def values(self):
        """Returns all label split bundles.

        Args:
            None.

        Returns:
            Dict-values view over LabelSplit objects.
        """
        return self.by_label.values()

    def items(self):
        """Returns (label, split) pairs.

        Args:
            None.

        Returns:
            Dict-items view of label names and LabelSplit objects.
        """
        return self.by_label.items()


def compute_label_statistics(df: pd.DataFrame, labels: list[str]) -> dict:
    """Computes per-label distribution statistics.

    Args:
        df: The full dataset containing label columns.
        labels: List of label column names to analyze.

    Returns:
        A dictionary keyed by label name, each containing:
            - total: total number of rows.
            - true_count: number of TRUE values.
            - false_count: number of FALSE values.
            - true_pct: percentage of TRUE values.
    """
    stats = {}
    for label in labels:
        total = len(df)
        true_count = int(df[label].sum())
        stats[label] = {
            "total": total,
            "true_count": true_count,
            "false_count": total - true_count,
            "true_pct": true_count / total * 100 if total else 0.0,
        }
    return stats


def _can_stratify(series: pd.Series) -> tuple[bool, str]:
    """Returns whether stratification is valid for a binary series.

    Args:
        series: Binary label series considered for stratified splitting.

    Returns:
        Tuple of (can_stratify, reason). reason is empty when stratification is valid.
    """
    value_counts = series.value_counts(dropna=False)
    if value_counts.empty:
        return False, "no rows available"
    if len(value_counts) < 2:
        return False, "only one class present"
    if int(value_counts.min()) < 2:
        return False, "at least one class has fewer than 2 rows"
    return True, ""


def _split_with_optional_stratify(
    df: pd.DataFrame,
    label: str,
    test_size: float,
    random_state: int,
    split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs train_test_split with optional stratification.

    Args:
        df: DataFrame to split.
        label: Label column used for stratification checks.
        test_size: Fraction (or absolute count) allocated to the second split.
        random_state: Seed for reproducibility.
        split_name: Human-readable split name for log/error messages.

    Returns:
        Tuple of (train_df, test_df) from train_test_split.
    """
    n_rows = len(df)
    if n_rows < 2:
        raise ValueError(
            f"Label '{label}' | {split_name}: need at least 2 rows to split, got {n_rows}."
        )
    if isinstance(test_size, float):
        test_rows = math.ceil(n_rows * test_size)
    else:
        test_rows = int(test_size)
    train_rows = n_rows - test_rows
    if test_rows < 1 or train_rows < 1:
        raise ValueError(
            f"Label '{label}' | {split_name}: invalid split for n={n_rows}, "
            f"test_size={test_size} (would produce train={train_rows}, test={test_rows})."
        )
    can_stratify, reason = _can_stratify(df[label])
    kwargs = {"test_size": test_size, "random_state": random_state}
    if can_stratify:
        kwargs["stratify"] = df[label]
    else:
        logger.warning(
            f"Label '{label}' | {split_name}: using non-stratified split "
            f"because {reason}."
        )
    return train_test_split(df, **kwargs)


def compute_cooccurrence_matrix(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Computes a label co-occurrence matrix counting rows where two labels are both TRUE.

    Args:
        df: The full dataset containing label columns.
        labels: List of label column names to analyze.

    Returns:
        A square DataFrame of shape (len(labels), len(labels)) with integer counts.
    """
    co = pd.DataFrame(index=labels, columns=labels, dtype=int)
    for l1 in labels:
        for l2 in labels:
            co.loc[l1, l2] = int((df[l1] & df[l2]).sum())
    return co


def compute_min_test_positives(
    confidence_level: float = CONFIDENCE_LEVEL,
    max_ci_width: float = MAX_F1_CI_WIDTH,
    f1_estimate: float = F1_CONSERVATIVE_ESTIMATE,
) -> int:
    """Computes the minimum number of positive test examples for a reliable F1 estimate.

    Formula: n_min = z^2 * f1 * (1 - f1) / (half_width^2)

    Args:
        confidence_level: Desired confidence level for the F1 CI.
        max_ci_width:     Maximum acceptable total width of the F1 CI.
        f1_estimate:      Assumed F1 value; 0.5 is most conservative.

    Returns:
        Minimum number of TRUE examples required in the test set.
    """
    z = Z_SCORES.get(confidence_level, DEFAULT_Z_SCORE)
    half_width = max_ci_width / 2
    return math.ceil(z**2 * f1_estimate * (1 - f1_estimate) / half_width**2)


def compute_test_size_per_label(
    stats: dict,
    labels: list[str],
    min_test_positives: int,
) -> dict:
    """Determines the test set size for each label from label statistics alone.

    Args:
        stats:              Output of compute_label_statistics.
        labels:             Label column names.
        min_test_positives: Minimum TRUE examples required in the test set.

    Returns:
        A dict keyed by label name, each containing:
            - test_size: float fraction to hold out as the test set.
            - projected_test_positives: estimated TRUE count in the test set.
            - sufficient: bool indicating whether the CI threshold is met.
    """
    test_sizes = {}
    for label in labels:
        true_count = stats[label]["true_count"]
        required = min_test_positives / true_count if true_count > 0 else MAX_TEST_SIZE
        clamped = max(MIN_TEST_SIZE, min(required, MAX_TEST_SIZE))
        projected = int(true_count * clamped)
        sufficient = projected >= min_test_positives
        if not sufficient:
            logger.warning(
                f"Label '{label}': {true_count} total positives. "
                f"Even at MAX_TEST_SIZE={MAX_TEST_SIZE}, test set yields only "
                f"~{projected} positives, below the required {min_test_positives}. "
                f"Consider collecting more data for this label."
            )
        else:
            logger.info(
                f"Label '{label}': test_size={clamped:.2f}, "
                f"~{projected} test positives (satisfies CI requirement)."
            )

        test_sizes[label] = {
            "test_size": clamped,
            "projected_test_positives": projected,
            "sufficient": sufficient,
        }
    return test_sizes


def _lock_test_sets(
    df: pd.DataFrame,
    labels: list[str],
    test_size_config: dict,
    random_state: int,
) -> dict:
    """Carves out and locks the test set for each label.

    Args:
        df:               Full dataset with TEXT_COLUMN and label columns.
        labels:           Label column names.
        test_size_config: Output of compute_test_size_per_label.
        random_state:     Random seed.

    Returns:
        Dict keyed by label name mapping to (trainval_df, test_df) tuples.
    """
    locked = {}
    for label in labels:
        trainval, test = _split_with_optional_stratify(
            df[[TEXT_COLUMN, label]],
            label=label,
            test_size=test_size_config[label]["test_size"],
            random_state=random_state,
            split_name="test lock",
        )
        locked[label] = (trainval, test)
        pos = int(test[label].sum())
        logger.info(
            f"Label '{label}' | test locked: {len(test)} rows, "
            f"{pos} TRUE ({pos / len(test) * 100:.1f}%) | "
            f"trainval remaining: {len(trainval)} rows."
        )
    return locked


def _make_dspy_splits(
    locked: dict,
    labels: list[str],
    dspy_val_size: float,
    random_state: int,
) -> dict:
    """Creates DSPy train and val splits from the locked trainval data.

    The val fraction is expressed relative to the full dataset so it remains
    proportional regardless of how large the test set is per label.

    Args:
        locked:        Output of _lock_test_sets.
        labels:        Label column names.
        dspy_val_size: Fraction of the full dataset reserved for val.
        random_state:  Random seed.

    Returns:
        Dict keyed by label name, each containing 'train' and 'val' DataFrames.
    """
    result = {}
    for label in labels:
        trainval, test = locked[label]
        test_frac = len(test) / (len(trainval) + len(test))
        relative_val = min(dspy_val_size / (1.0 - test_frac), 0.5)
        train, val = _split_with_optional_stratify(
            trainval,
            label=label,
            test_size=relative_val,
            random_state=random_state,
            split_name="dspy train/val",
        )
        result[label] = {"train": train, "val": val}
        for name, split_df in result[label].items():
            pos = int(split_df[label].sum())
            logger.info(
                f"Label '{label}' | dspy {name}: {len(split_df)} rows, "
                f"{pos} TRUE ({pos / len(split_df) * 100:.1f}%)."
            )
    return result


def _make_transformer_splits(
    locked: dict,
    labels: list[str],
) -> dict:
    """Returns the transformer trainval pool from each locked split.

    Args:
        locked: Output of _lock_test_sets.
        labels: Label column names.

    Returns:
        Dict keyed by label name with one key:
            - trainval: Full pool available for transformer training.
    """
    result = {}
    for label in labels:
        trainval, _ = locked[label]
        result[label] = {"trainval": trainval}
        logger.info(
            f"Label '{label}' | transformer trainval: {len(trainval)} rows, "
            f"{int(trainval[label].sum())} TRUE ({trainval[label].mean() * 100:.1f}%)."
        )
    return result


def run(df: pd.DataFrame) -> Splits:
    """Runs the full split pipeline for all labels.

    Locks test sets first (identical across both split types), then produces
    DSPy train/val splits plus transformer trainval pools.

    Args:
        df: Full dataset with TEXT_COLUMN and one boolean column per label.

    Returns:
        A Splits object. Access per-label data via attribute::

            splits.romantic.test
            splits.romantic.dspy.train
            splits.romantic.transformer.trainval

        Key access also works for loops: ``splits["romantic"]``.
    """
    if df.empty:
        raise ValueError("Dataset is empty. Provide at least one row before splitting.")
    stats = compute_label_statistics(df, LABELS)
    for label, s in stats.items():
        logger.info(
            f"Label '{label}': TRUE={s['true_count']} ({s['true_pct']:.1f}%), "
            f"FALSE={s['false_count']}."
        )
    co = compute_cooccurrence_matrix(df, LABELS)
    logger.info(f"Co-occurrence matrix:\n{co.to_string()}")
    min_test_positives = compute_min_test_positives()
    logger.info(f"Minimum test positives required per label: {min_test_positives}.")
    test_size_config = compute_test_size_per_label(stats, LABELS, min_test_positives)
    logger.info("Locking test sets.")
    locked = _lock_test_sets(df, LABELS, test_size_config, RANDOM_STATE)
    dspy_splits = _make_dspy_splits(locked, LABELS, DSPY_VAL_SIZE, RANDOM_STATE)
    transformer_splits = _make_transformer_splits(locked, LABELS)
    per_label = {}
    for label in LABELS:
        per_label[label] = LabelSplit(
            test=locked[label][1],
            dspy=DspySplit(
                train=dspy_splits[label]["train"],
                val=dspy_splits[label]["val"],
            ),
            transformer=TransformerSplit(
                trainval=transformer_splits[label]["trainval"],
            ),
        )
    return Splits(by_label=per_label)
