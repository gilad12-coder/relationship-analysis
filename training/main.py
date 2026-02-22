"""
Main Pipeline Entry Point
=========================

Runs the full relationship classification pipeline end to end:

  1. Load dataset.
  2. Split (produces both DSPy and transformer splits with a shared test set).
  3. Build dspy.Example objects.
  4. Grid search over all (generation_model, reflection_model) pairs,
     running GEPA independently for each label, ranked by val F1.
  5. Evaluate the best program per label on its locked test set once.

The test set is locked from label statistics alone before any training
decisions are made, and is identical across both split types.

Usage
-----
    python train.py
    python train.py --auto light
    python train.py --gen_model openai/gpt-4o --reflection_model openai/gpt-4o
"""

import argparse
import json
import os
import sys

import dspy
import pandas as pd
from loguru import logger

from training.config import (
    DATASET_PATH,
    DATASET_SHEET,
    FINAL_EVAL_CSV,
    GEPA_AUTO,
    GENERATION_MODELS,
    GRID_SEARCH_CSV,
    LABEL_TRUE_VALUES,
    LABELS,
    LOG_FILE,
    LOG_LEVEL_CONSOLE,
    LOG_LEVEL_FILE,
    LOG_ROTATION,
    PROGRAMS_MANIFEST,
    PROGRAMS_SUBDIR,
    REFLECTION_MODELS,
    RESULTS_DIR,
    SUPPORTED_EXTENSIONS,
    TEXT_COLUMN,
)
from training.data import dataset_builder, split_pipeline
from training.evaluation import evaluate
from training.optimization import grid_search, optimize

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level=LOG_LEVEL_CONSOLE,
)
os.makedirs(RESULTS_DIR, exist_ok=True)
logger.add(
    os.path.join(RESULTS_DIR, LOG_FILE),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    level=LOG_LEVEL_FILE,
    rotation=LOG_ROTATION,
)


def _normalise_label_column(series: pd.Series) -> pd.Series:
    """Converts a raw label column into a boolean Series.

    Treats any value in constants.LABEL_TRUE_VALUES as True, everything else
    as False.

    Args:
        series: Raw column from the loaded DataFrame.

    Returns:
        Boolean Series of the same length.
    """
    return series.apply(lambda v: v in LABEL_TRUE_VALUES)


def load_dataset() -> pd.DataFrame:
    """Loads and normalises the dataset from config.DATASET_PATH.

    Supports .xlsx and .csv files. Each label column is normalised
    to a boolean via constants.LABEL_TRUE_VALUES.

    Args:
        None.

    Returns:
        DataFrame with TEXT_COLUMN and one boolean column per label.

    Raises:
        ValueError: If the file extension is unsupported or required columns
                    are missing.
    """
    ext = os.path.splitext(DATASET_PATH)[1].lower()
    logger.info(f"Loading dataset from '{DATASET_PATH}'.")
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )
    if ext == ".xlsx":
        df = pd.read_excel(DATASET_PATH, sheet_name=DATASET_SHEET)
    else:
        df = pd.read_csv(DATASET_PATH)
    missing = [c for c in [TEXT_COLUMN] + LABELS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns missing from '{DATASET_PATH}': {missing}. "
            f"Check TEXT_COLUMN and LABELS in training/config/settings.py."
        )
    for label in LABELS:
        df[label] = _normalise_label_column(df[label])
    n_before = len(df)
    df = df.dropna(subset=[TEXT_COLUMN]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning(f"Dropped {n_dropped} rows with null values in '{TEXT_COLUMN}'.")
    logger.info(f"Dataset loaded: {len(df):,} rows.")
    return df


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for the training entrypoint.

    Args:
        None.

    Returns:
        argparse.Namespace with --auto, --gen_model, and --reflection_model.
    """
    parser = argparse.ArgumentParser(
        description="Relationship classification pipeline."
    )
    parser.add_argument(
        "--auto",
        choices=["light", "medium", "heavy"],
        default=GEPA_AUTO,
        help="GEPA budget. Default: %(default)s.",
    )
    parser.add_argument(
        "--gen_model",
        type=str,
        default=None,
        help="Fix a generation model and skip the grid search.",
    )
    parser.add_argument(
        "--reflection_model",
        type=str,
        default=None,
        help="Fix a reflection model and skip the grid search.",
    )
    return parser.parse_args()


def _run_fixed_dspy_pair(
    datasets: dict,
    gen_model: str,
    reflection_model: str,
    auto: str,
) -> dict:
    """Runs GEPA for all labels with a single fixed model pair, skipping grid search.

    Supports resuming: labels with a saved program on disk are loaded
    instead of re-optimized.

    Args:
        datasets:         Output of dataset_builder.build().
        gen_model:        DSPy generation model string.
        reflection_model: DSPy reflection model string.
        auto:             GEPA budget.

    Returns:
        Dict keyed by label name, each value an OptimizationResult.
    """
    logger.info(f"Fixed pair: gen={gen_model} | reflection={reflection_model}.")
    best_per_label = {}
    for label in LABELS:
        program_dir = os.path.join(RESULTS_DIR, PROGRAMS_SUBDIR, label)
        if os.path.isdir(program_dir):
            logger.info(f"[{label}] Saved program found. Loading instead of re-optimizing.")
            best_per_label[label] = optimize.OptimizationResult(
                label=label,
                gen_model=gen_model,
                reflection_model=reflection_model,
                optimized_program=dspy.load(program_dir, allow_pickle=True),
                val_f1=0.0,
                val_accuracy=0.0,
            )
            continue
        best_per_label[label] = optimize.run(
            label=label,
            gen_model=gen_model,
            reflection_model=reflection_model,
            trainset=datasets[label]["trainset"],
            valset=datasets[label]["valset"],
            auto=auto,
        )
    return best_per_label


def _label_eval_exists(label: str) -> bool:
    """Checks if a label already has a row in the final evaluation CSV.

    Args:
        label: Label name.

    Returns:
        True if the eval CSV exists and contains a row for this label.
    """
    eval_path = os.path.join(RESULTS_DIR, FINAL_EVAL_CSV)
    if not os.path.isfile(eval_path):
        return False
    eval_df = pd.read_csv(eval_path)
    return label in eval_df["label"].values


def _save_program(label: str, result, manifest: dict) -> None:
    """Saves a single label's optimized program to disk and updates the manifest.

    Args:
        label:    Label name.
        result:   OptimizationResult for this label.
        manifest: Shared manifest dict (mutated in place, then written to disk).

    Returns:
        None.
    """
    programs_dir = os.path.join(RESULTS_DIR, PROGRAMS_SUBDIR)
    label_dir = os.path.join(programs_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    result.optimized_program.save(label_dir, save_program=True)
    manifest[label] = result.gen_model
    logger.info(f"[{label}] Program saved to {label_dir}.")
    manifest_path = os.path.join(programs_dir, PROGRAMS_MANIFEST)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest saved to {manifest_path}.")


def _run_dspy(args: argparse.Namespace, splits: dict) -> None:
    """Executes the full DSPy/GEPA pipeline branch.

    Args:
        args:   Parsed CLI arguments.
        splits: Output of split_pipeline.run().

    Returns:
        None.
    """
    logger.info("Step 3: Building dspy.Example datasets.")
    datasets = dataset_builder.build(splits)
    fixed_pair = args.gen_model is not None or args.reflection_model is not None
    if fixed_pair:
        if args.gen_model is None or args.reflection_model is None:
            logger.error(
                "--gen_model and --reflection_model must be supplied together."
            )
            sys.exit(1)
        if args.gen_model.startswith("anthropic/"):
            logger.error(
                "Anthropic generation models are not supported in this project."
            )
            sys.exit(1)
        if args.reflection_model.startswith("anthropic/"):
            logger.error(
                "Anthropic reflection models are not supported in this project."
            )
            sys.exit(1)
        if args.gen_model not in GENERATION_MODELS:
            logger.warning(
                f"--gen_model '{args.gen_model}' not in config.GENERATION_MODELS."
            )
        if args.reflection_model not in REFLECTION_MODELS:
            logger.warning(
                f"--reflection_model '{args.reflection_model}' not in config.REFLECTION_MODELS."
            )
        logger.info("Step 4: Running fixed model pair (grid search skipped).")
        best_per_label = _run_fixed_dspy_pair(
            datasets, args.gen_model, args.reflection_model, args.auto
        )
    else:
        n_pairs = len(GENERATION_MODELS) * len(REFLECTION_MODELS)
        logger.info(
            f"Step 4: Grid search — {len(GENERATION_MODELS)} gen x "
            f"{len(REFLECTION_MODELS)} reflection = {n_pairs} pairs x "
            f"{len(LABELS)} labels = {n_pairs * len(LABELS)} trials."
        )
        best_per_label = grid_search.run(datasets=datasets, auto=args.auto)
    # Save programs immediately after optimization (enables resume if eval crashes).
    manifest_path = os.path.join(RESULTS_DIR, PROGRAMS_SUBDIR, PROGRAMS_MANIFEST)
    manifest = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    for label in LABELS:
        _save_program(label, best_per_label[label], manifest)

    # Evaluate on locked test sets (skip labels already evaluated).
    logger.info("Step 5: Final evaluation on locked test sets.")
    for label in LABELS:
        if _label_eval_exists(label):
            logger.info(f"[{label}] Evaluation already exists. Skipping.")
            continue
        evaluate.run_dspy_label(
            label=label,
            result=best_per_label[label],
            datasets=datasets,
        )


def main() -> None:
    """Runs the full training pipeline from data load to program export.

    Args:
        None.

    Returns:
        None.
    """
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logger.info("Pipeline starting.")
    logger.info("Step 1: Loading dataset.")
    df = load_dataset()
    logger.info("Step 2: Splitting dataset.")
    splits = split_pipeline.run(df)
    split_pipeline.save(splits)
    _run_dspy(args, splits)
    logger.info(
        f"Pipeline complete. Results in '{RESULTS_DIR}/': "
        f"{GRID_SEARCH_CSV} {FINAL_EVAL_CSV}."
    )


if __name__ == "__main__":
    main()
