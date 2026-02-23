"""
Main Pipeline Entry Point
=========================

Runs the full relationship classification pipeline end to end:

  1. Load dataset.
  2. Split (produces both DSPy and transformer splits with a shared selection holdout).
  3. Build dspy.Example objects.
  4. Grid search over all (generation_model, reflection_model) pairs,
     running GEPA independently for each label, ranked by holdout F1.
  5a. Baseline evaluation on locked holdout sets.
  5b. Optimised evaluation on locked holdout sets.
  6. Generate summary.md.

Usage
-----
    python train.py
    python train.py --auto light
    python train.py --gen_model openai/gpt-4o --reflection_model openai/gpt-4o
"""

import argparse
import hashlib
import json
import os
import sys

import dspy
import pandas as pd
from loguru import logger

from training.config import (
    CONFIDENCE_LEVEL,
    DATASET_PATH,
    DATASET_SHEET,
    DSPY_VAL_SIZE,
    EVAL_CSV,
    GEPA_AUTO,
    GENERATION_MODELS,
    GRID_SEARCH_CSV,
    LABEL_TRUE_VALUES,
    LABELS,
    LOG_FILE,
    LOG_LEVEL_CONSOLE,
    LOG_LEVEL_FILE,
    LOG_ROTATION,
    MAX_F1_CI_WIDTH,
    MAX_HOLDOUT_SIZE,
    MIN_HOLDOUT_SIZE,
    PROGRAMS_MANIFEST,
    PROGRAMS_SUBDIR,
    RANDOM_STATE,
    REFLECTION_MODELS,
    RESULTS_DIR,
    SUMMARY_MD,
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


def _dataset_hash(df: pd.DataFrame) -> str:
    """Computes a short content hash of the normalised dataset and split config.

    Covers both data changes and split-config changes (val size, holdout
    bounds, random state, CI settings) so any parameter that affects the
    splits also invalidates cached artifacts.

    Args:
        df: Normalised DataFrame (after label conversion and NA drop).

    Returns:
        A 16-character hex digest.
    """
    cols = [TEXT_COLUMN] + LABELS
    content = df[cols].sort_values(cols, kind="mergesort").to_csv(index=False)
    config_str = (
        f"RANDOM_STATE={RANDOM_STATE}|DSPY_VAL_SIZE={DSPY_VAL_SIZE}|"
        f"MIN_HOLDOUT_SIZE={MIN_HOLDOUT_SIZE}|MAX_HOLDOUT_SIZE={MAX_HOLDOUT_SIZE}|"
        f"CONFIDENCE_LEVEL={CONFIDENCE_LEVEL}|MAX_F1_CI_WIDTH={MAX_F1_CI_WIDTH}"
    )
    return hashlib.sha256((content + config_str).encode()).hexdigest()[:16]


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
    dataset_hash: str,
) -> dict:
    """Runs GEPA for all labels with a single fixed model pair, skipping grid search.

    Supports resuming: labels with a saved program on disk are loaded
    instead of re-optimized, provided the manifest matches the requested
    model pair and current dataset hash.

    Args:
        datasets:         Output of dataset_builder.build().
        gen_model:        DSPy generation model string.
        reflection_model: DSPy reflection model string.
        auto:             GEPA budget.
        dataset_hash:     Hash of the current dataset for staleness detection.

    Returns:
        Dict keyed by label name, each value an OptimizationResult.
    """
    logger.info(f"Fixed pair: gen={gen_model} | reflection={reflection_model}.")
    manifest_path = os.path.join(RESULTS_DIR, PROGRAMS_SUBDIR, PROGRAMS_MANIFEST)
    manifest = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    best_per_label = {}
    for label in LABELS:
        program_dir = os.path.join(RESULTS_DIR, PROGRAMS_SUBDIR, label)
        if os.path.isdir(program_dir):
            entry = manifest.get(label, {})
            saved_gen = entry.get("gen_model", "") if isinstance(entry, dict) else entry
            saved_refl = entry.get("reflection_model", "") if isinstance(entry, dict) else ""
            saved_hash = entry.get("dataset_hash", "") if isinstance(entry, dict) else ""
            if saved_gen == gen_model and saved_refl == reflection_model and saved_hash == dataset_hash:
                logger.info(f"[{label}] Saved program matches requested pair and dataset. Loading.")
                best_per_label[label] = optimize.OptimizationResult(
                    label=label,
                    gen_model=gen_model,
                    reflection_model=reflection_model,
                    optimized_program=dspy.load(program_dir),
                    val_f1=0.0,
                    val_accuracy=0.0,
                )
                continue
            reason = []
            if saved_gen != gen_model or saved_refl != reflection_model:
                reason.append(f"model pair (gen={saved_gen}, reflection={saved_refl})")
            if saved_hash != dataset_hash:
                reason.append("dataset content")
            logger.warning(
                f"[{label}] Saved program stale — {' and '.join(reason)} changed. Re-optimizing."
            )
        best_per_label[label] = optimize.run(
            label=label,
            gen_model=gen_model,
            reflection_model=reflection_model,
            trainset=datasets[label]["trainset"],
            valset=datasets[label]["valset"],
            auto=auto,
        )
    return best_per_label


_REQUIRED_EVAL_COLUMNS = {"holdout_f1", "holdout_precision", "holdout_recall", "holdout_accuracy"}


def _eval_exists(
    label: str,
    stage: str,
    gen_model: str = None,
    reflection_model: str = None,
    dataset_hash: str = None,
) -> bool:
    """Checks if a label+stage already has a matching row in the evaluation CSV.

    When model identifiers or dataset_hash are provided, the saved row must
    match them. Returns False if the CSV is missing the expected holdout
    metric columns (legacy schema), model columns, or dataset_hash column,
    so stale rows are never reused.

    Args:
        label:            Label name.
        stage:            'baseline' or 'optimized'.
        gen_model:        If set, the row must match this gen_model.
        reflection_model: If set, the row must match this reflection_model.
        dataset_hash:     If set, the row must match this dataset hash.

    Returns:
        True if a matching row exists.
    """
    eval_path = os.path.join(RESULTS_DIR, EVAL_CSV)
    if not os.path.isfile(eval_path):
        return False
    eval_df = pd.read_csv(eval_path)
    if not _REQUIRED_EVAL_COLUMNS.issubset(eval_df.columns):
        return False
    if gen_model and "gen_model" not in eval_df.columns:
        return False
    if reflection_model and "reflection_model" not in eval_df.columns:
        return False
    if dataset_hash and "dataset_hash" not in eval_df.columns:
        return False
    mask = (eval_df["label"] == label) & (eval_df["stage"] == stage)
    if gen_model:
        mask = mask & (eval_df["gen_model"] == gen_model)
    if reflection_model:
        mask = mask & (eval_df["reflection_model"] == reflection_model)
    if dataset_hash:
        mask = mask & (eval_df["dataset_hash"] == dataset_hash)
    return mask.any()


def _save_program(label: str, result, manifest: dict, dataset_hash: str) -> None:
    """Saves a single label's optimized program to disk and updates the manifest.

    Args:
        label:        Label name.
        result:       OptimizationResult for this label.
        manifest:     Shared manifest dict (mutated in place, then written to disk).
        dataset_hash: Hash of the dataset used for this optimization.

    Returns:
        None.
    """
    programs_dir = os.path.join(RESULTS_DIR, PROGRAMS_SUBDIR)
    label_dir = os.path.join(programs_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    result.optimized_program.save(label_dir, save_program=True)
    manifest[label] = {
        "gen_model": result.gen_model,
        "reflection_model": result.reflection_model,
        "dataset_hash": dataset_hash,
    }
    logger.info(f"[{label}] Program saved to {label_dir}.")
    manifest_path = os.path.join(programs_dir, PROGRAMS_MANIFEST)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest saved to {manifest_path}.")


def _write_summary_md() -> None:
    """Generates a human-readable Markdown summary of evaluation results.

    Reads the evaluation CSV and writes one section per label with a
    baseline vs optimised comparison table.
    """
    eval_path = os.path.join(RESULTS_DIR, EVAL_CSV)
    if not os.path.isfile(eval_path):
        logger.warning("No evaluation CSV found; skipping summary.md generation.")
        return

    eval_df = pd.read_csv(eval_path)
    lines = ["# Results Summary", ""]

    for label in LABELS:
        label_rows = eval_df[eval_df["label"] == label]
        if label_rows.empty:
            continue

        baseline = label_rows[label_rows["stage"] == "baseline"]
        optimized = label_rows[label_rows["stage"] == "optimized"]

        lines.append(f"## {label}")
        lines.append("")

        if not baseline.empty and not optimized.empty:
            b = baseline.iloc[0]
            o = optimized.iloc[0]
            lines.append("| Metric    | Baseline | Optimized |")
            lines.append("|-----------|----------|-----------|")
            lines.append(f"| F1        | {b['holdout_f1']:.4f}   | {o['holdout_f1']:.4f}    |")
            lines.append(f"| Precision | {b['holdout_precision']:.4f}   | {o['holdout_precision']:.4f}    |")
            lines.append(f"| Recall    | {b['holdout_recall']:.4f}   | {o['holdout_recall']:.4f}    |")
            lines.append(f"| Accuracy  | {b['holdout_accuracy']:.4f}   | {o['holdout_accuracy']:.4f}    |")
            lines.append("")
            gen = o.get("gen_model", "")
            refl = o.get("reflection_model", "")
            lines.append(f"Generation model: `{gen}` | Reflection model: `{refl}`")
            lines.append(f"Holdout set: {int(o['n_holdout'])} examples ({int(o['n_holdout_positive'])} positive)")
        elif not optimized.empty:
            o = optimized.iloc[0]
            lines.append(f"- F1: {o['holdout_f1']:.4f} | P: {o['holdout_precision']:.4f} | R: {o['holdout_recall']:.4f}")
        elif not baseline.empty:
            b = baseline.iloc[0]
            lines.append(f"- Baseline F1: {b['holdout_f1']:.4f} | P: {b['holdout_precision']:.4f} | R: {b['holdout_recall']:.4f}")

        lines.append("")
        lines.append("---")
        lines.append("")

    summary_path = os.path.join(RESULTS_DIR, SUMMARY_MD)
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Summary written to {summary_path}.")


def _run_dspy(args: argparse.Namespace, splits: dict, dataset_hash: str) -> None:
    """Executes the full DSPy/GEPA pipeline branch.

    Args:
        args:         Parsed CLI arguments.
        splits:       Output of split_pipeline.run().
        dataset_hash: Hash of the current dataset for staleness detection.

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
            datasets, args.gen_model, args.reflection_model, args.auto, dataset_hash
        )
    else:
        n_pairs = len(GENERATION_MODELS) * len(REFLECTION_MODELS)
        logger.info(
            f"Step 4: Grid search — {len(GENERATION_MODELS)} gen x "
            f"{len(REFLECTION_MODELS)} reflection = {n_pairs} pairs x "
            f"{len(LABELS)} labels = {n_pairs * len(LABELS)} trials."
        )
        best_per_label = grid_search.run(datasets=datasets, auto=args.auto, dataset_hash=dataset_hash)
    if fixed_pair:
        # Save programs after fixed-pair optimization (grid search saves incrementally).
        manifest_path = os.path.join(RESULTS_DIR, PROGRAMS_SUBDIR, PROGRAMS_MANIFEST)
        manifest = {}
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
        for label in LABELS:
            _save_program(label, best_per_label[label], manifest, dataset_hash)

    # Baseline evaluation (un-optimised programs on holdout set).
    logger.info("Step 5a: Baseline evaluation on locked holdout sets.")
    for label in LABELS:
        gen = best_per_label[label].gen_model
        if _eval_exists(label, "baseline", gen_model=gen, dataset_hash=dataset_hash):
            logger.info(f"[{label}] Baseline evaluation already exists. Skipping.")
            continue
        evaluate.run_baseline_label(
            label=label,
            gen_model=gen,
            datasets=datasets,
            dataset_hash=dataset_hash,
        )

    # Evaluate optimised programs on locked holdout sets.
    logger.info("Step 5b: Optimised evaluation on locked holdout sets.")
    for label in LABELS:
        gen = best_per_label[label].gen_model
        refl = best_per_label[label].reflection_model
        if _eval_exists(label, "optimized", gen_model=gen, reflection_model=refl, dataset_hash=dataset_hash):
            logger.info(f"[{label}] Optimised evaluation already exists. Skipping.")
            continue
        evaluate.run_dspy_label(
            label=label,
            result=best_per_label[label],
            datasets=datasets,
            dataset_hash=dataset_hash,
        )

    # Generate human-readable summary.
    logger.info("Step 6: Generating summary.")
    _write_summary_md()


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
    ds_hash = _dataset_hash(df)
    logger.info(f"Dataset hash: {ds_hash}.")
    logger.info("Step 2: Splitting dataset.")
    splits = split_pipeline.run(df)
    split_pipeline.save(splits)
    _run_dspy(args, splits, ds_hash)
    logger.info(
        f"Pipeline complete. Results in '{RESULTS_DIR}/': "
        f"{GRID_SEARCH_CSV} {EVAL_CSV} {SUMMARY_MD}."
    )


if __name__ == "__main__":
    main()
