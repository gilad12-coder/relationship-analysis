"""
Grid Search
===========

Runs GEPA optimization for every combination of (generation_model,
reflection_model) from the configured model lists in training/config/settings.py,
independently for each of the six relationship labels.

Each trial is scored on the validation set. The trial with the highest
validation F1 per label is selected as the best configuration.

No test data is used here. The grid search output is a dict keyed by label name
containing the best OptimizationResult for that label, which is then passed to
evaluate.py for a single final test evaluation.

Results from all trials are also saved to a CSV for inspection.

Design note on the val set
--------------------------
The same valset is used both by GEPA for Pareto frontier tracking during
optimization and by the grid search for ranking model pairs afterward. This is
a mild overlap — the valset scores are not fully independent of the optimization
process — but it is standard practice when no separate hold-out set exists for
model selection. The test set remains strictly clean for final evaluation.
"""

import itertools
import os

import dspy
import pandas as pd
from loguru import logger

from training.config import (
    GEPA_AUTO,
    GENERATION_MODELS,
    GRID_SEARCH_CSV,
    LABELS,
    PROGRAMS_SUBDIR,
    REFLECTION_MODELS,
    RESULTS_DIR,
)
from training.optimization import optimize


def _load_existing_csv() -> pd.DataFrame | None:
    """Loads the existing grid search CSV if it exists.

    Returns:
        DataFrame of previous results, or None if no CSV exists.
    """
    path = os.path.join(RESULTS_DIR, GRID_SEARCH_CSV)
    if os.path.isfile(path):
        return pd.read_csv(path)
    return None


def _label_has_all_trials(existing_df: pd.DataFrame | None, label: str) -> bool:
    """Checks if all expected grid search trials exist for a label.

    Args:
        existing_df: Existing grid search results DataFrame, or None.
        label:       Label name.

    Returns:
        True if all (gen_model, reflection_model) pairs have results for this label.
    """
    if existing_df is None:
        return False
    label_rows = existing_df[existing_df["label"] == label]
    expected = set(itertools.product(GENERATION_MODELS, REFLECTION_MODELS))
    actual = set(zip(label_rows["gen_model"], label_rows["reflection_model"]))
    return expected <= actual


def run(datasets: dict, auto: str = None) -> dict:
    """Runs the full grid search over all model pairs and all labels.

    Supports resuming: labels whose trials already exist in the grid search
    CSV are skipped. Programs are loaded from disk when available; otherwise
    only the best pair is re-run.

    Args:
        datasets: Output of dataset_builder.build(), keyed by label name.
                  Each entry must contain 'trainset', 'valset', and 'test_df'.
        auto:     GEPA budget override. If None, uses the value in config.GEPA_AUTO.

    Returns:
        A dict keyed by label name, each containing the best OptimizationResult
        (highest val_f1) across all model pair trials for that label.
    """
    if not LABELS:
        raise ValueError(
            "LABELS is empty. Configure at least one label in training/config/settings.py."
        )
    if not GENERATION_MODELS:
        raise ValueError(
            "GENERATION_MODELS is empty. Configure at least one generation model in training/config/settings.py."
        )
    if not REFLECTION_MODELS:
        raise ValueError(
            "REFLECTION_MODELS is empty. Configure at least one reflection model in training/config/settings.py."
        )

    effective_auto = auto if auto is not None else GEPA_AUTO

    os.makedirs(RESULTS_DIR, exist_ok=True)
    existing_df = _load_existing_csv()
    all_results: list[optimize.OptimizationResult] = []
    model_pairs = list(itertools.product(GENERATION_MODELS, REFLECTION_MODELS))
    total_trials = len(LABELS) * len(model_pairs)
    trial_num = 0
    for label in LABELS:
        if _label_has_all_trials(existing_df, label):
            logger.info(f"[{label}] All grid search trials found in CSV. Skipping.")
            label_rows = existing_df[existing_df["label"] == label]
            for _, row in label_rows.iterrows():
                all_results.append(optimize.OptimizationResult(
                    label=row["label"],
                    gen_model=row["gen_model"],
                    reflection_model=row["reflection_model"],
                    optimized_program=None,
                    val_f1=row["val_f1"],
                    val_accuracy=row["val_accuracy"],
                ))
            trial_num += len(model_pairs)
            continue
        trainset = datasets[label]["trainset"]
        valset = datasets[label]["valset"]
        for gen_model, reflection_model in model_pairs:
            trial_num += 1
            logger.info(
                f"Trial {trial_num}/{total_trials} | label={label} | "
                f"gen={gen_model} | reflection={reflection_model}"
            )

            result = optimize.run(
                label=label,
                gen_model=gen_model,
                reflection_model=reflection_model,
                trainset=trainset,
                valset=valset,
                auto=effective_auto,
            )
            all_results.append(result)
        _save_summary(all_results)
    best_per_label = _select_best(all_results)

    # Restore programs for labels that were skipped.
    for label, result in best_per_label.items():
        if result.optimized_program is not None:
            continue
        program_dir = os.path.join(RESULTS_DIR, PROGRAMS_SUBDIR, label)
        if os.path.isdir(program_dir):
            logger.info(f"[{label}] Loading saved program from {program_dir}.")
            result.optimized_program = dspy.load(program_dir, allow_pickle=True)
        else:
            logger.info(
                f"[{label}] No saved program found. Re-running best pair: "
                f"gen={result.gen_model} | reflection={result.reflection_model}."
            )
            rerun = optimize.run(
                label=label,
                gen_model=result.gen_model,
                reflection_model=result.reflection_model,
                trainset=datasets[label]["trainset"],
                valset=datasets[label]["valset"],
                auto=effective_auto,
            )
            best_per_label[label] = rerun

    _log_best(best_per_label)
    return best_per_label


def _save_summary(results: list[optimize.OptimizationResult]) -> None:
    """Saves all trial results as a CSV to RESULTS_DIR/grid_search_results.csv.

    Excludes the compiled program objects (not serialisable). Includes label,
    gen_model, reflection_model, val_f1, and val_accuracy for every trial.

    Args:
        results: List of all OptimizationResult objects from the grid search.

    Returns:
        None.
    """
    if not results:
        raise ValueError("No grid search results to save.")
    rows = []
    for r in results:
        rows.append(
            {
                "label": r.label,
                "gen_model": r.gen_model,
                "reflection_model": r.reflection_model,
                "val_f1": r.val_f1,
                "val_accuracy": r.val_accuracy,
            }
        )
    df = pd.DataFrame(rows).sort_values(["label", "val_f1"], ascending=[True, False])
    path = os.path.join(RESULTS_DIR, GRID_SEARCH_CSV)
    df.to_csv(path, index=False)
    logger.info(f"Grid search summary saved to {path}.")
    logger.info(f"\n{df.to_string(index=False)}")


def _select_best(
    results: list[optimize.OptimizationResult],
) -> dict:
    """Selects the trial with the highest val_f1 per label.

    Ties are broken by val_accuracy, then by the order results were produced
    (earlier wins, which corresponds to the first model pair in the product).

    Args:
        results: All OptimizationResult objects from the grid search.

    Returns:
        Dict keyed by label name, value is the best OptimizationResult for
        that label.
    """
    best: dict[str, optimize.OptimizationResult] = {}
    for result in results:
        label = result.label
        if label not in best:
            best[label] = result
        else:
            current_best = best[label]
            if (result.val_f1, result.val_accuracy) > (
                current_best.val_f1,
                current_best.val_accuracy,
            ):
                best[label] = result

    return best


def _log_best(best_per_label: dict) -> None:
    """Logs the best model pair per label in a readable table.

    Args:
        best_per_label: Output of _select_best, keyed by label name.

    Returns:
        None.
    """
    logger.info("Best model pair per label:")
    rows = []
    for label, result in best_per_label.items():
        rows.append(
            {
                "label": label,
                "gen_model": result.gen_model,
                "reflection_model": result.reflection_model,
                "val_f1": f"{result.val_f1:.4f}",
                "val_accuracy": f"{result.val_accuracy:.4f}",
            }
        )
    df = pd.DataFrame(rows)
    logger.info(f"\n{df.to_string(index=False)}")
