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

import pandas as pd
from loguru import logger

from training.config import (
    GEPA_AUTO,
    GENERATION_MODELS,
    GRID_SEARCH_CSV,
    LABELS,
    REFLECTION_MODELS,
    RESULTS_DIR,
)
from training.optimization import optimize


def run(datasets: dict, auto: str = None) -> dict:
    """Runs the full grid search over all model pairs and all labels.

    Iterates every (gen_model, reflection_model) pair for every label, runs a
    GEPA optimization trial, records the val F1, and selects the best pair per
    label. Saves a summary CSV of all results to RESULTS_DIR.

    Args:
        datasets: Output of dataset_builder.build(), keyed by label name.
                  Each entry must contain 'trainset', 'valset', and 'test_df'.
        auto:     GEPA budget override. If None, uses the value in config.GEPA_AUTO.

    Returns:
        A dict keyed by label name, each containing the best OptimizationResult
        (highest val_f1) across all 9 model pair trials for that label.
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
    all_results: list[optimize.OptimizationResult] = []
    model_pairs = list(itertools.product(GENERATION_MODELS, REFLECTION_MODELS))
    total_trials = len(LABELS) * len(model_pairs)
    trial_num = 0
    for label in LABELS:
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
