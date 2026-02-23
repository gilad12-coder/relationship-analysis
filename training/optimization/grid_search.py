"""
Grid Search
===========

Runs GEPA optimization for every combination of (generation_model,
reflection_model) from the configured model lists in training/config/settings.py,
independently for each of the six relationship labels.

Each trial is evaluated on both the validation and selection holdout sets. The
trial with the highest holdout F1 per label is selected as the best
configuration.

Results from all trials are saved to a CSV for inspection.
"""

import itertools
import json
import os

import dspy
import pandas as pd
from loguru import logger
from tqdm import tqdm

from training.config import (
    GEPA_AUTO,
    GENERATION_MODELS,
    GRID_SEARCH_CSV,
    LABELS,
    PROGRAMS_MANIFEST,
    PROGRAMS_SUBDIR,
    REFLECTION_MODELS,
    RESULTS_DIR,
)
from training.data.dataset_builder import df_to_examples
from training.optimization import optimize
from training.optimization.optimize import _make_lm, _score_on_valset


_REQUIRED_CSV_COLUMNS = {"holdout_f1", "holdout_precision", "holdout_recall", "holdout_accuracy"}


def _configured_model_pairs() -> set[tuple[str, str]]:
    """Returns the currently configured (gen_model, reflection_model) pairs.

    Args:
        None.

    Returns:
        Set of all configured model-pair tuples.
    """
    return set(itertools.product(GENERATION_MODELS, REFLECTION_MODELS))


def _load_existing_csv() -> pd.DataFrame | None:
    """Loads the existing grid search CSV if it exists.

    Returns:
        DataFrame of previous results, or None if no CSV exists.
    """
    path = os.path.join(RESULTS_DIR, GRID_SEARCH_CSV)
    if os.path.isfile(path):
        return pd.read_csv(path)
    return None


def _label_has_all_trials(
    existing_df: pd.DataFrame | None, label: str, dataset_hash: str
) -> bool:
    """Checks if all expected grid search trials exist for a label.

    Returns False if the CSV is missing required holdout metric columns (legacy
    format), or if the stored dataset hash does not match the current one,
    so stale rows are never reused.

    Args:
        existing_df:  Existing grid search results DataFrame, or None.
        label:        Label name.
        dataset_hash: Hash of the current dataset.

    Returns:
        True if all (gen_model, reflection_model) pairs have complete results
        built from the same dataset.
    """
    if existing_df is None:
        return False
    if not _REQUIRED_CSV_COLUMNS.issubset(existing_df.columns):
        return False
    if "dataset_hash" not in existing_df.columns:
        return False
    label_rows = existing_df[existing_df["label"] == label]
    if label_rows.empty:
        return False
    if not (label_rows["dataset_hash"] == dataset_hash).all():
        return False
    expected = _configured_model_pairs()
    actual = set(zip(label_rows["gen_model"], label_rows["reflection_model"]))
    return expected == actual


def _label_rows_for_configured_pairs(existing_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Returns CSV rows for one label restricted to currently configured model pairs.

    Args:
        existing_df: Existing grid-search results DataFrame.
        label:       Label name.

    Returns:
        DataFrame containing only rows for the label and configured model pairs.
    """
    configured_pairs = _configured_model_pairs()
    label_rows = existing_df[existing_df["label"] == label].copy()
    mask = [(gen, refl) in configured_pairs for gen, refl in zip(label_rows["gen_model"], label_rows["reflection_model"])]
    return label_rows[mask]


def _load_program_manifest() -> dict:
    """Loads the saved programs manifest from disk.

    Args:
        None.

    Returns:
        Manifest dict keyed by label, or an empty dict if not found/invalid.
    """
    path = os.path.join(RESULTS_DIR, PROGRAMS_SUBDIR, PROGRAMS_MANIFEST)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        logger.warning(f"Ignoring invalid manifest at {path}: expected JSON object.")
        return {}
    return manifest


def _manifest_matches_trial(
    manifest: dict,
    label: str,
    gen_model: str,
    reflection_model: str,
    dataset_hash: str,
) -> bool:
    """Checks whether a manifest entry matches a selected trial and dataset hash.

    Args:
        manifest:          Loaded programs manifest.
        label:             Label name.
        gen_model:         Selected generation model.
        reflection_model:  Selected reflection model.
        dataset_hash:      Current dataset hash.

    Returns:
        True when manifest metadata matches the selected trial exactly.
    """
    entry = manifest.get(label, {})
    saved_gen = entry.get("gen_model", "") if isinstance(entry, dict) else entry
    saved_refl = entry.get("reflection_model", "") if isinstance(entry, dict) else ""
    saved_hash = entry.get("dataset_hash", "") if isinstance(entry, dict) else ""
    return (
        saved_gen == gen_model
        and saved_refl == reflection_model
        and saved_hash == dataset_hash
    )


def _manifest_entry_matches_fast_path(
    manifest: dict,
    label: str,
    dataset_hash: str,
    configured_pairs: set[tuple[str, str]],
) -> bool:
    """Checks whether a manifest entry is valid for full fast-path resume.

    Args:
        manifest:         Loaded programs manifest.
        label:            Label name.
        dataset_hash:     Current dataset hash.
        configured_pairs: Currently configured (gen_model, reflection_model) pairs.

    Returns:
        True if the entry exists, matches the dataset hash, belongs to current
        configured model pairs, and has a program directory on disk.
    """
    entry = manifest.get(label)
    if not isinstance(entry, dict):
        return False
    gen_model = entry.get("gen_model", "")
    reflection_model = entry.get("reflection_model", "")
    if entry.get("dataset_hash", "") != dataset_hash:
        return False
    if (gen_model, reflection_model) not in configured_pairs:
        return False
    program_dir = os.path.join(RESULTS_DIR, PROGRAMS_SUBDIR, label)
    return os.path.isdir(program_dir)


def _save_program(
    label: str, result: optimize.OptimizationResult, dataset_hash: str
) -> None:
    """Saves a label's best program to disk and updates the manifest.

    Args:
        label:        Label name.
        result:       OptimizationResult with the program to save.
        dataset_hash: Hash of the dataset used for this optimization.
    """
    programs_dir = os.path.join(RESULTS_DIR, PROGRAMS_SUBDIR)
    label_dir = os.path.join(programs_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    result.optimized_program.save(label_dir, save_program=True)
    manifest = _load_program_manifest()
    manifest[label] = {
        "gen_model": result.gen_model,
        "reflection_model": result.reflection_model,
        "dataset_hash": dataset_hash,
    }
    manifest_path = os.path.join(programs_dir, PROGRAMS_MANIFEST)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"[{label}] Program saved to {label_dir}.")


def run(datasets: dict, auto: str = None, dataset_hash: str = "") -> dict:
    """Runs the full grid search over all model pairs and all labels.

    Supports resuming: labels whose trials already exist in the grid search
    CSV are skipped, provided they match the current dataset hash. Programs
    are loaded from disk when available; otherwise only the best pair is re-run.

    Args:
        datasets:     Output of dataset_builder.build(), keyed by label name.
                      Each entry must contain 'trainset', 'valset', and 'holdout_df'.
        auto:         GEPA budget override. If None, uses the value in config.GEPA_AUTO.
        dataset_hash: Hash of the current dataset for staleness detection.

    Returns:
        A dict keyed by label name, each containing the best OptimizationResult
        (highest holdout_f1) across all model pair trials for that label.
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

    # Fast path: if all labels already have programs with matching dataset hash,
    # load them and skip the grid search entirely.
    manifest = _load_program_manifest()
    configured_pairs = _configured_model_pairs()
    if dataset_hash and manifest and all(
        _manifest_entry_matches_fast_path(
            manifest=manifest,
            label=label,
            dataset_hash=dataset_hash,
            configured_pairs=configured_pairs,
        )
        for label in LABELS
    ):
        logger.info("All programs match current dataset hash. Loading from disk.")
        best = {}
        for label in LABELS:
            entry = manifest[label]
            program_dir = os.path.join(RESULTS_DIR, PROGRAMS_SUBDIR, label)
            best[label] = optimize.OptimizationResult(
                label=label,
                gen_model=entry["gen_model"],
                reflection_model=entry["reflection_model"],
                optimized_program=dspy.load(program_dir, allow_pickle=True),
                val_f1=0.0,
                val_accuracy=0.0,
            )
            logger.info(f"[{label}] Loaded from {program_dir}.")
        _log_best(best)
        return best

    os.makedirs(RESULTS_DIR, exist_ok=True)
    existing_df = _load_existing_csv()
    all_results: list[optimize.OptimizationResult] = []
    model_pairs = list(itertools.product(GENERATION_MODELS, REFLECTION_MODELS))
    total_trials = len(LABELS) * len(model_pairs)
    pbar = tqdm(total=total_trials, desc="Grid search", unit="trial")
    for label in LABELS:
        if _label_has_all_trials(existing_df, label, dataset_hash):
            logger.info(f"[{label}] All grid search trials found in CSV. Skipping.")
            label_rows = _label_rows_for_configured_pairs(existing_df, label)
            label_rows = label_rows.sort_values(
                ["holdout_f1", "holdout_accuracy"], ascending=[False, False]
            ).drop_duplicates(subset=["gen_model", "reflection_model"], keep="first")
            for _, row in label_rows.iterrows():
                all_results.append(optimize.OptimizationResult(
                    label=row["label"],
                    gen_model=row["gen_model"],
                    reflection_model=row["reflection_model"],
                    optimized_program=None,
                    val_f1=row["val_f1"],
                    val_accuracy=row["val_accuracy"],
                    holdout_f1=row["holdout_f1"],
                    holdout_precision=row["holdout_precision"],
                    holdout_recall=row["holdout_recall"],
                    holdout_accuracy=row["holdout_accuracy"],
                ))
            pbar.update(len(model_pairs))
            # Persist skipped-label rows too, so mixed resume runs keep full CSV coverage.
            _save_summary(all_results, dataset_hash)
        else:
            trainset = datasets[label]["trainset"]
            valset = datasets[label]["valset"]
            holdout_set = df_to_examples(datasets[label]["holdout_df"], label)
            for gen_model, reflection_model in model_pairs:
                gen_short = gen_model.split("/")[-1]
                refl_short = reflection_model.split("/")[-1]
                pbar.set_postfix_str(f"{label} | {gen_short}+{refl_short}")
                logger.info(
                    f"Trial {pbar.n + 1}/{total_trials} | label={label} | "
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
                # Evaluate on selection holdout (LM is still configured from optimize.run).
                dspy.configure(lm=_make_lm(gen_model))
                h_f1, h_p, h_r, h_acc = _score_on_valset(
                    result.optimized_program, holdout_set, label
                )
                result.holdout_f1 = h_f1
                result.holdout_precision = h_p
                result.holdout_recall = h_r
                result.holdout_accuracy = h_acc
                pbar.write(
                    f"  {label} | {gen_short}+{refl_short} | "
                    f"F1={h_f1:.4f} P={h_p:.4f} R={h_r:.4f}"
                )
                all_results.append(result)
                _save_summary(all_results, dataset_hash)
                pbar.update(1)
            # Save the best program for this label immediately.
            label_results = [r for r in all_results if r.label == label and r.optimized_program is not None]
            if label_results:
                best_for_label = max(label_results, key=lambda r: (r.holdout_f1, r.holdout_accuracy))
                _save_program(label, best_for_label, dataset_hash)
    pbar.close()
    best_per_label = _select_best(all_results)

    # Restore programs for labels that were skipped.
    manifest = _load_program_manifest()
    for label, result in best_per_label.items():
        if result.optimized_program is not None:
            continue
        program_dir = os.path.join(RESULTS_DIR, PROGRAMS_SUBDIR, label)
        if os.path.isdir(program_dir) and _manifest_matches_trial(
            manifest=manifest,
            label=label,
            gen_model=result.gen_model,
            reflection_model=result.reflection_model,
            dataset_hash=dataset_hash,
        ):
            logger.info(f"[{label}] Loading saved program from {program_dir}.")
            result.optimized_program = dspy.load(program_dir, allow_pickle=True)
        else:
            if os.path.isdir(program_dir):
                logger.warning(
                    f"[{label}] Saved program metadata does not match selected pair/hash. "
                    f"Re-running best pair: gen={result.gen_model} | reflection={result.reflection_model}."
                )
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


def _save_summary(results: list[optimize.OptimizationResult], dataset_hash: str) -> None:
    """Saves all trial results as a CSV to RESULTS_DIR/grid_search_results.csv.

    Args:
        results:      List of all OptimizationResult objects from the grid search.
        dataset_hash: Hash of the dataset used for these trials.

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
                "holdout_f1": r.holdout_f1,
                "holdout_precision": r.holdout_precision,
                "holdout_recall": r.holdout_recall,
                "holdout_accuracy": r.holdout_accuracy,
                "dataset_hash": dataset_hash,
            }
        )
    df = pd.DataFrame(rows).sort_values(["label", "holdout_f1"], ascending=[True, False])
    path = os.path.join(RESULTS_DIR, GRID_SEARCH_CSV)
    df.to_csv(path, index=False)
    logger.info(f"Grid search summary saved to {path}.")
    logger.info(f"\n{df.to_string(index=False)}")


def _select_best(
    results: list[optimize.OptimizationResult],
) -> dict:
    """Selects the trial with the highest holdout_f1 per label.

    Ties are broken by holdout_accuracy, then by the order results were produced.

    Args:
        results: All OptimizationResult objects from the grid search.

    Returns:
        Dict keyed by label name, value is the best OptimizationResult.
    """
    best: dict[str, optimize.OptimizationResult] = {}
    for result in results:
        label = result.label
        if label not in best:
            best[label] = result
        else:
            current_best = best[label]
            if (result.holdout_f1, result.holdout_accuracy) > (
                current_best.holdout_f1,
                current_best.holdout_accuracy,
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
    logger.info("Best model pair per label (selected by holdout_f1):")
    rows = []
    for label, result in best_per_label.items():
        rows.append(
            {
                "label": label,
                "gen_model": result.gen_model,
                "reflection_model": result.reflection_model,
                "val_f1": f"{result.val_f1:.4f}",
                "holdout_f1": f"{result.holdout_f1:.4f}",
                "holdout_P": f"{result.holdout_precision:.4f}",
                "holdout_R": f"{result.holdout_recall:.4f}",
            }
        )
    df = pd.DataFrame(rows)
    logger.info(f"\n{df.to_string(index=False)}")
