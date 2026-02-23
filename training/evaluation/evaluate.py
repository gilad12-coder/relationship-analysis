"""
Final Evaluation
================

Evaluates each label's classifier on its locked selection holdout set — both
the baseline (un-optimised) program and the best GEPA-optimised program.

Output:
  - RESULTS_DIR/EVAL_CSV — one row per (label, stage), all metrics.
  - RESULTS_DIR/PREDICTIONS_SUBDIR/{stage}_{label}_predictions.csv — per-example predictions.
"""

import os

import dspy
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score, precision_score, recall_score

from training.config import (
    EVAL_CSV,
    METRIC_ZERO_DIVISION,
    POSITIVE_CLASS,
    PREDICTIONS_SUBDIR,
    RESULTS_DIR,
)
from training.data.dataset_builder import df_to_examples
from training.optimization.optimize import OptimizationResult, _make_lm, _make_program


def _compute_metrics(
    label: str,
    y_true: list[int],
    y_pred: list[int],
    texts: list[str],
    extra: dict,
) -> tuple[dict, pd.DataFrame]:
    """Computes classification metrics and builds a predictions DataFrame.

    Args:
        label:  Label name.
        y_true: Ground truth binary integers.
        y_pred: Predicted binary integers.
        texts:  Raw text strings, aligned with y_true/y_pred.
        extra:  Additional fields to include in the metrics dict (e.g. model info).

    Returns:
        Tuple of (metrics_dict, predictions_df).
    """
    if not y_true:
        raise ValueError(f"[{label}] Empty evaluation set; cannot compute metrics.")
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=METRIC_ZERO_DIVISION)
    precision = precision_score(
        y_true, y_pred, pos_label=1, zero_division=METRIC_ZERO_DIVISION
    )
    recall = recall_score(
        y_true, y_pred, pos_label=1, zero_division=METRIC_ZERO_DIVISION
    )
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    logger.info(
        f"[{label}] HOLDOUT | "
        f"F1={f1:.4f} | P={precision:.4f} | R={recall:.4f} | Acc={accuracy:.4f}"
    )
    metrics = {
        "label": label,
        "holdout_f1": f1,
        "holdout_precision": precision,
        "holdout_recall": recall,
        "holdout_accuracy": accuracy,
        "n_holdout": len(y_true),
        "n_holdout_positive": sum(y_true),
        **extra,
    }
    preds_df = pd.DataFrame(
        {
            "text": texts,
            "gold": [POSITIVE_CLASS if t == 1 else "false" for t in y_true],
            "predicted": [POSITIVE_CLASS if p == 1 else "false" for p in y_pred],
            "correct": [t == p for t, p in zip(y_true, y_pred)],
        }
    )
    return metrics, preds_df


def _save_label_results(
    label: str, stage: str, metrics: dict, preds_df: pd.DataFrame
) -> None:
    """Saves one label's evaluation results to disk.

    Writes the predictions CSV and appends/updates the summary eval CSV
    so that results are persisted after each label finishes.

    Args:
        label:    Label name.
        stage:    'baseline' or 'optimized'.
        metrics:  Metrics dict for this label.
        preds_df: Predictions DataFrame for this label.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    preds_dir = os.path.join(RESULTS_DIR, PREDICTIONS_SUBDIR)
    os.makedirs(preds_dir, exist_ok=True)

    metrics["stage"] = stage

    path = os.path.join(preds_dir, f"{stage}_{label}_predictions.csv")
    preds_df.to_csv(path, index=False)
    logger.info(f"[{label}] {stage} predictions saved to {path}.")

    summary_path = os.path.join(RESULTS_DIR, EVAL_CSV)
    new_row = pd.DataFrame([metrics])
    if os.path.isfile(summary_path):
        existing = pd.read_csv(summary_path)
        existing = existing[
            ~((existing["label"] == label) & (existing["stage"] == stage))
        ]
        summary_df = pd.concat([existing, new_row], ignore_index=True)
    else:
        summary_df = new_row
    tmp_path = summary_path + ".tmp"
    summary_df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, summary_path)
    logger.info(f"[{label}] {stage} evaluation saved to {summary_path}.")


def run_dspy_label(
    label: str, result: OptimizationResult, datasets: dict, dataset_hash: str = ""
) -> dict:
    """Evaluates one label's GEPA-optimised program on its locked holdout set and saves results.

    Args:
        label:        Label name.
        result:       OptimizationResult for this label.
        datasets:     Output of dataset_builder.build(), keyed by label name.
        dataset_hash: Hash of the current dataset for staleness detection.

    Returns:
        Metrics dict for this label.
    """
    holdout_df: pd.DataFrame = datasets[label]["holdout_df"]
    holdout_set = df_to_examples(holdout_df, label)
    y_true, y_pred, texts = [], [], []
    dspy.configure(lm=_make_lm(result.gen_model))
    for idx, example in enumerate(holdout_set):
        try:
            prediction = result.optimized_program(text=example.text)
            pred_label = str(prediction.label).strip().lower()
        except Exception as exc:
            raise RuntimeError(
                f"[{label}] Holdout inference failed at example index {idx}: {exc}"
            ) from exc

        y_true.append(1 if example.label == POSITIVE_CLASS else 0)
        y_pred.append(1 if pred_label == POSITIVE_CLASS else 0)
        texts.append(example.text)
    try:
        optimized_prompt = result.optimized_program.classify.signature.__doc__ or ""
    except AttributeError:
        optimized_prompt = ""
    extra = {
        "gen_model": result.gen_model,
        "reflection_model": result.reflection_model,
        "optimized_prompt": optimized_prompt,
        "dataset_hash": dataset_hash,
    }
    metrics, preds_df = _compute_metrics(label, y_true, y_pred, texts, extra)
    _save_label_results(label, "optimized", metrics, preds_df)
    return metrics


def run_baseline_label(
    label: str, gen_model: str, datasets: dict, dataset_hash: str = ""
) -> dict:
    """Evaluates one label's un-optimised (baseline) program on its locked holdout set.

    Args:
        label:        Label name.
        gen_model:    DSPy model string to use for inference.
        datasets:     Output of dataset_builder.build(), keyed by label name.
        dataset_hash: Hash of the current dataset for staleness detection.

    Returns:
        Metrics dict for this label.
    """
    holdout_df: pd.DataFrame = datasets[label]["holdout_df"]
    holdout_set = df_to_examples(holdout_df, label)
    y_true, y_pred, texts = [], [], []
    program = _make_program(label)
    dspy.configure(lm=_make_lm(gen_model))
    for idx, example in enumerate(holdout_set):
        try:
            prediction = program(text=example.text)
            pred_label = str(prediction.label).strip().lower()
        except Exception as exc:
            raise RuntimeError(
                f"[{label}] Baseline holdout inference failed at example index {idx}: {exc}"
            ) from exc
        y_true.append(1 if example.label == POSITIVE_CLASS else 0)
        y_pred.append(1 if pred_label == POSITIVE_CLASS else 0)
        texts.append(example.text)
    extra = {"gen_model": gen_model, "dataset_hash": dataset_hash, "reflection_model": "", "optimized_prompt": ""}
    logger.info(f"[{label}] BASELINE (gen={gen_model}):")
    metrics, preds_df = _compute_metrics(label, y_true, y_pred, texts, extra)
    _save_label_results(label, "baseline", metrics, preds_df)
    return metrics
