"""
Final Evaluation
================

Evaluates each label's best GEPA-optimised classifier on its locked test set.
This module is the only place in the pipeline where the test set is used. It
must be called exactly once.

Output:
  - RESULTS_DIR/FINAL_EVAL_CSV          — one row per label, all metrics.
  - RESULTS_DIR/PREDICTIONS_SUBDIR/{label}_predictions.csv — per-example predictions.
"""

import os

import pandas as pd
from loguru import logger
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from training.config import (
    FINAL_EVAL_CSV,
    LABELS,
    METRIC_ZERO_DIVISION,
    POSITIVE_CLASS,
    PREDICTIONS_SUBDIR,
    RESULTS_DIR,
)
from training.data.dataset_builder import df_to_examples
from training.optimization.optimize import OptimizationResult, _temporary_lm


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
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["negative", "positive"],
        zero_division=METRIC_ZERO_DIVISION,
    )
    logger.info(
        f"[{label}] FINAL TEST | "
        f"F1={f1:.4f} | P={precision:.4f} | R={recall:.4f} | Acc={accuracy:.4f}"
    )
    logger.info(f"[{label}] Classification report:\n{report}")
    logger.info(f"[{label}] Confusion matrix:\n{cm}")
    metrics = {
        "label": label,
        "test_f1": f1,
        "test_precision": precision,
        "test_recall": recall,
        "test_accuracy": accuracy,
        "n_test": len(y_true),
        "n_test_positive": sum(y_true),
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


def _save_results(rows: list[dict], preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Saves the final evaluation CSV and per-label prediction files.

    Args:
        rows:  List of metrics dicts, one per label.
        preds: Dict keyed by label name, value is the predictions DataFrame.

    Returns:
        Summary DataFrame with one row per label.
    """
    preds_dir = os.path.join(RESULTS_DIR, PREDICTIONS_SUBDIR)
    os.makedirs(preds_dir, exist_ok=True)
    for label, preds_df in preds.items():
        path = os.path.join(preds_dir, f"{label}_predictions.csv")
        preds_df.to_csv(path, index=False)
        logger.info(f"[{label}] Predictions saved to {path}.")
    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(RESULTS_DIR, FINAL_EVAL_CSV)
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Final evaluation saved to {summary_path}.")
    logger.info(f"\n{summary_df.to_string(index=False)}")
    return summary_df


def run_dspy(best_per_label: dict, datasets: dict) -> pd.DataFrame:
    """Evaluates all GEPA-optimised programs on their locked test sets.

    Args:
        best_per_label: Output of grid_search.run(), keyed by label name,
                        each value an OptimizationResult.
        datasets:       Output of dataset_builder.build(), keyed by label name.

    Returns:
        Summary DataFrame with one row per label.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    preds = {}
    for label in LABELS:
        result: OptimizationResult = best_per_label[label]
        test_df: pd.DataFrame = datasets[label]["test_df"]
        testset = df_to_examples(test_df, label)
        y_true, y_pred, texts = [], [], []
        with _temporary_lm(result.gen_model):
            for idx, example in enumerate(testset):
                try:
                    prediction = result.optimized_program(text=example.text)
                    pred_label = str(prediction.label).strip().lower()
                except Exception as exc:
                    raise RuntimeError(
                        f"[{label}] Test inference failed at example index {idx}: {exc}"
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
        }
        metrics, preds_df = _compute_metrics(label, y_true, y_pred, texts, extra)
        rows.append(metrics)
        preds[label] = preds_df
    return _save_results(rows, preds)
