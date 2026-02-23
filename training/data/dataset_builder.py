"""
Dataset Builder
===============

Converts the per-label DSPy train/val/holdout DataFrames produced by
split_pipeline into the dspy.Example lists consumed by GEPA optimization
and evaluation.
"""

import dspy
import pandas as pd
from loguru import logger

from training.config import LABELS, TEXT_COLUMN


def df_to_examples(df: pd.DataFrame, label: str) -> list[dspy.Example]:
    """Converts a DataFrame slice into a list of dspy.Example objects.

    Args:
        df:    DataFrame with a TEXT_COLUMN column and a boolean label column.
        label: Name of the label column.

    Returns:
        List of dspy.Example with input key 'text' and output field 'label'
        ('true' or 'false').
    """
    examples = []
    for _, row in df.iterrows():
        example = dspy.Example(
            text=str(row[TEXT_COLUMN]),
            label="true" if bool(row[label]) else "false",
        ).with_inputs("text")
        examples.append(example)
    return examples


def build(splits: dict) -> dict:
    """Builds GEPA-ready example lists from the split pipeline output.

    Args:
        splits: Output of split_pipeline.run(). Keyed by label name, each
                a LabelSplit with .holdout, .dspy.train, .dspy.val.

    Returns:
        Dict keyed by label name, each containing:
            - trainset:    list[dspy.Example] for GEPA reflective updates.
            - valset:      list[dspy.Example] for Pareto tracking and model selection.
            - holdout_df:  pd.DataFrame, locked selection holdout passed through untouched.
    """
    datasets = {}
    for label in LABELS:
        train_df = splits[label].dspy.train
        val_df = splits[label].dspy.val
        holdout_df = splits[label].holdout
        trainset = df_to_examples(train_df, label)
        valset = df_to_examples(val_df, label)
        datasets[label] = {
            "trainset": trainset,
            "valset": valset,
            "holdout_df": holdout_df,
        }
        n_train_pos = sum(1 for e in trainset if e.label == "true")
        n_val_pos = sum(1 for e in valset if e.label == "true")
        logger.info(
            f"Label '{label}': "
            f"trainset={len(trainset)} ({n_train_pos} pos) | "
            f"valset={len(valset)} ({n_val_pos} pos) | "
            f"holdout={len(holdout_df)} ({int(holdout_df[label].sum())} pos)."
        )
    return datasets
