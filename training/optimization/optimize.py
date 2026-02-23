"""
Optimization
============

Runs a single GEPA optimization for one (label, generation_model,
reflection_model) combination and returns the optimized program together with
its F1 score on the validation set.

This module is the unit of work for the grid search. It is intentionally
stateless: it receives all data and models as arguments, configures DSPy
locally for this run, and restores the prior LM configuration afterward so
sequential calls do not interfere with each other.
"""

from dataclasses import dataclass

import dspy
from loguru import logger
from sklearn.metrics import f1_score, precision_score, recall_score

from training.config import (
    GEPA_AUTO,
    GEPA_TRACK_STATS,
    LM_API_KEY,
    LM_BASE_URL,
    LM_MAX_TOKENS,
    LM_REASONING_EFFORT,
    LM_TEMPERATURE,
    METRIC_ZERO_DIVISION,
    POSITIVE_CLASS,
)
from training.labels.metrics import LABEL_REGISTRY


@dataclass
class OptimizationResult:
    """Holds the output of a single GEPA optimization run.

    Attributes:
        label:            The classification label this run targeted.
        gen_model:        DSPy model string used for generation.
        reflection_model: DSPy model string used for GEPA reflection.
        optimized_program: The compiled dspy.Module with an optimized prompt.
        val_f1:           F1 score on the validation set (positive class).
        val_accuracy:     Accuracy on the validation set.
    """

    label: str
    gen_model: str
    reflection_model: str
    optimized_program: dspy.Module
    val_f1: float
    val_accuracy: float
    holdout_f1: float = 0.0
    holdout_precision: float = 0.0
    holdout_recall: float = 0.0
    holdout_accuracy: float = 0.0


def _supports_reasoning_effort(model_name: str) -> bool:
    """Checks whether a model supports the reasoning_effort parameter.

    Args:
        model_name: DSPy-compatible model identifier (e.g. 'openai/o4-mini').

    Returns:
        True if the model is known to support reasoning_effort.
    """
    # Strip provider prefix (e.g. 'openai/o4-mini' -> 'o4-mini').
    name = model_name.split("/")[-1] if "/" in model_name else model_name
    return name.startswith("o")


def _lm_kwargs(model_name: str = "") -> dict:
    """Returns shared keyword arguments for a dspy.LM() call.

    Args:
        model_name: The model identifier, used to decide whether to include
                    reasoning_effort.

    Returns:
        Dict of LM configuration values derived from training config.
    """
    kwargs = {
        "api_key": LM_API_KEY,
        "temperature": LM_TEMPERATURE,
        "max_tokens": LM_MAX_TOKENS,
    }
    if LM_BASE_URL:
        kwargs["base_url"] = LM_BASE_URL
    if LM_REASONING_EFFORT and _supports_reasoning_effort(model_name):
        kwargs["reasoning_effort"] = LM_REASONING_EFFORT
    return kwargs


def _make_lm(model_name: str) -> dspy.LM:
    """Creates a dspy.LM instance with shared config kwargs.

    Args:
        model_name: A DSPy-compatible model identifier string.

    Returns:
        A configured dspy.LM instance.
    """
    return dspy.LM(model_name, **_lm_kwargs(model_name))


def _make_program(label: str) -> dspy.Module:
    """Creates a fresh ChainOfThought program from the label's registered signature.

    A new instance is created for each run so that GEPA never modifies shared
    state between optimization trials.

    Args:
        label: One of the six relationship labels.

    Returns:
        A new dspy.Module wrapping ChainOfThought over the label's signature.
    """
    signature = LABEL_REGISTRY[label]["signature"]

    class BinaryClassifier(dspy.Module):

        def __init__(self):
            """Initialises the per-label ChainOfThought classifier.

            Args:
                None.

            Returns:
                None.
            """
            self.classify = dspy.ChainOfThought(signature)

        def forward(self, text: str) -> dspy.Prediction:
            """Runs a single forward pass for relationship classification.

            Args:
                text: Input text to classify.

            Returns:
                DSPy prediction containing a binary label.
            """
            return self.classify(text=text)

    return BinaryClassifier()


def _score_on_valset(
    program: dspy.Module,
    valset: list[dspy.Example],
    label: str,
) -> tuple[float, float, float, float]:
    """Evaluates a compiled program on a set of examples.

    Predictions with values other than POSITIVE_CLASS/'false' are treated as
    negative. F1 is computed for the positive class (config.POSITIVE_CLASS).
    Inference failures raise to avoid silently corrupting metrics.

    Args:
        program: A compiled DSPy Module.
        valset:  List of dspy.Example with 'text' input and 'label' output.
        label:   Label name, used for contextual error messages.

    Returns:
        Tuple of (f1, precision, recall, accuracy).
    """
    y_true, y_pred = [], []
    for idx, example in enumerate(valset):
        try:
            pred = program(text=example.text)
            pred_label = str(pred.label).strip().lower()
        except Exception as exc:
            raise RuntimeError(
                f"Inference failed for label '{label}' at example index {idx}: {exc}"
            ) from exc

        y_true.append(1 if example.label == POSITIVE_CLASS else 0)
        y_pred.append(1 if pred_label == POSITIVE_CLASS else 0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=METRIC_ZERO_DIVISION)
    precision = precision_score(
        y_true, y_pred, pos_label=1, zero_division=METRIC_ZERO_DIVISION
    )
    recall = recall_score(
        y_true, y_pred, pos_label=1, zero_division=METRIC_ZERO_DIVISION
    )
    accuracy = (
        sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0
    )
    return f1, precision, recall, accuracy


def run(
    label: str,
    gen_model: str,
    reflection_model: str,
    trainset: list[dspy.Example],
    valset: list[dspy.Example],
    auto: str = None,
) -> OptimizationResult:
    """Runs one full GEPA optimization for a label and model pair.

    Configures DSPy to use gen_model for generation and reflection_model for
    GEPA's reflection step. Compiles the program on trainset, then scores on
    valset. The test set is never used here.

    Args:
        label:            The relationship label to optimize.
        gen_model:        DSPy-compatible model identifier for generation.
        reflection_model: DSPy-compatible model identifier for GEPA reflection.
        trainset:         dspy.Example list for GEPA reflective updates.
        valset:           dspy.Example list for post-compilation scoring.
        auto:             GEPA budget. Defaults to config.GEPA_AUTO.

    Returns:
        An OptimizationResult with the compiled program and val metrics.
    """
    effective_auto = auto if auto is not None else GEPA_AUTO
    logger.info(
        f"Starting GEPA run | label='{label}' | gen='{gen_model}' | "
        f"reflection='{reflection_model}' | auto='{effective_auto}'"
    )
    program = _make_program(label)
    metric = LABEL_REGISTRY[label]["metric"]
    gen_lm = _make_lm(gen_model)
    reflection_lm = _make_lm(reflection_model)
    dspy.configure(lm=gen_lm)
    optimizer = dspy.GEPA(
        metric=metric,
        auto=effective_auto,
        track_stats=GEPA_TRACK_STATS,
        reflection_lm=reflection_lm,
    )
    optimized_program = optimizer.compile(
        student=program,
        trainset=trainset,
        valset=valset,
    )
    val_f1, _val_p, _val_r, val_accuracy = _score_on_valset(optimized_program, valset, label)
    logger.info(
        f"GEPA run complete | label='{label}' | gen='{gen_model}' | "
        f"reflection='{reflection_model}' | val_f1={val_f1:.4f} | val_acc={val_accuracy:.4f}"
    )
    return OptimizationResult(
        label=label,
        gen_model=gen_model,
        reflection_model=reflection_model,
        optimized_program=optimized_program,
        val_f1=val_f1,
        val_accuracy=val_accuracy,
    )
