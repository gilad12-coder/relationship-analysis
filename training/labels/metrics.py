"""Per-label GEPA metrics and the label registry.

Each metric function returns a ``dspy.Prediction(score, feedback)`` where
the feedback is a natural-language explanation of the error, giving GEPA
actionable signal for reflective prompt evolution.

``LABEL_REGISTRY`` maps each label name to its signature class and metric
function, making it straightforward to iterate over all labels in the
optimisation pipeline.
"""

import dspy

from training.labels.signatures import (
    FamilySignature,
    FriendshipSignature,
    IrrelevantSignature,
    ProfessionalSignature,
    RomanticSignature,
    UnknownSignature,
)


def romantic_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
) -> dspy.Prediction:
    """GEPA metric for the romantic label.

    Args:
        gold: The ground truth dspy.Example with a 'label' field ('true'/'false').
        pred: The model's dspy.Prediction with a 'label' field.
        trace: Optional execution trace provided by DSPy.

    Returns:
        A dspy.Prediction with 'score' (1.0 or 0.0) and 'feedback' (str).
    """
    gold_label = str(gold.label).strip().lower()
    pred_label = str(pred.label).strip().lower()
    correct = gold_label == pred_label
    if correct:
        feedback = (
            f"Correct. The text was correctly classified as '{gold_label}' "
            f"for the romantic label."
        )
        score = 1.0
    elif gold_label == "true" and pred_label == "false":
        feedback = (
            "False negative. The correct label is 'true': this text expresses "
            "a romantic relationship, defined as evidence of love, attraction, "
            "emotional intimacy, or intimate partnership between people. "
            "The prediction was 'false'."
        )
        score = 0.0
    elif gold_label == "false" and pred_label == "true":
        feedback = (
            "False positive. The correct label is 'false': this text does not "
            "express a romantic relationship. A romantic relationship requires "
            "evidence of love, attraction, emotional intimacy, or intimate "
            "partnership between people, which is absent here. "
            "The prediction was 'true'."
        )
        score = 0.0
    else:
        feedback = (
            f"Invalid prediction '{pred_label}'. "
            f"The model must return exactly 'true' or 'false'. "
            f"The correct answer was '{gold_label}'."
        )
        score = 0.0
    return dspy.Prediction(score=score, feedback=feedback)


def family_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
) -> dspy.Prediction:
    """GEPA metric for the family label.

    Args:
        gold: The ground truth dspy.Example with a 'label' field ('true'/'false').
        pred: The model's dspy.Prediction with a 'label' field.
        trace: Optional execution trace provided by DSPy.

    Returns:
        A dspy.Prediction with 'score' (1.0 or 0.0) and 'feedback' (str).
    """
    gold_label = str(gold.label).strip().lower()
    pred_label = str(pred.label).strip().lower()
    correct = gold_label == pred_label
    if correct:
        feedback = (
            f"Correct. The text was correctly classified as '{gold_label}' "
            f"for the family label."
        )
        score = 1.0
    elif gold_label == "true" and pred_label == "false":
        feedback = (
            "False negative. The correct label is 'true': this text expresses "
            "a family relationship, defined as a bond between parents and children, "
            "siblings, grandparents, spouses in a family context, or any relatives "
            "by blood or marriage. The prediction was 'false'."
        )
        score = 0.0
    elif gold_label == "false" and pred_label == "true":
        feedback = (
            "False positive. The correct label is 'false': this text does not "
            "express a family relationship. A family relationship requires a bond "
            "between relatives by blood or marriage, which is absent here. "
            "The prediction was 'true'."
        )
        score = 0.0
    else:
        feedback = (
            f"Invalid prediction '{pred_label}'. "
            f"The model must return exactly 'true' or 'false'. "
            f"The correct answer was '{gold_label}'."
        )
        score = 0.0
    return dspy.Prediction(score=score, feedback=feedback)


def friendship_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
) -> dspy.Prediction:
    """GEPA metric for the friendship label.

    Args:
        gold: The ground truth dspy.Example with a 'label' field ('true'/'false').
        pred: The model's dspy.Prediction with a 'label' field.
        trace: Optional execution trace provided by DSPy.

    Returns:
        A dspy.Prediction with 'score' (1.0 or 0.0) and 'feedback' (str).
    """
    gold_label = str(gold.label).strip().lower()
    pred_label = str(pred.label).strip().lower()
    correct = gold_label == pred_label
    if correct:
        feedback = (
            f"Correct. The text was correctly classified as '{gold_label}' "
            f"for the friendship label."
        )
        score = 1.0
    elif gold_label == "true" and pred_label == "false":
        feedback = (
            "False negative. The correct label is 'true': this text expresses "
            "a friendship, defined as a close, voluntary, platonic bond between "
            "people based on mutual affection, trust, and shared experience, "
            "without romantic or familial obligations. The prediction was 'false'."
        )
        score = 0.0
    elif gold_label == "false" and pred_label == "true":
        feedback = (
            "False positive. The correct label is 'false': this text does not "
            "express a friendship. A friendship requires a voluntary platonic bond "
            "based on mutual affection and shared experience, independent of "
            "romantic or familial ties, which is absent here. "
            "The prediction was 'true'."
        )
        score = 0.0
    else:
        feedback = (
            f"Invalid prediction '{pred_label}'. "
            f"The model must return exactly 'true' or 'false'. "
            f"The correct answer was '{gold_label}'."
        )
        score = 0.0
    return dspy.Prediction(score=score, feedback=feedback)


def professional_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
) -> dspy.Prediction:
    """GEPA metric for the professional label.

    Args:
        gold: The ground truth dspy.Example with a 'label' field ('true'/'false').
        pred: The model's dspy.Prediction with a 'label' field.
        trace: Optional execution trace provided by DSPy.

    Returns:
        A dspy.Prediction with 'score' (1.0 or 0.0) and 'feedback' (str).
    """
    gold_label = str(gold.label).strip().lower()
    pred_label = str(pred.label).strip().lower()
    correct = gold_label == pred_label
    if correct:
        feedback = (
            f"Correct. The text was correctly classified as '{gold_label}' "
            f"for the professional label."
        )
        score = 1.0
    elif gold_label == "true" and pred_label == "false":
        feedback = (
            "False negative. The correct label is 'true': this text expresses "
            "a professional relationship, defined as a relationship rooted in a "
            "work or organisational context, such as colleagues, employer and "
            "employee, business partners, or client and service provider. "
            "The prediction was 'false'."
        )
        score = 0.0
    elif gold_label == "false" and pred_label == "true":
        feedback = (
            "False positive. The correct label is 'false': this text does not "
            "express a professional relationship. A professional relationship "
            "requires a work or organisational context as the defining basis of "
            "the bond between the people described, which is absent here. "
            "The prediction was 'true'."
        )
        score = 0.0
    else:
        feedback = (
            f"Invalid prediction '{pred_label}'. "
            f"The model must return exactly 'true' or 'false'. "
            f"The correct answer was '{gold_label}'."
        )
        score = 0.0
    return dspy.Prediction(score=score, feedback=feedback)


def unknown_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
) -> dspy.Prediction:
    """GEPA metric for the unknown label.

    Args:
        gold: The ground truth dspy.Example with a 'label' field ('true'/'false').
        pred: The model's dspy.Prediction with a 'label' field.
        trace: Optional execution trace provided by DSPy.

    Returns:
        A dspy.Prediction with 'score' (1.0 or 0.0) and 'feedback' (str).
    """
    gold_label = str(gold.label).strip().lower()
    pred_label = str(pred.label).strip().lower()
    correct = gold_label == pred_label
    if correct:
        feedback = (
            f"Correct. The text was correctly classified as '{gold_label}' "
            f"for the unknown label."
        )
        score = 1.0
    elif gold_label == "true" and pred_label == "false":
        feedback = (
            "False negative. The correct label is 'true': this text contains "
            "a relationship between people, but the available context is "
            "insufficient to determine whether it is romantic, familial, "
            "professional, or a friendship. The presence of a relationship is "
            "clear, but its type is not. The prediction was 'false'."
        )
        score = 0.0
    elif gold_label == "false" and pred_label == "true":
        feedback = (
            "False positive. The correct label is 'false': this text does not "
            "meet the criteria for the unknown label. The unknown label requires "
            "that a relationship is detectably present but its type cannot be "
            "determined from context. Either no relationship is present, or the "
            "relationship type is identifiable here. The prediction was 'true'."
        )
        score = 0.0
    else:
        feedback = (
            f"Invalid prediction '{pred_label}'. "
            f"The model must return exactly 'true' or 'false'. "
            f"The correct answer was '{gold_label}'."
        )
        score = 0.0
    return dspy.Prediction(score=score, feedback=feedback)


def irrelevant_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
) -> dspy.Prediction:
    """GEPA metric for the irrelevant label.

    Args:
        gold: The ground truth dspy.Example with a 'label' field ('true'/'false').
        pred: The model's dspy.Prediction with a 'label' field.
        trace: Optional execution trace provided by DSPy.

    Returns:
        A dspy.Prediction with 'score' (1.0 or 0.0) and 'feedback' (str).
    """
    gold_label = str(gold.label).strip().lower()
    pred_label = str(pred.label).strip().lower()
    correct = gold_label == pred_label
    if correct:
        feedback = (
            f"Correct. The text was correctly classified as '{gold_label}' "
            f"for the irrelevant label."
        )
        score = 1.0
    elif gold_label == "true" and pred_label == "false":
        feedback = (
            "False negative. The correct label is 'true': this text contains "
            "no interpersonal relationship content of any kind — not romantic, "
            "familial, professional, friendship, or unknown. The text does not "
            "describe or imply a bond between people. The prediction was 'false'."
        )
        score = 0.0
    elif gold_label == "false" and pred_label == "true":
        feedback = (
            "False positive. The correct label is 'false': this text does "
            "contain interpersonal relationship content and should not be "
            "classified as irrelevant. The irrelevant label requires the complete "
            "absence of any relational bond between people, which is not the case "
            "here. The prediction was 'true'."
        )
        score = 0.0
    else:
        feedback = (
            f"Invalid prediction '{pred_label}'. "
            f"The model must return exactly 'true' or 'false'. "
            f"The correct answer was '{gold_label}'."
        )
        score = 0.0
    return dspy.Prediction(score=score, feedback=feedback)


LABEL_REGISTRY: dict[str, dict] = {
    "romantic": {
        "signature": RomanticSignature,
        "metric": romantic_metric,
    },
    "family": {
        "signature": FamilySignature,
        "metric": family_metric,
    },
    "friendship": {
        "signature": FriendshipSignature,
        "metric": friendship_metric,
    },
    "professional": {
        "signature": ProfessionalSignature,
        "metric": professional_metric,
    },
    "unknown": {
        "signature": UnknownSignature,
        "metric": unknown_metric,
    },
    "irrelevant": {
        "signature": IrrelevantSignature,
        "metric": irrelevant_metric,
    },
}
