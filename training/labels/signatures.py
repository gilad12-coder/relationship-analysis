"""Per-label DSPy Signatures for binary relationship classification.

Each signature defines the task description that GEPA will optimise. One
signature class per label, each with a ``text`` input and ``label`` output.
"""

import dspy


class RomanticSignature(dspy.Signature):
    """Classify whether the given text expresses a romantic relationship.
    A romantic relationship involves love, attraction, emotional intimacy,
    or intimate partnership between people. Return 'true' if the text
    expresses this, 'false' if it does not."""

    text: str = dspy.InputField(desc="The text to classify.")
    label: str = dspy.OutputField(desc="'true' or 'false'.")


class FamilySignature(dspy.Signature):
    """Classify whether the given text expresses a family relationship.
    A family relationship includes bonds between parents and children, siblings,
    grandparents, spouses in a family context, or any other relatives by blood
    or marriage. Return 'true' if the text expresses this, 'false' if it does not."""

    text: str = dspy.InputField(desc="The text to classify.")
    label: str = dspy.OutputField(desc="'true' or 'false'.")


class FriendshipSignature(dspy.Signature):
    """Classify whether the given text expresses a friendship relationship.
    A friendship is a close, voluntary, platonic bond between people based on
    mutual affection, trust, and shared experience, without romantic or familial
    obligations. Return 'true' if the text expresses this, 'false' if it does not."""

    text: str = dspy.InputField(desc="The text to classify.")
    label: str = dspy.OutputField(desc="'true' or 'false'.")


class ProfessionalSignature(dspy.Signature):
    """Classify whether the given text expresses a professional relationship.
    A professional relationship is defined by a work or organizational context,
    such as colleagues, employer and employee, business partners, client and
    service provider, or mentor and mentee in a professional setting.
    Return 'true' if the text expresses this, 'false' if it does not."""

    text: str = dspy.InputField(desc="The text to classify.")
    label: str = dspy.OutputField(desc="'true' or 'false'.")


class UnknownSignature(dspy.Signature):
    """Classify whether the given text expresses a relationship whose type
    cannot be determined. This label applies when there is clear evidence that
    a relationship exists between two or more people, but the text does not
    provide enough context to identify whether it is romantic, familial,
    professional, or a friendship. Return 'true' if the relationship is present
    but unidentifiable, 'false' otherwise."""

    text: str = dspy.InputField(desc="The text to classify.")
    label: str = dspy.OutputField(desc="'true' or 'false'.")


class IrrelevantSignature(dspy.Signature):
    """Classify whether the given text contains no relationship content at all.
    This label applies when the text does not describe or imply any interpersonal
    relationship between people — romantic, familial, professional, or otherwise.
    The text may be factual, descriptive, or about non-relational topics entirely.
    Return 'true' if the text is irrelevant to any relationship, 'false' if it
    contains any relational content."""

    text: str = dspy.InputField(desc="The text to classify.")
    label: str = dspy.OutputField(desc="'true' or 'false'.")
