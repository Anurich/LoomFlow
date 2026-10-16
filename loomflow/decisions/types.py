"""Question and answer types for System One decisions.

These mirror the TypeSafe wire API exactly (``typesafe_sdk`` uses the
same four primitive names), so code written against loomflow's types
translates 1:1 to the vendor docs — and so the LLM-backed fallback
must reproduce the same semantics, down to :class:`NoulDecision`
carrying a bare probability with NO separate confidence field.

A *question* is a specific, well-scoped judgment ("which team handles
this?", "is this urgent?") — something a knowledgeable person could
answer in seconds. Decompose bigger judgments into several atomic
questions and combine the answers in code; questions in one
``decide()`` call are evaluated together, so extra questions are
nearly free.

Images: any :class:`~loomflow.Image` can sit anywhere in the state,
a question's ``instructions``, or its ``criteria`` (a Choice option
described by an example photo, a Score level, a Noul outcome) — as a
value, alone or nested in dicts/lists. :class:`Box` locates something
in an image.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..core.types import Image, Usage

__all__ = [
    "Box",
    "BoxDecision",
    "Choice",
    "ChoiceDecision",
    "DecisionAnswer",
    "DecisionContent",
    "Decisions",
    "Noul",
    "NoulDecision",
    "Question",
    "Score",
    "ScoreDecision",
    "describe_content",
]


# Text, an image, or a JSON-like dict/list that may nest images —
# the wire API's ``Content`` shape for instructions and criteria.
DecisionContent = str | Image | Mapping[str, Any] | Sequence[Any]

# Noul outcome keys. The wire API spells them ``true`` / ``false``;
# ``yes`` / ``no`` are accepted aliases (loomflow's original spelling).
_NOUL_KEYS = {"true": "true", "yes": "true", "false": "false", "no": "false"}


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Choice:
    """Select ONE option from a defined set.

    ``criteria`` maps each option key to a description that helps the
    model discriminate — keep descriptions specific and mutually
    distinguishing. A description may be text, an example image, a
    dict like ``{"description": ..., "examples": [img1, img2]}``, or
    ``None`` when the key speaks for itself. Up to 255 options.
    Returns a :class:`ChoiceDecision`.
    """

    instructions: DecisionContent | None
    criteria: Mapping[str, DecisionContent | None]

    def __post_init__(self) -> None:
        if not self.criteria:
            raise ValueError("Choice needs at least one option in criteria")


@dataclass(frozen=True)
class Score:
    """Rate content against ORDERED, descriptive levels.

    ``criteria`` is an ordered list of level descriptions (text or
    images), worst (or lowest) first. Returns a :class:`ScoreDecision`
    whose ``score`` is a position along the levels and may fall
    between two.
    """

    instructions: DecisionContent | None
    criteria: Sequence[DecisionContent]

    def __post_init__(self) -> None:
        if not self.criteria:
            raise ValueError("Score needs at least one level in criteria")


@dataclass(frozen=True)
class Noul:
    """Evaluate a yes/no question.

    Returns a :class:`NoulDecision` whose ``noul`` is the probability
    that the answer is *yes* — near 1 a strong yes, near 0 a strong
    no, near 0.5 uncertain. Optional ``criteria`` describes what each
    outcome means, keyed ``"true"`` / ``"false"`` (``"yes"`` / ``"no"``
    accepted too); ``instructions`` may then be ``None``.
    """

    instructions: DecisionContent | None
    criteria: Mapping[str, DecisionContent | None] | None = None

    def __post_init__(self) -> None:
        if self.criteria is not None:
            unknown = [k for k in self.criteria if k.lower() not in _NOUL_KEYS]
            if unknown:
                raise ValueError(
                    f"Noul criteria keys must be 'true'/'false' (or "
                    f"'yes'/'no'); got {unknown}"
                )
        described = self.criteria is not None and any(
            v is not None for v in self.criteria.values()
        )
        if self.instructions is None and not described:
            raise ValueError(
                "Noul needs instructions or at least one described outcome"
            )

    def wire_criteria(self) -> dict[str, DecisionContent | None] | None:
        """``criteria`` with keys normalised to the wire's true/false."""
        if self.criteria is None:
            return None
        return {_NOUL_KEYS[k.lower()]: v for k, v in self.criteria.items()}


@dataclass(frozen=True)
class Box:
    """Locate something in an image — "the dog", "the `badge` in
    `photo`". Returns a :class:`BoxDecision`.

    Jev 1.14+ only (``JevModel("jev-preview")`` / ``"jev-1.14.0"``).
    A Box ALWAYS returns a box — there is no "not found" — so pair it
    with a :class:`Noul` ("is there a dog?") when the target may be
    absent. One Box per target.
    """

    instructions: DecisionContent


Question = Choice | Score | Noul | Box


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChoiceDecision:
    """Answer to a :class:`Choice`."""

    choice: str
    probabilities: Mapping[str, float]
    confidence: float
    """How peaked the distribution is, collapsed to [0, 1]."""


@dataclass(frozen=True)
class ScoreDecision:
    """Answer to a :class:`Score`."""

    score: float
    """Position along the levels (0-indexed; may fall between two)."""
    legend: Sequence[str]
    """The level descriptions, by position."""
    probabilities: Sequence[float]
    """Distribution across levels, same order as ``legend``."""
    confidence: float

    @property
    def normalized(self) -> float:
        """``score`` mapped onto [0, 1] regardless of level count —
        the shape architecture code (ToT pruning, thresholds) wants."""
        top = len(self.legend) - 1
        if top <= 0:
            return 0.0
        return min(1.0, max(0.0, self.score / top))


@dataclass(frozen=True)
class NoulDecision:
    """Answer to a :class:`Noul`.

    Deliberately has NO ``confidence`` field — the wire API returns a
    bare P(yes) for Noul, where distance from 0.5 *is* the certainty.
    Threshold code written against this type ports unchanged between
    the Jev backend and the LLM fallback.
    """

    noul: float
    """Probability that the answer is yes, in [0, 1]."""


@dataclass(frozen=True)
class BoxDecision:
    """Answer to a :class:`Box`: a bounding box as four edges, each a
    fraction (0-1) of the image's height (``top``/``bottom``) or width
    (``left``/``right``), measured from the top-left corner."""

    top: float
    left: float
    bottom: float
    right: float
    confidence: float | None = None
    """0-1, higher is surer; ``None`` when the backend gives none."""

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """``(left, top, right, bottom)`` in pixels — the tuple
        ``PIL.Image.crop`` takes."""
        return (
            round(self.left * width),
            round(self.top * height),
            round(self.right * width),
            round(self.bottom * height),
        )

    def padded(self, margin: float) -> BoxDecision:
        """Grow every edge outward by ``margin`` (a fraction of the
        image), clamped to [0, 1] — the cheap fix for a low-confidence
        box that may clip its target."""
        return BoxDecision(
            top=max(0.0, self.top - margin),
            left=max(0.0, self.left - margin),
            bottom=min(1.0, self.bottom + margin),
            right=min(1.0, self.right + margin),
            confidence=self.confidence,
        )

    def to_full_image(self, crop: BoxDecision) -> BoxDecision:
        """Map a box found inside ``crop`` back to full-image fractions
        (coarse-to-fine search: find the region, Box the crop, then
        convert)."""
        return BoxDecision(
            top=crop.top + self.top * crop.height,
            left=crop.left + self.left * crop.width,
            bottom=crop.top + self.bottom * crop.height,
            right=crop.left + self.right * crop.width,
            confidence=self.confidence,
        )


DecisionAnswer = ChoiceDecision | ScoreDecision | NoulDecision | BoxDecision


@dataclass(frozen=True)
class Decisions:
    """Result of one ``decide()`` call: named answers + usage.

    ``usage`` rides loomflow's normal :class:`~loomflow.Usage` spine,
    so budgets, telemetry, and cost accounting treat a decision call
    exactly like the model call it replaced. ``model`` is the exact
    version that answered (aliases like ``jev-latest`` move — log this
    to reproduce a decision); ``request_id`` is what TypeSafe support
    asks for. Both are ``None`` when the backend doesn't report them.

    Typed views (``nouls`` / ``choices`` / ``scores`` / ``boxes``)
    skip the ``isinstance`` narrowing ``decisions["name"]`` needs.
    """

    answers: Mapping[str, DecisionAnswer]
    usage: Usage = field(default_factory=Usage)
    model: str | None = None
    request_id: str | None = None

    def __getitem__(self, name: str) -> DecisionAnswer:
        return self.answers[name]

    @property
    def nouls(self) -> dict[str, NoulDecision]:
        return {k: a for k, a in self.answers.items() if isinstance(a, NoulDecision)}

    @property
    def choices(self) -> dict[str, ChoiceDecision]:
        return {k: a for k, a in self.answers.items() if isinstance(a, ChoiceDecision)}

    @property
    def scores(self) -> dict[str, ScoreDecision]:
        return {k: a for k, a in self.answers.items() if isinstance(a, ScoreDecision)}

    @property
    def boxes(self) -> dict[str, BoxDecision]:
        return {k: a for k, a in self.answers.items() if isinstance(a, BoxDecision)}


def describe_content(value: Any) -> str:
    """A short text label for question content — used for Score
    ``legend`` entries, where an image or dict level has no text."""
    if isinstance(value, str):
        return value
    if isinstance(value, Image):
        return "<image>"
    try:
        return json.dumps(
            value,
            default=lambda o: "<image>" if isinstance(o, Image) else str(o),
        )
    except (TypeError, ValueError):
        return str(value)
