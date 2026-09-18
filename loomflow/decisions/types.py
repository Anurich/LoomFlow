"""Question and answer types for System One decisions.

These mirror the TypeSafe wire API exactly (``typesafe_sdk`` uses the
same three primitive names), so code written against loomflow's types
translates 1:1 to the vendor docs — and so the LLM-backed fallback
must reproduce the same semantics, down to :class:`NoulDecision`
carrying a bare probability with NO separate confidence field.

A *question* is a specific, well-scoped judgment ("which team handles
this?", "is this urgent?") — something a knowledgeable person could
answer in seconds. Decompose bigger judgments into several atomic
questions and combine the answers in code; questions in one
``decide()`` call are evaluated together, so extra questions are
nearly free.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ..core.types import Usage

__all__ = [
    "Choice",
    "ChoiceDecision",
    "DecisionAnswer",
    "Decisions",
    "Noul",
    "NoulDecision",
    "Question",
    "Score",
    "ScoreDecision",
]


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Choice:
    """Select ONE option from a defined set.

    ``criteria`` maps each option key to a description that helps the
    model discriminate — keep descriptions specific and mutually
    distinguishing. Returns a :class:`ChoiceDecision`.
    """

    instructions: str
    criteria: Mapping[str, str]


@dataclass(frozen=True)
class Score:
    """Rate content against ORDERED, descriptive levels.

    ``criteria`` is an ordered list of level descriptions, worst (or
    lowest) first. Returns a :class:`ScoreDecision` whose ``score``
    is a position along the levels and may fall between two.
    """

    instructions: str
    criteria: Sequence[str]


@dataclass(frozen=True)
class Noul:
    """Evaluate a yes/no question.

    Returns a :class:`NoulDecision` whose ``noul`` is the probability
    that the answer is *yes* — near 1 a strong yes, near 0 a strong
    no, near 0.5 uncertain. Optional ``criteria`` clarifies what yes
    and no mean (keys ``"yes"`` / ``"no"``).
    """

    instructions: str
    criteria: Mapping[str, str] | None = None


Question = Choice | Score | Noul


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


DecisionAnswer = ChoiceDecision | ScoreDecision | NoulDecision


@dataclass(frozen=True)
class Decisions:
    """Result of one ``decide()`` call: named answers + usage.

    ``usage`` rides loomflow's normal :class:`~loomflow.Usage` spine,
    so budgets, telemetry, and cost accounting treat a decision call
    exactly like the model call it replaced.
    """

    answers: Mapping[str, DecisionAnswer]
    usage: Usage = field(default_factory=Usage)

    def __getitem__(self, name: str) -> DecisionAnswer:
        return self.answers[name]
