"""System One decisions — fast, typed, probabilistic judgments.

A :class:`DecisionModel` evaluates *questions* against *state* and
returns typed answers with probabilities, instead of generating text.
Backed by TypeSafe AI's Jev over the wire (:class:`JevModel`), by any
loomflow ``Model`` (:class:`LLMDecisionModel` — works today, no
waitlist), or by a scripted fake for tests.

Where it plugs in (all opt-in, ``decider=`` everywhere):

* ``Team.router(decider="jev")`` — the classification step
* ``Agent(run_until={"condition": ..., "decider": "jev"})`` — goal check
* ``TreeOfThoughts(evaluator_decider="jev")`` — thought scoring
* ``approval_handler=DecisionApprovalPolicy(decider="jev", ...)``
* ``guardrails=[DecisionGuard(decider="jev", question=Noul(...))]``

Images (Jev 1.13+) and Box localisation (Jev 1.14+): put a
``loomflow.Image`` anywhere in the state or a question, and ask
``Box("the dog")`` to get edges back::

    photo = Image.from_path("dog.jpg")
    async with JevModel("jev-preview") as jev:
        d = await jev.decide(
            {"photo": photo},
            questions={
                "dog": Noul("Is there a dog in `photo`?"),
                "where": Box("the dog in `photo`"),
            },
        )
    d.boxes["where"].to_pixels(width, height)

Import from this submodule (Tier-2 API, like model adapters)::

    from loomflow.decisions import (
        Choice, Score, Noul, Box, JevModel, DecisionApprovalPolicy,
    )
"""

from ..core.types import Image
from .base import (
    DecisionError,
    DecisionModel,
    DecisionState,
    resolve_decision_model,
)
from .jev import JevModel
from .llm import LLMDecisionModel
from .policies import DecisionApprovalPolicy, DecisionGuard
from .scripted import ScriptedDecisions
from .types import (
    Box,
    BoxDecision,
    Choice,
    ChoiceDecision,
    DecisionAnswer,
    DecisionContent,
    Decisions,
    Noul,
    NoulDecision,
    Question,
    Score,
    ScoreDecision,
)

__all__ = [
    "Box",
    "BoxDecision",
    "Choice",
    "ChoiceDecision",
    "DecisionAnswer",
    "DecisionApprovalPolicy",
    "DecisionContent",
    "DecisionError",
    "DecisionGuard",
    "DecisionModel",
    "DecisionState",
    "Decisions",
    "Image",
    "JevModel",
    "LLMDecisionModel",
    "Noul",
    "NoulDecision",
    "Question",
    "Score",
    "ScoreDecision",
    "ScriptedDecisions",
    "resolve_decision_model",
]
