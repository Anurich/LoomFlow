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

Import from this submodule (Tier-2 API, like model adapters)::

    from loomflow.decisions import (
        Choice, Score, Noul, JevModel, DecisionApprovalPolicy,
    )
"""

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
    Choice,
    ChoiceDecision,
    DecisionAnswer,
    Decisions,
    Noul,
    NoulDecision,
    Question,
    Score,
    ScoreDecision,
)

__all__ = [
    "Choice",
    "ChoiceDecision",
    "DecisionAnswer",
    "DecisionApprovalPolicy",
    "DecisionError",
    "DecisionGuard",
    "DecisionModel",
    "DecisionState",
    "Decisions",
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
