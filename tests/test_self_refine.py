"""SelfRefine architecture tests.

Covers:

* Protocol satisfaction.
* The ``stop_phrase`` early-exit path (critic says "no issues" → no
  refiner call, output preserved).
* The full critique → refine cycle (critic finds issues → refiner
  produces a new output → that becomes ``session.output``).
* ``max_rounds`` enforcement (no convergence within budget).
* Custom ``base`` architecture composes (a no-op base lets us test
  Self-Refine in isolation without running ReAct underneath).
* End-to-end via ``Agent.run`` with the resolver string
  (``architecture="self-refine"``).
* Budget gating between rounds.
* Architecture progress events surface through ``Agent.stream``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from loomflow import Agent, Architecture, ReAct, ScriptedModel, ScriptedTurn
from loomflow.architecture import AgentSession, Dependencies, SelfRefine
from loomflow.architecture.resolver import resolve_architecture
from loomflow.core.types import Event

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Protocol surface
# ---------------------------------------------------------------------------


def test_self_refine_satisfies_architecture_protocol() -> None:
    assert isinstance(SelfRefine(), Architecture)


def test_self_refine_name_is_kebab_case() -> None:
    assert SelfRefine().name == "self-refine"


def test_self_refine_declares_no_workers() -> None:
    """Self-Refine uses one model in three roles, not separate
    Agents — declared_workers stays empty."""
    assert SelfRefine().declared_workers() == {}


def test_resolver_handles_self_refine_string() -> None:
    arch = resolve_architecture("self-refine")
    assert isinstance(arch, SelfRefine)


def test_self_refine_rejects_max_rounds_lt_1() -> None:
    with pytest.raises(ValueError, match="max_rounds"):
        SelfRefine(max_rounds=0)


# ---------------------------------------------------------------------------
# A no-op base architecture so we can test Self-Refine without invoking
# ReAct's full machinery underneath. Round 0 just sets a starting output.
# ---------------------------------------------------------------------------


class _PrebakedBase:
    """Test fixture: round-0 architecture that sets a fixed output
    without calling the model. Lets us isolate Self-Refine behaviour."""

    name = "_prebaked"

    def __init__(self, initial_output: str) -> None:
        self._initial = initial_output

    def declared_workers(self) -> dict[str, Agent]:
        return {}

    async def run(
        self,
        session: AgentSession,
        deps: Dependencies,
        prompt: str,
    ) -> AsyncIterator[Event]:
        session.output = self._initial
        session.turns = 1
        # Yield nothing — purely sets state.
        if False:  # pragma: no cover — keeps this an async generator
            yield Event.budget_warning(session.id, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Early-exit on stop_phrase
# ---------------------------------------------------------------------------


async def test_self_refine_converges_on_stop_phrase() -> None:
    """Critic says exactly the stop phrase on round 1 → terminate;
    no refiner call; output unchanged from round 0."""
    base = _PrebakedBase(initial_output="initial good output")
    # Critic returns the stop phrase, so refiner should NEVER run.
    model = ScriptedModel([ScriptedTurn(text="no issues")])
    agent = Agent(
        "test",
        model=model,
        architecture=SelfRefine(base=base, max_rounds=3),
    )
    result = await agent.run("any prompt")
    assert result.output == "initial good output"
    # Generator round (1) + critic call (1) = 2 turns total.
    assert result.turns == 2


# ---------------------------------------------------------------------------
# Full critique → refine cycle
# ---------------------------------------------------------------------------


async def test_self_refine_one_full_cycle_then_converge() -> None:
    """Round 1: critic finds issues → refiner produces revision.
    Round 2: critic says 'no issues' → terminate. Final output is
    the refiner's revised version."""
    base = _PrebakedBase(initial_output="rough draft")
    model = ScriptedModel(
        [
            ScriptedTurn(text="missing examples; wording unclear"),
            ScriptedTurn(text="polished revision"),
            ScriptedTurn(text="no issues"),
        ]
    )
    agent = Agent(
        "test",
        model=model,
        architecture=SelfRefine(base=base, max_rounds=3),
    )
    result = await agent.run("write a thing")
    assert result.output == "polished revision"
    # 1 (base) + 1 (critic R1) + 1 (refiner R1) + 1 (critic R2) = 4
    assert result.turns == 4


# ---------------------------------------------------------------------------
# max_rounds enforcement
# ---------------------------------------------------------------------------


async def test_self_refine_respects_max_rounds() -> None:
    """Critic never says 'no issues' → refine until max_rounds, then
    return the latest refined output. session.output is whatever the
    last refiner produced."""
    base = _PrebakedBase(initial_output="v0")
    # max_rounds=2 → critic R1, refiner R1, critic R2, refiner R2.
    # Both critics find issues; both refiners produce a new version.
    model = ScriptedModel(
        [
            ScriptedTurn(text="issues 1"),
            ScriptedTurn(text="v1"),
            ScriptedTurn(text="issues 2"),
            ScriptedTurn(text="v2"),
        ]
    )
    agent = Agent(
        "test",
        model=model,
        architecture=SelfRefine(base=base, max_rounds=2),
    )
    result = await agent.run("task")
    assert result.output == "v2"
    # 1 (base) + 4 (two full critic+refiner pairs) = 5
    assert result.turns == 5


# ---------------------------------------------------------------------------
# Architecture-event stream
# ---------------------------------------------------------------------------


async def test_self_refine_emits_progress_events() -> None:
    """``architecture_event`` payloads include the namespaced ``name``
    so consumers can pattern-match Self-Refine progress without
    expanding ``EventKind``."""
    base = _PrebakedBase(initial_output="x")
    model = ScriptedModel(
        [
            ScriptedTurn(text="issues here"),
            ScriptedTurn(text="x improved"),
            ScriptedTurn(text="no issues"),
        ]
    )
    agent = Agent(
        "test",
        model=model,
        architecture=SelfRefine(base=base, max_rounds=3),
    )

    events = [event async for event in agent.stream("go")]
    arch_events = [e for e in events if e.kind == "architecture_event"]
    names = [e.payload["name"] for e in arch_events]

    # Generator round + R1 critic + R1 refiner + R2 critic + converged.
    assert "self_refine.round_started" in names
    assert "self_refine.critique" in names
    assert "self_refine.refined" in names
    assert "self_refine.converged" in names


# ---------------------------------------------------------------------------
# End-to-end via the resolver string
# ---------------------------------------------------------------------------


async def test_self_refine_via_string_arg_with_react_base() -> None:
    """``architecture="self-refine"`` defaults base=ReAct(); the
    initial generation runs through ReAct (echo model produces a
    deterministic text), then a critic + refiner cycle iterates."""
    # Echo produces "Echo: <prompt>" text-only, no tool calls. Then
    # the critic says no issues so we converge after one critic call.
    model = ScriptedModel(
        [
            ScriptedTurn(text="Hello world"),  # ReAct round 0
            ScriptedTurn(text="no issues"),  # Self-Refine critic R1
        ]
    )
    agent = Agent(
        "test",
        model=model,
        architecture="self-refine",
    )
    result = await agent.run("hi")
    assert result.output == "Hello world"
    # ReAct turn 1 (model call) + critic R1 = 2.
    assert result.turns == 2


# ---------------------------------------------------------------------------
# Budget gating
# ---------------------------------------------------------------------------


async def test_self_refine_budget_blocks_before_round_starts() -> None:
    """A blocking budget on the inter-round check terminates cleanly:
    interrupted=True with reason starting with ``budget:``. The
    pre-baked base sets ``session.output``; Self-Refine's very first
    ``allows_step`` blocks, so no critic / refiner runs and the
    output is preserved."""
    from loomflow.core.types import BudgetStatus

    class _AlwaysBlock:
        async def allows_step(self) -> BudgetStatus:
            return BudgetStatus.blocked_("test_block")

        async def consume(self, **kwargs: object) -> None:
            return None

    base = _PrebakedBase(initial_output="initial")
    # Model is never called — base doesn't model, and budget blocks
    # before Self-Refine's critic runs.
    model = ScriptedModel([ScriptedTurn(text="never reached")])
    agent = Agent(
        "test",
        model=model,
        architecture=SelfRefine(base=base, max_rounds=3),
        budget=_AlwaysBlock(),  # type: ignore[arg-type]
    )
    result = await agent.run("anything")
    assert result.interrupted
    assert result.interruption_reason is not None
    assert result.interruption_reason.startswith("budget:")
    # Output should still be the round-0 result; refiner never ran.
    assert result.output == "initial"


# ---------------------------------------------------------------------------
# Composition: SelfRefine wrapping an explicit ReAct(max_turns=...)
# ---------------------------------------------------------------------------


async def test_self_refine_passes_through_base_max_turns_override() -> None:
    """SelfRefine doesn't override the base architecture's max_turns —
    if the user passes ReAct(max_turns=1), the base uses that cap."""
    # Configure ReAct with max_turns=1; only one model call inside ReAct.
    # Then self-refine adds one critic call that converges.
    model = ScriptedModel(
        [
            ScriptedTurn(text="Initial"),
            ScriptedTurn(text="no issues"),
        ]
    )
    agent = Agent(
        "test",
        model=model,
        architecture=SelfRefine(base=ReAct(max_turns=1), max_rounds=3),
    )
    result = await agent.run("go")
    assert result.output == "Initial"
    # 1 (ReAct turn) + 1 (critic) = 2.
    assert result.turns == 2
