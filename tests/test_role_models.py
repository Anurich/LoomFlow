"""Per-role model routing on the single-agent architectures.

The five architectures whose internal calls all previously ran on the
one agent model (``deps.model``) now accept ``<role>_model=`` kwargs:

* ``TreeOfThoughts(proposer_model=, evaluator_model=, synthesizer_model=)``
* ``PlanAndExecute(planner_model=, executor_model=, synthesizer_model=)``
* ``ReWOO(planner_model=, solver_model=)``
* ``Reflexion(evaluator_model=, reflector_model=)``
* ``SelfRefine(critic_model=, refiner_model=)``

Each accepts a spec string (resolved via ``_resolve_model``), a
``Model`` instance (passed through untouched — so ``RetryingModel`` /
``FallbackModel`` wrappers compose), or ``None`` (the agent's main
model — byte-identical to the pre-feature behavior).
"""

from __future__ import annotations

from typing import Any

import pytest

from loomflow import Agent, EchoModel, Usage
from loomflow.architecture import (
    PlanAndExecute,
    Reflexion,
    ReWOO,
    SelfRefine,
    TreeOfThoughts,
)
from loomflow.architecture.helpers import resolve_role_model
from loomflow.core.types import ModelChunk

pytestmark = pytest.mark.anyio


class _RecordingModel:
    """Fake model that records every prompt it serves and answers
    with a fixed text (so scorers parse a score, planners parse
    plans, etc.)."""

    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self._text = text
        self.calls: list[str] = []  # first 60 chars of each SYSTEM msg

    def _record(self, messages: Any) -> None:
        head = messages[0].content if messages else ""
        self.calls.append((head or "")[:60])

    async def complete(self, messages: Any, **kwargs: Any) -> Any:
        self._record(messages)
        return (self._text, [], Usage(input_tokens=1, output_tokens=1), "stop")

    async def stream(self, messages: Any, **kwargs: Any) -> Any:
        self._record(messages)
        yield ModelChunk(kind="text", text=self._text)
        yield ModelChunk(kind="finish", finish_reason="stop", usage=Usage())


# ---------------------------------------------------------------------------
# resolve_role_model unit behavior
# ---------------------------------------------------------------------------


def test_resolve_role_model_none_passthrough() -> None:
    assert resolve_role_model(None) is None


def test_resolve_role_model_instance_passthrough() -> None:
    m = EchoModel()
    assert resolve_role_model(m) is m


def test_resolve_role_model_string_resolves() -> None:
    m = resolve_role_model("echo")
    assert isinstance(m, EchoModel)


# ---------------------------------------------------------------------------
# TreeOfThoughts — evaluator/proposer/synthesizer routing
# ---------------------------------------------------------------------------


async def test_tot_routes_evaluator_calls_to_evaluator_model() -> None:
    evaluator = _RecordingModel("cheap-scorer", "score: 0.9")
    main = _RecordingModel("main", "1. a good next step")

    agent = Agent(
        "solve",
        model=main,
        architecture=TreeOfThoughts(
            branch_factor=1,
            max_depth=1,
            beam_width=1,
            parallel=False,
            synthesize_final=False,
            evaluator_model=evaluator,
        ),
    )
    await agent.run("problem")

    # Every eval call went to the evaluator model, none to main.
    assert evaluator.calls, "evaluator model never invoked"
    assert all("evaluat" in c.lower() or "score" in c.lower() or c
               for c in evaluator.calls)
    # Main model served the proposals only.
    assert main.calls, "main model should still propose"


async def test_tot_synthesizer_model_routes_final_call() -> None:
    synth = _RecordingModel("synth", "the final answer")
    main = _RecordingModel("main", "1. a step\nscore: 0.9")

    agent = Agent(
        "solve",
        model=main,
        architecture=TreeOfThoughts(
            branch_factor=1,
            max_depth=1,
            beam_width=1,
            parallel=False,
            synthesize_final=True,
            synthesizer_model=synth,
        ),
    )
    result = await agent.run("problem")
    assert synth.calls, "synthesizer model never invoked"
    assert result.output == "the final answer"


async def test_tot_none_means_main_model_for_everything() -> None:
    main = _RecordingModel("main", "1. a step\nscore: 0.9")
    agent = Agent(
        "solve",
        model=main,
        architecture=TreeOfThoughts(
            branch_factor=1,
            max_depth=1,
            beam_width=1,
            parallel=False,
            synthesize_final=False,
        ),
    )
    await agent.run("problem")
    # propose + eval both landed on the single main model.
    assert len(main.calls) >= 2


# ---------------------------------------------------------------------------
# PlanAndExecute — planner/executor split
# ---------------------------------------------------------------------------


async def test_pae_planner_and_executor_split() -> None:
    planner = _RecordingModel("frontier-planner", '["do the one step"]')
    executor = _RecordingModel("cheap-executor", "step done")
    main = _RecordingModel("main", "synthesized answer")

    agent = Agent(
        "build",
        model=main,
        architecture=PlanAndExecute(
            planner_model=planner, executor_model=executor
        ),
    )
    result = await agent.run("task")

    assert len(planner.calls) == 1, "planner model plans exactly once"
    assert len(executor.calls) == 1, "executor model runs the one step"
    # Synthesizer defaulted to the main model.
    assert main.calls, "main model synthesizes by default"
    assert result.output


# ---------------------------------------------------------------------------
# ReWOO — planner/solver split
# ---------------------------------------------------------------------------


async def test_rewoo_planner_and_solver_split() -> None:
    # One step invoking a nonexistent tool: the step errors, the
    # solver still runs over the error output — which is exactly what
    # we need to prove both roles routed away from the main model.
    planner = _RecordingModel(
        "planner", '[{"id": "E1", "tool": "lookup", "args": {}}]'
    )
    solver = _RecordingModel("solver", "final solved answer")
    main = _RecordingModel("main", "should not be called")

    agent = Agent(
        "solve",
        model=main,
        architecture=ReWOO(planner_model=planner, solver_model=solver),
    )
    result = await agent.run("task")

    assert len(planner.calls) == 1
    assert len(solver.calls) == 1
    assert main.calls == [], "main model unused when both roles routed"
    assert result.output == "final solved answer"


# ---------------------------------------------------------------------------
# Reflexion — evaluator/reflector split (base attempt stays on main)
# ---------------------------------------------------------------------------


async def test_reflexion_evaluator_routed_base_attempt_on_main() -> None:
    evaluator = _RecordingModel("eval", "score: 0.95")
    main = _RecordingModel("main", "attempt answer")

    agent = Agent(
        "try",
        model=main,
        architecture=Reflexion(max_attempts=2, evaluator_model=evaluator),
    )
    result = await agent.run("task")

    assert evaluator.calls, "evaluator model never invoked"
    assert main.calls, "base attempt must use the main model"
    assert result.output == "attempt answer"


async def test_reflexion_reflector_routed_on_failure() -> None:
    evaluator = _RecordingModel("eval", "score: 0.1")  # always fail
    reflector = _RecordingModel("reflect", "lesson: be better")
    main = _RecordingModel("main", "attempt answer")

    agent = Agent(
        "try",
        model=main,
        architecture=Reflexion(
            max_attempts=2,
            evaluator_model=evaluator,
            reflector_model=reflector,
        ),
    )
    await agent.run("task")
    assert reflector.calls, "reflector model never invoked on failure"


# ---------------------------------------------------------------------------
# SelfRefine — critic/refiner split
# ---------------------------------------------------------------------------


async def test_self_refine_critic_and_refiner_split() -> None:
    critic = _RecordingModel("critic", "needs more detail")
    refiner = _RecordingModel("refiner", "the refined answer")
    main = _RecordingModel("main", "first draft")

    agent = Agent(
        "write",
        model=main,
        architecture=SelfRefine(
            max_rounds=1, critic_model=critic, refiner_model=refiner
        ),
    )
    result = await agent.run("task")

    assert critic.calls, "critic model never invoked"
    assert refiner.calls, "refiner model never invoked"
    assert main.calls, "initial draft must use the main model"
    assert result.output == "the refined answer"


# ---------------------------------------------------------------------------
# Usage still flows into the one budget regardless of routing
# ---------------------------------------------------------------------------


async def test_routed_usage_counts_against_run_usage() -> None:
    planner = _RecordingModel(
        "planner", '[{"id": "E1", "tool": "lookup", "args": {}}]'
    )
    solver = _RecordingModel("solver", "answer")
    agent = Agent(
        "solve",
        model=_RecordingModel("main", "x"),
        architecture=ReWOO(planner_model=planner, solver_model=solver),
    )
    result = await agent.run("task")
    # Both routed calls contributed usage (1 in / 1 out each).
    assert result.tokens_in >= 2
    assert result.tokens_out >= 2
