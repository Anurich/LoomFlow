"""31_model_routing.py — per-role models: plan expensive, execute cheap.

Model routing — "use the right model for the right task" — is the
single biggest lever on an AI bill. The two places loomflow gives it
to you:

**Between agents (always worked):** every seat in a ``Team`` is an
``Agent`` with its own ``model=`` — frontier coordinator, cheap
workers::

    Team.supervisor(
        model="claude-fable-5",                       # plans + reviews
        workers={"coder": Agent(..., model="claude-sonnet-4-6")},
    )

**Inside one agent (new in v0.11):** the single-agent architectures
now take ``<role>_model=`` kwargs, so their internal planner /
evaluator / solver calls can each run on a different model::

    PlanAndExecute(planner_model="claude-fable-5",     # thinks
                   executor_model="claude-haiku-4-5")  # types

    TreeOfThoughts(evaluator_model="claude-haiku-4-5") # scores 0-1

    ReWOO(planner_model=..., solver_model=...)
    Reflexion(evaluator_model=..., reflector_model=...)
    SelfRefine(critic_model=..., refiner_model=...)

Why it pays: output tokens cost ~5x input tokens, and execution is
output-heavy while planning is input-heavy. Frontier where judgment
lives, cheap where typing lives. ``None`` (the default) keeps every
call on the agent's main model — nothing changes unless you route.

This example runs OFFLINE (no API key): recording fakes stand in for
"frontier" and "cheap" so we can PROVE which model served which call.

Run with::

    python examples/31_model_routing.py
"""

from __future__ import annotations

from typing import Any

import anyio

from loomflow import Agent, Usage
from loomflow.architecture import PlanAndExecute, TreeOfThoughts
from loomflow.core.types import ModelChunk


class RecordingModel:
    """Stands in for a real model; records every call it serves."""

    def __init__(self, name: str, reply: str) -> None:
        self.name = name
        self._reply = reply
        self.calls = 0

    async def complete(self, messages: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return (self._reply, [], Usage(input_tokens=5, output_tokens=5), "stop")

    async def stream(self, messages: Any, **kwargs: Any) -> Any:
        self.calls += 1
        yield ModelChunk(kind="text", text=self._reply)
        yield ModelChunk(kind="finish", finish_reason="stop", usage=Usage())


async def plan_and_execute_demo() -> None:
    print("=" * 62)
    print("1) PlanAndExecute — frontier plans, cheap model executes")
    print("=" * 62)

    frontier = RecordingModel(
        "fable-5 (planner)",
        '["Outline the rate-limit API design",'
        ' "Implement the middleware on /api/orders"]',
    )
    cheap = RecordingModel("haiku-4-5 (executor)", "step complete")
    main = RecordingModel("opus-4-8 (synthesizer)", "Feature shipped.")

    agent = Agent(
        "You build features.",
        model=main,  # only the final synthesis uses the main model
        architecture=PlanAndExecute(
            planner_model=frontier,
            executor_model=cheap,
        ),
    )
    result = await agent.run("Add rate limiting to /api/orders")

    print(f"  output: {result.output}")
    print(f"  {frontier.name:26} served {frontier.calls} call (the plan)")
    print(f"  {cheap.name:26} served {cheap.calls} calls (the steps)")
    print(f"  {main.name:26} served {main.calls} call (the synthesis)")


async def tree_of_thoughts_demo() -> None:
    print()
    print("=" * 62)
    print("2) TreeOfThoughts — cheap evaluator scores the branches")
    print("=" * 62)

    scorer = RecordingModel("haiku-4-5 (evaluator)", "score: 0.9")
    main = RecordingModel("opus-4-8 (proposer)", "1. a promising next step")

    agent = Agent(
        "You solve hard problems.",
        model=main,
        architecture=TreeOfThoughts(
            branch_factor=2,
            max_depth=2,
            beam_width=1,
            parallel=False,
            synthesize_final=False,
            evaluator_model=scorer,  # the "rate this 0-1" calls
        ),
    )
    await agent.run("What is the optimal caching strategy here?")

    print(f"  {main.name:26} served {main.calls} calls (proposals)")
    print(f"  {scorer.name:26} served {scorer.calls} calls (0-1 scoring)")
    print()
    print("  Every scoring call moved off the frontier model. At ToT")
    print("  defaults (3x3x2) that's ~12 calls per run rerouted to a")
    print("  model ~25x cheaper — with zero effect on proposal quality.")


async def main() -> None:
    await plan_and_execute_demo()
    await tree_of_thoughts_demo()


if __name__ == "__main__":
    anyio.run(main)
