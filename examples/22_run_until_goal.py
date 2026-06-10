"""22_run_until_goal.py — run-until-done loop gated by a fast checker.

``Agent(run_until=...)`` keeps re-prompting the agent until a small,
fast **checker model** confirms a *measurable* stop condition holds —
loomflow's take on the ``/goal`` / "Ralph Wiggum" pattern. It wraps any
architecture (ReAct here) by hanging a :class:`~loomflow.GoalStopHook`
off the framework's existing stop-hook loop.

The whole point is the guardrails. An unbounded run-until loop is the
#1 autonomous-agent failure mode (it burns budget flailing on an
under-specified goal), so the hook is bounded three ways:

* ``max_iterations``  — hard cap on re-prompts.
* ``max_no_progress`` — bail when N consecutive passes change nothing.
* ``max_cost_usd``    — hard cost ceiling for the loop.

Forms::

    Agent(run_until="all tests pass and the endpoint returns 200")

    Agent(run_until={
        "condition":       "all tests pass and the endpoint returns 200",
        "checker":         "claude-haiku-4-5",  # cheap, fast checker
        "max_iterations":  20,
        "max_no_progress": 3,
        "max_cost_usd":    5.0,
    })

This example runs OFFLINE with :class:`ScriptedModel` (no API key): a
worker that takes two passes and a checker that says ``NOT_DONE`` then
``DONE``. With a real model you'd point ``checker`` at Haiku and write
a condition your agent's tools can actually verify (tests passing, a
file existing, an endpoint returning 200).

Run with::

    python examples/22_run_until_goal.py
"""

from __future__ import annotations

import anyio

from loomflow import Agent, ScriptedModel, ScriptedTurn


async def main() -> None:
    # The worker agent: two passes of "work". In a real run this is
    # your model with tools (edit files, run tests, curl an endpoint).
    worker = ScriptedModel(
        turns=[
            ScriptedTurn(text="First attempt — wired the endpoint."),
            ScriptedTurn(text="Second attempt — fixed the failing test."),
        ]
    )

    # The checker: a SEPARATE, cheap model asked DONE / NOT_DONE after
    # each pass. Here it withholds approval once, then confirms.
    checker = ScriptedModel(
        turns=[
            ScriptedTurn(text="NOT_DONE — a test is still failing."),
            ScriptedTurn(text="DONE — tests pass and endpoint returns 200."),
        ]
    )

    agent = Agent(
        "You are a build agent. Keep working until the goal is met.",
        model=worker,
        run_until={
            "condition": "all tests pass and the endpoint returns 200",
            "checker": checker,
            "max_iterations": 10,
            "max_no_progress": 3,
        },
    )

    result = await agent.run("Get the new endpoint green.")

    print(f"passes (turns):     {result.turns}")
    print(f"interrupted:        {result.interrupted}")
    print(f"final output:       {result.output!r}")
    # The hook records WHY the loop stopped under run_until.exit.
    # (condition_met | max_iterations | no_progress | cost_cap | budget:*)
    print("→ The loop ran a second pass only because the checker said")
    print("  NOT_DONE on the first — that's the run-until-done loop.")


if __name__ == "__main__":
    anyio.run(main)
