"""18_living_plan.py — TodoWrite-style structured plan.

The agent gets two new tools when ``living_plan=True`` is set:

* ``plan_write(goal, steps)`` — atomically rewrite the FULL plan.
  Each step is ``{description, status, finding}``. Tool returns the
  rendered plan back as markdown so it becomes load-bearing in the
  conversation.
* ``plan_read()`` — re-orient on current state without modifying.

When a workspace is ALSO wired, the plan mirrors to a
``kind="plan"`` note so future runs can ``recall_past_plans(query)``
and bootstrap from prior task plans.

Why this beats a free-form ``note(kind="plan")``:

* **Forced engagement** — full-list rewrite means the model can't
  silently drop steps.
* **No partial-update bugs** — one atomic call replaces the state.
* **Drift becomes structural** — every action maps to a step the
  agent itself wrote.

This example uses a :class:`ScriptedModel` so it runs offline (no
API key required). It walks through:

1. The agent calls ``plan_write`` to commit a plan.
2. Calls a tool to do "work" (echo a value).
3. Updates the plan to mark the step ``done``.
4. Emits a final message.

After the run, we inspect the workspace and confirm the plan note
is there with the correct kind / structure.

Run with::

    python examples/18_living_plan.py
"""

from __future__ import annotations

import asyncio

from loomflow import Agent, LivingPlan, tool
from loomflow.core.types import ToolCall
from loomflow.model.scripted import ScriptedModel, ScriptedTurn
from loomflow.workspace import InMemoryWorkspace


@tool
async def echo(msg: str) -> str:
    """Trivial tool the model "uses" to do work."""
    return f"echoed: {msg}"


def _build_scripted_model() -> ScriptedModel:
    """Hand-script four turns:

    1. ``plan_write`` — commit the initial plan with step 1 marked
       ``doing`` (we'll mark it ``done`` after the echo).
    2. ``echo`` — do the actual work.
    3. ``plan_write`` — rewrite the plan, mark step 1 ``done`` with
       a finding, mark the verify step ``done``.
    4. Final message — task complete.
    """
    return ScriptedModel(
        turns=[
            ScriptedTurn(
                text="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        tool="plan_write",
                        args={
                            "goal": "Echo the value 42 and verify",
                            "steps": [
                                {
                                    "description": "Call echo with 42",
                                    "status": "doing",
                                },
                                {
                                    "description": "Verify echo returned",
                                    "status": "todo",
                                },
                            ],
                        },
                    ),
                ],
            ),
            ScriptedTurn(
                text="",
                tool_calls=[
                    ToolCall(
                        id="call_2",
                        tool="echo",
                        args={"msg": "42"},
                    ),
                ],
            ),
            ScriptedTurn(
                text="",
                tool_calls=[
                    ToolCall(
                        id="call_3",
                        tool="plan_write",
                        args={
                            "goal": "Echo the value 42 and verify",
                            "steps": [
                                {
                                    "description": "Call echo with 42",
                                    "status": "done",
                                    "finding": "got 'echoed: 42'",
                                },
                                {
                                    "description": "Verify echo returned",
                                    "status": "done",
                                    "finding": "value present",
                                },
                            ],
                        },
                    ),
                ],
            ),
            ScriptedTurn(text="Done. 42 was echoed and verified."),
        ]
    )


async def main() -> None:
    workspace = InMemoryWorkspace()

    # The two key kwargs:
    #   * ``workspace=`` — the shared notebook the plan mirrors into.
    #   * ``living_plan=True`` — enables ``plan_write`` / ``plan_read``
    #     (and ``recall_past_plans`` because workspace is set).
    agent = Agent(
        "You are a careful agent. Use the living plan religiously.",
        model=_build_scripted_model(),
        tools=[echo],
        workspace=workspace,
        living_plan=True,
    )

    # Confirm the smart-default wiring registered the right tools.
    tool_names = {t.name for t in agent._tool_host._tools.values()}
    print("Tools wired:")
    for name in sorted(tool_names):
        print(f"  - {name}")
    assert "plan_write" in tool_names
    assert "plan_read" in tool_names
    assert "recall_past_plans" in tool_names

    print()
    print("Running agent...")
    result = await agent.run(
        "Echo 42 and verify", user_id="example-user"
    )
    print(f"  output: {result.output!r}")
    print(f"  turns:  {result.turns}")
    print()

    # The plan mirrored to the workspace. List + read it back.
    notes = await workspace.list_notes(user_id="example-user")
    print(f"Workspace has {len(notes)} note(s) after the run:")
    for note in notes:
        print(f"  - {note.slug}  ({note.kind}): {note.title}")

    plan_notes = [n for n in notes if n.kind == "plan"]
    assert len(plan_notes) == 1, "expected exactly one plan note"

    full_note = await workspace.read_note(
        plan_notes[0].slug, user_id="example-user"
    )
    assert full_note is not None
    print()
    print("--- plan note body (first 400 chars) ---")
    print(full_note.body[:400])

    # Pre-seed demo: pass a constructed LivingPlan instance so the
    # next run STARTS with a plan in place rather than the model
    # having to write one.
    print()
    print("=" * 50)
    print("Pre-seed demo:")
    print("=" * 50)
    seed = LivingPlan(goal="Pre-seeded plan demo", steps=[])
    seeded_agent = Agent(
        "test",
        model=_build_scripted_model(),
        tools=[echo],
        living_plan=seed,
    )
    print(f"  Pre-seeded goal: {seeded_agent._living_plan_spec.seed_plan.goal!r}")


if __name__ == "__main__":
    asyncio.run(main())
