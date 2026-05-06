"""18_plan_and_execute — Planner → step executor → synthesizer.

What it shows:
* The ``PlanAndExecute`` architecture commits to a plan upfront,
  then walks through it step-by-step. One planner call + N step
  calls + one synthesizer call.
* Cheaper than ReAct on multi-step tasks with predictable
  structure: ReAct re-plans before every action; PnE plans once.
* The plan is parsed from the planner's JSON output (markdown
  fences and bullet/numbered lists are tolerated for robustness).
* Each step's prompt includes the original task, the full plan,
  and prior step outputs — so the executor model has the context
  it needs to act on this step.
* Final answer comes from the synthesizer, which combines all
  step outputs into a coherent response.

Run:
    python examples/18_plan_and_execute.py
"""

from __future__ import annotations

import asyncio

from jeevesagent import (
    Agent,
    PlanAndExecute,
    ScriptedModel,
    ScriptedTurn,
)


async def main() -> None:
    # Toy task: research-and-summarize.
    # Planner produces a 3-step plan; each step is executed; final
    # synthesizer combines them.
    model = ScriptedModel(
        [
            # 1. Planner: returns a JSON list of step descriptions.
            ScriptedTurn(
                text=(
                    '["Identify the three core ReAct strengths",\n'
                    ' "List two production failure modes",\n'
                    ' "Compare to Plan-and-Execute on cost"]'
                )
            ),
            # 2. Step 1: identify strengths
            ScriptedTurn(
                text=(
                    "ReAct strengths: (1) simple debugging — linear "
                    "trajectory; (2) tool-use feedback per turn; "
                    "(3) widely studied — strong baseline."
                )
            ),
            # 3. Step 2: list failure modes
            ScriptedTurn(
                text=(
                    "Failure modes: (1) goal drift on long tasks; "
                    "(2) excess tokens on multi-step problems with "
                    "predictable structure."
                )
            ),
            # 4. Step 3: compare costs
            ScriptedTurn(
                text=(
                    "Plan-and-Execute saves ~30-50% on tasks with "
                    "predictable structure (single planner pass + N "
                    "step calls vs N×K ReAct turns)."
                )
            ),
            # 5. Synthesizer: combines into final
            ScriptedTurn(
                text=(
                    "ReAct's strengths (debuggability, tool-use, "
                    "research breadth) make it the right default. "
                    "Its failure modes — goal drift on long tasks, "
                    "token cost on predictable multi-step work — "
                    "are exactly where Plan-and-Execute is 30-50% "
                    "cheaper. Use ReAct first; switch to "
                    "Plan-and-Execute when costs hurt and the task "
                    "structure is stable."
                )
            ),
        ]
    )

    agent = Agent(
        "Research analyst",
        model=model,
        architecture=PlanAndExecute(),
    )

    print("=== Streaming events ===")
    async for event in agent.stream(
        "Compare ReAct vs Plan-and-Execute for production agent loops."
    ):
        if event.kind != "architecture_event":
            continue
        name = event.payload.get("name", "")
        if name == "plan.created":
            steps = event.payload["steps"]
            print(f"[plan: {len(steps)} steps]")
            for i, s in enumerate(steps, 1):
                print(f"  {i}. {s}")
        elif name == "plan.step_started":
            i = event.payload["step_index"]
            desc = event.payload["description"]
            print(f"\n[step {i + 1} started] {desc[:80]}")
        elif name == "plan.step_completed":
            i = event.payload["step_index"]
            output = event.payload["output"]
            print(f"[step {i + 1} → {output[:80]}...]")
        elif name == "plan.synthesizer_started":
            print("\n[synthesizing...]")
        elif name == "plan.completed":
            n = event.payload["num_steps"]
            print(f"\n[completed — {n} steps]")

    # Re-run for final-answer print.
    fresh_model = ScriptedModel(
        [
            ScriptedTurn(text='["s1", "s2"]'),
            ScriptedTurn(text="step 1 output"),
            ScriptedTurn(text="step 2 output"),
            ScriptedTurn(
                text="Final: combined insight from both steps."
            ),
        ]
    )
    fresh_agent = Agent(
        "analyst",
        model=fresh_model,
        architecture=PlanAndExecute(),
    )
    result = await fresh_agent.run("any multi-step task")
    print(f"\n=== Final answer ===\n{result.output}")


if __name__ == "__main__":
    asyncio.run(main())
