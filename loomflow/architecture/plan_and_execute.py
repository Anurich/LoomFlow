"""Plan-and-Execute: planner → step executor → synthesizer.

Wang et al. 2023 (`Plan-and-Solve Prompting`), He et al. 2025
(`Plan-then-Execute Pattern Implementation`). The 2026 production
pattern for cost-sensitive multi-step tasks where ReAct-style
"think before each action" wastes tokens — Plan-and-Execute
commits to a plan upfront and executes step-by-step.

Pattern (v1 simple)
-------------------

1. **Plan.** A single planner call produces a JSON list of step
   descriptions. Steps are short, ordered, dependency-free for v1.
2. **Execute.** Each step is a text-only model call seeded with the
   original problem, the plan, and prior step outputs. Sequential.
3. **Synthesize.** A final model call combines step outputs into
   the answer.

What's NOT in v1
----------------

* **Sub-architecture invocation per step.** Each step is a single
  text-only call, NOT a fresh ReAct (or any other architecture)
  invocation. That's the planned v0.5 work — it unblocks Deep Agent
  and a richer ReWOO too.
* **Replanning on failure.** ``max_replans`` is in the constructor
  but not yet wired; it'll land alongside the sub-architecture
  invocation primitive when steps can fail meaningfully.
* **Parallel step execution.** Steps run sequentially. DAG-aware
  parallel execution is straightforward to add once dependencies
  are first-class on :class:`PlanStep`.

Strengths
---------
* **Cheaper than ReAct** for tasks with predictable structure: one
  planner + N step calls + one synthesizer ≪ N×K ReAct turns.
* **Observable plan.** The plan is a structured Pydantic object you
  can log, audit, or override before execution.

Weaknesses
----------
* No tool use within steps in v1 (use ReAct or Supervisor for
  tool-heavy work).
* Plan quality bounds answer quality; bad planner = bad output.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from ..core.types import Event, Message, Role
from .base import AgentSession, Dependencies
from .helpers import (
    budget_gate,
    consume_usage,
    parse_fenced_json,
    resolve_role_model,
    strip_markdown_fences,
    text_only_model_call,
)

if TYPE_CHECKING:
    from ..agent.api import Agent
    from ..core.protocols import Model


DEFAULT_PLANNER_PROMPT = """\
You produce a step-by-step plan to solve the user's task. The plan
must be executable: each step should be a concrete sub-task that an
LLM can complete in a single response.

Output a JSON list of strings — each string is one step description.
Aim for 3-7 steps; fewer is better when the task is simple. Order
matters: each step can use prior step outputs.

Output ONLY the JSON list. No prose, no markdown fences."""


DEFAULT_EXECUTOR_PROMPT = """\
You are executing one step of a multi-step plan. Use the original
task and the prior step outputs to produce this step's output.

Output ONLY the result of this step (no preamble, no commentary)."""


DEFAULT_SYNTHESIZER_PROMPT = """\
You synthesize the final answer from a sequence of step outputs.
Combine them into a single coherent response that addresses the
original task. Be concise; don't repeat the steps verbatim."""


class PlanStep(BaseModel):
    """One step of a plan."""

    id: str
    description: str


class Plan(BaseModel):
    """A list of plan steps in execution order."""

    steps: list[PlanStep] = Field(default_factory=list)


class StepResult(BaseModel):
    """The output of executing one step."""

    step_id: str
    description: str
    output: str


class PlanAndExecute:
    """Planner → step executor → synthesizer."""

    name = "plan-and-execute"

    def __init__(
        self,
        *,
        max_steps: int = 8,
        planner_prompt: str | None = None,
        executor_prompt: str | None = None,
        synthesizer_prompt: str | None = None,
        planner_model: str | Model | None = None,
        executor_model: str | Model | None = None,
        synthesizer_model: str | Model | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self._max_steps = max_steps
        self._planner_prompt = (
            planner_prompt or DEFAULT_PLANNER_PROMPT
        )
        self._executor_prompt = (
            executor_prompt or DEFAULT_EXECUTOR_PROMPT
        )
        self._synthesizer_prompt = (
            synthesizer_prompt or DEFAULT_SYNTHESIZER_PROMPT
        )
        # Per-role model routing — the "plan with a frontier model,
        # execute with a cheap one" split, inside one agent. ``None``
        # = the agent's main model (``deps.model``).
        self._planner_model = resolve_role_model(planner_model)
        self._executor_model = resolve_role_model(executor_model)
        self._synthesizer_model = resolve_role_model(synthesizer_model)

    def declared_workers(self) -> dict[str, Agent]:
        return {}

    async def run(
        self,
        session: AgentSession,
        deps: Dependencies,
        prompt: str,
    ) -> AsyncIterator[Event]:
        # === 1. Plan ===
        yield Event.architecture_event(
            session.id, "plan.planner_started"
        )
        plan = await self._make_plan(deps, session, prompt)
        # Cap step count defensively — long plans are usually a
        # planner failure mode.
        if len(plan.steps) > self._max_steps:
            plan = Plan(steps=plan.steps[: self._max_steps])
        yield Event.architecture_event(
            session.id,
            "plan.created",
            steps=[s.description[:120] for s in plan.steps],
            num_steps=len(plan.steps),
        )

        if not plan.steps:
            session.output = (
                "Planner produced no steps; cannot execute."
            )
            yield Event.architecture_event(
                session.id, "plan.empty_plan"
            )
            return

        # === 2. Execute each step sequentially ===
        results: list[StepResult] = []
        for i, step in enumerate(plan.steps):
            blocked, gate_events = await budget_gate(deps, session)
            for gate_event in gate_events:
                yield gate_event
            if blocked:
                return

            yield Event.architecture_event(
                session.id,
                "plan.step_started",
                step_id=step.id,
                step_index=i,
                description=step.description,
            )
            output = await self._execute_step(
                deps, session, prompt, plan, step, results, i
            )
            results.append(
                StepResult(
                    step_id=step.id,
                    description=step.description,
                    output=output,
                )
            )
            yield Event.architecture_event(
                session.id,
                "plan.step_completed",
                step_id=step.id,
                step_index=i,
                output=output[:300],
            )

        # === 3. Synthesize ===
        yield Event.architecture_event(
            session.id, "plan.synthesizer_started"
        )
        final = await self._synthesize(deps, session, prompt, plan, results)
        session.output = final
        # Stash the full plan + step results for post-hoc analysis.
        session.metadata["plan"] = plan.model_dump()
        session.metadata["step_results"] = [
            r.model_dump() for r in results
        ]
        yield Event.architecture_event(
            session.id,
            "plan.completed",
            num_steps=len(plan.steps),
            final=final[:300],
        )

    # ---- helpers -----------------------------------------------------

    async def _make_plan(
        self,
        deps: Dependencies,
        session: AgentSession,
        prompt: str,
    ) -> Plan:
        msgs = [
            Message(role=Role.SYSTEM, content=self._planner_prompt),
            Message(role=Role.USER, content=prompt),
        ]
        text, usage = await text_only_model_call(
            deps, "plan_planner", msgs, model=self._planner_model
        )
        await consume_usage(deps, session, usage)
        return _parse_plan(text)

    async def _execute_step(
        self,
        deps: Dependencies,
        session: AgentSession,
        prompt: str,
        plan: Plan,
        step: PlanStep,
        prior_results: list[StepResult],
        step_index: int,
    ) -> str:
        prior_text = (
            "\n".join(
                f"Step {i + 1} ({r.step_id}): {r.output}"
                for i, r in enumerate(prior_results)
            )
            if prior_results
            else "(no prior steps)"
        )
        plan_text = "\n".join(
            f"  {i + 1}. {s.description}"
            for i, s in enumerate(plan.steps)
        )
        user_content = (
            f"Original task:\n{prompt}\n\n"
            f"Full plan:\n{plan_text}\n\n"
            f"Prior step outputs:\n{prior_text}\n\n"
            f"Now execute step {step_index + 1}:\n"
            f"{step.description}"
        )
        msgs = [
            Message(
                role=Role.SYSTEM,
                content=self._executor_prompt,
            ),
            Message(role=Role.USER, content=user_content),
        ]
        text, usage = await text_only_model_call(
            deps, f"plan_step_{step.id}", msgs, model=self._executor_model
        )
        await consume_usage(deps, session, usage)
        return text.strip()

    async def _synthesize(
        self,
        deps: Dependencies,
        session: AgentSession,
        prompt: str,
        plan: Plan,
        results: list[StepResult],
    ) -> str:
        results_text = "\n\n".join(
            f"Step {i + 1} ({r.step_id}): {r.description}\n"
            f"Output: {r.output}"
            for i, r in enumerate(results)
        )
        user_content = (
            f"Original task:\n{prompt}\n\n"
            f"Step outputs:\n{results_text}\n\n"
            f"Produce the final answer."
        )
        msgs = [
            Message(
                role=Role.SYSTEM,
                content=self._synthesizer_prompt,
            ),
            Message(role=Role.USER, content=user_content),
        ]
        # The synthesizer produces the run's final answer — forward
        # the caller's output_schema so native structured-output
        # adapters can constrain it.
        text, usage = await text_only_model_call(
            deps,
            "plan_synthesizer",
            msgs,
            model=self._synthesizer_model,
            output_schema=deps.output_schema,
        )
        await consume_usage(deps, session, usage)
        return text.strip()


# ---------------------------------------------------------------------------
# Plan parsing
# ---------------------------------------------------------------------------


def _parse_plan(text: str) -> Plan:
    """Parse the planner's JSON-list output into a :class:`Plan`.

    Tolerant: strips markdown code fences, accepts JSON arrays of
    strings or arrays of ``{description: ...}`` objects, falls back
    to splitting on newlines if no JSON found. Step ids are
    auto-assigned (``step_1``, ``step_2``, ...).
    """
    cleaned = strip_markdown_fences(text)

    descriptions: list[str] = []

    # Try a strict JSON parse first.
    parsed: object = parse_fenced_json(cleaned)

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, str):
                descriptions.append(item.strip())
            elif isinstance(item, dict):
                desc = item.get("description") or item.get(
                    "step"
                ) or item.get("name")
                if isinstance(desc, str):
                    descriptions.append(desc.strip())
        # JSON parsed → that IS the plan, even if empty. Don't run
        # the line-splitter fallback (it would otherwise interpret
        # the literal string ``"[]"`` as a single step).
    else:
        # Fallback: split on numbered or bulleted lines.
        for line in cleaned.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            cleaned_line = re.sub(
                r"^\s*(?:[\-\*]|\d+[\.\)])\s*", "", stripped
            )
            if cleaned_line:
                descriptions.append(cleaned_line)

    steps = [
        PlanStep(id=f"step_{i + 1}", description=d)
        for i, d in enumerate(descriptions)
        if d
    ]
    return Plan(steps=steps)
