"""ReWOO: Reasoning WithOut Observation — plan-then-tool-execute.

Xu et al. 2023 — `ReWOO: Decoupling Reasoning from Observations
for Efficient Augmented Language Models
<https://arxiv.org/abs/2305.18323>`_. The cost-saving sibling of
:class:`~loomflow.architecture.PlanAndExecute`: each step in the
plan is a real **tool call**, with ``{{En}}`` placeholder
substitution to reference prior step outputs. Independent steps
(no dependency on each other) run in **parallel**.

Total cost: **2 LLM calls + N tool calls**. ReAct on the same task
needs roughly N+1 LLM calls (one per turn). For tool-heavy workloads
where the planner can predict the call sequence upfront, ReWOO is
30-50% cheaper.

Pattern
-------

1. **Planner.** ONE LLM call. Output is a JSON list of steps. Each
   step has shape::

       {"id": "E1", "tool": "<tool_name>", "args": {...}}

   Args may reference prior steps via ``{{En}}`` placeholders —
   ``{"args": {"url": "{{E1}}"}}`` will use E1's output as the
   ``url`` arg when E2 runs.

2. **Worker.** Compute topological levels from the plan's
   placeholder dependencies. For each level, dispatch all steps
   in parallel via ``deps.tools.call(...)``. Substitute
   ``{{En}}`` placeholders in args from prior step outputs first.

3. **Solver.** ONE LLM call. Given the original task and the
   step→output map, produce the final answer.

Strengths
---------
* **Cheaper than ReAct on tool-heavy multi-step tasks.** Two LLM
  calls cap the LLM cost regardless of plan length.
* **Parallelism for free.** Independent steps run concurrently via
  ``anyio.create_task_group`` (same primitive Supervisor + ReAct
  use for parallel tool dispatch).
* **Observable plan.** The plan is a structured Pydantic object —
  log it, audit it, override it before execution.

Weaknesses
---------
* **Planner must predict accurately upfront.** No replanning on
  failure in v1. If a step fails, the worker logs the error and
  the solver sees it as the step's "output."
* **Limited to known tool names.** A planner that hallucinates a
  tool name produces a step that errors at dispatch time.
* **Placeholder substitution is string-typed.** Tool outputs get
  stringified. For structured-output tools, the planner has to
  treat outputs as opaque text.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import anyio
from pydantic import BaseModel, Field

from ..core.types import Event, Message, Role, ToolResult
from .base import AgentSession, Dependencies
from .helpers import add_usage, text_only_model_call

if TYPE_CHECKING:
    from ..agent.api import Agent


DEFAULT_PLANNER_PROMPT = """\
You produce a step-by-step plan to solve the user's task using the
available tools. Each step is a tool call.

Output ONLY a JSON array of step objects. Each step has:
- "id": a short identifier like "E1", "E2", ... unique within the plan
- "tool": the name of one of the available tools
- "args": a dict of arguments to pass to the tool

You may reference a prior step's output in args using `{{En}}`. The
worker will substitute `{{En}}` with the actual output of step En
before invoking the tool.

Available tools:
{tool_descriptions}

Output format example:
[
  {{"id": "E1", "tool": "web_search", "args": {{"query": "Tokyo weather"}}}},
  {{"id": "E2", "tool": "summarize", "args": {{"text": "{{{{E1}}}}"}}}}
]

Output ONLY the JSON array. No prose, no markdown fences.
"""


DEFAULT_SOLVER_PROMPT = """\
You synthesize the final answer from a sequence of tool-call
results. Use the original task and the step outputs to produce the
final answer. Be concise."""


class ReWOOStep(BaseModel):
    """One step of a ReWOO plan: id + tool + args."""

    id: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)

    @property
    def depends_on(self) -> list[str]:
        """Extract ``{{En}}`` step ids referenced in args."""
        return _extract_placeholders(self.args)


class ReWOOPlan(BaseModel):
    """A list of ReWOO steps (no required ordering — dependencies
    are inferred from ``{{En}}`` placeholders)."""

    steps: list[ReWOOStep] = Field(default_factory=list)


class ReWOOStepResult(BaseModel):
    step_id: str
    tool: str
    output: str
    error: str | None = None


class ReWOO:
    """Plan-then-tool-execute with placeholder substitution."""

    name = "rewoo"

    def __init__(
        self,
        *,
        max_steps: int = 8,
        planner_prompt: str | None = None,
        solver_prompt: str | None = None,
        parallel_levels: bool = True,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self._max_steps = max_steps
        self._planner_prompt = (
            planner_prompt or DEFAULT_PLANNER_PROMPT
        )
        self._solver_prompt = (
            solver_prompt or DEFAULT_SOLVER_PROMPT
        )
        self._parallel_levels = parallel_levels

    def declared_workers(self) -> dict[str, Agent]:
        return {}

    async def run(
        self,
        session: AgentSession,
        deps: Dependencies,
        prompt: str,
    ) -> AsyncIterator[Event]:
        # === 1. Planner ===
        yield Event.architecture_event(
            session.id, "rewoo.planner_started"
        )
        plan = await self._make_plan(deps, session, prompt)
        if len(plan.steps) > self._max_steps:
            plan = ReWOOPlan(steps=plan.steps[: self._max_steps])
        yield Event.architecture_event(
            session.id,
            "rewoo.plan_created",
            num_steps=len(plan.steps),
            steps=[
                {"id": s.id, "tool": s.tool, "depends_on": s.depends_on}
                for s in plan.steps
            ],
        )

        if not plan.steps:
            session.output = (
                "Planner produced no steps; cannot execute."
            )
            yield Event.architecture_event(
                session.id, "rewoo.empty_plan"
            )
            return

        # === 2. Topological worker ===
        levels = _topological_levels(plan.steps)
        if levels is None:
            # Cycle detected — planner-side bug.
            session.output = (
                "Planner produced a plan with cyclic dependencies; "
                "cannot execute."
            )
            yield Event.architecture_event(
                session.id, "rewoo.cyclic_plan"
            )
            return

        results: dict[str, ReWOOStepResult] = {}
        for level_index, level in enumerate(levels):
            status = await deps.budget.allows_step()
            if status.blocked:
                session.interrupted = True
                session.interruption_reason = (
                    f"budget:{status.reason}"
                )
                yield Event.budget_exceeded(session.id, status)
                return
            if status.warn:
                yield Event.budget_warning(session.id, status)

            yield Event.architecture_event(
                session.id,
                "rewoo.level_started",
                level=level_index,
                step_ids=[s.id for s in level],
            )

            level_results = await _execute_level(
                deps, session, level, results, self._parallel_levels
            )
            results.update(level_results)
            for step_id, sr in level_results.items():
                yield Event.architecture_event(
                    session.id,
                    "rewoo.step_completed",
                    step_id=step_id,
                    tool=sr.tool,
                    error=sr.error,
                    output=sr.output[:300],
                )

        # === 3. Solver ===
        yield Event.architecture_event(
            session.id, "rewoo.solver_started"
        )
        final = await self._solve(
            deps, session, prompt, plan, results
        )
        session.output = final
        session.metadata["rewoo_plan"] = plan.model_dump()
        session.metadata["rewoo_results"] = {
            sid: r.model_dump() for sid, r in results.items()
        }
        yield Event.architecture_event(
            session.id,
            "rewoo.completed",
            num_steps=len(plan.steps),
            final=final[:300],
        )

    # ---- helpers -----------------------------------------------------

    async def _make_plan(
        self,
        deps: Dependencies,
        session: AgentSession,
        prompt: str,
    ) -> ReWOOPlan:
        tool_defs = await deps.tools.list_tools()
        tool_descriptions = (
            "\n".join(
                f"  - {t.name}: {t.description}" for t in tool_defs
            )
            or "  (no tools registered)"
        )
        planner_text = self._planner_prompt.format(
            tool_descriptions=tool_descriptions
        )
        msgs = [
            Message(role=Role.SYSTEM, content=planner_text),
            Message(role=Role.USER, content=prompt),
        ]
        text, usage = await text_only_model_call(
            deps, "rewoo_planner", msgs
        )
        await deps.budget.consume(
            tokens_in=usage.input_tokens,
            tokens_out=usage.output_tokens,
            cost_usd=usage.cost_usd,
        )
        session.cumulative_usage = add_usage(
            session.cumulative_usage, usage
        )
        session.turns += 1
        return _parse_rewoo_plan(text)

    async def _solve(
        self,
        deps: Dependencies,
        session: AgentSession,
        prompt: str,
        plan: ReWOOPlan,
        results: dict[str, ReWOOStepResult],
    ) -> str:
        results_text = "\n\n".join(
            f"Step {sid} ({results[sid].tool}):\n"
            + (
                f"ERROR: {results[sid].error}"
                if results[sid].error
                else results[sid].output
            )
            for sid in [s.id for s in plan.steps]
            if sid in results
        )
        user_content = (
            f"Original task:\n{prompt}\n\n"
            f"Step outputs:\n{results_text}\n\n"
            f"Produce the final answer."
        )
        msgs = [
            Message(role=Role.SYSTEM, content=self._solver_prompt),
            Message(role=Role.USER, content=user_content),
        ]
        text, usage = await text_only_model_call(
            deps, "rewoo_solver", msgs
        )
        await deps.budget.consume(
            tokens_in=usage.input_tokens,
            tokens_out=usage.output_tokens,
            cost_usd=usage.cost_usd,
        )
        session.cumulative_usage = add_usage(
            session.cumulative_usage, usage
        )
        session.turns += 1
        return text.strip()


# ---------------------------------------------------------------------------
# Placeholder + topological helpers
# ---------------------------------------------------------------------------


_PLACEHOLDER_RE = re.compile(r"\{\{(E\d+)\}\}")


def _extract_placeholders(value: Any) -> list[str]:
    """Recursively walk ``value`` (a dict / list / str) and collect
    every ``{{En}}`` placeholder id. Used to compute step
    dependencies from ``ReWOOStep.args``."""
    found: set[str] = set()

    def _walk(v: Any) -> None:
        if isinstance(v, str):
            for m in _PLACEHOLDER_RE.finditer(v):
                found.add(m.group(1))
        elif isinstance(v, dict):
            for sub in v.values():
                _walk(sub)
        elif isinstance(v, list):
            for sub in v:
                _walk(sub)

    _walk(value)
    return sorted(found)


def _substitute_placeholders(
    value: Any, results: dict[str, ReWOOStepResult]
) -> Any:
    """Recursively replace ``{{En}}`` in ``value`` with the
    corresponding step's output text. Strings get substring
    substitution; non-string types are returned unchanged."""
    if isinstance(value, str):
        def _replace(match: re.Match[str]) -> str:
            sid = match.group(1)
            if sid in results:
                return results[sid].output
            return match.group(0)  # leave unresolved as-is

        return _PLACEHOLDER_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {
            k: _substitute_placeholders(v, results)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            _substitute_placeholders(v, results) for v in value
        ]
    return value


def _topological_levels(
    steps: list[ReWOOStep],
) -> list[list[ReWOOStep]] | None:
    """Group steps by topological level — steps in level N depend
    only on steps in levels 0..N-1.

    Returns ``None`` if the dependency graph has a cycle (planner
    bug). Steps that reference unknown ids are treated as having
    no dependency on those — they run as soon as their KNOWN
    dependencies are satisfied (the substitution leaves the
    placeholder literal in place at execution time).
    """
    by_id = {s.id: s for s in steps}
    remaining = {s.id for s in steps}
    levels: list[list[ReWOOStep]] = []
    placed: set[str] = set()

    while remaining:
        # Steps whose deps are all placed already (or unknown ids)
        # qualify for this level.
        current_level = [
            by_id[sid]
            for sid in remaining
            if all(
                dep in placed or dep not in by_id
                for dep in by_id[sid].depends_on
            )
        ]
        if not current_level:
            # Nothing made progress → cycle.
            return None
        # Stable order within a level: by step id (lexicographic).
        current_level.sort(key=lambda s: s.id)
        levels.append(current_level)
        for s in current_level:
            placed.add(s.id)
            remaining.discard(s.id)

    return levels


async def _execute_level(
    deps: Dependencies,
    session: AgentSession,
    level: list[ReWOOStep],
    prior_results: dict[str, ReWOOStepResult],
    parallel: bool,
) -> dict[str, ReWOOStepResult]:
    """Run every step in ``level`` (parallel or sequential) and
    return the new step results keyed by step id."""
    new_results: dict[str, ReWOOStepResult] = {}

    if parallel and len(level) > 1:
        async with anyio.create_task_group() as tg:
            for step in level:
                tg.start_soon(
                    _run_one_step,
                    deps,
                    session,
                    step,
                    prior_results,
                    new_results,
                )
    else:
        for step in level:
            await _run_one_step(
                deps, session, step, prior_results, new_results
            )
    return new_results


async def _run_one_step(
    deps: Dependencies,
    session: AgentSession,
    step: ReWOOStep,
    prior_results: dict[str, ReWOOStepResult],
    out_results: dict[str, ReWOOStepResult],
) -> None:
    """Resolve placeholders, dispatch the tool call, capture output
    or error into ``out_results[step.id]``."""
    resolved_args = _substitute_placeholders(
        step.args, prior_results
    )
    try:
        tool_result: ToolResult = await deps.runtime.step(
            f"rewoo_step_{step.id}",
            deps.tools.call,
            step.tool,
            resolved_args,
            call_id=f"rewoo_{step.id}",
        )
    except Exception as exc:  # noqa: BLE001 — surface as step error
        out_results[step.id] = ReWOOStepResult(
            step_id=step.id,
            tool=step.tool,
            output="",
            error=str(exc),
        )
        return

    if tool_result.ok:
        out_results[step.id] = ReWOOStepResult(
            step_id=step.id,
            tool=step.tool,
            output=str(tool_result.output),
        )
    else:
        out_results[step.id] = ReWOOStepResult(
            step_id=step.id,
            tool=step.tool,
            output="",
            error=tool_result.error
            or tool_result.reason
            or "tool failed",
        )


# ---------------------------------------------------------------------------
# Plan parsing
# ---------------------------------------------------------------------------


def _parse_rewoo_plan(text: str) -> ReWOOPlan:
    """Parse a ReWOO plan from JSON. Tolerant of markdown code
    fences. Returns an empty plan on parse failure (caller's
    ``empty_plan`` branch handles termination)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return ReWOOPlan(steps=[])

    if not isinstance(parsed, list):
        return ReWOOPlan(steps=[])

    steps: list[ReWOOStep] = []
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            continue
        sid = str(item.get("id") or f"E{i + 1}")
        tool = item.get("tool")
        if not isinstance(tool, str) or not tool:
            continue
        args = item.get("args", {})
        if not isinstance(args, dict):
            args = {}
        steps.append(ReWOOStep(id=sid, tool=tool, args=args))

    return ReWOOPlan(steps=steps)
