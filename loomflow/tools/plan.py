"""TodoWrite-style structured plan tools.

A **living plan** is a small, mutable list of steps the agent
rewrites atomically on every update (Claude Code TodoWrite pattern,
2026 mainstream). The plan becomes the load-bearing artifact in
the conversation — every plan-tool call returns the FULL rendered
plan back as the tool result, so the model re-orients on the
current state every time it touches the plan. Drift becomes
structurally hard because every action maps to a step the agent
itself wrote.

Why atomic full-list rewrites beat fine-grained delta updates
(``plan_update_step(2, "done")`` etc.):

* **Forced engagement** — the model must serialize the whole plan
  on every change. Forgetting a step is visible immediately.
* **No partial-update bugs** — one tool call replaces the whole
  state. No "did I add the step before or after the doing one?".
* **Smaller API surface** — two tools (``plan_write``,
  ``plan_read``) instead of five.

Storage:

* Per-run state lives in :data:`~loomflow.core.context._ambient_
  living_plan_var` as a :class:`_LivingPlanState`. The contextvar
  binds tools to per-run state so concurrent ``agent.run()``
  invocations on the same :class:`Agent` instance have isolated
  plans (loomflow's standard concurrency pattern, mirroring
  ``_ambient_workspace_var`` / ``_ambient_memory_var``).
* Optional workspace mirror: when ``Agent(workspace=...)`` is set,
  every successful ``plan_write`` also persists to a ``kind="plan"``
  note so future runs can :func:`recall_past_plans` and bootstrap
  from prior task plans. Multi-tenant ``user_id`` flows through
  :func:`get_run_context` at write time.

Lenient input coercion:

The ``steps`` argument of :func:`plan_write` accepts four shapes
(provider serializations vary):

* native ``list[dict]`` — the ideal shape
* JSON-string of a list — ``'[{"description": ...}, ...]'``
* JSON-string of an object — ``'{"steps": [...]}'``
* free-form numbered text — ``'1. step a\\n2. step b'``

This mirrors the lenient-by-default tool-input convention loomflow
applies elsewhere (str → int coercion for timeouts, etc.).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import anyio

from ..core.context import (
    _ambient_living_plan_var,
    get_run_context,
)
from .registry import Tool, tool

__all__ = [
    "LivingPlan",
    "LivingPlanStep",
    "VALID_STATUSES",
    "get_active_plan",
    "living_plan_prompt_section",
    "make_plan_tools",
    "make_recall_past_plans_tool",
]

# Canonical status set. ``todo`` is the default for new steps;
# ``doing`` / ``blocked`` are the "in-flight" states; the rest are
# terminal. Order is the rendering order used by ``LivingPlan.render``.
VALID_STATUSES: tuple[str, ...] = (
    "todo",
    "doing",
    "done",
    "blocked",
    "skipped",
)


# Soft-coercions for status values the model may invent. Maps
# common synonyms to canonical statuses; unknown values fall back
# to ``"todo"`` (safest — keeps the step active).
_STATUS_SYNONYMS: dict[str, str] = {
    "in_progress": "doing",
    "in-progress": "doing",
    "wip": "doing",
    "started": "doing",
    "running": "doing",
    "complete": "done",
    "completed": "done",
    "finished": "done",
    "ok": "done",
    "failed": "blocked",
    "error": "blocked",
    "stuck": "blocked",
    "fail": "blocked",
    "skip": "skipped",
}


@dataclass(slots=True)
class LivingPlanStep:
    """One step of a :class:`LivingPlan`.

    The agent never constructs this directly — it passes a list of
    ``dict[str, Any]`` to ``plan_write``, which builds the
    :class:`LivingPlanStep` instances after status coercion. The
    dataclass is exposed for tests + custom architectures reading
    the active plan via :func:`get_active_plan`.
    """

    description: str
    status: str = "todo"
    finding: str = ""

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            low = self.status.lower()
            self.status = _STATUS_SYNONYMS.get(low, "todo")


@dataclass(slots=True)
class LivingPlan:
    """The task's living plan. Rewritten atomically on every
    :func:`plan_write` call.

    Read inside custom architectures / hooks via
    :func:`get_active_plan`. Pre-seed a run's plan by constructing
    one explicitly and passing it as
    ``Agent(living_plan=LivingPlan(...))``.
    """

    goal: str = ""
    steps: list[LivingPlanStep] = field(default_factory=list)

    def render(self) -> str:
        """Compact markdown table the agent reads after every
        plan-tool call. Includes a progress summary so the
        ``done/total`` ratio is one glance away."""
        if not self.steps and not self.goal:
            return (
                "(no plan yet — call `plan_write(goal, steps)` "
                "to create one)"
            )
        lines = [f"**GOAL:** {self.goal}", ""]
        lines.append("| # | Status | Description | Finding |")
        lines.append("|---|--------|-------------|---------|")
        # Compact status badges so the table stays scannable on
        # narrow tool-result displays.
        badge = {
            "todo": "todo",
            "doing": "DOING",
            "done": "DONE",
            "blocked": "BLOCKED",
            "skipped": "skipped",
        }
        for i, step in enumerate(self.steps, 1):
            desc = step.description.replace("|", "\\|")
            finding = step.finding.replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {i} | {badge.get(step.status, step.status)} "
                f"| {desc} | {finding} |"
            )
        done = sum(1 for s in self.steps if s.status == "done")
        blocked = sum(1 for s in self.steps if s.status == "blocked")
        lines.append("")
        lines.append(
            f"**Progress:** {done}/{len(self.steps)} done"
            + (f", {blocked} blocked" if blocked else "")
        )
        return "\n".join(lines)


@dataclass(slots=True)
class _LivingPlanState:
    """Per-run state for the living-plan tools.

    Stored on :data:`_ambient_living_plan_var` for the duration of
    one :meth:`Agent.run` invocation. Holds the plan itself plus
    the captured workspace-mirror slug (so subsequent
    ``plan_write`` calls within the run update the same note rather
    than creating duplicates).
    """

    plan: LivingPlan = field(default_factory=LivingPlan)
    mirror_slug: str | None = None


def get_active_plan() -> LivingPlan | None:
    """Return the :class:`LivingPlan` for the currently-running
    agent, or ``None`` if living-plan is not enabled for this run.

    Custom architectures and hooks call this to inspect the plan
    state — e.g. a hook that emits a telemetry event when a step
    transitions to ``done``, or an architecture that injects the
    rendered plan into the next user message.
    """
    state = _ambient_living_plan_var.get()
    if state is None:
        return None
    assert isinstance(state, _LivingPlanState)
    return state.plan


def _coerce_steps(value: Any) -> list[dict[str, Any]] | str:
    """Try to coerce the model's serialization of ``steps`` into a
    native list-of-dicts. Returns the list on success, or an error
    message string the tool returns verbatim to the model.

    Handles four input shapes (see module docstring). Anything else
    yields a string error message — the tool propagates that back
    as the tool result so the model sees actionable feedback.
    """
    if isinstance(value, list):
        return [s for s in value if isinstance(s, dict)]
    if not isinstance(value, str):
        return (
            "ERROR: `steps` must be a list of dicts. "
            f"Got: {type(value).__name__}"
        )
    text = value.strip()
    parsed: Any = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Numbered-text fallback. Each non-empty line that starts
        # with a number + "." / ")" / ":" or a "- " / "* " bullet
        # becomes one ``todo`` step. Anything else is ignored.
        out: list[dict[str, Any]] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            for prefix in ("- ", "* "):
                if line.startswith(prefix):
                    line = line[len(prefix):]
                    break
            else:
                cut = 0
                while cut < len(line) and line[cut].isdigit():
                    cut += 1
                if cut and cut < len(line) and line[cut] in ".):":
                    line = line[cut + 1:].strip()
            if line:
                out.append({"description": line, "status": "todo"})
        return out
    # Unwrap ``{"steps": [...]}`` if the model wrapped a list in
    # an outer object.
    if isinstance(parsed, dict) and "steps" in parsed:
        parsed = parsed["steps"]
    if not isinstance(parsed, list):
        return (
            "ERROR: parsed `steps` is not a list. "
            f"Got: {type(parsed).__name__}"
        )
    return [s for s in parsed if isinstance(s, dict)]


def _current_user_id() -> str | None:
    """Read the multi-tenant partition key from the ambient
    :class:`~loomflow.RunContext`. ``None`` outside an active run."""
    return get_run_context().user_id


def make_plan_tools(
    *,
    workspace: Any | None = None,
    task_id: str | None = None,
    author: str = "agent",
) -> list[Tool]:
    """Build ``plan_write`` + ``plan_read`` tools.

    The tools read per-run state from the ambient
    :data:`_ambient_living_plan_var` contextvar (set by
    :meth:`Agent.run` at run start). This means:

    * Concurrent ``agent.run()`` calls on the same :class:`Agent`
      have isolated plan state.
    * Tools can be constructed once at :meth:`Agent.__init__` and
      reused across many runs.
    * Outside an active run (test code, direct ``@tool``
      invocation), the tools return an "not enabled" error
      message rather than raise.

    ``workspace`` is optional. When provided, every successful
    ``plan_write`` mirrors the plan to a ``kind="plan"`` note in
    the workspace (first call ``write_note``; subsequent calls
    ``update_note`` using the slug captured in the per-run state).
    Multi-tenant ``user_id`` flows through :func:`get_run_context`.

    ``task_id`` is embedded in the mirror note's title for
    cross-task discoverability. Falls back to the active
    :class:`~loomflow.RunContext`'s ``run_id``.

    ``author`` is baked into workspace writes so the agent never
    has to attribute itself. :class:`Team` builders override this
    with the worker's role name.
    """

    async def _mirror_to_workspace(state: _LivingPlanState) -> None:
        if workspace is None:
            return
        body = (
            state.plan.render()
            + "\n\n```json\n"
            + json.dumps(
                {
                    "goal": state.plan.goal,
                    "steps": [asdict(s) for s in state.plan.steps],
                },
                indent=2,
            )
            + "\n```"
        )
        run_id = get_run_context().run_id
        title_id = task_id or run_id or "run"
        title = f"Plan {title_id}: {state.plan.goal[:60]}"
        try:
            if state.mirror_slug is not None:
                await workspace.update_note(
                    author=author,
                    slug=state.mirror_slug,
                    body=body,
                    user_id=_current_user_id(),
                )
            else:
                note = await workspace.write_note(
                    author=author,
                    title=title,
                    body=body,
                    kind="plan",
                    user_id=_current_user_id(),
                    run_id=run_id or None,
                )
                state.mirror_slug = note.slug
        except anyio.get_cancelled_exc_class():
            # Cancellation must propagate — never swallow.
            raise
        except Exception:  # noqa: BLE001 — mirroring is best-effort
            # Disk full, permissions, transient I/O — the
            # in-memory plan is the source of truth. Don't fail
            # the tool call over a missing mirror.
            pass

    @tool
    async def plan_write(
        goal: str, steps: list[dict[str, Any]] | str
    ) -> str:
        """Create or rewrite the LIVING PLAN for this run.

        Pass the COMPLETE updated list of steps every call — this
        tool atomically replaces the prior plan. Each step is a
        dict with:

        * ``description`` (str, required) — what the step does.
        * ``status`` (str, optional) — ``todo`` | ``doing`` |
          ``done`` | ``blocked`` | ``skipped``. Defaults to
          ``todo``. Synonyms like ``in_progress``, ``failed``,
          ``WIP`` are auto-normalized.
        * ``finding`` (str, optional) — 1-line note about what
          happened on this step. Most useful for ``done`` and
          ``blocked`` steps.

        Returns the rendered plan as a markdown table — read it.
        That table is your source of truth for what's next.

        Workflow:

        1. **Plan first.** After your initial orient calls, call
           ``plan_write(goal=..., steps=[{description: ..., status:
           "todo"}, ...])``. 3-7 steps is the sweet spot.
        2. **Update as you go.** Before each significant action,
           rewrite the plan with that step's status = ``doing``.
           After the action, rewrite again with ``done`` (and a
           1-line ``finding``) or ``blocked`` (with a finding
           describing the blocker).
        3. **Replan on discovery.** If you learn something that
           changes the strategy, just call ``plan_write`` again
           with the new list. Insert, remove, reorder — it's a
           full rewrite.
        4. **Verify before done.** A step like "Run the validator
           and confirm pass" should be in your plan. Do NOT mark
           it done until the validator actually passes.
        """
        state = _ambient_living_plan_var.get()
        if not isinstance(state, _LivingPlanState):
            return (
                "ERROR: living_plan is not enabled for this run. "
                "Set Agent(living_plan=True) to enable."
            )
        coerced = _coerce_steps(steps)
        if isinstance(coerced, str):
            # Coerce returned an error message; return verbatim
            # so the model sees actionable feedback.
            return coerced
        state.plan = LivingPlan(
            goal=str(goal),
            steps=[
                LivingPlanStep(
                    description=str(s.get("description", "")),
                    status=str(s.get("status", "todo")),
                    finding=str(s.get("finding", "")),
                )
                for s in coerced
            ],
        )
        await _mirror_to_workspace(state)
        return state.plan.render()

    @tool
    async def plan_read() -> str:
        """Return the current LIVING PLAN as a markdown table.

        Call this to re-orient on what's left without making any
        changes. Most of the time the return value of
        ``plan_write`` is enough — call ``plan_read`` only when
        the plan state is no longer fresh in your context.
        """
        state = _ambient_living_plan_var.get()
        if not isinstance(state, _LivingPlanState):
            return (
                "ERROR: living_plan is not enabled for this run."
            )
        return state.plan.render()

    return [plan_write, plan_read]


def make_recall_past_plans_tool(
    workspace: Any,
    *,
    author: str = "agent",
) -> Tool:
    """Build ``recall_past_plans(query)`` — cross-task plan lineage.

    Once a few tasks have completed and mirrored their plans to
    the workspace, a new run can search past plans by query and
    bootstrap from ones that match. Past plans show which
    strategies succeeded (``done`` steps) and which got
    ``blocked`` — invaluable head start.

    Multi-tenant: scoped to the active ``user_id`` from
    :func:`get_run_context`. Plans from user A are invisible to
    user B even with the same query.
    """

    @tool
    async def recall_past_plans(query: str, limit: int = 3) -> str:
        """Search past runs' LIVING PLANS by free-text ``query``.

        Returns up to ``limit`` matching plans with their slugs,
        titles, and a preview. Read the full plan with
        ``read_note(slug)``.

        Call this at task start with key terms from your task
        ("conda env conflict", "pytest pin", "permission chmod")
        to see if a past run already worked on similar ground.
        Past plans expose what strategies worked and what got
        stuck — invaluable head start.
        """
        cap = max(1, min(10, int(limit)))
        # ``search_notes`` has no ``kind=`` filter, so pull a
        # widened result and filter to ``kind="plan"`` post-hoc.
        try:
            hits = await workspace.search_notes(
                query, limit=cap * 3, user_id=_current_user_id()
            )
        except anyio.get_cancelled_exc_class():
            raise
        except Exception as exc:  # noqa: BLE001 — fail soft
            return f"recall_past_plans failed: {exc}"

        # ``NoteMatch`` exposes the projection on ``match.summary``
        # (slug / title / kind / tags / lede) plus ``match.snippet``
        # for the body excerpt around the match. The full body
        # itself is fetched lazily via ``read_note``.
        plan_hits = [
            m for m in hits if getattr(m.summary, "kind", "") == "plan"
        ][:cap]
        if not plan_hits:
            return (
                f"No past plans match '{query}'. Expected early "
                "in the run — later tasks benefit more."
            )
        out = [f"Top {len(plan_hits)} past plan(s) matching '{query}':", ""]
        for match in plan_hits:
            summary = match.summary
            preview = (
                (match.snippet or summary.lede or "")[:200]
                .replace("\n", " ")
            )
            out.append(f"- **{summary.slug}** — {summary.title}")
            out.append(f"  > {preview}...")
            out.append(f"  Read full: `read_note('{summary.slug}')`")
            out.append("")
        # Author is captured but not used in the body — kept on
        # the signature for symmetry with other workspace-tool
        # factories. Suppress unused warning via reference.
        _ = author
        return "\n".join(out)

    return recall_past_plans


def living_plan_prompt_section(*, has_workspace_mirror: bool) -> str:
    """The markdown chunk :meth:`Agent.__init__` appends to the
    system prompt when ``living_plan=`` is wired.

    Tells the model that the plan tools exist and HOW to use them
    (TodoWrite discipline). Slightly different wording when
    ``has_workspace_mirror`` is True — the model is told the plan
    persists and can be recalled by future tasks.
    """
    body = (
        "## Living plan (your scratchpad)\n\n"
        "You have a structured **living plan** — a TodoWrite-style "
        "list of steps with statuses. Use it religiously:\n\n"
        "- ``plan_write(goal, steps)`` — REWRITE the FULL plan "
        "atomically every time something changes. Each step is "
        '``{"description": "...", "status": "todo"|"doing"|"done"|'
        '"blocked"|"skipped", "finding": "..."}``.\n'
        "- ``plan_read()`` — re-orient on current state.\n\n"
        "**Discipline**:\n\n"
        "1. After orient calls, call ``plan_write`` with 3-7 steps. "
        "The LAST step must be a VERIFY step naming the success "
        "criterion (validator command, expected state, etc.).\n"
        "2. Before each significant action, ``plan_write`` with "
        "that step's status = ``doing``.\n"
        "3. After the action, ``plan_write`` again with status = "
        "``done`` (and a 1-line ``finding``) or ``blocked`` (with "
        "a finding describing the blocker).\n"
        "4. If discovery requires changes, just rewrite the plan. "
        "Insert / remove / reorder steps. The plan is a living "
        "document, not a frozen contract.\n"
        "5. NEVER declare a task done before the VERIFY step is "
        "actually ``done`` (not just optimistically marked so).\n"
    )
    if has_workspace_mirror:
        body += (
            "\nYour plan persists to the shared notebook as a "
            '``kind="plan"`` note. Future runs can call '
            "``recall_past_plans(query)`` to find similar prior "
            "plans and bootstrap from what worked.\n"
        )
    return body
