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

The ``steps`` argument of :func:`plan_write` is coerced from
whatever the model serialised — provider serializations vary
wildly, and weaker models (gpt-4.1-mini, etc.) are especially
loose. Accepted shapes:

* native ``list[dict]`` — the ideal shape
* ``list[str]`` — a list of plain-text descriptions (each becomes
  a ``todo`` step)
* a list with elements that are themselves JSON-string dicts
* a bare ``dict`` — either a ``{"steps": [...]}`` wrapper or a
  single step the model forgot to wrap in a list
* JSON-string of any of the above
* free-form numbered text — ``'1. step a\\n2. step b'``

This mirrors the lenient-by-default tool-input convention loomflow
applies elsewhere (str → int coercion for timeouts, etc.). The
rule: salvage anything salvageable; only error when there is
genuinely nothing to work with.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import anyio

from ..core.context import (
    _ambient_living_plan_var,
    _ambient_workspace_var,
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


# Trailing status markers the model tends to append to a step's
# DESCRIPTION instead of setting the ``status`` field — e.g.
# "Summarise app.py DONE". Left unparsed, the status stays ``todo``,
# the plan reads 0/N done forever, and the agent re-plans in a loop.
# We promote such a marker to the status field. Deliberately
# conservative: the words are matched CASE-SENSITIVELY in all-caps
# (so a description merely ending in lowercase "done" is NOT a false
# positive), and a separator before the marker is required. Bare
# "TODO" is intentionally excluded — it's a common code term, and
# ``todo`` is the harmless default anyway (use ``[ ]`` for explicit
# todo).
_TRAILING_MARKER_RE = re.compile(
    r"[\s\-–—:|]+(DONE|DOING|BLOCKED|SKIPPED|\[[ xX]\]|✓|✗)[\s.)\]]*$"
)
_MARKER_STATUS: dict[str, str] = {
    "DONE": "done",
    "[x]": "done",
    "[X]": "done",
    "✓": "done",
    "DOING": "doing",
    "BLOCKED": "blocked",
    "SKIPPED": "skipped",
    "✗": "skipped",
    "[ ]": "todo",
}


def _split_trailing_marker(text: str) -> tuple[str, str | None]:
    """Return ``(description_without_marker, status_or_None)``.

    Recognises a status marker appended to the END of a step
    description (the "... DONE" failure mode). Returns the cleaned
    description plus the canonical status, or ``(text, None)`` when no
    explicit trailing marker is present."""
    m = _TRAILING_MARKER_RE.search(text)
    if m is None:
        return text, None
    status = _MARKER_STATUS.get(m.group(1))
    if status is None:
        return text, None
    return text[: m.start()].rstrip(), status


@dataclass(slots=True)
class LivingPlanStep:
    """One step of a :class:`LivingPlan`.

    The agent never constructs this directly — it passes a list of
    ``dict[str, Any]`` to ``plan_write``, which builds the
    :class:`LivingPlanStep` instances after status coercion. The
    dataclass is exposed for tests + custom architectures reading
    the active plan via :func:`get_active_plan`.

    ``verified_by`` is the list of ``ToolCall.id``\\ s that did this
    step's work. ``plan_write`` validates that DONE transitions
    either reference real tool calls fired in the current turn
    (each id usable by at most one step — kills the "one call,
    many claims" failure mode where a single edit is claimed to
    cover N items) OR carry a non-empty ``finding`` (≥20 chars)
    that justifies why no tool work was needed (analytical steps,
    pure-reasoning items, etc.). Existing callers that omit
    ``verified_by`` get the finding-fallback automatically — back-
    compat preserved during the soft cutover in 0.10.x.
    """

    description: str
    status: str = "todo"
    finding: str = ""
    verified_by: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Promote a status marker the model appended to the description
        # (e.g. "... DONE") into the status field — but only when the
        # field is still the default ``todo`` (never override a status
        # the model set explicitly). Strip the marker either way so the
        # rendered description stays clean.
        cleaned, marker_status = _split_trailing_marker(self.description)
        if marker_status is not None:
            self.description = cleaned
            if self.status == "todo":
                self.status = marker_status
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

    ``observed_tool_call_ids`` accumulates every non-plan tool call
    fired in the current Agent.run() turn (architectures call
    :func:`record_tool_call` to add). ``plan_write`` reads this set
    when validating ``verified_by`` references — the set of ids it
    will accept for new DONE transitions. Architectures that
    haven't been updated to call ``record_tool_call`` leave this
    empty, in which case ``plan_write`` falls back to requiring a
    non-empty ``finding`` on every DONE transition (soft cutover —
    no behaviour regression for un-upgraded architectures).
    """

    plan: LivingPlan = field(default_factory=LivingPlan)
    mirror_slug: str | None = None
    observed_tool_call_ids: set[str] = field(default_factory=set)


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


def record_tool_call(call_id: str) -> None:
    """Architecture hook: record that a (non-plan_write) tool call
    fired this turn so subsequent ``plan_write`` calls can verify
    DONE transitions reference real work.

    Call AFTER each tool dispatch in the architecture's main loop,
    excluding ``plan_write`` itself (recording plan_write would let
    the model self-verify steps by simply re-calling the plan tool).
    No-op when living_plan isn't enabled for this run.

    Soft cutover: architectures that haven't been updated to call
    this still work — ``plan_write`` falls back to requiring a
    non-empty ``finding`` (≥20 chars) on every DONE transition
    when the observed-id set is empty. Upgraded architectures get
    the stronger ``verified_by`` check + double-claim prevention.
    """
    state = _ambient_living_plan_var.get()
    if isinstance(state, _LivingPlanState):
        state.observed_tool_call_ids.add(call_id)


def _coerce_one_step(item: Any) -> dict[str, Any] | None:
    """Coerce a single step-ish element into a step dict, or
    ``None`` if it can't be salvaged.

    Weak models serialise step lists inconsistently — a list of
    plain strings, a list of stringified-JSON dicts, a mix — so
    EVERY element of a ``steps`` list goes through here:

    * a ``dict`` is used as-is;
    * a string that parses as a JSON object becomes that dict;
    * any other non-empty string becomes a ``todo`` step described
      by the text;
    * anything else (int, None, …) yields ``None`` and is dropped.
    """
    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        s = item.strip()
        if not s:
            return None
        if s[0] in "{[":
            try:
                inner = json.loads(s)
            except json.JSONDecodeError:
                inner = None
            if isinstance(inner, dict):
                return inner
        return {"description": s, "status": "todo"}
    return None


def _coerce_numbered_text(text: str) -> list[dict[str, Any]]:
    """Last-resort fallback: turn free-form numbered / bulleted
    text into ``todo`` steps. Each non-empty line that starts with
    a number + ``.`` / ``)`` / ``:`` or a ``- `` / ``* `` bullet
    becomes one step (the prefix is stripped); other non-empty
    lines are kept verbatim as steps too."""
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


def _coerce_steps(value: Any) -> list[dict[str, Any]] | str:
    """Coerce the model's serialization of ``steps`` into a native
    list-of-dicts. Returns the list on success, or an error string
    the tool returns verbatim so the model sees actionable feedback.

    See the module docstring for the accepted shapes. The guiding
    rule: salvage anything salvageable; only error when the input
    is genuinely empty or structurally hopeless.
    """
    # A bare dict: either a ``{"steps": [...]}`` wrapper, or a
    # single step the model forgot to wrap in a list.
    if isinstance(value, dict):
        value = value["steps"] if "steps" in value else [value]
    if isinstance(value, list):
        out = [
            step
            for step in (_coerce_one_step(s) for s in value)
            if step is not None
        ]
        if not out and value:
            return (
                "ERROR: `steps` had items but none were usable. "
                "Each step should be an object with a "
                "`description` (and optional `status` / `finding`), "
                "or a plain string description."
            )
        return out
    if not isinstance(value, str):
        return (
            "ERROR: `steps` must be a list of step objects. "
            f"Got: {type(value).__name__}"
        )
    text = value.strip()
    if not text:
        return "ERROR: `steps` was empty."
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Not JSON — treat as free-form numbered / bulleted text.
        return _coerce_numbered_text(text)
    # Parsed cleanly — recurse so the list / dict / wrapper logic
    # above handles it uniformly. (Recursion terminates: each JSON
    # decode strips a layer, and a non-JSON string hits the
    # numbered-text branch.)
    return _coerce_steps(parsed)


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
        # Explicit workspace wins; otherwise inherit from the
        # ambient ``_ambient_workspace_var`` set by a parent
        # :class:`Workflow`. Mirrors the workspace-tools ambient
        # pattern so ``Workflow(workspace=ws)`` + child
        # ``Agent(living_plan=True)`` (no explicit workspace) works
        # symmetrically: the plan persists into the workflow's
        # shared notebook without per-agent ``workspace=`` boilerplate.
        effective_workspace = workspace
        if effective_workspace is None:
            effective_workspace = _ambient_workspace_var.get()
        if effective_workspace is None:
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
                await effective_workspace.update_note(
                    author=author,
                    slug=state.mirror_slug,
                    body=body,
                    user_id=_current_user_id(),
                )
            else:
                note = await effective_workspace.write_note(
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
          ``blocked`` steps. **Required (≥20 chars) on any DONE
          step that has empty ``verified_by``** (see below).
        * ``verified_by`` (list[str], optional) — the
          ``tool_call_id``\\ (s) that did this step's work. Each
          id must reference a real tool call fired earlier in
          this turn (you can see ids on tool_result messages in
          your context). Each id can verify AT MOST ONE step —
          claiming the same tool call for multiple steps is
          rejected. Strong-mode protection against "I marked
          all 5 done after one edit" hallucinations.

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
           it done until the validator actually passes. Use
           ``verified_by=[<tool_call_id>]`` to record the call
           that did the work; the tool REJECTS DONE transitions
           with no verifying tool call AND no explanatory finding.
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

        # === Strong verification for DONE transitions ===
        # Before mutating state, validate every DONE step: it
        # either references a real tool_call from this turn
        # (each id usable by at most one step → no
        # "one edit, five claims" gaming) OR carries a
        # non-empty finding explaining why no tool work was
        # needed. The previous plan's already-done steps are
        # exempt — we only validate NEW transitions to DONE,
        # not historical ones the model is re-asserting.
        prior_done_descriptions = {
            s.description for s in state.plan.steps
            if s.status == "done"
        }
        observed = state.observed_tool_call_ids
        claimed_by_step: dict[str, str] = {}
        for s in coerced:
            status = str(s.get("status", "todo"))
            # Coerce status with the same logic the dataclass
            # would apply, so synonyms (in_progress → doing) get
            # normalised before the check.
            if status not in VALID_STATUSES:
                status = _STATUS_SYNONYMS.get(status.lower(), "todo")
            if status != "done":
                continue
            desc = str(s.get("description", ""))
            if desc in prior_done_descriptions:
                # Already-done in the prior plan — re-asserting,
                # not transitioning. Skip verification.
                continue
            verified_by = s.get("verified_by") or []
            if not isinstance(verified_by, list):
                return (
                    f"ERROR: step {desc!r} has malformed "
                    f"verified_by — must be a list of tool_call "
                    f"id strings, got {type(verified_by).__name__}."
                )
            finding = str(s.get("finding", ""))
            if verified_by:
                # Strong path: each id must be real + unique.
                for tc_id in verified_by:
                    tc_id_s = str(tc_id)
                    if observed and tc_id_s not in observed:
                        # Only enforce real-id check when the
                        # architecture is recording calls. If
                        # observed is empty (pre-upgrade
                        # architecture) we skip this check and
                        # fall back to finding-required, below.
                        return (
                            f"ERROR: step {desc!r}: verified_by "
                            f"references unknown tool_call_id "
                            f"{tc_id_s!r}. Available ids fired "
                            f"this turn (most recent shown): "
                            f"{sorted(observed)[-10:] or '(none)'}."
                        )
                    if tc_id_s in claimed_by_step:
                        return (
                            f"ERROR: step {desc!r}: tool_call_id "
                            f"{tc_id_s!r} was already claimed by "
                            f"step {claimed_by_step[tc_id_s]!r}. "
                            "Each tool call can verify at most "
                            "one step — split the work or "
                            "transition the duplicate to 'skipped'."
                        )
                    claimed_by_step[tc_id_s] = desc
            else:
                # No tool work claimed — require a substantive
                # finding so analytical/no-tool steps are still
                # honest. 20 chars rules out empty / "ok" / "done"
                # without being so strict it blocks real one-line
                # justifications.
                if len(finding.strip()) < 20:
                    return (
                        f"ERROR: step {desc!r} is DONE with empty "
                        "verified_by — provide either (a) "
                        "verified_by=[<tool_call_id>] referencing "
                        "the call that did the work, or (b) a "
                        "finding ≥20 chars explaining why no tool "
                        "work was needed. Hallucinated completion "
                        "is the most common plan-failure mode; "
                        "this check rejects it at write time."
                    )

        state.plan = LivingPlan(
            goal=str(goal),
            steps=[
                LivingPlanStep(
                    description=str(s.get("description", "")),
                    status=str(s.get("status", "todo")),
                    finding=str(s.get("finding", "")),
                    verified_by=list(s.get("verified_by") or []),
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
        "that step's status = ``doing``. **Exactly ONE step should "
        "be ``doing`` at any time** (not zero, not two) — this "
        "lets the framework + reviewers tell at a glance what "
        "you're actually working on right now.\n"
        "3. After the action, ``plan_write`` IMMEDIATELY with "
        "status = ``done`` (and a 1-line ``finding``) or "
        "``blocked`` (with a finding describing the blocker). "
        "Don't batch completions; mark each step ``done`` the "
        "moment it's finished so the next step can move to "
        "``doing``.\n"
        "4. If discovery requires changes, just rewrite the plan. "
        "Insert / remove / reorder steps. The plan is a living "
        "document, not a frozen contract.\n"
        "5. NEVER declare a task done before the VERIFY step is "
        "actually ``done`` (not just optimistically marked so).\n"
        "6. **You may not emit a final response to the user while "
        "any step is ``doing`` or ``todo``.** The framework will "
        "detect this and re-prompt you. Mark every step "
        "``done`` / ``skipped`` / ``blocked`` first, then respond.\n"
    )
    if has_workspace_mirror:
        body += (
            "\n**Plan + research as separate artifacts** (the "
            "Anthropic multi-agent research pattern — keep them "
            "split, don't conflate):\n\n"
            "- The **plan** (``plan_write``) is your strategy / "
            "todo list. The ``finding`` field on each step is a "
            "1-line SUMMARY only.\n"
            "- The **research** (``note(kind=\"finding\", ...)``) "
            "is the actual collected information — analysis, "
            "evidence, version compatibility tables, error "
            "diagnoses, anything substantive. Multi-paragraph is "
            "fine; this is the real research log.\n\n"
            "When a step involves substantive research / analysis, "
            "**write the full findings as a ``note(kind=\"finding\", "
            "title=\"...\", content=\"...\")`` FIRST**, then put a "
            "1-line pointer in the step's ``finding`` field "
            "(e.g. ``\"see note: env-conflict-analysis\"``). The "
            "plan stays scannable; the research log stays "
            "complete; both persist for future tasks via "
            "``recall_past_plans`` (for the plan) and "
            "``search_notes`` (for the findings).\n\n"
            "Your plan persists to the shared notebook as a "
            '``kind="plan"`` note. Future runs can call '
            "``recall_past_plans(query)`` to find similar prior "
            "plans and bootstrap from what worked.\n"
        )
    return body


# ---------------------------------------------------------------------------
# Stop hook — framework's Ralph-loop default when ``living_plan=True``
# ---------------------------------------------------------------------------


def make_plan_stop_hook(*, name: str = "living_plan") -> Any:
    """Build a :class:`StopHook` that re-prompts the model when the
    plan still has incomplete steps after the architecture exits.

    This is the framework's answer to the ReAct exit bug: the model
    emits text like "Now let me scaffold the backend..." between
    delegations, ReAct treats that as final-answer and exits, the
    plan sits with steps still in ``doing``/``todo``. The hook reads
    the active plan via :func:`get_active_plan` and, if any step is
    incomplete, returns a :class:`StopHookResult` naming the
    specific step. ``Agent._loop`` injects the message as a fresh
    user turn and re-runs the architecture.

    Status taxonomy for the check:

    * ``doing`` / ``todo`` → incomplete, **trigger continuation**.
    * ``done`` / ``skipped`` → terminal-good, don't trigger.
    * ``blocked`` → terminal-bad-but-deliberate, don't trigger.
      ``blocked`` means the model explicitly said "I can't do this
      without user input." Continuing would be the wrong call —
      the human needs to unblock.

    First ``doing`` step wins (matches the TodoWriteTool rule that
    only ONE step should be ``doing`` at a time). If only ``todo``
    steps exist, the first ``todo`` is named instead — covers the
    case where the model planned but never started.

    Auto-registered by :class:`Agent.__init__` when
    ``living_plan=True``; opt out via
    ``living_plan={"auto_stop_hook": False}``.
    """
    # Local import: stop_hooks lives in the agent package, which
    # imports this module — circular at module-load time without
    # the deferred import.
    from ..agent.stop_hooks import StopHookResult

    class _LivingPlanStopHook:
        # Implements the StopHook Protocol structurally; no need
        # to subclass anything — Protocol is runtime_checkable.
        name = ""

        def __init__(self, hook_name: str) -> None:
            self.name = hook_name

        async def __call__(
            self,
            session: Any,
            deps: Any,
            *,
            iteration: int,
        ) -> StopHookResult | None:
            plan = get_active_plan()
            if plan is None or not plan.steps:
                return None
            # Prefer ``doing`` over ``todo`` — if the model marked a
            # step in-progress, that's a deliberate commitment the
            # framework should hold it to.
            doing = next(
                (
                    (i, s)
                    for i, s in enumerate(plan.steps, start=1)
                    if s.status == "doing"
                ),
                None,
            )
            if doing is not None:
                idx, step = doing
                directive = (
                    f"Your plan's step {idx} "
                    f"({step.description!r}) is still marked "
                    "`doing`. Either complete it (call "
                    "`plan_write` with status=`done`), explicitly "
                    "mark it `blocked` with a finding the user "
                    "needs to resolve, or `skipped` with a "
                    "reason. You may not finish the turn while a "
                    "step is `doing`."
                )
                return StopHookResult(
                    inject_message=directive,
                    reason=f"plan_step_{idx}_doing",
                )
            todo = next(
                (
                    (i, s)
                    for i, s in enumerate(plan.steps, start=1)
                    if s.status == "todo"
                ),
                None,
            )
            if todo is not None:
                idx, step = todo
                directive = (
                    f"Your plan's step {idx} "
                    f"({step.description!r}) is still `todo` "
                    "and you emitted a final response. Start it "
                    "now: mark it `doing` via `plan_write`, then "
                    "do the work. The plan is a contract — finish "
                    "every step before returning the final answer."
                )
                return StopHookResult(
                    inject_message=directive,
                    reason=f"plan_step_{idx}_todo",
                )
            return None

    return _LivingPlanStopHook(name)
