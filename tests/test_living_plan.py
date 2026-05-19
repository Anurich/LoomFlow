"""LivingPlan tools — tool behavior, lenient coercion, workspace
mirror, multi-tenant safety, smart-default agent integration."""

from __future__ import annotations

import pytest

from loomflow import (
    Agent,
    LivingPlan,
    LivingPlanStep,
    RunContext,
    Tool,
    set_run_context,
    tool,
)
from loomflow.core.context import _ambient_living_plan_var
from loomflow.tools.plan import (
    VALID_STATUSES,
    _coerce_steps,
    _LivingPlanState,
    get_active_plan,
    make_plan_tools,
    make_recall_past_plans_tool,
    record_tool_call,
)
from loomflow.workspace import InMemoryWorkspace

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# LivingPlanStep status coercion
# ---------------------------------------------------------------------------


def test_step_canonical_status() -> None:
    """All canonical statuses survive ``__post_init__`` unchanged."""
    for s in VALID_STATUSES:
        step = LivingPlanStep(description="x", status=s)
        assert step.status == s


def test_step_status_synonyms_normalized() -> None:
    """Common model-invented synonyms collapse to canonical values."""
    cases = {
        "in_progress": "doing",
        "in-progress": "doing",
        "WIP": "doing",
        "started": "doing",
        "running": "doing",
        "complete": "done",
        "completed": "done",
        "finished": "done",
        "ok": "done",
        "failed": "blocked",
        "error": "blocked",
        "stuck": "blocked",
        "skip": "skipped",
    }
    for given, expected in cases.items():
        assert LivingPlanStep(description="x", status=given).status == expected


def test_step_unknown_status_falls_back_to_todo() -> None:
    """Defensive: unknown statuses become ``todo`` (active),
    not silently rejected. The agent gets a working step."""
    step = LivingPlanStep(description="x", status="frobnicated")
    assert step.status == "todo"


# ---------------------------------------------------------------------------
# LivingPlan rendering
# ---------------------------------------------------------------------------


def test_empty_plan_renders_hint() -> None:
    out = LivingPlan().render()
    assert "no plan yet" in out


def test_plan_renders_table() -> None:
    plan = LivingPlan(
        goal="Test goal",
        steps=[
            LivingPlanStep(description="step a", status="done"),
            LivingPlanStep(description="step b", status="doing"),
        ],
    )
    rendered = plan.render()
    assert "Test goal" in rendered
    assert "step a" in rendered
    assert "step b" in rendered
    assert "1/2 done" in rendered


# ---------------------------------------------------------------------------
# _coerce_steps — the four shapes
# ---------------------------------------------------------------------------


def test_coerce_native_list() -> None:
    out = _coerce_steps([{"description": "a"}, {"description": "b"}])
    assert isinstance(out, list)
    assert len(out) == 2


def test_coerce_json_string_of_list() -> None:
    out = _coerce_steps('[{"description": "a"}, {"description": "b"}]')
    assert isinstance(out, list)
    assert len(out) == 2


def test_coerce_json_object_with_steps_key() -> None:
    """Models sometimes wrap the list in ``{"steps": [...]}`` —
    unwrap that shape transparently."""
    out = _coerce_steps('{"steps": [{"description": "a"}]}')
    assert isinstance(out, list)
    assert len(out) == 1


def test_coerce_numbered_text() -> None:
    out = _coerce_steps("1. first step\n2. second step\n- third step")
    assert isinstance(out, list)
    assert len(out) == 3
    assert out[0]["description"] == "first step"
    assert out[2]["description"] == "third step"


def test_coerce_invalid_type_returns_error_message() -> None:
    out = _coerce_steps(42)  # not list / str
    assert isinstance(out, str)
    assert "must be a list" in out


# ---------------------------------------------------------------------------
# _coerce_steps — weak-model shapes (gpt-4.1-mini etc. are loose)
# ---------------------------------------------------------------------------


def test_coerce_list_of_plain_strings() -> None:
    """gpt-4.1-mini emits ``steps`` as a list of bare description
    strings. Each must become a ``todo`` step — the old coercion
    filtered non-dicts out and produced an empty plan."""
    out = _coerce_steps(["add sub() to calc.py", "wire it into compute()"])
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0]["description"] == "add sub() to calc.py"
    assert out[0]["status"] == "todo"


def test_coerce_bare_dict_single_step() -> None:
    """A single step dict the model forgot to wrap in a list is
    wrapped, not rejected."""
    out = _coerce_steps({"description": "do the thing", "status": "doing"})
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0]["description"] == "do the thing"


def test_coerce_list_of_stringified_dicts() -> None:
    """Some models double-encode: a list whose elements are each a
    JSON-string of a step dict."""
    out = _coerce_steps(
        ['{"description": "a", "status": "done"}', '{"description": "b"}']
    )
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0]["status"] == "done"


def test_coerce_json_string_of_list_of_strings() -> None:
    """JSON-string wrapping a list of plain strings — combines the
    string-parse path with the list-of-strings path."""
    out = _coerce_steps('["first", "second"]')
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[1]["description"] == "second"


def test_coerce_list_with_items_but_none_usable_errors() -> None:
    """A non-empty list that yields nothing salvageable is an
    actionable error, not a silent empty plan."""
    out = _coerce_steps([1, 2, None])
    assert isinstance(out, str)
    assert "none were usable" in out


# ---------------------------------------------------------------------------
# plan_write / plan_read via direct .execute (ambient state set
# manually via set_run_context + the contextvar)
# ---------------------------------------------------------------------------


async def _call_tool(tool_obj: Tool, args: dict) -> str:
    """Invoke a Tool and return the output string.

    :meth:`Tool.execute` returns the raw function value (``Any``),
    not a wrapped ``ToolResult`` — wrappers are added by the agent
    loop, not the tool itself.
    """
    return str(await tool_obj.execute(args))


async def _with_plan_state() -> _LivingPlanState:
    """Install a fresh plan state on the ambient contextvar and
    return it. Tests that want to inspect post-call state read this
    object back."""
    state = _LivingPlanState()
    _ambient_living_plan_var.set(state)
    return state


async def test_plan_write_creates_plan() -> None:
    state = await _with_plan_state()
    [plan_write, _] = make_plan_tools()
    out = await _call_tool(
        plan_write,
        {
            "goal": "Test goal",
            "steps": [{"description": "step a", "status": "todo"}],
        },
    )
    assert "Test goal" in out
    assert state.plan.goal == "Test goal"
    assert len(state.plan.steps) == 1
    assert state.plan.steps[0].description == "step a"


async def test_plan_write_atomic_rewrite() -> None:
    """Second ``plan_write`` REPLACES prior steps — TodoWrite-style
    atomic full-list rewrite. No merging."""
    state = await _with_plan_state()
    [plan_write, _] = make_plan_tools()
    await _call_tool(
        plan_write,
        {"goal": "g", "steps": [{"description": "a"}, {"description": "b"}]},
    )
    assert len(state.plan.steps) == 2
    await _call_tool(
        plan_write, {"goal": "g", "steps": [{"description": "c"}]}
    )
    assert len(state.plan.steps) == 1
    assert state.plan.steps[0].description == "c"


async def test_plan_read_returns_current() -> None:
    await _with_plan_state()
    [plan_write, plan_read] = make_plan_tools()
    await _call_tool(
        plan_write, {"goal": "g", "steps": [{"description": "a"}]}
    )
    read_out = await _call_tool(plan_read, {})
    assert "g" in read_out
    assert "a" in read_out


async def test_plan_tools_without_state_return_helpful_error() -> None:
    """Calling plan_write outside an active run (no contextvar
    state) must NOT raise — it returns an actionable error string
    so the model sees what to do."""
    _ambient_living_plan_var.set(None)
    [plan_write, plan_read] = make_plan_tools()
    write_result = await plan_write.execute(
        {"goal": "g", "steps": [{"description": "a"}]}
    )
    assert "not enabled" in str(write_result).lower()
    read_result = await plan_read.execute({})
    assert "not enabled" in str(read_result).lower()


async def test_plan_write_coerces_string_steps() -> None:
    """Real-world bug: Anthropic sometimes serializes the list as
    a JSON string. The tool must accept it."""
    state = await _with_plan_state()
    [plan_write, _] = make_plan_tools()
    await _call_tool(
        plan_write,
        {
            "goal": "g",
            "steps": '[{"description": "a", "status": "todo"}]',
        },
    )
    assert len(state.plan.steps) == 1
    assert state.plan.steps[0].description == "a"


# ---------------------------------------------------------------------------
# Workspace mirror
# ---------------------------------------------------------------------------


async def test_plan_write_mirrors_to_workspace() -> None:
    """When a workspace is provided, ``plan_write`` writes a
    ``kind="plan"`` note. Subsequent writes update the SAME note
    via the captured slug."""
    await _with_plan_state()
    ws = InMemoryWorkspace()
    [plan_write, _] = make_plan_tools(
        workspace=ws, task_id="t-001", author="agent"
    )
    async with set_run_context(RunContext(user_id="alice", run_id="r1")):
        await _call_tool(
            plan_write,
            {"goal": "g1", "steps": [{"description": "a"}]},
        )
        # First write should create a note.
        notes = await ws.list_notes(user_id="alice")
        assert len(notes) == 1
        assert notes[0].kind == "plan"
        first_slug = notes[0].slug

        await _call_tool(
            plan_write,
            {"goal": "g1", "steps": [{"description": "a", "status": "done"}]},
        )
        # Second write should UPDATE the same note, not create a new one.
        notes = await ws.list_notes(user_id="alice")
        assert len(notes) == 1
        assert notes[0].slug == first_slug


async def test_plan_mirror_falls_back_to_ambient_workspace() -> None:
    """When the plan tools were built without an explicit workspace
    but a parent :class:`Workflow` set ``_ambient_workspace_var``,
    the mirror should target the ambient workspace. This makes
    ``Workflow(workspace=ws)`` + nested ``Agent(living_plan=True)``
    (no explicit workspace) work symmetrically with how the
    workspace tools themselves inherit the ambient."""
    from loomflow.core.context import _ambient_workspace_var
    await _with_plan_state()
    ambient_ws = InMemoryWorkspace()
    # NO workspace passed to make_plan_tools — relies on the ambient.
    [plan_write, _] = make_plan_tools(author="agent")
    token = _ambient_workspace_var.set(ambient_ws)
    try:
        async with set_run_context(RunContext(user_id="alice", run_id="r1")):
            await _call_tool(
                plan_write,
                {"goal": "ambient-test", "steps": [{"description": "a"}]},
            )
        notes = await ambient_ws.list_notes(user_id="alice")
        assert len(notes) == 1
        assert notes[0].kind == "plan"
    finally:
        _ambient_workspace_var.reset(token)


async def test_plan_mirror_failure_does_not_break_tool() -> None:
    """If the workspace mirror raises, the plan tool must still
    return successfully — the in-memory plan is the source of
    truth, mirroring is best-effort."""

    class _BrokenWorkspace:
        async def write_note(self, **kwargs: object) -> object:
            raise RuntimeError("disk full")

    state = await _with_plan_state()
    [plan_write, _] = make_plan_tools(
        workspace=_BrokenWorkspace(), author="agent"
    )
    out = await _call_tool(
        plan_write,
        {"goal": "g", "steps": [{"description": "a"}]},
    )
    # In-memory plan still updated.
    assert state.plan.goal == "g"
    # And the tool still returned the rendered plan.
    assert "g" in out


# ---------------------------------------------------------------------------
# Multi-tenant: user_id partition
# ---------------------------------------------------------------------------


async def test_user_id_partitions_workspace_mirror() -> None:
    """Plans written under user A are invisible to user B's
    workspace queries — the standard loomflow multi-tenant
    partition rules apply via :func:`get_run_context`."""
    await _with_plan_state()
    ws = InMemoryWorkspace()
    [plan_write, _] = make_plan_tools(workspace=ws, author="agent")

    async with set_run_context(RunContext(user_id="alice", run_id="r1")):
        await _call_tool(
            plan_write, {"goal": "alice-goal", "steps": [{"description": "a"}]}
        )

    # Bob sees nothing.
    bob_notes = await ws.list_notes(user_id="bob")
    assert bob_notes == []
    # Alice sees her plan.
    alice_notes = await ws.list_notes(user_id="alice")
    assert len(alice_notes) == 1


# ---------------------------------------------------------------------------
# recall_past_plans
# ---------------------------------------------------------------------------


async def test_recall_past_plans_filters_to_kind_plan() -> None:
    """``recall_past_plans`` returns only ``kind="plan"`` notes,
    not arbitrary findings / decisions that happened to match the
    query string."""
    ws = InMemoryWorkspace()
    async with set_run_context(RunContext(user_id="alice", run_id="r0")):
        await ws.write_note(
            author="agent", title="Plan: do X", body="step 1: X", kind="plan",
            user_id="alice",
        )
        await ws.write_note(
            author="agent", title="Random finding", body="X happened here",
            kind="finding", user_id="alice",
        )

    recall = make_recall_past_plans_tool(ws)
    async with set_run_context(RunContext(user_id="alice", run_id="r1")):
        out = str(await recall.execute({"query": "X"}))
    assert "do X" in out
    assert "Random finding" not in out


async def test_recall_past_plans_empty_returns_friendly_message() -> None:
    ws = InMemoryWorkspace()
    recall = make_recall_past_plans_tool(ws)
    async with set_run_context(RunContext(user_id="alice", run_id="r1")):
        out = str(await recall.execute({"query": "anything"}))
    assert "No past plans match" in out


# ---------------------------------------------------------------------------
# get_active_plan helper
# ---------------------------------------------------------------------------


async def test_get_active_plan_returns_state() -> None:
    state = await _with_plan_state()
    state.plan.goal = "x"
    plan = get_active_plan()
    assert plan is state.plan


async def test_get_active_plan_returns_none_outside_run() -> None:
    _ambient_living_plan_var.set(None)
    assert get_active_plan() is None


# ---------------------------------------------------------------------------
# Agent integration — smart default semantics
# ---------------------------------------------------------------------------


@tool
def _stub_tool() -> str:
    """Trivial tool used to make the agent count as "tool-using"."""
    return "ok"


def test_agent_default_disabled() -> None:
    """v0.10.0 default for ``living_plan=`` is OPT-IN (False).
    Agents do NOT get plan tools unless they ask for them."""
    a = Agent("t", model="echo", tools=[_stub_tool])
    assert a._living_plan_spec.enabled is False
    names = [t.name for t in a._tool_host._tools.values()]
    assert "plan_write" not in names
    assert "plan_read" not in names


def test_agent_living_plan_true_wires_tools() -> None:
    a = Agent("t", model="echo", tools=[_stub_tool], living_plan=True)
    names = [t.name for t in a._tool_host._tools.values()]
    assert "plan_write" in names
    assert "plan_read" in names


def test_agent_living_plan_false_skips_wiring() -> None:
    a = Agent("t", model="echo", tools=[_stub_tool], living_plan=False)
    assert a._living_plan_spec.enabled is False
    names = [t.name for t in a._tool_host._tools.values()]
    assert "plan_write" not in names


def test_agent_living_plan_with_workspace_includes_recall() -> None:
    ws = InMemoryWorkspace()
    a = Agent(
        "t", model="echo", tools=[_stub_tool], workspace=ws, living_plan=True
    )
    names = [t.name for t in a._tool_host._tools.values()]
    assert "plan_write" in names
    assert "recall_past_plans" in names


def test_agent_living_plan_pre_seeded() -> None:
    seed = LivingPlan(goal="pre-seeded", steps=[])
    a = Agent("t", model="echo", tools=[_stub_tool], living_plan=seed)
    assert a._living_plan_spec.enabled is True
    assert a._living_plan_spec.seed_plan is seed


def test_agent_living_plan_appends_prompt_section() -> None:
    a = Agent("t", model="echo", tools=[_stub_tool], living_plan=True)
    assert "Living plan" in a._instructions
    assert "plan_write" in a._instructions


# ---------------------------------------------------------------------------
# Strong verification on DONE transitions (0.10.19+)
# ---------------------------------------------------------------------------
#
# The contract: a step transitioning to DONE must either reference
# a real tool_call_id from this turn via ``verified_by``, OR carry
# a substantive ``finding`` (≥20 chars) for analytical steps that
# don't involve tools. Each tool_call_id may verify AT MOST ONE
# step. Soft-cutover: when ``record_tool_call`` has never been
# called for this run (architectures not yet upgraded), the
# verified_by real-id check is skipped and only the finding
# fallback applies.


async def test_done_with_empty_verified_by_and_no_finding_rejected() -> None:
    """The headline bug: marking a step DONE with neither tool
    call evidence nor a finding is plan-theater. Reject it."""
    await _with_plan_state()
    [plan_write, _] = make_plan_tools()
    out = await _call_tool(
        plan_write,
        {
            "goal": "g",
            "steps": [{"description": "fix it", "status": "done"}],
        },
    )
    assert out.startswith("ERROR"), (
        f"expected ERROR, got: {out!r}"
    )
    assert "verified_by" in out
    assert "hallucinated completion" in out.lower()


async def test_done_with_substantive_finding_accepted() -> None:
    """Finding ≥20 chars is the analytical-step path — accepted
    even with empty verified_by because the step had no tool
    work to point at."""
    state = await _with_plan_state()
    [plan_write, _] = make_plan_tools()
    out = await _call_tool(
        plan_write,
        {
            "goal": "g",
            "steps": [
                {
                    "description": "analyze the user's intent",
                    "status": "done",
                    "finding": (
                        "user wants a JSON output not YAML — "
                        "no tool call needed for this conclusion"
                    ),
                }
            ],
        },
    )
    assert not out.startswith("ERROR")
    assert state.plan.steps[0].status == "done"


async def test_done_with_short_finding_rejected() -> None:
    """A 1-word finding like 'done' or 'ok' is exactly what we
    want to reject — too vague to be honest verification."""
    await _with_plan_state()
    [plan_write, _] = make_plan_tools()
    out = await _call_tool(
        plan_write,
        {
            "goal": "g",
            "steps": [
                {
                    "description": "do it",
                    "status": "done",
                    "finding": "ok",
                }
            ],
        },
    )
    assert out.startswith("ERROR")
    assert "20 chars" in out or "≥20" in out


async def test_done_with_verified_by_real_call_id_accepted() -> None:
    """The strong path: model references a real tool_call_id that
    was recorded this turn → DONE accepted, the id is now claimed."""
    state = await _with_plan_state()
    record_tool_call("call_abc123")
    [plan_write, _] = make_plan_tools()
    out = await _call_tool(
        plan_write,
        {
            "goal": "g",
            "steps": [
                {
                    "description": "edit foo.py",
                    "status": "done",
                    "verified_by": ["call_abc123"],
                }
            ],
        },
    )
    assert not out.startswith("ERROR"), out
    assert state.plan.steps[0].verified_by == ["call_abc123"]


async def test_done_with_verified_by_unknown_id_rejected() -> None:
    """Strong path with a hallucinated id: model claims work
    happened via a tool_call_id that doesn't exist in this turn's
    journal. Reject + tell the model what's actually available."""
    await _with_plan_state()
    record_tool_call("call_real")
    [plan_write, _] = make_plan_tools()
    out = await _call_tool(
        plan_write,
        {
            "goal": "g",
            "steps": [
                {
                    "description": "edit",
                    "status": "done",
                    "verified_by": ["call_fake"],
                }
            ],
        },
    )
    assert out.startswith("ERROR")
    assert "call_fake" in out
    assert "call_real" in out  # helpful "available ids" listing


async def test_one_call_cannot_verify_two_steps() -> None:
    """The 'one edit, five claims' bug: model marks N steps
    DONE all referencing the same tool call. Each call_id can
    only verify ONE step — others must split the work or
    skip honestly."""
    await _with_plan_state()
    record_tool_call("call_one")
    [plan_write, _] = make_plan_tools()
    out = await _call_tool(
        plan_write,
        {
            "goal": "g",
            "steps": [
                {
                    "description": "step a",
                    "status": "done",
                    "verified_by": ["call_one"],
                },
                {
                    "description": "step b",
                    "status": "done",
                    "verified_by": ["call_one"],
                },
            ],
        },
    )
    assert out.startswith("ERROR")
    assert "already claimed" in out
    assert "call_one" in out


async def test_re_asserting_prior_done_is_not_re_verified() -> None:
    """When the model re-submits the plan with a step that was
    ALREADY done in the prior version, don't re-validate that
    step — it's a re-assertion, not a transition. (Otherwise
    every subsequent plan_write would need to re-cite the
    original tool calls, which doesn't match how models
    incrementally update.)"""
    state = await _with_plan_state()
    record_tool_call("call_first")
    [plan_write, _] = make_plan_tools()
    # Round 1: mark step done with a real call.
    out1 = await _call_tool(
        plan_write,
        {
            "goal": "g",
            "steps": [
                {
                    "description": "first step",
                    "status": "done",
                    "verified_by": ["call_first"],
                },
            ],
        },
    )
    assert not out1.startswith("ERROR")
    assert state.plan.steps[0].status == "done"
    # Round 2: re-submit WITHOUT verified_by, just re-asserting
    # the step is done. Should be accepted — not a new
    # transition.
    out2 = await _call_tool(
        plan_write,
        {
            "goal": "g",
            "steps": [
                {"description": "first step", "status": "done"},
                {
                    "description": "second step",
                    "status": "todo",
                },
            ],
        },
    )
    assert not out2.startswith("ERROR"), out2


async def test_soft_cutover_no_recorded_calls_requires_finding() -> None:
    """Backwards compat: when the architecture hasn't been
    updated to call ``record_tool_call`` (observed set empty),
    the real-id check is skipped — but the finding fallback
    still fires. So pre-upgrade architectures get the medium-
    strength verification instead of nothing."""
    await _with_plan_state()
    # NOTE: deliberately NOT calling record_tool_call → observed set empty.
    [plan_write, _] = make_plan_tools()
    out = await _call_tool(
        plan_write,
        {
            "goal": "g",
            "steps": [{"description": "do it", "status": "done"}],
        },
    )
    # Without finding → still rejected.
    assert out.startswith("ERROR")
    # With sufficient finding → accepted even though no calls recorded.
    out2 = await _call_tool(
        plan_write,
        {
            "goal": "g",
            "steps": [
                {
                    "description": "do it",
                    "status": "done",
                    "finding": (
                        "soft-cutover path: pre-upgrade arch with "
                        "no recorded calls, finding sufficient"
                    ),
                }
            ],
        },
    )
    assert not out2.startswith("ERROR")


async def test_skipped_and_blocked_not_verified() -> None:
    """``skipped`` / ``blocked`` are HONEST non-completion
    statuses — they shouldn't trigger the verification check.
    Only DONE transitions need evidence."""
    state = await _with_plan_state()
    [plan_write, _] = make_plan_tools()
    out = await _call_tool(
        plan_write,
        {
            "goal": "g",
            "steps": [
                {"description": "a", "status": "skipped"},
                {"description": "b", "status": "blocked"},
            ],
        },
    )
    assert not out.startswith("ERROR")
    assert state.plan.steps[0].status == "skipped"
    assert state.plan.steps[1].status == "blocked"


async def test_record_tool_call_noop_without_living_plan() -> None:
    """``record_tool_call`` is called from architectures
    unconditionally; when living_plan isn't enabled the
    contextvar is unset → should be a no-op, not a crash."""
    # Explicitly install None so the contextvar is unset.
    _ambient_living_plan_var.set(None)
    # Should not raise.
    record_tool_call("call_xyz")
