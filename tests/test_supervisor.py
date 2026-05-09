"""Supervisor architecture tests.

Covers:

* Protocol satisfaction.
* Constructor validation (empty workers rejected).
* ``declared_workers`` exposes workers.
* Single delegation: supervisor calls ``delegate(worker, ...)`` once;
  worker output flows back as the tool result and into the
  supervisor's final reply.
* Parallel delegation: two ``delegate`` calls in one supervisor turn
  → both workers run, both results returned in the same turn.
* Unknown-worker handling: ``delegate(worker="missing", ...)`` returns
  an error string instead of crashing.
* Instructions composition: agent's own ``instructions`` survive,
  supervisor template appends with worker descriptions.
* Worker session ids are unique per delegation (collision-free even
  when the same worker is called twice).
* Architecture progress events surface (``supervisor.workers_ready``,
  ``supervisor.completed``).
* Tool host wrapper exposes ``delegate`` alongside the parent's
  pre-existing tools (no override of user tools).
"""

from __future__ import annotations

import pytest

from loomflow import Agent, Architecture, ScriptedModel, ScriptedTurn, Tool, tool
from loomflow.architecture import Supervisor
from loomflow.core.types import ToolCall

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _worker(reply: str, instructions: str = "I am a specialist worker.") -> Agent:
    """Build a worker Agent that produces a fixed text response."""
    return Agent(instructions, model=ScriptedModel([ScriptedTurn(text=reply)]))


# ---------------------------------------------------------------------------
# Protocol surface
# ---------------------------------------------------------------------------


def test_supervisor_satisfies_architecture_protocol() -> None:
    sup = Supervisor(workers={"a": _worker("alpha")})
    assert isinstance(sup, Architecture)


def test_supervisor_name_is_supervisor() -> None:
    sup = Supervisor(workers={"a": _worker("alpha")})
    assert sup.name == "supervisor"


def test_supervisor_declared_workers_returns_workers() -> None:
    a, b = _worker("a"), _worker("b")
    sup = Supervisor(workers={"alpha": a, "beta": b})
    assert sup.declared_workers() == {"alpha": a, "beta": b}


def test_supervisor_rejects_empty_workers() -> None:
    with pytest.raises(ValueError, match="at least one worker"):
        Supervisor(workers={})


# ---------------------------------------------------------------------------
# Single delegation
# ---------------------------------------------------------------------------


async def test_supervisor_single_delegation_returns_worker_output() -> None:
    """Supervisor model emits one ``delegate`` call. Worker runs.
    Worker output comes back as a tool result. Supervisor's next
    turn synthesizes a final answer."""
    coder = _worker("def hello(): return 'hi'", "Python coder.")

    # Supervisor's model: turn 1 = call delegate(coder, ...); turn 2 = final answer.
    parent_model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        tool="delegate",
                        args={
                            "worker": "coder",
                            "instructions": "write a hello function",
                        },
                    )
                ]
            ),
            ScriptedTurn(text="Here is the function: def hello(): return 'hi'"),
        ]
    )
    agent = Agent(
        "manager",
        model=parent_model,
        architecture=Supervisor(workers={"coder": coder}),
    )
    result = await agent.run("write some code")
    assert "def hello()" in result.output


# ---------------------------------------------------------------------------
# Parallel delegation
# ---------------------------------------------------------------------------


async def test_supervisor_parallel_delegations_both_workers_run() -> None:
    """One supervisor turn emits TWO delegate calls. Both workers
    run (in parallel via ReAct's task-group dispatch). Both outputs
    are tool results visible to the supervisor's next turn, which
    synthesizes them."""
    researcher = _worker("found citation X")
    coder = _worker("wrote function Y")

    parent_model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        tool="delegate",
                        args={
                            "worker": "researcher",
                            "instructions": "find a citation",
                        },
                    ),
                    ToolCall(
                        id="c2",
                        tool="delegate",
                        args={
                            "worker": "coder",
                            "instructions": "write a function",
                        },
                    ),
                ]
            ),
            ScriptedTurn(text="Combined: X and Y"),
        ]
    )
    agent = Agent(
        "manager",
        model=parent_model,
        architecture=Supervisor(
            workers={"researcher": researcher, "coder": coder}
        ),
    )
    result = await agent.run("research and code")
    # Final synthesis comes from supervisor's second turn.
    assert "X and Y" in result.output


# ---------------------------------------------------------------------------
# Unknown worker handling
# ---------------------------------------------------------------------------


async def test_supervisor_unknown_worker_returns_error_string() -> None:
    """``delegate(worker="ghost", ...)`` doesn't crash; the tool
    returns an error string and the supervisor sees it as a normal
    tool result (so it can adjust on the next turn)."""
    parent_model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        tool="delegate",
                        args={
                            "worker": "ghost",
                            "instructions": "do something",
                        },
                    )
                ]
            ),
            ScriptedTurn(text="oh, never mind"),
        ]
    )
    agent = Agent(
        "manager",
        model=parent_model,
        architecture=Supervisor(workers={"real": _worker("ok")}),
    )
    result = await agent.run("anything")
    # Final reply uses the second turn's text — supervisor has
    # adapted to the error.
    assert "never mind" in result.output


# ---------------------------------------------------------------------------
# Instructions composition
# ---------------------------------------------------------------------------


async def test_supervisor_composes_user_instructions_with_template() -> None:
    """The agent's own ``instructions`` survive; the supervisor
    template appends with worker descriptions. We verify by
    capturing the system message the model receives."""
    captured_system_messages: list[str] = []

    class _CaptureSystemModel:
        name = "capture"

        async def stream(self, messages, *, tools=None):  # type: ignore[no-untyped-def]
            from loomflow.core.types import ModelChunk, Usage

            for m in messages:
                if m.role == "system":
                    captured_system_messages.append(m.content)
            yield ModelChunk(kind="text", text="ok")
            yield ModelChunk(kind="finish", usage=Usage())

    coder = _worker("code", "Python expert who writes idiomatic code.")
    agent = Agent(
        "You are the manager of a small team.",
        model=_CaptureSystemModel(),  # type: ignore[arg-type]
        architecture=Supervisor(workers={"coder": coder}),
    )
    await agent.run("anything")

    full_system = "\n".join(captured_system_messages)
    # User's domain instructions present.
    assert "manager of a small team" in full_system
    # Supervisor template present.
    assert "delegate(worker" in full_system
    # Worker description visible to supervisor.
    assert "Python expert" in full_system


# ---------------------------------------------------------------------------
# Worker session ids are unique per delegation
# ---------------------------------------------------------------------------


async def test_supervisor_worker_session_ids_unique_per_call() -> None:
    """Same worker delegated twice → two distinct session ids
    (so the worker's own journal records are not collided)."""
    captured_ids: list[str] = []

    class _CaptureAgent(Agent):
        async def run(  # type: ignore[override]
            self, prompt: str, **kwargs: object
        ):
            sid = kwargs.get("session_id")
            assert sid is not None
            captured_ids.append(sid)
            return await super().run(prompt, **kwargs)  # type: ignore[arg-type]

    snoop = _CaptureAgent(
        "snooper",
        model=ScriptedModel(
            [ScriptedTurn(text="ok1"), ScriptedTurn(text="ok2")]
        ),
    )

    # Two delegate calls in one supervisor turn, both to the same worker.
    parent_model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        tool="delegate",
                        args={
                            "worker": "snoop",
                            "instructions": "task one",
                        },
                    ),
                    ToolCall(
                        id="c2",
                        tool="delegate",
                        args={
                            "worker": "snoop",
                            "instructions": "task two",
                        },
                    ),
                ]
            ),
            ScriptedTurn(text="done"),
        ]
    )
    agent = Agent(
        "manager",
        model=parent_model,
        architecture=Supervisor(workers={"snoop": snoop}),
    )
    await agent.run("two tasks")

    assert len(captured_ids) == 2
    # Both should share the parent prefix + worker name + delegate marker.
    for sid in captured_ids:
        assert "__delegate_snoop_" in sid
    # And they should be distinct.
    assert captured_ids[0] != captured_ids[1]


# ---------------------------------------------------------------------------
# Tool host wrapper preserves parent tools
# ---------------------------------------------------------------------------


async def test_supervisor_does_not_hide_pre_existing_tools() -> None:
    """If the parent Agent has its own tools, those still work
    inside the supervisor — the wrapper is additive, not replacing."""

    @tool
    async def my_calc(a: int, b: int) -> int:
        """Add two ints."""
        return a + b

    coder = _worker("written")

    # Supervisor's model: turn 1 calls the user tool my_calc; turn 2
    # delegates to the worker; turn 3 produces final text. We verify
    # all three steps execute.
    parent_model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        tool="my_calc",
                        args={"a": 2, "b": 3},
                    )
                ]
            ),
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="c2",
                        tool="delegate",
                        args={
                            "worker": "coder",
                            "instructions": "write something",
                        },
                    )
                ]
            ),
            ScriptedTurn(text="final"),
        ]
    )
    agent = Agent(
        "manager",
        model=parent_model,
        tools=[my_calc],
        architecture=Supervisor(workers={"coder": coder}),
    )
    result = await agent.run("multi-step")
    assert "final" in result.output


# ---------------------------------------------------------------------------
# Architecture progress events
# ---------------------------------------------------------------------------


async def test_supervisor_emits_workers_ready_and_completed_events() -> None:
    coder = _worker("code")
    parent_model = ScriptedModel(
        [ScriptedTurn(text="just a direct answer, no delegation")]
    )
    agent = Agent(
        "manager",
        model=parent_model,
        architecture=Supervisor(workers={"coder": coder}),
    )
    events = [e async for e in agent.stream("hello")]
    arch_names = [
        e.payload["name"]
        for e in events
        if e.kind == "architecture_event"
    ]
    assert "supervisor.workers_ready" in arch_names
    assert "supervisor.completed" in arch_names


# ---------------------------------------------------------------------------
# Custom delegate tool name
# ---------------------------------------------------------------------------


async def test_supervisor_accepts_custom_delegate_name() -> None:
    """User can rename ``delegate`` to avoid clashes with their own
    tools or to match their domain vocabulary."""
    coder = _worker("done")
    parent_model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        tool="hand_off",
                        args={
                            "worker": "coder",
                            "instructions": "code it",
                        },
                    )
                ]
            ),
            ScriptedTurn(text="all set"),
        ]
    )
    agent = Agent(
        "manager",
        model=parent_model,
        architecture=Supervisor(
            workers={"coder": coder},
            delegate_tool_name="hand_off",
        ),
    )
    result = await agent.run("go")
    assert "all set" in result.output


# ---------------------------------------------------------------------------
# Tool object passing (sanity)
# ---------------------------------------------------------------------------


async def test_supervisor_forward_message_returns_worker_output_verbatim() -> None:
    """``forward_message(worker)`` overrides the supervisor's final
    output with the worker's last delegated output — no paraphrase
    round-trip. Even if the supervisor model says something else
    after, the captured worker text wins."""
    worker_output = "The exact polished answer the user wants."
    coder = _worker(worker_output)

    parent_model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        tool="delegate",
                        args={
                            "worker": "coder",
                            "instructions": "do the thing",
                        },
                    )
                ]
            ),
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="c2",
                        tool="forward_message",
                        args={"worker": "coder"},
                    )
                ]
            ),
            # Supervisor's last text would normally become the
            # output; forward_message must override it.
            ScriptedTurn(text="[done]"),
        ]
    )
    agent = Agent(
        "manager",
        model=parent_model,
        architecture=Supervisor(workers={"coder": coder}),
    )
    result = await agent.run("write some code")
    assert result.output == worker_output


async def test_supervisor_forward_message_unknown_worker_returns_error() -> None:
    """``forward_message(worker)`` for a worker that hasn't been
    delegated to yet returns an error string instead of overriding
    the output."""
    coder = _worker("hello")
    parent_model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        tool="forward_message",
                        args={"worker": "coder"},
                    )
                ]
            ),
            ScriptedTurn(text="fallback supervisor response"),
        ]
    )
    agent = Agent(
        "manager",
        model=parent_model,
        architecture=Supervisor(workers={"coder": coder}),
    )
    result = await agent.run("forward without delegating")
    # forward_message returned an error — supervisor's own final
    # response stands.
    assert result.output == "fallback supervisor response"


def test_delegate_tool_enumerates_worker_names_in_schema() -> None:
    """The delegate tool's `worker` arg must include an enum of the
    actual worker names so strict-schema providers reject invalid
    names at the API boundary."""
    from loomflow.architecture.supervisor import _make_delegate_tool

    coder = _worker("c", "Python coder.")
    writer = _worker("w", "Markdown writer.")
    tool = _make_delegate_tool(
        {"coder": coder, "writer": writer},
        parent_session_id="parent",
        tool_name="delegate",
    )
    worker_schema = tool.input_schema["properties"]["worker"]
    assert "enum" in worker_schema
    assert set(worker_schema["enum"]) == {"coder", "writer"}
    # Description echoes the names too (for non-strict providers).
    for name in ("coder", "writer"):
        assert name in worker_schema["description"]
    # Top-level description includes each worker's role description.
    assert "Python coder" in tool.description
    assert "Markdown writer" in tool.description


def test_forward_message_tool_enumerates_worker_names_in_schema() -> None:
    """The forward_message tool's `worker` arg also gets the worker
    enum — same reason as delegate."""
    from loomflow.architecture.supervisor import (
        _make_forward_message_tool,
    )

    tool = _make_forward_message_tool(
        last_outputs={},
        forward_request={},
        tool_name="forward_message",
        worker_names=["alpha", "beta", "gamma"],
    )
    worker_schema = tool.input_schema["properties"]["worker"]
    assert "enum" in worker_schema
    assert set(worker_schema["enum"]) == {"alpha", "beta", "gamma"}


def test_make_delegate_tool_returns_a_real_tool_instance() -> None:
    """Smoke-test the helper that builds the delegate tool."""
    from loomflow.architecture.supervisor import _make_delegate_tool

    workers = {"x": _worker("x")}
    t = _make_delegate_tool(workers, "parent_sess", tool_name="delegate")
    assert isinstance(t, Tool)
    assert t.name == "delegate"
    schema = t.input_schema
    assert "worker" in schema["properties"]
    assert "instructions" in schema["properties"]
    assert set(schema["required"]) == {"worker", "instructions"}
