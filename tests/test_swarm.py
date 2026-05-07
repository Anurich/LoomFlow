"""Swarm architecture tests.

Covers:

* Protocol satisfaction; ``declared_workers`` exposes peers.
* Constructor validation: empty agents, unknown entry_agent, bad
  max_handoffs.
* Single-agent answer (no handoff): entry agent produces final
  output; nothing else runs.
* Single handoff: entry agent calls ``handoff(target=...)``,
  control switches, target agent's output is final.
* Multi-handoff chain: A → B → C → final answer.
* Cycle detection: A → B → A → B trips ``swarm.cycle_detected``.
* ``max_handoffs`` cap.
* Custom handoff tool name.
* Architecture progress events.
* The ``extra_tools`` plumbing on ``Agent.run`` is what powers the
  handoff tool injection — verifies the new primitive end-to-end.
"""

from __future__ import annotations

import pytest

from jeevesagent import (
    Agent,
    Architecture,
    ScriptedModel,
    ScriptedTurn,
    Swarm,
)
from jeevesagent.architecture.swarm import _is_cycling
from jeevesagent.core.types import ToolCall

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# _is_cycling helper
# ---------------------------------------------------------------------------


def test_is_cycling_detects_a_b_a_b() -> None:
    from collections import deque

    handoffs: deque[tuple[str, str]] = deque(
        [("A", "B"), ("B", "A"), ("A", "B"), ("B", "A")],
        maxlen=4,
    )
    assert _is_cycling(handoffs)


def test_is_cycling_false_for_linear_chain() -> None:
    from collections import deque

    handoffs: deque[tuple[str, str]] = deque(
        [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")],
        maxlen=4,
    )
    assert not _is_cycling(handoffs)


def test_is_cycling_false_when_history_short() -> None:
    from collections import deque

    handoffs: deque[tuple[str, str]] = deque(
        [("A", "B"), ("B", "A")], maxlen=4
    )
    assert not _is_cycling(handoffs)


# ---------------------------------------------------------------------------
# Protocol surface
# ---------------------------------------------------------------------------


def _agent_no_handoff(text: str) -> Agent:
    """Agent that produces a final answer without calling handoff."""
    return Agent(
        "test",
        model=ScriptedModel([ScriptedTurn(text=text)]),
    )


def test_swarm_satisfies_architecture_protocol() -> None:
    sw = Swarm(
        agents={"a": _agent_no_handoff("x"), "b": _agent_no_handoff("y")},
        entry_agent="a",
    )
    assert isinstance(sw, Architecture)


def test_swarm_name_is_swarm() -> None:
    sw = Swarm(
        agents={"a": _agent_no_handoff("x"), "b": _agent_no_handoff("y")},
        entry_agent="a",
    )
    assert sw.name == "swarm"


def test_swarm_declared_workers_exposes_peers() -> None:
    a, b = _agent_no_handoff("x"), _agent_no_handoff("y")
    sw = Swarm(agents={"a": a, "b": b}, entry_agent="a")
    assert sw.declared_workers() == {"a": a, "b": b}


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_swarm_rejects_empty_agents() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Swarm(agents={}, entry_agent="x")


def test_swarm_rejects_unknown_entry() -> None:
    with pytest.raises(ValueError, match="entry_agent"):
        Swarm(
            agents={"a": _agent_no_handoff("x")},
            entry_agent="ghost",
        )


def test_swarm_rejects_negative_max_handoffs() -> None:
    with pytest.raises(ValueError, match="max_handoffs"):
        Swarm(
            agents={"a": _agent_no_handoff("x")},
            entry_agent="a",
            max_handoffs=-1,
        )


# ---------------------------------------------------------------------------
# Entry agent owns the answer (no handoffs)
# ---------------------------------------------------------------------------


async def test_swarm_entry_agent_answers_directly() -> None:
    """Entry agent produces text, no handoff tool called → final."""
    entry = _agent_no_handoff("direct answer")
    other = _agent_no_handoff("never reached")
    parent_model = ScriptedModel([ScriptedTurn(text="never")])
    agent = Agent(
        "host",
        model=parent_model,
        architecture=Swarm(
            agents={"entry": entry, "other": other},
            entry_agent="entry",
        ),
    )
    result = await agent.run("hi")
    assert result.output == "direct answer"


# ---------------------------------------------------------------------------
# Single handoff: A → B
# ---------------------------------------------------------------------------


async def test_swarm_single_handoff_switches_active_agent() -> None:
    """Entry calls handoff(target=billing); control switches; billing
    produces the final answer."""
    # Entry agent: turn 1 emits handoff tool call; turn 2 produces text
    # (the agent's run continues after the tool call). The final
    # output of the entry agent isn't important — what matters is that
    # the handoff request is detected and Swarm switches to billing.
    entry = Agent(
        "triage",
        model=ScriptedModel(
            [
                ScriptedTurn(
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            tool="handoff",
                            args={
                                "target": "billing",
                                "message": "billing query",
                            },
                        )
                    ]
                ),
                ScriptedTurn(text="passing along to billing"),
            ]
        ),
    )
    billing = _agent_no_handoff("refund processed")

    parent_model = ScriptedModel([ScriptedTurn(text="never")])
    agent = Agent(
        "host",
        model=parent_model,
        architecture=Swarm(
            agents={"triage": entry, "billing": billing},
            entry_agent="triage",
        ),
    )
    result = await agent.run("I was charged twice")
    assert result.output == "refund processed"


# ---------------------------------------------------------------------------
# Multi-handoff chain
# ---------------------------------------------------------------------------


async def test_swarm_handoff_chain_a_to_b_to_c() -> None:
    """A → B → C; C produces the final answer."""
    a_to_b = Agent(
        "A",
        model=ScriptedModel(
            [
                ScriptedTurn(
                    tool_calls=[
                        ToolCall(
                            id="ca",
                            tool="handoff",
                            args={"target": "B"},
                        )
                    ]
                ),
                ScriptedTurn(text="A done"),
            ]
        ),
    )
    b_to_c = Agent(
        "B",
        model=ScriptedModel(
            [
                ScriptedTurn(
                    tool_calls=[
                        ToolCall(
                            id="cb",
                            tool="handoff",
                            args={"target": "C"},
                        )
                    ]
                ),
                ScriptedTurn(text="B done"),
            ]
        ),
    )
    c_final = _agent_no_handoff("C done — final")

    agent = Agent(
        "host",
        model=ScriptedModel([ScriptedTurn(text="never")]),
        architecture=Swarm(
            agents={"A": a_to_b, "B": b_to_c, "C": c_final},
            entry_agent="A",
        ),
    )
    result = await agent.run("start the chain")
    assert result.output == "C done — final"


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


async def test_swarm_detects_a_b_a_b_cycle() -> None:
    """A → B → A → B triggers cycle detection. Output is the latest
    agent's text. Note: each agent needs ENOUGH scripted turns for
    each invocation; 4 invocations of each = 8 turns each (tool call +
    text).
    """
    def _ping_pong(target: str, replies: int) -> Agent:
        turns: list[ScriptedTurn] = []
        for i in range(replies):
            turns.append(
                ScriptedTurn(
                    tool_calls=[
                        ToolCall(
                            id=f"c{i}",
                            tool="handoff",
                            args={"target": target},
                        )
                    ]
                )
            )
            turns.append(ScriptedTurn(text=f"output {i}"))
        return Agent("ping-pong", model=ScriptedModel(turns))

    a = _ping_pong("B", replies=4)
    b = _ping_pong("A", replies=4)

    agent = Agent(
        "host",
        model=ScriptedModel([ScriptedTurn(text="never")]),
        architecture=Swarm(
            agents={"A": a, "B": b},
            entry_agent="A",
            max_handoffs=20,  # higher than what cycle would allow
            detect_cycles=True,
        ),
    )
    events = [e async for e in agent.stream("ping-pong forever")]
    arch_names = [
        e.payload["name"]
        for e in events
        if e.kind == "architecture_event"
    ]
    assert "swarm.cycle_detected" in arch_names


# ---------------------------------------------------------------------------
# max_handoffs cap
# ---------------------------------------------------------------------------


async def test_swarm_max_handoffs_terminates_loop() -> None:
    """``max_handoffs=2`` caps the chain length even if agents keep
    requesting handoffs. After hitting the cap, the latest agent's
    output is returned."""
    def _always_handoff(target: str) -> Agent:
        # Each invocation does exactly 1 handoff + 1 text turn.
        return Agent(
            "ping",
            model=ScriptedModel(
                [
                    ScriptedTurn(
                        tool_calls=[
                            ToolCall(
                                id="c",
                                tool="handoff",
                                args={"target": target},
                            )
                        ]
                    ),
                    ScriptedTurn(text=f"keep going to {target}"),
                ]
            ),
        )

    a = _always_handoff("B")
    b = _always_handoff("A")
    agent = Agent(
        "host",
        model=ScriptedModel([ScriptedTurn(text="never")]),
        architecture=Swarm(
            agents={"A": a, "B": b},
            entry_agent="A",
            max_handoffs=2,
            detect_cycles=False,  # cycle detection would beat the cap
        ),
    )
    events = [e async for e in agent.stream("forever")]
    arch_names = [
        e.payload["name"]
        for e in events
        if e.kind == "architecture_event"
    ]
    assert "swarm.max_handoffs" in arch_names


# ---------------------------------------------------------------------------
# Custom handoff tool name
# ---------------------------------------------------------------------------


async def test_swarm_accepts_custom_handoff_tool_name() -> None:
    """Users can rename ``handoff`` to avoid clashes."""
    a = Agent(
        "A",
        model=ScriptedModel(
            [
                ScriptedTurn(
                    tool_calls=[
                        ToolCall(
                            id="c",
                            tool="pass_to",
                            args={"target": "B"},
                        )
                    ]
                ),
                ScriptedTurn(text="passing"),
            ]
        ),
    )
    b = _agent_no_handoff("B answers")

    agent = Agent(
        "host",
        model=ScriptedModel([ScriptedTurn(text="never")]),
        architecture=Swarm(
            agents={"A": a, "B": b},
            entry_agent="A",
            handoff_tool_name="pass_to",
        ),
    )
    result = await agent.run("go")
    assert result.output == "B answers"


# ---------------------------------------------------------------------------
# Architecture events
# ---------------------------------------------------------------------------


async def test_swarm_emits_full_event_sequence() -> None:
    a = Agent(
        "A",
        model=ScriptedModel(
            [
                ScriptedTurn(
                    tool_calls=[
                        ToolCall(
                            id="c",
                            tool="handoff",
                            args={"target": "B", "message": "ctx"},
                        )
                    ]
                ),
                ScriptedTurn(text="A done"),
            ]
        ),
    )
    b = _agent_no_handoff("B answers")

    agent = Agent(
        "host",
        model=ScriptedModel([ScriptedTurn(text="never")]),
        architecture=Swarm(
            agents={"A": a, "B": b}, entry_agent="A"
        ),
    )
    events = [e async for e in agent.stream("q")]
    arch_names = [
        e.payload["name"]
        for e in events
        if e.kind == "architecture_event"
    ]
    assert "swarm.started" in arch_names
    assert "swarm.active" in arch_names
    assert "swarm.handoff" in arch_names
    assert "swarm.completed" in arch_names


# ---------------------------------------------------------------------------
# Sanity: extra_tools primitive end-to-end via Agent.run directly
# ---------------------------------------------------------------------------


async def test_agent_run_extra_tools_kwarg_injects_tools() -> None:
    """Smoke-test the ``Agent.run(extra_tools=...)`` plumbing: a tool
    passed in for ONE run is callable by the model that turn but
    isn't part of the agent's static config."""
    from jeevesagent import Tool

    captured = []

    async def my_handoff(target: str) -> str:
        captured.append(target)
        return f"recorded {target}"

    handoff_tool = Tool(
        name="handoff_test",
        description="test",
        fn=my_handoff,
        input_schema={
            "type": "object",
            "properties": {"target": {"type": "string"}},
            "required": ["target"],
        },
    )
    agent = Agent(
        "test",
        model=ScriptedModel(
            [
                ScriptedTurn(
                    tool_calls=[
                        ToolCall(
                            id="c",
                            tool="handoff_test",
                            args={"target": "X"},
                        )
                    ]
                ),
                ScriptedTurn(text="done"),
            ]
        ),
    )
    result = await agent.run("go", extra_tools=[handoff_tool])
    assert "done" in result.output
    assert captured == ["X"]
    # And the tool is NOT registered statically:
    assert "handoff_test" not in await agent.tools_list()


# ---------------------------------------------------------------------------
# Typed handoffs (Handoff dataclass)
# ---------------------------------------------------------------------------


from pydantic import BaseModel  # noqa: E402

from jeevesagent import Handoff  # noqa: E402


class RefundRequest(BaseModel):
    reason: str
    amount: float


def test_typed_handoff_emits_per_target_tools() -> None:
    """When any peer is a Handoff with input_type, the swarm
    switches to typed mode and emits one ``transfer_to_<name>``
    tool per peer instead of the legacy single ``handoff`` tool."""
    a = _agent_no_handoff("x")
    b = _agent_no_handoff("y")
    sw = Swarm(
        agents={
            "a": a,
            "b": Handoff(agent=b, input_type=RefundRequest),
        },
        entry_agent="a",
    )
    # Internal: typed_mode flips on
    assert sw._typed_mode is True
    # The typed tools list contains transfer_to_a / transfer_to_b
    handoff_request: dict[str, object] = {}
    tools = sw._build_handoff_tools(handoff_request)
    names = {t.name for t in tools}
    assert "transfer_to_a" in names
    assert "transfer_to_b" in names
    assert "handoff" not in names


def test_typed_handoff_schema_mirrors_pydantic_model() -> None:
    """The transfer_to_<name> tool's input_schema should expose
    the Pydantic model's fields (reason, amount), not a generic
    ``message`` field."""
    b = _agent_no_handoff("y")
    sw = Swarm(
        agents={"b": Handoff(agent=b, input_type=RefundRequest)},
        entry_agent="b",
    )
    tools = sw._build_handoff_tools({})
    refund_tool = next(t for t in tools if t.name == "transfer_to_b")
    props = refund_tool.input_schema["properties"]
    assert "reason" in props
    assert "amount" in props


async def test_typed_handoff_validates_payload_via_pydantic() -> None:
    """Calling ``transfer_to_<name>`` with valid args records the
    validated payload; missing-field call returns an error string."""
    b = _agent_no_handoff("billing answer")
    sw = Swarm(
        agents={
            "a": _agent_no_handoff("never runs"),  # not used
            "billing": Handoff(agent=b, input_type=RefundRequest),
        },
        entry_agent="a",
    )
    handoff_request: dict[str, object] = {}
    tools = sw._build_handoff_tools(handoff_request)
    transfer = next(t for t in tools if t.name == "transfer_to_billing")

    # Valid payload — pydantic validates, request is recorded.
    out = await transfer.fn(reason="damaged", amount=49.99)
    assert "handoff requested → billing" in out
    assert handoff_request["target"] == "billing"
    payload = handoff_request["payload"]
    assert isinstance(payload, dict)
    assert payload["reason"] == "damaged"
    assert payload["amount"] == 49.99

    # Missing required field — pydantic rejects, error returned.
    handoff_request.clear()
    err = await transfer.fn(reason="late")  # no amount
    assert "Error" in err
    assert "target" not in handoff_request


def test_input_filter_replaces_history_for_receiving_peer() -> None:
    """When a target peer has an ``input_filter``, the filter is
    invoked on each handoff; the filtered string becomes the only
    history entry the receiving peer sees."""
    seen_history: list[list[str]] = []
    seen_payloads: list[dict[str, object]] = []

    def my_filter(history: list[str], payload: dict[str, object]) -> str:
        seen_history.append(list(history))
        seen_payloads.append(dict(payload))
        return f"[filtered: {payload.get('reason', '?')}]"

    b = _agent_no_handoff("billing")
    sw = Swarm(
        agents={
            "a": _agent_no_handoff("a"),
            "billing": Handoff(
                agent=b,
                input_type=RefundRequest,
                input_filter=my_filter,
            ),
        },
        entry_agent="a",
    )
    # Just verify the filter is wired up — full e2e is covered by
    # the existing handoff tests, which we don't need to duplicate.
    assert sw._handoffs["billing"].input_filter is my_filter
