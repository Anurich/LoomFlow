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

from loomflow import Agent, Architecture, ScriptedModel, ScriptedTurn
from loomflow.architecture import Swarm
from loomflow.architecture.swarm import _is_cycling
from loomflow.core.types import ToolCall

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# _is_cycling helper
# ---------------------------------------------------------------------------


def test_is_cycling_detects_three_round_trips() -> None:
    from collections import deque

    handoffs: deque[tuple[str, str]] = deque(
        [("A", "B"), ("B", "A")] * 3,
        maxlen=6,
    )
    assert _is_cycling(handoffs)


def test_is_cycling_allows_two_round_trips() -> None:
    # Handing back once or twice is normal swarm behaviour.
    from collections import deque

    handoffs: deque[tuple[str, str]] = deque(
        [("A", "B"), ("B", "A"), ("A", "B"), ("B", "A")],
        maxlen=6,
    )
    assert not _is_cycling(handoffs)


def test_is_cycling_false_for_linear_chain() -> None:
    from collections import deque

    handoffs: deque[tuple[str, str]] = deque(
        [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "G")],
        maxlen=6,
    )
    assert not _is_cycling(handoffs)


def test_is_cycling_false_when_history_short() -> None:
    from collections import deque

    handoffs: deque[tuple[str, str]] = deque(
        [("A", "B"), ("B", "A")], maxlen=6
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


async def test_swarm_detects_ping_pong_cycle() -> None:
    """Three A↔B round trips trip cycle detection. The peer just
    handed the conversation gets a final turn with no handoff tools,
    and the run is flagged as interrupted."""

    def _ping_pong(target: str, replies: int, final: str) -> Agent:
        # A handoff ends the turn, so each activation is one model call.
        turns = [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id=f"c{i}", tool="handoff", args={"target": target})
                ]
            )
            for i in range(replies)
        ]
        turns.append(ScriptedTurn(text=final))
        return Agent("ping-pong", model=ScriptedModel(turns))

    a = _ping_pong("B", replies=3, final="A answers at last")
    b = _ping_pong("A", replies=3, final="B answers at last")

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
    assert arch_names.count("swarm.handoff") == 6
    assert "swarm.cycle_detected" in arch_names
    result = events[-1].payload["result"]
    assert result["output"] == "A answers at last"
    assert result["interrupted"] is True
    assert result["interruption_reason"] == "swarm:cycle_detected"


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
    from loomflow import Tool

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

from loomflow.architecture import Handoff  # noqa: E402


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
    # One tool per OTHER peer: an agent never gets a tool for itself.
    assert {t.name for t in sw._build_handoff_tools("a", {})} == {"transfer_to_b"}
    assert {t.name for t in sw._build_handoff_tools("b", {})} == {"transfer_to_a"}


def test_typed_handoff_schema_mirrors_pydantic_model() -> None:
    """The transfer_to_<name> tool's input_schema should expose
    the Pydantic model's fields (reason, amount), not a generic
    ``message`` field."""
    b = _agent_no_handoff("y")
    sw = Swarm(
        agents={"a": _agent_no_handoff("x"), "b": Handoff(agent=b, input_type=RefundRequest)},
        entry_agent="a",
    )
    tools = sw._build_handoff_tools("a", {})
    refund_tool = next(t for t in tools if t.name == "transfer_to_b")
    props = refund_tool.input_schema["properties"]
    assert "reason" in props
    assert "amount" in props


async def test_typed_handoff_validates_payload_via_pydantic() -> None:
    """Calling ``transfer_to_<name>`` with valid args records the
    validated payload; a missing field fails the call (so the turn
    continues and the model sees the error)."""
    b = _agent_no_handoff("billing answer")
    sw = Swarm(
        agents={
            "a": _agent_no_handoff("never runs"),  # not used
            "billing": Handoff(agent=b, input_type=RefundRequest),
        },
        entry_agent="a",
    )
    handoff_request: dict[str, object] = {}
    tools = sw._build_handoff_tools("a", handoff_request)
    transfer = next(t for t in tools if t.name == "transfer_to_billing")
    assert transfer.ends_turn

    # Valid payload — pydantic validates, request is recorded.
    out = await transfer.fn(reason="damaged", amount=49.99)
    assert "handoff requested → billing" in out
    assert handoff_request["target"] == "billing"
    payload = handoff_request["payload"]
    assert isinstance(payload, dict)
    assert payload["reason"] == "damaged"
    assert payload["amount"] == 49.99

    # Missing required field — pydantic rejects, the call fails.
    handoff_request.clear()
    with pytest.raises(ValueError, match="invalid handoff payload"):
        await transfer.fn(reason="late")  # no amount
    assert "target" not in handoff_request


def test_legacy_handoff_tool_enumerates_peer_names_in_schema() -> None:
    """The legacy single-handoff tool must enumerate the peers the
    active agent may hand off to (never itself) in its JSON Schema
    enum, so strict-schema providers reject invalid targets at the
    API boundary instead of bouncing through our error path."""
    a = _agent_no_handoff("a")
    b = _agent_no_handoff("b")
    c = _agent_no_handoff("c")
    sw = Swarm(
        agents={"alpha": a, "bravo": b, "charlie": c},
        entry_agent="alpha",
    )
    assert sw._typed_mode is False
    (tool,) = sw._build_handoff_tools("alpha", {})
    assert tool.ends_turn
    target_schema = tool.input_schema["properties"]["target"]
    assert "enum" in target_schema
    assert set(target_schema["enum"]) == {"bravo", "charlie"}
    # Description should also list the names so models that don't
    # honour enum still see them in plain text.
    desc = tool.input_schema["properties"]["target"]["description"]
    for name in ("bravo", "charlie"):
        assert name in desc
    assert "alpha" not in desc


def test_legacy_handoff_tool_description_lists_each_peers_role() -> None:
    """The handoff tool's top-level description should include each
    peer's instructions so the model can reason about who to pick."""
    triage = Agent(
        "Triage incoming requests",
        model=ScriptedModel([ScriptedTurn(text="ok")]),
    )
    billing = Agent(
        "Handle refunds and invoices",
        model=ScriptedModel([ScriptedTurn(text="ok")]),
    )
    sw = Swarm(
        agents={"triage": triage, "billing": billing},
        entry_agent="triage",
    )
    (from_triage,) = sw._build_handoff_tools("triage", {})
    assert "Handle refunds" in from_triage.description
    (from_billing,) = sw._build_handoff_tools("billing", {})
    assert "Triage incoming" in from_billing.description


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


# ---------------------------------------------------------------------------
# Behaviour aligned with the reference swarms (OpenAI Swarm, OpenAI
# Agents SDK handoffs, LangGraph Swarm)
# ---------------------------------------------------------------------------


class _Recording(ScriptedModel):
    """ScriptedModel that keeps the last user message of every call."""

    def __init__(self, turns: list[ScriptedTurn]) -> None:
        super().__init__(turns)
        self.prompts: list[str] = []

    def _note(self, messages: object) -> None:
        last_user = next(
            (m.content for m in reversed(list(messages)) if m.role == "user"),  # type: ignore[attr-defined]
            "",
        )
        self.prompts.append(str(last_user))

    async def complete(self, messages, *args, **kwargs):  # type: ignore[no-untyped-def]
        self._note(messages)
        return await super().complete(messages, *args, **kwargs)

    async def stream(self, messages, *args, **kwargs):  # type: ignore[no-untyped-def]
        self._note(messages)
        async for chunk in super().stream(messages, *args, **kwargs):
            yield chunk


def _handoff_turn(target: str, message: str = "") -> ScriptedTurn:
    return ScriptedTurn(
        tool_calls=[
            ToolCall(
                id=f"h-{target}",
                tool="handoff",
                args={"target": target, "message": message},
            )
        ]
    )


def _swarm_host(**kwargs: object) -> Agent:
    return Agent(
        "host",
        model=ScriptedModel([ScriptedTurn(text="never")]),
        architecture=Swarm(**kwargs),  # type: ignore[arg-type]
    )


async def test_handoff_ends_the_turn_immediately() -> None:
    # Reference behaviour: the next agent takes over at once; the
    # handing-off agent makes no further model call.
    triage = _Recording([_handoff_turn("billing"), ScriptedTurn(text="unused")])
    billing = _Recording([ScriptedTurn(text="refund issued")])
    host = _swarm_host(
        agents={
            "triage": Agent("triage", model=triage),
            "billing": Agent("billing", model=billing),
        },
        entry_agent="triage",
    )
    result = await host.run("charged twice")
    assert result.output == "refund issued"
    assert len(triage.prompts) == 1
    assert triage.remaining == 1


async def test_peer_can_hand_back_to_the_previous_peer() -> None:
    a = _Recording(
        [_handoff_turn("B", "billing issue"), ScriptedTurn(text="A: final answer")]
    )
    b = _Recording([_handoff_turn("A", "needs triage again")])
    host = _swarm_host(
        agents={"A": Agent("triage", model=a), "B": Agent("billing", model=b)},
        entry_agent="A",
    )
    events = [e async for e in host.stream("charged twice")]
    handoffs = [
        (e.payload["from_agent"], e.payload["to_agent"])
        for e in events
        if e.kind == "architecture_event" and e.payload["name"] == "swarm.handoff"
    ]
    assert handoffs == [("A", "B"), ("B", "A")]
    result = events[-1].payload["result"]
    assert result["output"] == "A: final answer"
    assert result["interrupted"] is False
    # A sees the whole thread when it gets control back.
    assert "[Handoff: A → B] billing issue" in a.prompts[-1]
    assert "[Handoff: B → A] needs triage again" in a.prompts[-1]


async def test_next_peer_sees_previous_peers_tool_calls() -> None:
    from loomflow import tool

    @tool
    async def lookup_order(order_id: str) -> str:
        """Look up an order."""
        return "order 123: charged twice, 49.00"

    triage = _Recording(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="l1", tool="lookup_order", args={"order_id": "123"})
                ]
            ),
            _handoff_turn("billing"),
        ]
    )
    billing = _Recording([ScriptedTurn(text="ok")])
    host = _swarm_host(
        agents={
            "triage": Agent("triage", model=triage, tools=[lookup_order]),
            "billing": Agent("billing", model=billing),
        },
        entry_agent="triage",
    )
    await host.run("charged twice on order 123")
    seen = billing.prompts[0]
    assert 'triage called lookup_order({"order_id": "123"})' in seen
    assert "order 123: charged twice, 49.00" in seen
    # The handoff call itself shows up as the marker, not as a tool line.
    assert "called handoff" not in seen


async def test_pass_full_history_false_keeps_request_and_latest_handoff() -> None:
    a = _Recording(
        [ScriptedTurn(text="A found order 123", tool_calls=_handoff_turn("B", "refund").tool_calls)]
    )
    b = _Recording([_handoff_turn("C", "please confirm")])
    c = _Recording([ScriptedTurn(text="done")])
    host = _swarm_host(
        agents={
            "A": Agent("a", model=a),
            "B": Agent("b", model=b),
            "C": Agent("c", model=c),
        },
        entry_agent="A",
        pass_full_history=False,
    )
    await host.run("I was charged twice")
    assert b.prompts[0].startswith("I was charged twice")
    assert "A found order 123" in b.prompts[0]
    assert "[Handoff: A → B] refund" in b.prompts[0]
    # C gets the request and B's handoff, not A's older segment.
    assert c.prompts[0].startswith("I was charged twice")
    assert "[Handoff: B → C] please confirm" in c.prompts[0]
    assert "A found order 123" not in c.prompts[0]


async def test_agent_cannot_hand_off_to_itself() -> None:
    a = _Recording([_handoff_turn("A"), ScriptedTurn(text="A answers")])
    host = _swarm_host(
        agents={"A": Agent("a", model=a), "B": _agent_no_handoff("B")},
        entry_agent="A",
    )
    events = [e async for e in host.stream("hi")]
    names = [e.payload["name"] for e in events if e.kind == "architecture_event"]
    assert "swarm.handoff" not in names
    failed = [
        e.payload["result"]
        for e in events
        if e.kind == "tool_result" and not e.payload["result"]["ok"]
    ]
    assert failed and "cannot hand off to 'A'" in failed[0]["error"]
    assert events[-1].payload["result"]["output"] == "A answers"


def test_handoffs_map_limits_targets() -> None:
    sw = Swarm(
        agents={
            "triage": _agent_no_handoff("t"),
            "billing": _agent_no_handoff("b"),
            "tech": _agent_no_handoff("x"),
        },
        entry_agent="triage",
        handoffs={"triage": ["billing", "tech"], "billing": ["triage"]},
    )
    (from_triage,) = sw._build_handoff_tools("triage", {})
    assert from_triage.input_schema["properties"]["target"]["enum"] == ["billing", "tech"]
    (from_billing,) = sw._build_handoff_tools("billing", {})
    assert from_billing.input_schema["properties"]["target"]["enum"] == ["triage"]
    # Left out of the map: no handoff tool, so tech has to answer.
    assert sw._build_handoff_tools("tech", {}) == []


@pytest.mark.parametrize(
    ("handoffs", "message"),
    [
        ({"nobody": ["a"]}, "unknown peers: nobody"),
        ({"a": ["nobody"]}, "unknown peers: nobody"),
        ({"a": ["a"]}, "lists 'a' itself"),
    ],
)
def test_handoffs_map_is_validated(handoffs: dict[str, list[str]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Swarm(
            agents={"a": _agent_no_handoff("a"), "b": _agent_no_handoff("b")},
            entry_agent="a",
            handoffs=handoffs,
        )


async def test_next_message_in_a_session_goes_to_the_last_active_peer() -> None:
    triage = _Recording([_handoff_turn("billing"), ScriptedTurn(text="triage again")])
    billing = _Recording([ScriptedTurn(text="refund issued"), ScriptedTurn(text="5 days")])
    host = _swarm_host(
        agents={
            "triage": Agent("triage", model=triage),
            "billing": Agent("billing", model=billing),
        },
        entry_agent="triage",
    )
    first = await host.run("charged twice", session_id="conv-1")
    second = await host.run("when will it arrive?", session_id="conv-1")
    assert (first.output, second.output) == ("refund issued", "5 days")
    # A different conversation still starts at the entry agent.
    other = await host.run("hello", session_id="conv-2")
    assert other.output == "triage again"


async def test_resume_last_agent_can_be_turned_off() -> None:
    triage = _Recording([_handoff_turn("billing"), ScriptedTurn(text="triage answers")])
    billing = _Recording([ScriptedTurn(text="refund issued")])
    host = _swarm_host(
        agents={
            "triage": Agent("triage", model=triage),
            "billing": Agent("billing", model=billing),
        },
        entry_agent="triage",
        resume_last_agent=False,
    )
    await host.run("charged twice", session_id="conv-1")
    second = await host.run("thanks", session_id="conv-1")
    assert second.output == "triage answers"


async def test_max_handoffs_gives_a_final_answer_and_flags_the_run() -> None:
    a = _Recording([_handoff_turn("B")])
    b = _Recording([_handoff_turn("A"), ScriptedTurn(text="B answers without handing off")])
    host = _swarm_host(
        agents={"A": Agent("a", model=a), "B": Agent("b", model=b)},
        entry_agent="A",
        max_handoffs=1,
    )
    result = await host.run("go")
    assert result.output == "B answers without handing off"
    assert result.interrupted is True
    assert result.interruption_reason == "swarm:max_handoffs"
    # The final turn had no handoff tool to call.
    assert "No more handoffs are possible" in b.prompts[-1]


async def test_ends_turn_tool_stops_a_plain_agent_and_skips_output_schema() -> None:
    from pydantic import BaseModel as _BaseModel

    from loomflow import tool

    class Answer(_BaseModel):
        text: str

    @tool(ends_turn=True)
    async def escalate(reason: str) -> str:
        """Hand the case to a human."""
        return "escalated"

    model = _Recording(
        [
            ScriptedTurn(
                text="Escalating.",
                tool_calls=[ToolCall(id="e1", tool="escalate", args={"reason": "angry"})],
            ),
            ScriptedTurn(text="should not be called"),
        ]
    )
    agent = Agent("support", model=model, tools=[escalate], output_schema=Answer)
    result = await agent.run("I want a human")
    assert result.output == "Escalating."
    assert model.remaining == 1
