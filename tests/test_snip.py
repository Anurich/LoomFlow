"""Tests for the conversation-snip feature (0.10.16).

Snip is the cheap always-on context-budget defence: before each
architecture invocation, ``session.messages`` is trimmed to the
last N user-anchored turn groups. Pure list slicing — no LLM call.

Coverage:

* ``snip_messages`` helper — unit tests for slicing rules:
  empty input, no-op cases, leading system preservation,
  user-boundary cutting, dropped-count return value.
* End-to-end via ``Agent`` — confirm ``snip_window=N`` actually
  bounds history across multiple ``agent.run()`` calls and emits
  the ``messages_snipped`` architecture event.
"""

from __future__ import annotations

import pytest

from loomflow import Agent, ScriptedModel, ScriptedTurn
from loomflow.agent.snip import snip_messages
from loomflow.core.types import Message, Role

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# snip_messages — direct unit tests
# ---------------------------------------------------------------------------


def _msg(role: Role, content: str) -> Message:
    return Message(role=role, content=content)


def test_snip_empty_messages_is_noop() -> None:
    out, dropped = snip_messages([], keep_last_n_turns=3)
    assert out == []
    assert dropped == 0


def test_snip_zero_window_is_disabled() -> None:
    """``keep_last_n_turns=0`` disables snipping — list passes
    through unchanged. Mirror of the ``snip_window=0`` default
    on Agent."""
    msgs = [_msg(Role.USER, "hello"), _msg(Role.ASSISTANT, "hi")]
    out, dropped = snip_messages(msgs, keep_last_n_turns=0)
    assert out is msgs  # same object — no allocation
    assert dropped == 0


def test_snip_negative_window_is_disabled() -> None:
    msgs = [_msg(Role.USER, "hello")]
    out, dropped = snip_messages(msgs, keep_last_n_turns=-1)
    assert out is msgs
    assert dropped == 0


def test_snip_under_window_no_drop() -> None:
    """Two turns, keep_last_n=3 → nothing to drop."""
    msgs = [
        _msg(Role.USER, "q1"),
        _msg(Role.ASSISTANT, "a1"),
        _msg(Role.USER, "q2"),
        _msg(Role.ASSISTANT, "a2"),
    ]
    out, dropped = snip_messages(msgs, keep_last_n_turns=3)
    assert out == msgs
    assert dropped == 0


def test_snip_keeps_last_n_user_anchored_turns() -> None:
    """Three turns, keep_last_n=2 → drop the first user-anchored
    group (q1 + a1), keep q2/a2 + q3/a3."""
    msgs = [
        _msg(Role.USER, "q1"),
        _msg(Role.ASSISTANT, "a1"),
        _msg(Role.USER, "q2"),
        _msg(Role.ASSISTANT, "a2"),
        _msg(Role.USER, "q3"),
        _msg(Role.ASSISTANT, "a3"),
    ]
    out, dropped = snip_messages(msgs, keep_last_n_turns=2)
    assert [m.content for m in out] == ["q2", "a2", "q3", "a3"]
    assert dropped == 2


def test_snip_preserves_leading_system_head() -> None:
    """Leading system messages (rare — architectures usually
    rebuild system content per turn — but possible) survive
    every snip."""
    msgs = [
        _msg(Role.SYSTEM, "be helpful"),
        _msg(Role.SYSTEM, "memory: X"),
        _msg(Role.USER, "q1"),
        _msg(Role.ASSISTANT, "a1"),
        _msg(Role.USER, "q2"),
        _msg(Role.ASSISTANT, "a2"),
    ]
    out, dropped = snip_messages(msgs, keep_last_n_turns=1)
    assert [m.role for m in out] == [
        Role.SYSTEM, Role.SYSTEM, Role.USER, Role.ASSISTANT,
    ]
    assert [m.content for m in out] == [
        "be helpful", "memory: X", "q2", "a2",
    ]
    assert dropped == 2


def test_snip_no_user_messages_is_noop() -> None:
    """A system-only or assistant-only history has no user anchor
    to slice at → return unchanged."""
    msgs = [
        _msg(Role.SYSTEM, "instructions"),
        _msg(Role.ASSISTANT, "I'm thinking..."),
    ]
    out, dropped = snip_messages(msgs, keep_last_n_turns=1)
    assert out == msgs
    assert dropped == 0


def test_snip_preserves_tool_result_groupings() -> None:
    """Tool results stay grouped with their user message — the
    snip never leaves an orphan tool_result without its preceding
    tool_call. (Anchored at user boundaries, so the slice always
    starts at a user message.)"""
    msgs = [
        _msg(Role.USER, "q1"),
        _msg(Role.ASSISTANT, "calling tool"),
        _msg(Role.TOOL, "tool result 1"),
        _msg(Role.ASSISTANT, "a1"),
        _msg(Role.USER, "q2"),
        _msg(Role.ASSISTANT, "calling tool again"),
        _msg(Role.TOOL, "tool result 2"),
        _msg(Role.ASSISTANT, "a2"),
    ]
    out, dropped = snip_messages(msgs, keep_last_n_turns=1)
    # Should start with q2 (no orphan TOOL from q1's group).
    assert out[0].role == Role.USER
    assert out[0].content == "q2"
    assert dropped == 4
    # Confirm none of the kept messages is a stray tool_result
    # without a preceding tool-call assistant.
    saw_user = False
    for m in out:
        if m.role == Role.USER:
            saw_user = True
        if m.role == Role.TOOL:
            assert saw_user, (
                "snip left a tool_result before its user anchor"
            )


# ---------------------------------------------------------------------------
# End-to-end via Agent
# ---------------------------------------------------------------------------


def test_agent_negative_snip_window_rejected() -> None:
    with pytest.raises(
        ValueError, match="snip_window must be >= 0"
    ):
        Agent("you help", model="echo", snip_window=-1)


def test_agent_snip_window_default_is_zero() -> None:
    """Default 0 → snipping disabled → fully back-compat."""
    agent = Agent("you help", model="echo")
    assert agent._snip_window == 0


def test_agent_snip_window_propagates() -> None:
    agent = Agent("you help", model="echo", snip_window=3)
    assert agent._snip_window == 3


async def test_agent_snip_fires_on_rehydrated_history() -> None:
    """End-to-end: accumulate history across multiple ``agent.run()``
    calls (via :class:`InMemoryMemory` + a stable session_id so the
    architecture rehydrates prior turns into ``session.messages``),
    then verify ``snip_window=1`` drops the rehydrated older turns
    and emits the ``messages_snipped`` event."""
    from loomflow import InMemoryMemory

    coord = ScriptedModel(
        turns=[
            ScriptedTurn(text="r1"),
            ScriptedTurn(text="r2"),
            ScriptedTurn(text="r3"),
            ScriptedTurn(text="r4"),
        ]
    )
    mem = InMemoryMemory()
    agent = Agent(
        "you help",
        model=coord,
        memory=mem,
        snip_window=1,
    )
    snip_events: list[dict] = []
    # Same session_id across all runs → architecture rehydrates
    # prior conversation into ``session.messages`` at the top of
    # each run, so by the 3rd/4th call there's enough history to
    # trigger the snip.
    sid = "snip-test-session"
    for prompt in ["q1", "q2", "q3", "q4"]:
        async for event in agent.stream(prompt, session_id=sid):
            kind = getattr(event, "kind", None)
            payload = getattr(event, "payload", None)
            if kind is None or payload is None:
                continue
            if str(kind).endswith("architecture_event") and (
                payload.get("name") == "messages_snipped"
            ):
                snip_events.append(payload)
    assert snip_events, (
        "expected at least one messages_snipped event after "
        "multi-run rehydrated history accumulated"
    )
    payload = snip_events[-1]
    assert payload["window_turns"] == 1
    assert payload["dropped"] >= 1
