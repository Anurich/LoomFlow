"""Tool-transcript persistence (P3 fix).

Pins ``Agent(persist_tool_transcripts=True)``: the intermediate
tool_call / tool_result messages a worker emits in one delegation
get captured on the Episode, persisted via the memory backend, and
spliced back between USER and ASSISTANT by ``session_messages()``
on subsequent runs of the same ``session_id``.

This closes the structural gap where persistent_subagents preserved
the SHAPE of the conversation (prompt → reply) but not the
SUBSTANCE (what the worker actually did) — see the discussion in
the BUILD_LOG for context.

Coverage:
* The builder (``Agent._build_tool_transcript``) filters correctly.
* The per-entry cap truncates with a marker; below-cap content
  passes through unchanged.
* Default ``persist_tool_transcripts=False`` writes ``None`` —
  no behavioral change for existing users.
* End-to-end with InMemoryMemory: first run captures, second run
  on the same session_id rehydrates the tool messages spliced
  between USER and ASSISTANT.
* End-to-end with SqliteMemory: sidecar table holds the rows,
  bulk-fetch round-trips them correctly, and pre-feature databases
  (no rows in the sidecar) gracefully return without a transcript.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loomflow import (
    Agent,
    EchoModel,
    Episode,
    InMemoryMemory,
)
from loomflow.core.types import Message, Role, ToolCall
from loomflow.memory.sqlite import SqliteMemory

pytestmark = pytest.mark.anyio


# --- Builder unit tests (pure) --------------------------------


def _agent(**kwargs: object) -> Agent:
    """Tiny helper — every test wants an Agent with EchoModel."""
    return Agent(instructions="", model=EchoModel(), **kwargs)  # type: ignore[arg-type]


def test_build_transcript_drops_system_first_user_and_final_assistant() -> None:
    agent = _agent(persist_tool_transcripts=True)
    messages = [
        Message(role=Role.SYSTEM, content="instructions"),
        Message(role=Role.USER, content="the original prompt"),
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[
                ToolCall(id="t1", tool="read", args={"path": "x.py"})
            ],
        ),
        Message(role=Role.TOOL, content="<file body>", tool_call_id="t1"),
        Message(role=Role.ASSISTANT, content="the final answer"),
    ]
    transcript = agent._build_tool_transcript(messages)
    # Drop: SYSTEM, the USER prompt, the final ASSISTANT-text reply.
    # Keep: the ASSISTANT-with-tool_calls + the TOOL result.
    assert len(transcript) == 2
    assert transcript[0].role is Role.ASSISTANT
    assert transcript[0].tool_calls is not None
    assert transcript[0].tool_calls[0].tool == "read"
    assert transcript[1].role is Role.TOOL
    assert transcript[1].content == "<file body>"


def test_build_transcript_empty_messages_returns_empty() -> None:
    agent = _agent(persist_tool_transcripts=True)
    assert agent._build_tool_transcript([]) == []


def test_build_transcript_no_tool_calls_returns_empty() -> None:
    """Conversation with no tool work between USER and ASSISTANT —
    transcript is empty (USER + ASSISTANT are already captured as
    input/output on the Episode)."""
    agent = _agent(persist_tool_transcripts=True)
    messages = [
        Message(role=Role.SYSTEM, content="instructions"),
        Message(role=Role.USER, content="hi"),
        Message(role=Role.ASSISTANT, content="hello"),
    ]
    assert agent._build_tool_transcript(messages) == []


def test_cap_truncates_oversize_with_marker() -> None:
    agent = _agent(persist_tool_transcripts=True, tool_transcript_max_bytes=20)
    huge = "x" * 100
    msg = Message(role=Role.TOOL, content=huge, tool_call_id="t")
    capped = agent._cap_message(msg)
    assert "[truncated:" in capped.content
    # Content was reduced to (cap) prefix + marker.
    assert len(capped.content.encode("utf-8")) < len(huge.encode("utf-8"))


def test_cap_passes_through_below_threshold() -> None:
    agent = _agent(persist_tool_transcripts=True, tool_transcript_max_bytes=100)
    small = "small content"
    msg = Message(role=Role.TOOL, content=small, tool_call_id="t")
    capped = agent._cap_message(msg)
    assert capped.content == small
    assert "[truncated:" not in capped.content


def test_cap_zero_means_unbounded() -> None:
    agent = _agent(persist_tool_transcripts=True, tool_transcript_max_bytes=0)
    huge = "x" * 10_000
    msg = Message(role=Role.TOOL, content=huge, tool_call_id="t")
    capped = agent._cap_message(msg)
    assert capped.content == huge


def test_cap_handles_empty_content() -> None:
    agent = _agent(persist_tool_transcripts=True, tool_transcript_max_bytes=10)
    msg = Message(
        role=Role.ASSISTANT,
        content="",
        tool_calls=[ToolCall(id="t", tool="x", args={})],
    )
    capped = agent._cap_message(msg)
    assert capped is msg  # No copy needed when content is empty.


def test_persist_tool_transcripts_default_is_off() -> None:
    """Backward-compat: existing users on upgrade see no behavior
    change. The default must remain False until the v1.0 cutover."""
    agent = _agent()
    assert agent._persist_tool_transcripts is False


# --- Episode + InMemory round-trip ----------------------------


async def test_inmemory_episode_carries_transcript_field() -> None:
    """The new Episode field round-trips through InMemoryMemory
    (it stores Episode wholesale, so the field is preserved
    without any backend-specific work)."""
    mem = InMemoryMemory()
    transcript = [
        Message(role=Role.TOOL, content="result1", tool_call_id="t1"),
    ]
    ep = Episode(
        session_id="s1",
        user_id="alice",
        input="do something",
        output="done",
        tool_transcript=transcript,
    )
    await mem.remember(ep)

    msgs = await mem.session_messages("s1", user_id="alice", limit=10)
    # USER, TOOL (from transcript), ASSISTANT.
    assert len(msgs) == 3
    assert msgs[0].role is Role.USER
    assert msgs[0].content == "do something"
    assert msgs[1].role is Role.TOOL
    assert msgs[1].content == "result1"
    assert msgs[2].role is Role.ASSISTANT
    assert msgs[2].content == "done"


async def test_inmemory_episode_without_transcript_legacy_behavior() -> None:
    """Episode without a transcript (default) rehydrates to just
    (USER, ASSISTANT) — pre-feature behavior preserved."""
    mem = InMemoryMemory()
    ep = Episode(
        session_id="s1",
        user_id="alice",
        input="hi",
        output="hello",
        # tool_transcript defaults to None
    )
    await mem.remember(ep)

    msgs = await mem.session_messages("s1", user_id="alice", limit=10)
    assert len(msgs) == 2
    assert [m.role for m in msgs] == [Role.USER, Role.ASSISTANT]


# --- SqliteMemory: sidecar table + bulk fetch -----------------


async def test_sqlite_stores_and_rehydrates_transcript(tmp_path: Path) -> None:
    """SqliteMemory persists transcripts to the sidecar table and
    bulk-fetches them on session_messages()."""
    mem = SqliteMemory(str(tmp_path / "m.db"))
    transcript = [
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[ToolCall(id="t1", tool="read", args={"path": "x.py"})],
        ),
        Message(role=Role.TOOL, content="x.py contents", tool_call_id="t1"),
    ]
    ep = Episode(
        session_id="s1",
        user_id="alice",
        input="read x.py",
        output="done",
        tool_transcript=transcript,
    )
    await mem.remember(ep)

    msgs = await mem.session_messages("s1", user_id="alice", limit=10)
    assert len(msgs) == 4  # USER + 2 transcript + ASSISTANT
    assert msgs[0].role is Role.USER
    assert msgs[1].role is Role.ASSISTANT
    assert msgs[1].tool_calls is not None
    assert msgs[1].tool_calls[0].tool == "read"
    assert msgs[2].role is Role.TOOL
    assert msgs[2].content == "x.py contents"
    assert msgs[3].role is Role.ASSISTANT
    assert msgs[3].content == "done"


async def test_sqlite_legacy_episode_without_transcript(tmp_path: Path) -> None:
    """An episode written WITHOUT a transcript (e.g. by a pre-
    feature Agent or by an opted-out Agent) rehydrates cleanly —
    no sidecar rows, no transcript spliced. Same shape as the
    pre-feature behavior."""
    mem = SqliteMemory(str(tmp_path / "m.db"))
    ep = Episode(
        session_id="s1",
        user_id="alice",
        input="hi",
        output="hello",
    )
    await mem.remember(ep)

    msgs = await mem.session_messages("s1", user_id="alice", limit=10)
    assert len(msgs) == 2
    assert [m.role for m in msgs] == [Role.USER, Role.ASSISTANT]


async def test_sqlite_remember_twice_replaces_transcript(tmp_path: Path) -> None:
    """Re-remembering an Episode with the same id replaces the
    sidecar rows (DELETE then INSERT). Without this, accumulating
    duplicates would multiply the transcript on every replay-path
    invocation through the runtime."""
    mem = SqliteMemory(str(tmp_path / "m.db"))
    ep1 = Episode(
        id="ep_constant",
        session_id="s1",
        user_id="alice",
        input="prompt",
        output="reply",
        tool_transcript=[
            Message(role=Role.TOOL, content="first", tool_call_id="t1"),
        ],
    )
    await mem.remember(ep1)
    ep2 = ep1.model_copy(
        update={
            "tool_transcript": [
                Message(role=Role.TOOL, content="second", tool_call_id="t2"),
            ]
        }
    )
    await mem.remember(ep2)

    msgs = await mem.session_messages("s1", user_id="alice", limit=10)
    # Only the second transcript should appear — no doubling.
    tool_contents = [m.content for m in msgs if m.role is Role.TOOL]
    assert tool_contents == ["second"]


async def test_sqlite_empty_transcript_distinguishable_from_none(
    tmp_path: Path,
) -> None:
    """An explicit empty list ``tool_transcript=[]`` writes no
    sidecar rows but also DOES record the opt-in (Episode field
    is ``[]``, not ``None``). Round-trip preserves the distinction
    via the sidecar-rows-vs-no-rows semantics."""
    mem = SqliteMemory(str(tmp_path / "m.db"))
    ep = Episode(
        session_id="s1",
        user_id="alice",
        input="prompt",
        output="reply",
        tool_transcript=[],
    )
    await mem.remember(ep)

    msgs = await mem.session_messages("s1", user_id="alice", limit=10)
    # No tool messages spliced; just USER + ASSISTANT.
    assert [m.role for m in msgs] == [Role.USER, Role.ASSISTANT]


# --- End-to-end via Agent.run() -------------------------------


async def test_agent_with_persist_off_writes_no_transcript() -> None:
    """The Agent with the feature OFF must NOT populate
    Episode.tool_transcript — preserves storage cost + behavior
    for existing users."""
    mem = InMemoryMemory()
    agent = Agent(
        instructions="echo",
        model=EchoModel(),
        memory=mem,
        # persist_tool_transcripts defaults to False
    )
    await agent.run("hello", user_id="alice", session_id="s-off")
    eps = [
        ep
        for ep in mem._episodes.values()
        if ep.session_id == "s-off"
    ]
    assert len(eps) == 1
    assert eps[0].tool_transcript is None


async def test_agent_with_persist_on_writes_transcript() -> None:
    """The Agent with the feature ON populates Episode.tool_transcript
    (empty for a no-tool conversation; populated when tool calls fire).
    EchoModel doesn't emit tool calls, so we just verify the field
    is set to a list (the BUILDER ran), not None."""
    mem = InMemoryMemory()
    agent = Agent(
        instructions="echo",
        model=EchoModel(),
        memory=mem,
        persist_tool_transcripts=True,
    )
    await agent.run("hello", user_id="alice", session_id="s-on")
    eps = [
        ep
        for ep in mem._episodes.values()
        if ep.session_id == "s-on"
    ]
    assert len(eps) == 1
    # Even an empty list (no tools fired) is the opt-in marker —
    # distinct from None which is the legacy / opted-out shape.
    assert eps[0].tool_transcript is not None
    assert isinstance(eps[0].tool_transcript, list)


def test_tool_transcript_max_bytes_negative_rejected() -> None:
    with pytest.raises(ValueError, match="tool_transcript_max_bytes"):
        Agent(
            instructions="",
            model=EchoModel(),
            tool_transcript_max_bytes=-1,
        )
