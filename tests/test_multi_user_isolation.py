"""Multi-user isolation regression tests.

These guard the core M1 contract: ``user_id`` is a hard namespace
boundary on every memory primitive (working blocks not in scope —
those are global per-Agent — but Episodes and Facts ARE).

Every test exercises a default ``InMemoryMemory`` (with the default
``InMemoryFactStore``) and asserts that data persisted under one
``user_id`` never leaks into a recall scoped to a different one.
The contextvar-backed ``RunContext`` is also exercised end-to-end
through ``Agent.run`` so we know the wiring works, not just the
backend.
"""

from __future__ import annotations

from typing import Any

import pytest

from jeevesagent import (
    Agent,
    Episode,
    Fact,
    RunContext,
    get_run_context,
    set_run_context,
    tool,
)
from jeevesagent.core.types import ToolCall
from jeevesagent.memory.facts import InMemoryFactStore
from jeevesagent.memory.inmemory import InMemoryMemory
from jeevesagent.model.scripted import ScriptedModel, ScriptedTurn

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# RunContext primitive — basic propagation
# ---------------------------------------------------------------------------


async def test_default_run_context_is_empty() -> None:
    """Outside an active run, ``get_run_context`` returns the default
    empty context — never raises."""
    ctx = get_run_context()
    assert ctx.user_id is None
    assert ctx.session_id is None
    assert ctx.run_id == ""
    assert ctx.metadata == {}


async def test_set_run_context_installs_then_restores() -> None:
    """``set_run_context`` is reentrant: nested blocks restore the
    prior context on exit."""
    outer = RunContext(user_id="alice", session_id="s1", metadata={"k": 1})
    inner = RunContext(user_id="bob", session_id="s2")

    async with set_run_context(outer):
        assert get_run_context() == outer
        async with set_run_context(inner):
            assert get_run_context() == inner
        assert get_run_context() == outer
    assert get_run_context() == RunContext()


async def test_run_context_metadata_get_helper() -> None:
    ctx = RunContext(metadata={"locale": "en", "tenant": "acme"})
    assert ctx.get("locale") == "en"
    assert ctx.get("missing", "fallback") == "fallback"


async def test_run_context_with_overrides_preserves_unset_fields() -> None:
    """Sentinel-default kwargs let callers distinguish "leave alone"
    from "explicitly set to None"."""
    base = RunContext(user_id="alice", session_id="s1", metadata={"k": 1})
    derived = base.with_overrides(session_id="s2")
    assert derived.user_id == "alice"  # preserved
    assert derived.session_id == "s2"  # overridden
    assert derived.metadata == {"k": 1}

    # Explicit None overrides:
    cleared = base.with_overrides(user_id=None)
    assert cleared.user_id is None
    assert cleared.session_id == "s1"


# ---------------------------------------------------------------------------
# InMemoryMemory — episode partition
# ---------------------------------------------------------------------------


async def test_episodes_partition_by_user_id() -> None:
    """Episodes stored under user A must not surface in a recall
    scoped to user B."""
    mem = InMemoryMemory()
    await mem.remember(
        Episode(session_id="sA", user_id="alice", input="hi", output="x")
    )
    await mem.remember(
        Episode(session_id="sB", user_id="bob", input="hi", output="y")
    )

    alice_only = await mem.recall("hi", user_id="alice")
    bob_only = await mem.recall("hi", user_id="bob")

    assert len(alice_only) == 1
    assert alice_only[0].user_id == "alice"
    assert len(bob_only) == 1
    assert bob_only[0].user_id == "bob"


async def test_anonymous_bucket_does_not_leak_into_named_users() -> None:
    """``user_id=None`` is its own bucket — episodes stored without a
    user_id never surface for a named-user query (and vice versa)."""
    mem = InMemoryMemory()
    await mem.remember(
        Episode(session_id="s0", user_id=None, input="hi", output="anon")
    )
    await mem.remember(
        Episode(session_id="s1", user_id="alice", input="hi", output="alice")
    )

    anon = await mem.recall("hi", user_id=None)
    alice = await mem.recall("hi", user_id="alice")
    bob = await mem.recall("hi", user_id="bob")  # never persisted

    assert [e.output for e in anon] == ["anon"]
    assert [e.output for e in alice] == ["alice"]
    assert bob == []


# ---------------------------------------------------------------------------
# InMemoryFactStore — fact partition + namespaced supersession
# ---------------------------------------------------------------------------


async def test_facts_partition_by_user_id() -> None:
    facts = InMemoryFactStore()
    await facts.append(
        Fact(user_id="alice", subject="x", predicate="loves", object="pizza")
    )
    await facts.append(
        Fact(user_id="bob", subject="x", predicate="loves", object="sushi")
    )

    alice = await facts.recall_text("loves", user_id="alice")
    bob = await facts.recall_text("loves", user_id="bob")

    assert len(alice) == 1 and alice[0].object == "pizza"
    assert len(bob) == 1 and bob[0].object == "sushi"


async def test_supersession_is_namespace_scoped() -> None:
    """Adding a new fact for the same (subject, predicate) under one
    user must NOT invalidate another user's currently-valid claim."""
    facts = InMemoryFactStore()

    alice_v1 = Fact(
        user_id="alice", subject="x", predicate="lives_in", object="Berlin"
    )
    bob_v1 = Fact(
        user_id="bob", subject="x", predicate="lives_in", object="Berlin"
    )
    await facts.append(alice_v1)
    await facts.append(bob_v1)

    # New claim from alice should close her old fact, not bob's.
    alice_v2 = Fact(
        user_id="alice", subject="x", predicate="lives_in", object="Lisbon"
    )
    await facts.append(alice_v2)

    # Bob's original fact must still be currently-valid.
    bob_facts = await facts.query(user_id="bob", subject="x")
    assert len(bob_facts) == 1
    assert bob_facts[0].object == "Berlin"
    assert bob_facts[0].valid_until is None  # still currently valid

    # Alice has both: an invalidated v1 and a current v2.
    alice_facts = await facts.query(user_id="alice", subject="x")
    by_obj = {f.object: f for f in alice_facts}
    assert by_obj["Berlin"].valid_until is not None  # superseded
    assert by_obj["Lisbon"].valid_until is None  # current


# ---------------------------------------------------------------------------
# End-to-end: Agent.run propagates user_id into Memory
# ---------------------------------------------------------------------------


async def test_agent_run_persists_episode_with_user_id() -> None:
    """``Agent.run(user_id=...)`` should tag the persisted episode
    with that user_id, so subsequent recalls scoped to that user
    surface it (and recalls scoped elsewhere do not)."""
    mem = InMemoryMemory()
    agent = Agent(
        "You are helpful.",
        model=ScriptedModel([ScriptedTurn(text="ok")]),
        memory=mem,
    )

    await agent.run("first message", user_id="alice", session_id="s1")

    # The persisted episode lives in alice's bucket.
    alice = await mem.recall("first", user_id="alice")
    bob = await mem.recall("first", user_id="bob")

    assert len(alice) == 1
    assert alice[0].user_id == "alice"
    assert bob == []


async def test_two_agents_share_memory_without_leaking_users() -> None:
    """A single ``InMemoryMemory`` shared across logical "users" must
    keep their histories disjoint — the M1 multi-tenant safety
    contract end-to-end."""
    mem = InMemoryMemory()
    # Same Agent instance used twice, simulating a server using one
    # Agent across requests for many users.
    agent = Agent(
        "You are helpful.",
        model=ScriptedModel(
            [
                ScriptedTurn(text="hi alice"),
                ScriptedTurn(text="hi bob"),
            ]
        ),
        memory=mem,
    )

    await agent.run("alice's question", user_id="alice", session_id="sA")
    await agent.run("bob's question", user_id="bob", session_id="sB")

    alice_recall = await mem.recall("question", user_id="alice")
    bob_recall = await mem.recall("question", user_id="bob")

    assert len(alice_recall) == 1
    assert alice_recall[0].input == "alice's question"
    assert len(bob_recall) == 1
    assert bob_recall[0].input == "bob's question"


# ---------------------------------------------------------------------------
# Context propagation into tools
# ---------------------------------------------------------------------------


async def test_tools_see_user_id_via_get_run_context() -> None:
    """A ``@tool`` invoked during ``Agent.run`` should see the same
    ``user_id`` via ``get_run_context()`` that the run was scoped
    to. Verifies the contextvar is installed before architecture
    iteration begins and survives parallel-tool dispatch."""
    seen: list[str | None] = []

    @tool
    async def report_user() -> str:
        """Return the user_id seen by this tool."""
        ctx = get_run_context()
        seen.append(ctx.user_id)
        return ctx.user_id or "anonymous"

    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[ToolCall(id="c1", tool="report_user", args={})]
            ),
            ScriptedTurn(text="done"),
        ]
    )
    agent = Agent("hi", model=model, tools=[report_user])

    await agent.run("who am i?", user_id="alice")
    assert seen == ["alice"]


async def test_tool_metadata_is_visible_via_get_run_context() -> None:
    """The free-form ``metadata`` bag rides along on ``RunContext``
    and is reachable from tools without any extra plumbing."""
    captured: list[str | None] = []

    @tool
    async def report_locale() -> str:
        """Return the locale from run metadata."""
        ctx = get_run_context()
        captured.append(ctx.get("locale"))
        return captured[-1] or "?"

    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[ToolCall(id="c1", tool="report_locale", args={})]
            ),
            ScriptedTurn(text="done"),
        ]
    )
    agent = Agent("hi", model=model, tools=[report_locale])

    await agent.run(
        "any",
        user_id="alice",
        metadata={"locale": "en-US"},
    )
    assert captured == ["en-US"]


async def test_run_context_resets_after_run() -> None:
    """The contextvar must be restored to the default after the run
    exits — no leak into subsequent unrelated code."""
    agent = Agent(
        "hi",
        model=ScriptedModel([ScriptedTurn(text="ok")]),
    )
    await agent.run("anything", user_id="alice")
    assert get_run_context() == RunContext()


# ---------------------------------------------------------------------------
# context= kwarg path
# ---------------------------------------------------------------------------


async def test_agent_run_accepts_full_run_context_object() -> None:
    """Passing a constructed ``RunContext`` via ``context=`` should
    work the same as flat kwargs — useful when forwarding context
    across multi-agent boundaries."""
    mem = InMemoryMemory()
    agent = Agent(
        "hi",
        model=ScriptedModel([ScriptedTurn(text="ok")]),
        memory=mem,
    )

    ctx = RunContext(
        user_id="alice",
        session_id="conv_42",
        metadata={"tenant": "acme"},
    )
    await agent.run("hello", context=ctx)

    alice = await mem.recall("hello", user_id="alice")
    assert len(alice) == 1
    assert alice[0].session_id == "conv_42"


# ---------------------------------------------------------------------------
# M2 — session_messages + rehydration
# ---------------------------------------------------------------------------


async def test_session_messages_partitions_by_session_and_user() -> None:
    """``session_messages`` returns only THIS session's turns, in
    order, oldest-first; respects the ``user_id`` partition; never
    leaks turns from another session or user."""
    mem = InMemoryMemory()
    # Alice: two turns in conv_a
    await mem.remember(
        Episode(
            session_id="conv_a", user_id="alice",
            input="hi", output="hello",
        )
    )
    await mem.remember(
        Episode(
            session_id="conv_a", user_id="alice",
            input="my food is pizza", output="noted",
        )
    )
    # Bob: one turn in conv_b
    await mem.remember(
        Episode(
            session_id="conv_b", user_id="bob",
            input="anything", output="any",
        )
    )
    # Alice on a DIFFERENT session — must not surface in conv_a.
    await mem.remember(
        Episode(
            session_id="conv_x", user_id="alice",
            input="other thread", output="other",
        )
    )

    msgs = await mem.session_messages("conv_a", user_id="alice")
    # Two episodes × {USER, ASSISTANT} = 4 messages, oldest first.
    assert len(msgs) == 4
    assert [m.content for m in msgs] == [
        "hi", "hello", "my food is pizza", "noted",
    ]

    # Bob's session — separate, single episode → 2 messages.
    bob_msgs = await mem.session_messages("conv_b", user_id="bob")
    assert [m.content for m in bob_msgs] == ["anything", "any"]

    # Wrong-user query against alice's session: nothing.
    leaked = await mem.session_messages("conv_a", user_id="bob")
    assert leaked == []


async def test_session_messages_limit_keeps_most_recent_turns() -> None:
    """``limit`` caps the number of returned messages, preferring the
    MOST recent turns (since older history is the first to drop on
    long-running threads)."""
    mem = InMemoryMemory()
    for i in range(5):
        await mem.remember(
            Episode(
                session_id="s", user_id="alice",
                input=f"q{i}", output=f"a{i}",
            )
        )
    # limit=4 → 2 most-recent turns × 2 messages each = 4 messages.
    msgs = await mem.session_messages("s", user_id="alice", limit=4)
    assert len(msgs) == 4
    assert [m.content for m in msgs] == ["q3", "a3", "q4", "a4"]


async def test_session_id_reused_continues_conversation() -> None:
    """Reusing the same ``session_id`` across two ``Agent.run`` calls
    must rehydrate the prior turn so the model sees real chat
    history, not just a semantic-recall summary."""

    captured_messages: list[list[Any]] = []

    class _CapturingScripted(ScriptedModel):
        async def complete(self, messages: Any, **kwargs: Any) -> Any:
            captured_messages.append(list(messages))
            return await super().complete(messages, **kwargs)

    model = _CapturingScripted([
        ScriptedTurn(text="my favourite is pizza"),
        ScriptedTurn(text="pizza"),
    ])
    agent = Agent("hi", model=model, memory=InMemoryMemory())

    await agent.run(
        "tell me your favourite food",
        user_id="alice",
        session_id="conv_a",
    )
    await agent.run(
        "what is my favourite food?",
        user_id="alice",
        session_id="conv_a",  # SAME session — must rehydrate.
    )

    # The second run's seed messages must contain the FIRST run's
    # exchange as user/assistant turns, NOT just a system "Relevant
    # past episodes:" note.
    second_run_messages = captured_messages[1]
    contents = [m.content for m in second_run_messages]
    assert "tell me your favourite food" in contents
    assert "my favourite is pizza" in contents


async def test_different_session_ids_do_not_continue_conversation() -> None:
    """A *different* ``session_id`` must NOT pull in another
    session's chat history — the conversation is fresh."""
    captured_messages: list[list[Any]] = []

    class _CapturingScripted(ScriptedModel):
        async def complete(self, messages: Any, **kwargs: Any) -> Any:
            captured_messages.append(list(messages))
            return await super().complete(messages, **kwargs)

    model = _CapturingScripted([
        ScriptedTurn(text="pizza"),
        ScriptedTurn(text="I do not know"),
    ])
    agent = Agent("hi", model=model, memory=InMemoryMemory())

    await agent.run(
        "my favourite is pizza",
        user_id="alice",
        session_id="conv_a",
    )
    await agent.run(
        "what is my favourite food?",
        user_id="alice",
        session_id="conv_b",  # DIFFERENT session — fresh thread.
    )

    second = captured_messages[1]
    contents = [m.content for m in second]
    # The second run sees only its own user prompt as USER content
    # (plus possibly system messages from recall — which is a
    # separate layer, not the rehydrated message log).
    assert "my favourite is pizza" not in contents


async def test_other_user_session_history_not_rehydrated() -> None:
    """Rehydration is namespace-partitioned: even with the same
    ``session_id`` value, a different ``user_id`` MUST get a fresh
    conversation."""
    captured_messages: list[list[Any]] = []

    class _CapturingScripted(ScriptedModel):
        async def complete(self, messages: Any, **kwargs: Any) -> Any:
            captured_messages.append(list(messages))
            return await super().complete(messages, **kwargs)

    model = _CapturingScripted([
        ScriptedTurn(text="alice's pizza"),
        ScriptedTurn(text="don't know about bob"),
    ])
    agent = Agent("hi", model=model, memory=InMemoryMemory())

    await agent.run(
        "alice tells the bot her favourite is pizza",
        user_id="alice",
        session_id="shared_id",
    )
    await agent.run(
        "what is my favourite food?",
        user_id="bob",
        session_id="shared_id",  # same session_id, different user_id
    )

    second = captured_messages[1]
    contents = [m.content for m in second]
    # Alice's content must not appear anywhere in Bob's seed.
    assert all("pizza" not in c for c in contents if isinstance(c, str))


# ---------------------------------------------------------------------------
# M3 — IsolationWarning footgun protection
# ---------------------------------------------------------------------------


async def test_isolation_warning_fires_on_mixed_bucket_recall() -> None:
    """When a memory contains data for one or more named users and a
    recall is run with ``user_id=None``, ``IsolationWarning`` must
    fire — guarding the very common "forgot to pass user_id"
    mistake."""
    import warnings

    from jeevesagent import IsolationWarning

    mem = InMemoryMemory()
    await mem.remember(
        Episode(session_id="s", user_id="alice", input="hi", output="hello")
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IsolationWarning)
        await mem.recall("hi", user_id=None)

    assert any(issubclass(w.category, IsolationWarning) for w in caught)


async def test_isolation_warning_does_not_fire_when_only_anonymous() -> None:
    """No warning when the store has only anonymous data — the
    None-bucket query is unambiguously correct in that case."""
    import warnings

    from jeevesagent import IsolationWarning

    mem = InMemoryMemory()
    await mem.remember(
        Episode(session_id="s", user_id=None, input="hi", output="hello")
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IsolationWarning)
        await mem.recall("hi", user_id=None)

    assert not any(
        issubclass(w.category, IsolationWarning) for w in caught
    )


async def test_isolation_warning_does_not_fire_when_user_id_passed() -> None:
    """Passing the right ``user_id`` (even one that has no stored
    data yet) is a clean call — no warning."""
    import warnings

    from jeevesagent import IsolationWarning

    mem = InMemoryMemory()
    await mem.remember(
        Episode(session_id="s", user_id="alice", input="hi", output="hello")
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IsolationWarning)
        # Bob has no episodes; the query is correct even if empty.
        await mem.recall("hi", user_id="bob")

    assert not any(
        issubclass(w.category, IsolationWarning) for w in caught
    )


async def test_isolation_warning_can_be_promoted_to_error() -> None:
    """Apps that want strict isolation enforcement promote the
    warning to an exception via the standard ``warnings`` filter."""
    import warnings

    from jeevesagent import IsolationWarning

    mem = InMemoryMemory()
    await mem.remember(
        Episode(session_id="s", user_id="alice", input="hi", output="hello")
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", IsolationWarning)
        with pytest.raises(IsolationWarning):
            await mem.recall("hi", user_id=None)


# ---------------------------------------------------------------------------
# M3 — Multi-agent context inheritance
# ---------------------------------------------------------------------------


async def test_subagent_inherits_parent_user_id_via_contextvar() -> None:
    """A sub-agent invoked via ``SubagentInvocation`` (used by every
    multi-agent architecture) must see the parent's ``user_id``
    through ``get_run_context()`` without any explicit plumbing."""
    from jeevesagent.architecture.helpers import SubagentInvocation

    seen: list[str | None] = []

    @tool
    async def report_user() -> str:
        ctx = get_run_context()
        seen.append(ctx.user_id)
        return ctx.user_id or "anonymous"

    # Sub-agent that calls a tool to check what user_id it sees.
    sub = Agent(
        "be brief",
        model=ScriptedModel([
            ScriptedTurn(
                tool_calls=[ToolCall(id="c1", tool="report_user", args={})]
            ),
            ScriptedTurn(text="done"),
        ]),
        tools=[report_user],
    )

    # Parent installs the contextvar via set_run_context, then
    # invokes the sub the way an architecture would.
    parent_ctx = RunContext(user_id="alice", session_id="parent_session")
    async with set_run_context(parent_ctx):
        invocation = SubagentInvocation(sub, "go")
        async for _ in invocation.events():
            pass

    assert seen == ["alice"]


async def test_subagent_gets_fresh_session_id_when_one_provided() -> None:
    """``SubagentInvocation(session_id="...")`` must override the
    parent's session_id so the worker has its own conversation
    thread, even while inheriting the parent's user_id."""
    from jeevesagent.architecture.helpers import SubagentInvocation

    sub_mem = InMemoryMemory()
    sub = Agent(
        "be brief",
        model=ScriptedModel([ScriptedTurn(text="hi")]),
        memory=sub_mem,
    )

    parent_ctx = RunContext(user_id="alice", session_id="parent_session")
    async with set_run_context(parent_ctx):
        invocation = SubagentInvocation(
            sub, "say hi", session_id="worker_session"
        )
        async for _ in invocation.events():
            pass

    # The sub-agent persisted an episode tagged with worker_session,
    # NOT parent_session — and with alice's user_id.
    alice_ep = await sub_mem.recall("hi", user_id="alice")
    assert len(alice_ep) == 1
    assert alice_ep[0].session_id == "worker_session"
