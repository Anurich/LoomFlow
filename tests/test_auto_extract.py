"""Tests for the M8 auto-extraction wrapper.

Covers:

* :class:`AutoExtractMemory` wraps the protocol cleanly — every
  method forwards through to the inner backend
* Calling ``remember(episode)`` triggers a Consolidator pass that
  appends extracted facts to ``inner.facts`` and tags them with
  the episode's ``user_id``
* Extraction failures are best-effort — they NEVER break the run;
  the underlying ``remember`` succeeds regardless
* Agent's ``auto_extract=`` kwarg picks the right default for
  in-tree network adapters vs test fakes
"""

from __future__ import annotations

from typing import Any

import pytest

from jeevesagent import Agent, Episode, Fact
from jeevesagent.core.types import ModelChunk, Usage
from jeevesagent.memory.auto_extract import AutoExtractMemory
from jeevesagent.memory.consolidator import Consolidator
from jeevesagent.memory.inmemory import InMemoryMemory

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Test models — return canned consolidator output
# ---------------------------------------------------------------------------


class _ExtractingModel:
    """Model that returns a canned JSON array of facts. Used so the
    Consolidator's parse path runs end-to-end without a real LLM."""

    name = "extracting"

    def __init__(self, facts_json: str) -> None:
        self._facts_json = facts_json
        self.calls = 0

    async def stream(self, messages: Any, **kwargs: Any) -> Any:
        self.calls += 1
        # Stream a single text chunk + finish.
        yield ModelChunk(kind="text", text=self._facts_json)
        yield ModelChunk(kind="finish", finish_reason="stop", usage=Usage())


class _FailingModel:
    """Model that always raises on stream; verifies that extraction
    failures don't break ``remember``."""

    name = "failing"

    async def stream(self, messages: Any, **kwargs: Any) -> Any:
        raise RuntimeError("model is down")
        yield  # unreachable; satisfies the AsyncIterator protocol typing


# ---------------------------------------------------------------------------
# Wrapper behaviour
# ---------------------------------------------------------------------------


async def test_remember_extracts_and_appends_facts() -> None:
    """The headline contract: an episode goes in, extracted facts
    arrive in the inner fact store under the episode's ``user_id``."""
    inner = InMemoryMemory()
    facts_json = (
        '[{"subject": "alice", "predicate": "lives_in", '
        '"object": "Berlin", "confidence": 0.9}]'
    )
    consolidator = Consolidator(model=_ExtractingModel(facts_json))
    wrapped = AutoExtractMemory(inner, consolidator)

    await wrapped.remember(
        Episode(
            session_id="s",
            user_id="alice",
            input="I just moved to Berlin.",
            output="Got it!",
        )
    )

    # Fact is in the underlying store, partitioned to alice.
    alice_facts = await inner.facts.query(user_id="alice", limit=10)
    bob_facts = await inner.facts.query(user_id="bob", limit=10)
    assert len(alice_facts) == 1
    assert alice_facts[0].subject == "alice"
    assert alice_facts[0].object == "Berlin"
    assert alice_facts[0].user_id == "alice"  # inherits from episode
    assert bob_facts == []  # partition holds


async def test_extraction_failure_does_not_break_remember() -> None:
    """The wrapper is a best-effort enhancement; the underlying
    ``remember`` ALWAYS completes even if extraction blows up."""
    inner = InMemoryMemory()
    consolidator = Consolidator(model=_FailingModel())
    wrapped = AutoExtractMemory(inner, consolidator)

    # No exception expected.
    eid = await wrapped.remember(
        Episode(
            session_id="s",
            user_id="alice",
            input="anything",
            output="ok",
        )
    )
    assert eid  # episode persisted
    # Episode is in the inner store.
    episodes = await inner.recall("anything", user_id="alice")
    assert len(episodes) == 1


async def test_no_extraction_when_inner_has_no_fact_store() -> None:
    """The wrapper installs cleanly even when ``inner.facts is None``
    — extraction silently skips, ``remember`` still works."""

    class _NoFactsMemory(InMemoryMemory):
        def __init__(self) -> None:
            super().__init__()
            self.facts = None  # type: ignore[assignment]

    inner = _NoFactsMemory()
    consolidator_calls = []

    class _RecordingConsolidator(Consolidator):
        async def consolidate(  # type: ignore[override]
            self, episodes: Any, *, store: Any
        ) -> list[Fact]:
            consolidator_calls.append(1)
            return []

    wrapped = AutoExtractMemory(
        inner,
        _RecordingConsolidator(model=_ExtractingModel("[]")),
    )
    await wrapped.remember(
        Episode(session_id="s", user_id="alice", input="x", output="y")
    )
    # Skipped because inner.facts is None.
    assert consolidator_calls == []


async def test_wrapper_forwards_protocol_methods() -> None:
    """Every Memory method on the wrapper hits the inner backend.
    Sanity-check; doesn't enumerate every method but catches the
    common ones."""
    inner = InMemoryMemory()
    wrapped = AutoExtractMemory(
        inner, Consolidator(model=_ExtractingModel("[]"))
    )

    await wrapped.update_block("preferences", "dark mode")
    blocks = await wrapped.working()
    assert any(b.name == "preferences" for b in blocks)

    await wrapped.remember(
        Episode(
            session_id="s", user_id="alice",
            input="my favourite is jazz", output="ok",
        )
    )
    msgs = await wrapped.session_messages("s", user_id="alice")
    assert any(m.content == "my favourite is jazz" for m in msgs)
    profile = await wrapped.profile(user_id="alice")
    assert profile.episode_count == 1


async def test_wrapper_facts_property_forwards() -> None:
    """``wrapped.facts`` is the same FactStore as ``inner.facts``,
    so power users can hit the bi-temporal store directly without
    knowing about the wrapper."""
    inner = InMemoryMemory()
    wrapped = AutoExtractMemory(
        inner, Consolidator(model=_ExtractingModel("[]"))
    )
    assert wrapped.facts is inner.facts


# ---------------------------------------------------------------------------
# Agent integration — default policy
# ---------------------------------------------------------------------------


async def test_agent_does_not_auto_wrap_scripted_model_by_default() -> None:
    """ScriptedModel / EchoModel are test fakes; auto-extract default
    is OFF for them so tests stay deterministic. Verified by the
    type of ``agent._wrapped_memory`` — should equal
    ``agent._memory``, no AutoExtractMemory layer."""
    from jeevesagent.model.scripted import ScriptedModel, ScriptedTurn

    agent = Agent(
        "...", model=ScriptedModel([ScriptedTurn(text="ok")])
    )
    assert agent._memory is agent._wrapped_memory
    assert "AutoExtractMemory" not in type(agent._wrapped_memory).__name__


async def test_agent_explicit_auto_extract_true_wraps_even_on_fakes() -> None:
    """Explicit opt-in works regardless of model type — the
    framework respects the caller's choice."""
    from jeevesagent.model.scripted import ScriptedModel, ScriptedTurn

    agent = Agent(
        "...",
        model=ScriptedModel([ScriptedTurn(text="ok")]),
        auto_extract=True,
    )
    assert isinstance(agent._wrapped_memory, AutoExtractMemory)
    assert agent._memory is not agent._wrapped_memory


async def test_agent_explicit_auto_extract_false_skips_wrapping() -> None:
    """Explicit opt-out also works for the case where someone is
    already running a real network model but wants today's
    behaviour back."""
    from jeevesagent.model.scripted import ScriptedModel, ScriptedTurn

    agent = Agent(
        "...",
        model=ScriptedModel([ScriptedTurn(text="ok")]),
        auto_extract=False,
    )
    assert agent._memory is agent._wrapped_memory


async def test_agent_memory_property_returns_inner_not_wrapper() -> None:
    """``agent.memory`` (the public introspection accessor) returns
    the user-supplied / resolver-built backend — not the
    auto-extract wrapper. Tests + ``agent.memory.profile(...)``
    style code can keep working as if no wrapping happened."""
    from jeevesagent.model.scripted import ScriptedModel, ScriptedTurn

    inner = InMemoryMemory()
    agent = Agent(
        "...",
        model=ScriptedModel([ScriptedTurn(text="ok")]),
        memory=inner,
        auto_extract=True,
    )
    # ``_memory`` (which ``agent.memory`` returns) is the inner;
    # ``_wrapped_memory`` is the autopilot.
    assert agent.memory is inner
    assert isinstance(agent._wrapped_memory, AutoExtractMemory)


# ---------------------------------------------------------------------------
# End-to-end with a scripted model that returns extractable JSON
# ---------------------------------------------------------------------------


async def test_agent_run_auto_extracts_facts_into_memory() -> None:
    """The headline UX: one ``agent.run`` and structured facts
    appear in memory, partitioned by user_id, ready for the next
    run's recall to surface."""
    from jeevesagent.core.types import ToolCall as _Unused  # noqa: F401

    # The agent's own model + the consolidator's model are the
    # SAME instance here (default behaviour). The Scripted model
    # returns its scripted text for the agent turn first, then the
    # JSON for the auto-extract pass.
    class _DualScripted:
        name = "dual"

        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages: Any, **kwargs: Any) -> Any:
            self.calls += 1
            # Agent loop's call.
            return ("ok", [], Usage(input_tokens=5, output_tokens=1), "stop")

        async def stream(self, messages: Any, **kwargs: Any) -> Any:
            self.calls += 1
            # Consolidator's stream call — return a JSON fact array.
            facts = (
                '[{"subject": "alice", "predicate": "loves", '
                '"object": "jazz", "confidence": 0.9}]'
            )
            yield ModelChunk(kind="text", text=facts)
            yield ModelChunk(
                kind="finish", finish_reason="stop", usage=Usage()
            )

    inner = InMemoryMemory()
    agent = Agent(
        "Be helpful.",
        model=_DualScripted(),
        memory=inner,
        auto_extract=True,
    )
    await agent.run("I love jazz music.", user_id="alice")

    # The fact extracted from the run is in the inner store under
    # alice's partition.
    facts = await inner.facts.query(user_id="alice", limit=10)
    assert any(
        f.subject == "alice" and f.predicate == "loves" and f.object == "jazz"
        for f in facts
    )
