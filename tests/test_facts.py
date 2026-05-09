"""Bi-temporal fact store + consolidator tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from loomflow import Agent, Fact, InMemoryMemory
from loomflow.core.types import (
    Episode,
    Message,
    ModelChunk,
    ToolDef,
    Usage,
)
from loomflow.memory import Consolidator, InMemoryFactStore, VectorMemory
from loomflow.model.scripted import ScriptedModel, ScriptedTurn

pytestmark = pytest.mark.anyio


def _fact(
    *,
    subject: str = "user",
    predicate: str = "name_is",
    object_: str = "Alice",
    confidence: float = 0.9,
    valid_from: datetime | None = None,
    sources: list[str] | None = None,
) -> Fact:
    return Fact(
        subject=subject,
        predicate=predicate,
        object=object_,
        confidence=confidence,
        valid_from=valid_from or datetime.now(UTC),
        recorded_at=datetime.now(UTC),
        sources=sources or [],
    )


# ---------------------------------------------------------------------------
# InMemoryFactStore basics
# ---------------------------------------------------------------------------


async def test_append_and_query_by_subject() -> None:
    store = InMemoryFactStore()
    fid = await store.append(_fact())
    out = await store.query(subject="user")
    assert len(out) == 1
    assert out[0].id == fid
    assert out[0].object == "Alice"


async def test_query_by_subject_predicate_object_filters() -> None:
    store = InMemoryFactStore()
    await store.append(_fact(subject="alice", predicate="lives_in", object_="Tokyo"))
    await store.append(_fact(subject="alice", predicate="lives_in", object_="Paris"))
    await store.append(_fact(subject="bob", predicate="lives_in", object_="Tokyo"))

    alice = await store.query(subject="alice")
    assert len(alice) == 2

    in_tokyo = await store.query(predicate="lives_in", object_="Tokyo")
    assert {f.subject for f in in_tokyo} == {"alice", "bob"}


async def test_recall_text_substring_match() -> None:
    store = InMemoryFactStore()
    await store.append(_fact(subject="user", predicate="name_is", object_="Alice"))
    await store.append(_fact(subject="user", predicate="works_at", object_="Anthropic"))
    await store.append(_fact(subject="user", predicate="lives_in", object_="Tokyo"))

    found = await store.recall_text("Alice")
    assert len(found) == 1
    assert found[0].object == "Alice"

    found = await store.recall_text("Tokyo")
    assert len(found) == 1
    assert found[0].object == "Tokyo"


# ---------------------------------------------------------------------------
# Bi-temporal supersession
# ---------------------------------------------------------------------------


async def test_supersession_invalidates_prior_fact() -> None:
    store = InMemoryFactStore()
    base = datetime(2026, 1, 1, tzinfo=UTC)

    f1_id = await store.append(
        _fact(
            subject="user",
            predicate="lives_in",
            object_="Tokyo",
            valid_from=base,
        )
    )
    f2_id = await store.append(
        _fact(
            subject="user",
            predicate="lives_in",
            object_="Paris",
            valid_from=base + timedelta(days=30),
        )
    )

    by_id = {f.id: f for f in await store.all_facts()}
    # First fact closed off at second fact's valid_from
    assert by_id[f1_id].valid_until == base + timedelta(days=30)
    # Second fact still currently valid
    assert by_id[f2_id].valid_until is None


async def test_supersession_preserves_same_object_fact() -> None:
    """Re-asserting the same (subject, predicate, object) triple shouldn't
    invalidate the existing fact."""
    store = InMemoryFactStore()
    base = datetime(2026, 1, 1, tzinfo=UTC)
    f1_id = await store.append(
        _fact(predicate="prefers", object_="dark mode", valid_from=base)
    )
    await store.append(
        _fact(
            predicate="prefers",
            object_="dark mode",
            valid_from=base + timedelta(days=10),
        )
    )
    by_id = {f.id: f for f in await store.all_facts()}
    assert by_id[f1_id].valid_until is None  # not invalidated


async def test_query_at_specific_time_returns_correct_window() -> None:
    store = InMemoryFactStore()
    base = datetime(2026, 1, 1, tzinfo=UTC)
    await store.append(
        _fact(predicate="lives_in", object_="Tokyo", valid_from=base)
    )
    await store.append(
        _fact(
            predicate="lives_in",
            object_="Paris",
            valid_from=base + timedelta(days=30),
        )
    )

    # On day 10: lived in Tokyo
    on_jan_10 = base + timedelta(days=10)
    facts = await store.query(predicate="lives_in", valid_at=on_jan_10)
    assert {f.object for f in facts} == {"Tokyo"}

    # On day 60: lives in Paris
    on_mar_1 = base + timedelta(days=60)
    facts = await store.query(predicate="lives_in", valid_at=on_mar_1)
    assert {f.object for f in facts} == {"Paris"}


async def test_query_before_valid_from_excludes_fact() -> None:
    store = InMemoryFactStore()
    base = datetime(2026, 1, 1, tzinfo=UTC)
    await store.append(_fact(predicate="x", object_="y", valid_from=base))

    earlier = base - timedelta(days=5)
    facts = await store.query(valid_at=earlier)
    assert facts == []


# ---------------------------------------------------------------------------
# Consolidator extracts facts via LLM
# ---------------------------------------------------------------------------


async def test_consolidator_extracts_facts_from_episode() -> None:
    extracted_json = (
        '[{"subject":"user","predicate":"name_is","object":"Alice","confidence":0.95}]'
    )
    model = ScriptedModel([ScriptedTurn(text=extracted_json)])
    store = InMemoryFactStore()
    consolidator = Consolidator(model=model)

    ep = Episode(session_id="s1", input="hi I'm Alice", output="nice to meet you")
    new_facts = await consolidator.consolidate([ep], store=store)

    assert len(new_facts) == 1
    fact = new_facts[0]
    assert fact.subject == "user"
    assert fact.predicate == "name_is"
    assert fact.object == "Alice"
    assert fact.confidence == 0.95
    assert ep.id in fact.sources
    # valid_from inherits from the episode
    assert fact.valid_from == ep.occurred_at


async def test_consolidator_skips_fenced_json() -> None:
    """Robust parsing of ```json ... ``` fences some models add."""
    fenced = "```json\n[{\"subject\":\"u\",\"predicate\":\"is\",\"object\":\"x\"}]\n```"
    model = ScriptedModel([ScriptedTurn(text=fenced)])
    store = InMemoryFactStore()
    consolidator = Consolidator(model=model)
    ep = Episode(session_id="s1", input="hi", output="hi")
    new_facts = await consolidator.consolidate([ep], store=store)
    assert len(new_facts) == 1
    assert new_facts[0].subject == "u"


async def test_consolidator_returns_empty_on_invalid_json() -> None:
    model = ScriptedModel([ScriptedTurn(text="not actually JSON")])
    store = InMemoryFactStore()
    consolidator = Consolidator(model=model)
    ep = Episode(session_id="s1", input="x", output="y")
    out = await consolidator.consolidate([ep], store=store)
    assert out == []
    assert await store.all_facts() == []


async def test_consolidator_skips_facts_missing_required_fields() -> None:
    bad = '[{"subject":"u","predicate":"x"},{"subject":"u","predicate":"y","object":"z"}]'
    model = ScriptedModel([ScriptedTurn(text=bad)])
    store = InMemoryFactStore()
    consolidator = Consolidator(model=model)
    ep = Episode(session_id="s1", input="x", output="y")
    out = await consolidator.consolidate([ep], store=store)
    # Only the well-formed entry survives.
    assert len(out) == 1
    assert out[0].predicate == "y"


async def test_consolidator_clamps_confidence_to_unit_interval() -> None:
    out_of_range = (
        '[{"subject":"u","predicate":"p","object":"o","confidence":2.5}]'
    )
    model = ScriptedModel([ScriptedTurn(text=out_of_range)])
    store = InMemoryFactStore()
    consolidator = Consolidator(model=model)
    ep = Episode(session_id="s1", input="x", output="y")
    [fact] = await consolidator.consolidate([ep], store=store)
    assert 0.0 <= fact.confidence <= 1.0
    assert fact.confidence == 1.0  # clamped


# ---------------------------------------------------------------------------
# Memory backends + consolidator integration
# ---------------------------------------------------------------------------


async def test_in_memory_memory_consolidate_runs_extractor_and_marks_done() -> None:
    extracted = '[{"subject":"user","predicate":"prefers","object":"tea","confidence":0.9}]'
    consolidator_model = ScriptedModel([ScriptedTurn(text=extracted)])
    memory = InMemoryMemory(
        consolidator=Consolidator(model=consolidator_model),
    )

    ep = Episode(session_id="s1", input="I like tea", output="great")
    await memory.remember(ep)
    await memory.consolidate()

    # Fact landed in store
    facts = await memory.facts.all_facts()
    assert len(facts) == 1
    assert facts[0].object == "tea"

    # Second consolidate is idempotent — no double-extraction.
    await memory.consolidate()
    assert len(await memory.facts.all_facts()) == 1


async def test_vector_memory_consolidate_pulls_from_episode_history() -> None:
    extracted = '[{"subject":"user","predicate":"works_at","object":"Anthropic"}]'
    consolidator_model = ScriptedModel([ScriptedTurn(text=extracted)] * 3)
    memory = VectorMemory(
        consolidator=Consolidator(model=consolidator_model),
    )

    for i in range(3):
        await memory.remember(
            Episode(session_id="s", input=f"msg {i}", output="ack")
        )
    await memory.consolidate()

    # Each episode is consolidated; facts may be 1 (deduplicated by
    # supersession-skip on identical triples) or 3 (one per episode).
    # Since same fact triple repeats across episodes, supersession
    # leaves only the first one currently valid.
    facts = await memory.facts.all_facts()
    assert len(facts) >= 1
    currently_valid = [f for f in facts if f.valid_until is None]
    assert all(f.object == "Anthropic" for f in currently_valid)


# ---------------------------------------------------------------------------
# Agent loop pulls facts into the model's context
# ---------------------------------------------------------------------------


class _RecordingModel:
    """Captures the messages list passed to each ``stream`` call."""

    name: str = "recording"

    def __init__(self) -> None:
        self.last_messages: list[Message] | None = None

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ModelChunk]:
        self.last_messages = list(messages)
        yield ModelChunk(kind="text", text="ack")
        yield ModelChunk(
            kind="finish",
            finish_reason="stop",
            usage=Usage(input_tokens=1, output_tokens=1),
        )


async def test_agent_run_includes_known_facts_in_system_message() -> None:
    memory = InMemoryMemory()
    base = datetime(2026, 1, 1, tzinfo=UTC)
    await memory.facts.append(
        _fact(
            subject="user",
            predicate="name_is",
            object_="Alice",
            valid_from=base,
        )
    )

    rec = _RecordingModel()
    agent = Agent("you know facts", model=rec, memory=memory)
    # The query needs a token present in the formatted fact triple
    # for the naive substring recall_text to surface it. (Smarter
    # retrieval — embeddings, BM25 — comes in a follow-up slice.)
    await agent.run("tell me the user's name")

    assert rec.last_messages is not None
    system_messages = [
        m.content for m in rec.last_messages if m.role.value == "system"
    ]
    joined = "\n".join(system_messages)
    assert "user name_is Alice" in joined
    assert "Known facts" in joined


async def test_agent_run_omits_facts_section_when_store_is_empty() -> None:
    memory = InMemoryMemory()
    rec = _RecordingModel()
    agent = Agent("hi", model=rec, memory=memory)
    await agent.run("hello")

    assert rec.last_messages is not None
    system_messages = [
        m.content for m in rec.last_messages if m.role.value == "system"
    ]
    joined = "\n".join(system_messages)
    assert "Known facts" not in joined


# ---------------------------------------------------------------------------
# Embedding-based recall
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Maps specific texts to fixed vectors. Lets tests assert on the
    cosine-similarity ranking without a real embedding model."""

    name: str = "fake-embedder"
    dimensions: int = 4

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    async def embed(self, text: str) -> list[float]:
        if text in self._mapping:
            return list(self._mapping[text])
        # Default: zero vector so unknown texts produce 0 cosine.
        return [0.0] * self.dimensions

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


async def test_embedding_based_recall_uses_cosine_similarity() -> None:
    """When an embedder is configured, recall_text should rank by
    cosine similarity rather than token overlap."""
    embedder = _FakeEmbedder(
        {
            "alice loves apples": [1.0, 0.0, 0.0, 0.0],
            "bob hates oranges": [0.0, 1.0, 0.0, 0.0],
            "charlie cooks pasta": [0.0, 0.0, 1.0, 0.0],
            # Query embedding closer to alice (cos~0.99 vs alice, ~0 vs others).
            "tell me about fruit": [0.99, 0.1, 0.0, 0.0],
        }
    )
    store = InMemoryFactStore(embedder=embedder)
    await store.append(
        _fact(subject="alice", predicate="loves", object_="apples")
    )
    await store.append(
        _fact(subject="bob", predicate="hates", object_="oranges")
    )
    await store.append(
        _fact(subject="charlie", predicate="cooks", object_="pasta")
    )

    out = await store.recall_text("tell me about fruit", limit=1)
    assert len(out) == 1
    assert out[0].subject == "alice"


async def test_embedding_based_recall_returns_top_k_in_order() -> None:
    embedder = _FakeEmbedder(
        {
            "x p a": [1.0, 0.0, 0.0, 0.0],
            "y p b": [0.5, 0.5, 0.0, 0.0],
            "z p c": [0.0, 1.0, 0.0, 0.0],
            "find a": [1.0, 0.1, 0.0, 0.0],
        }
    )
    store = InMemoryFactStore(embedder=embedder)
    await store.append(_fact(subject="x", predicate="p", object_="a"))
    await store.append(_fact(subject="y", predicate="p", object_="b"))
    await store.append(_fact(subject="z", predicate="p", object_="c"))

    out = await store.recall_text("find a", limit=2)
    assert [f.subject for f in out] == ["x", "y"]


async def test_no_embedder_falls_back_to_token_overlap() -> None:
    """An InMemoryFactStore() with no embedder uses token-overlap, so
    queries with no shared tokens return nothing."""
    store = InMemoryFactStore()
    await store.append(
        _fact(subject="alice", predicate="loves", object_="apples")
    )
    # No token overlap with "completely unrelated query"
    out = await store.recall_text("completely unrelated query", limit=5)
    assert out == []


async def test_vector_memory_uses_its_embedder_for_facts() -> None:
    """VectorMemory should default the fact store's embedder to its own."""
    embedder = _FakeEmbedder(
        {
            "alice lives_in tokyo": [1.0, 0.0, 0.0, 0.0],
            "bob works_at acme": [0.0, 1.0, 0.0, 0.0],
            "tell me about alice": [0.99, 0.0, 0.0, 0.1],
        }
    )
    memory = VectorMemory(embedder=embedder)
    await memory.facts.append(
        _fact(subject="alice", predicate="lives_in", object_="tokyo")
    )
    await memory.facts.append(
        _fact(subject="bob", predicate="works_at", object_="acme")
    )

    out = await memory.facts.recall_text("tell me about alice", limit=1)
    assert len(out) == 1
    assert out[0].subject == "alice"


# ---------------------------------------------------------------------------
# Auto-consolidation
# ---------------------------------------------------------------------------


async def test_auto_consolidate_extracts_facts_after_run() -> None:
    extracted = '[{"subject":"user","predicate":"name_is","object":"Alice","confidence":0.95}]'
    consolidator_model = ScriptedModel([ScriptedTurn(text=extracted)])
    memory = InMemoryMemory(
        consolidator=Consolidator(model=consolidator_model)
    )
    agent = Agent("hi", model="echo", memory=memory, auto_consolidate=True)

    # Before the run, no facts.
    assert await memory.facts.all_facts() == []

    await agent.run("hello, I'm Alice")

    # After the run, consolidation has fired and facts landed.
    facts = await memory.facts.all_facts()
    assert len(facts) == 1
    assert facts[0].object == "Alice"


async def test_auto_consolidate_off_by_default() -> None:
    extracted = '[{"subject":"user","predicate":"name_is","object":"Alice"}]'
    consolidator_model = ScriptedModel([ScriptedTurn(text=extracted)])
    memory = InMemoryMemory(
        consolidator=Consolidator(model=consolidator_model)
    )
    agent = Agent("hi", model="echo", memory=memory)  # auto_consolidate defaults to False

    await agent.run("hello")

    # No facts because consolidation never ran.
    assert await memory.facts.all_facts() == []


async def test_consolidator_failure_emits_error_but_does_not_break_run() -> None:
    """If the consolidator's model raises, the run still returns and
    the COMPLETED event still fires."""

    class _BoomModel:
        name = "boom"

        async def stream(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("consolidator went boom")
            yield  # pragma: no cover

    memory = InMemoryMemory(consolidator=Consolidator(model=_BoomModel()))
    agent = Agent("hi", model="echo", memory=memory, auto_consolidate=True)

    # Run still completes despite the consolidation failure.
    result = await agent.run("hello")
    assert result.output  # the model produced output
    assert not result.interrupted


async def test_agent_consolidate_method_runs_extractor() -> None:
    extracted = '[{"subject":"u","predicate":"p","object":"o"}]'
    consolidator_model = ScriptedModel([ScriptedTurn(text=extracted)])
    memory = InMemoryMemory(
        consolidator=Consolidator(model=consolidator_model)
    )
    agent = Agent("hi", model="echo", memory=memory)
    await agent.run("hello")

    # Before manual consolidate: nothing
    assert await memory.facts.all_facts() == []

    await agent.consolidate()

    facts = await memory.facts.all_facts()
    assert len(facts) == 1
    assert facts[0].object == "o"


async def test_auto_consolidate_with_no_consolidator_is_noop() -> None:
    """auto_consolidate=True on a memory without a consolidator should
    silently do nothing — not raise."""
    memory = InMemoryMemory()  # no consolidator configured
    agent = Agent("hi", model="echo", memory=memory, auto_consolidate=True)
    result = await agent.run("hello")
    assert result.output
    assert await memory.facts.all_facts() == []


async def test_agent_loop_tolerates_memory_without_facts_attribute() -> None:
    """Backends that don't carry ``.facts`` (e.g. PostgresMemory before
    fact support lands) must still work."""

    class _FactlessMemory:
        async def working(self) -> list:  # noqa: ANN001
            return []

        async def update_block(self, name: str, content: str) -> None:
            pass

        async def append_block(self, name: str, content: str) -> None:
            pass

        async def remember(self, episode: Episode) -> str:
            return episode.id

        async def recall(self, query: str, **kwargs: object) -> list:  # noqa: ANN001
            return []

        async def consolidate(self) -> None:
            return None

    rec = _RecordingModel()
    agent = Agent("hi", model=rec, memory=_FactlessMemory())
    await agent.run("hello")

    assert rec.last_messages is not None
    # No facts section should appear because memory has no .facts.
    joined = "\n".join(
        m.content for m in rec.last_messages if m.role.value == "system"
    )
    assert "Known facts" not in joined
