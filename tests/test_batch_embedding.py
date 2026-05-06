"""Verify the consolidator + InMemoryFactStore batch their embedder
calls into a single ``embed_batch`` round-trip when many facts land
together — instead of N individual ``embed`` calls.
"""

from __future__ import annotations

import pytest

from jeevesagent import Consolidator, InMemoryFactStore, ScriptedModel, ScriptedTurn
from jeevesagent.core.types import Episode

pytestmark = pytest.mark.anyio


class _CountingEmbedder:
    """Embedder that tracks how many times each method is called."""

    name: str = "counting"
    dimensions: int = 4

    def __init__(self) -> None:
        self.embed_calls = 0
        self.embed_batch_calls = 0
        self.embed_batch_total_texts = 0

    async def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        return [float(len(text))] * self.dimensions

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.embed_batch_calls += 1
        self.embed_batch_total_texts += len(texts)
        return [[float(len(t))] * self.dimensions for t in texts]


# ---------------------------------------------------------------------------
# InMemoryFactStore.append_many uses embed_batch
# ---------------------------------------------------------------------------


async def test_append_many_calls_embed_batch_once() -> None:
    embedder = _CountingEmbedder()
    store = InMemoryFactStore(embedder=embedder)

    from datetime import UTC, datetime

    from jeevesagent import Fact

    now = datetime.now(UTC)
    facts = [
        Fact(
            subject=f"s{i}",
            predicate="p",
            object=f"o{i}",
            valid_from=now,
            recorded_at=now,
        )
        for i in range(5)
    ]

    await store.append_many(facts)

    # All 5 embeddings came from a single batch call.
    assert embedder.embed_batch_calls == 1
    assert embedder.embed_batch_total_texts == 5
    # No per-fact embed() calls.
    assert embedder.embed_calls == 0


async def test_append_many_with_no_embedder_skips_embedding() -> None:
    store = InMemoryFactStore()  # no embedder

    from datetime import UTC, datetime

    from jeevesagent import Fact

    now = datetime.now(UTC)
    facts = [
        Fact(
            subject="s",
            predicate="p",
            object=f"o{i}",
            valid_from=now,
            recorded_at=now,
        )
        for i in range(3)
    ]
    ids = await store.append_many(facts)
    assert len(ids) == 3


async def test_append_many_empty_is_noop() -> None:
    embedder = _CountingEmbedder()
    store = InMemoryFactStore(embedder=embedder)
    ids = await store.append_many([])
    assert ids == []
    assert embedder.embed_batch_calls == 0


# ---------------------------------------------------------------------------
# Consolidator uses append_many
# ---------------------------------------------------------------------------


async def test_consolidator_batches_through_append_many() -> None:
    """Two extracted facts from one episode should hit the embedder
    via a single batch call, not two singletons."""
    extracted = (
        '[{"subject":"u","predicate":"p1","object":"o1","confidence":0.9},'
        '{"subject":"u","predicate":"p2","object":"o2","confidence":0.9}]'
    )
    consolidator_model = ScriptedModel([ScriptedTurn(text=extracted)])

    embedder = _CountingEmbedder()
    store = InMemoryFactStore(embedder=embedder)
    consolidator = Consolidator(model=consolidator_model)

    ep = Episode(session_id="s", input="hello", output="ack")
    new_facts = await consolidator.consolidate([ep], store=store)

    assert len(new_facts) == 2
    assert embedder.embed_batch_calls == 1
    assert embedder.embed_batch_total_texts == 2
    assert embedder.embed_calls == 0


async def test_consolidator_falls_back_to_per_fact_append_when_no_append_many() -> None:
    """A custom FactStore without ``append_many`` still works — the
    consolidator falls back to per-fact ``append``."""
    extracted = '[{"subject":"u","predicate":"p","object":"o"}]'
    consolidator_model = ScriptedModel([ScriptedTurn(text=extracted)])

    appended: list[object] = []

    class _AppendOnlyStore:
        async def append(self, fact):  # type: ignore[no-untyped-def]
            appended.append(fact)
            return fact.id

        # Intentionally NO append_many.

        async def query(self, **kwargs):  # type: ignore[no-untyped-def]
            return list(appended)

        async def recall_text(self, query, **kwargs):  # type: ignore[no-untyped-def]
            return []

        async def all_facts(self):  # type: ignore[no-untyped-def]
            return list(appended)

        async def aclose(self) -> None:
            return None

    consolidator = Consolidator(model=consolidator_model)
    ep = Episode(session_id="s", input="x", output="y")
    await consolidator.consolidate([ep], store=_AppendOnlyStore())  # type: ignore[arg-type]
    assert len(appended) == 1
