"""ChromaFactStore tests using an EphemeralClient (no persistence).

Skipped automatically if ``chromadb`` isn't installed. Each test gets
its own UUID-suffixed collection because Chroma's ``EphemeralClient``
shares state across instances inside a single process.
"""

from __future__ import annotations

import pytest

pytest.importorskip("chromadb")

import uuid  # noqa: E402
from datetime import UTC, datetime, timedelta  # noqa: E402

from loomflow.core.types import Fact  # noqa: E402
from loomflow.memory import ChromaMemory  # noqa: E402
from loomflow.memory.chroma_facts import ChromaFactStore  # noqa: E402

pytestmark = pytest.mark.anyio


def _fact(
    *,
    subject: str = "user",
    predicate: str = "name_is",
    object_: str = "Alice",
    valid_from: datetime | None = None,
) -> Fact:
    base = valid_from or datetime.now(UTC)
    return Fact(
        subject=subject,
        predicate=predicate,
        object=object_,
        valid_from=base,
        recorded_at=datetime.now(UTC),
    )


def _fresh_store() -> ChromaFactStore:
    return ChromaFactStore.ephemeral(
        collection_name=f"jeeves_test_facts_{uuid.uuid4().hex}",
    )


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


async def test_append_and_query_roundtrip() -> None:
    store = _fresh_store()
    fid = await store.append(
        _fact(subject="user", predicate="name_is", object_="Alice")
    )
    out = await store.query(subject="user")
    assert len(out) == 1
    assert out[0].id == fid
    assert out[0].object == "Alice"


async def test_query_filters_by_predicate_and_object() -> None:
    store = _fresh_store()
    await store.append(_fact(subject="alice", predicate="lives_in", object_="Tokyo"))
    await store.append(_fact(subject="alice", predicate="lives_in", object_="Paris"))
    await store.append(_fact(subject="bob", predicate="lives_in", object_="Tokyo"))

    in_tokyo = await store.query(predicate="lives_in", object_="Tokyo")
    assert {f.subject for f in in_tokyo} == {"alice", "bob"}


async def test_sources_round_trip_through_metadata() -> None:
    store = _fresh_store()
    await store.append(
        Fact(
            subject="x",
            predicate="p",
            object="o",
            valid_from=datetime.now(UTC),
            recorded_at=datetime.now(UTC),
            sources=["ep_a", "ep_b"],
        )
    )
    out = await store.query(subject="x")
    assert out[0].sources == ["ep_a", "ep_b"]


# ---------------------------------------------------------------------------
# Bi-temporal supersession
# ---------------------------------------------------------------------------


async def test_supersession_closes_off_prior_fact() -> None:
    store = _fresh_store()
    base = datetime(2026, 1, 1, tzinfo=UTC)

    f1_id = await store.append(
        _fact(predicate="lives_in", object_="Tokyo", valid_from=base)
    )
    f2_id = await store.append(
        _fact(
            predicate="lives_in",
            object_="Paris",
            valid_from=base + timedelta(days=30),
        )
    )

    by_id = {f.id: f for f in await store.all_facts()}
    assert by_id[f1_id].valid_until is not None
    assert by_id[f2_id].valid_until is None


async def test_supersession_skips_same_object_fact() -> None:
    store = _fresh_store()
    base = datetime(2026, 1, 1, tzinfo=UTC)
    f1 = await store.append(
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
    assert by_id[f1].valid_until is None


async def test_query_at_specific_time_returns_correct_window() -> None:
    store = _fresh_store()
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

    on_jan_10 = base + timedelta(days=10)
    facts = await store.query(predicate="lives_in", valid_at=on_jan_10)
    assert {f.object for f in facts} == {"Tokyo"}

    on_mar_1 = base + timedelta(days=60)
    facts = await store.query(predicate="lives_in", valid_at=on_mar_1)
    assert {f.object for f in facts} == {"Paris"}


# ---------------------------------------------------------------------------
# Embedding-based recall
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    name: str = "fake"
    dimensions: int = 4

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    async def embed(self, text: str) -> list[float]:
        return list(self._mapping.get(text, [0.0] * self.dimensions))

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


async def test_recall_text_uses_cosine_similarity() -> None:
    embedder = _FakeEmbedder(
        {
            "alice loves apples": [1.0, 0.0, 0.0, 0.0],
            "bob hates oranges": [0.0, 1.0, 0.0, 0.0],
            "fruit query": [0.99, 0.1, 0.0, 0.0],
        }
    )
    store = ChromaFactStore.ephemeral(
        embedder=embedder,
        collection_name=f"jeeves_test_facts_{uuid.uuid4().hex}",
    )
    await store.append(
        _fact(subject="alice", predicate="loves", object_="apples")
    )
    await store.append(
        _fact(subject="bob", predicate="hates", object_="oranges")
    )

    out = await store.recall_text("fruit query", limit=1)
    assert len(out) == 1
    assert out[0].subject == "alice"


# ---------------------------------------------------------------------------
# ChromaMemory(with_facts=True) integration
# ---------------------------------------------------------------------------


async def test_chroma_memory_with_facts_attaches_chroma_fact_store() -> None:
    mem = ChromaMemory.ephemeral(
        with_facts=True,
        collection_name=f"jeeves_test_episodes_{uuid.uuid4().hex}",
        facts_collection_name=f"jeeves_test_facts_{uuid.uuid4().hex}",
    )
    assert mem.facts is not None
    # Should be able to round-trip a fact through the attached store.
    await mem.facts.append(
        _fact(subject="user", predicate="name_is", object_="Alice")
    )
    out = await mem.facts.query(subject="user")
    assert len(out) == 1
    assert out[0].object == "Alice"


async def test_chroma_memory_default_has_no_fact_store() -> None:
    mem = ChromaMemory.ephemeral(
        collection_name=f"jeeves_test_episodes_{uuid.uuid4().hex}",
    )
    assert mem.facts is None
