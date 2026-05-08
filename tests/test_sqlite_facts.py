"""SqliteFactStore tests — same shape contract as InMemoryFactStore,
plus persistence across instances against the same DB file."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from jeevesagent import Fact, SqliteFactStore
from jeevesagent.memory.sqlite_facts import _to_epoch

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
    base = valid_from or datetime.now(UTC)
    return Fact(
        subject=subject,
        predicate=predicate,
        object=object_,
        confidence=confidence,
        valid_from=base,
        recorded_at=datetime.now(UTC),
        sources=sources or [],
    )


# ---------------------------------------------------------------------------
# Roundtrip + DDL
# ---------------------------------------------------------------------------


async def test_init_creates_facts_table_and_indexes(tmp_path: Path) -> None:
    db = tmp_path / "f.db"
    store = SqliteFactStore(db)
    # Implicit assertion: no exception. Verify by listing tables.
    import sqlite3

    with sqlite3.connect(db) as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN "
                "('table', 'index')"
            )
        }
    assert "facts" in names
    assert "facts_subject_idx" in names
    assert "facts_user_subject_predicate_idx" in names
    assert store.path == db


async def test_append_and_query_roundtrip(tmp_path: Path) -> None:
    store = SqliteFactStore(tmp_path / "f.db")
    fid = await store.append(_fact())
    out = await store.query(subject="user")
    assert len(out) == 1
    assert out[0].id == fid
    assert out[0].object == "Alice"


async def test_query_by_subject_predicate_object(tmp_path: Path) -> None:
    store = SqliteFactStore(tmp_path / "f.db")
    await store.append(_fact(subject="alice", predicate="lives_in", object_="Tokyo"))
    await store.append(_fact(subject="alice", predicate="lives_in", object_="Paris"))
    await store.append(_fact(subject="bob", predicate="lives_in", object_="Tokyo"))

    in_tokyo = await store.query(predicate="lives_in", object_="Tokyo")
    assert {f.subject for f in in_tokyo} == {"alice", "bob"}


async def test_recall_text_token_overlap(tmp_path: Path) -> None:
    store = SqliteFactStore(tmp_path / "f.db")
    await store.append(_fact(subject="user", predicate="name_is", object_="Alice"))
    await store.append(_fact(subject="user", predicate="works_at", object_="Anthropic"))
    await store.append(_fact(subject="user", predicate="lives_in", object_="Tokyo"))

    found = await store.recall_text("Alice")
    assert len(found) == 1
    assert found[0].object == "Alice"


async def test_sources_round_trip_through_json(tmp_path: Path) -> None:
    store = SqliteFactStore(tmp_path / "f.db")
    await store.append(_fact(sources=["ep_1", "ep_2"]))
    out = await store.query(subject="user")
    assert out[0].sources == ["ep_1", "ep_2"]


# ---------------------------------------------------------------------------
# Bi-temporal supersession
# ---------------------------------------------------------------------------


async def test_supersession_closes_off_prior_fact(tmp_path: Path) -> None:
    store = SqliteFactStore(tmp_path / "f.db")
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
    # The Sqlite store stores valid_until as an epoch float; the loaded
    # value should be equal modulo float<->datetime conversion.
    assert by_id[f1_id].valid_until is not None
    assert abs(
        _to_epoch(by_id[f1_id].valid_until)
        - _to_epoch(base + timedelta(days=30))
    ) < 1.0
    assert by_id[f2_id].valid_until is None


async def test_supersession_preserves_same_object_fact(tmp_path: Path) -> None:
    store = SqliteFactStore(tmp_path / "f.db")
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


async def test_query_at_specific_time(tmp_path: Path) -> None:
    store = SqliteFactStore(tmp_path / "f.db")
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
# Persistence across instances
# ---------------------------------------------------------------------------


async def test_facts_survive_a_fresh_instance(tmp_path: Path) -> None:
    db = tmp_path / "f.db"
    s1 = SqliteFactStore(db)
    fid = await s1.append(_fact(subject="alice", object_="x"))

    s2 = SqliteFactStore(db)  # simulates a process restart
    out = await s2.query(subject="alice")
    assert len(out) == 1
    assert out[0].id == fid
    assert out[0].object == "x"


async def test_supersession_persists_across_restart(tmp_path: Path) -> None:
    db = tmp_path / "f.db"
    base = datetime(2026, 1, 1, tzinfo=UTC)

    s1 = SqliteFactStore(db)
    f1 = await s1.append(
        _fact(predicate="role", object_="engineer", valid_from=base)
    )

    # Process restart…
    s2 = SqliteFactStore(db)
    await s2.append(
        _fact(
            predicate="role",
            object_="manager",
            valid_from=base + timedelta(days=365),
        )
    )

    by_id = {f.id: f for f in await s2.all_facts()}
    assert by_id[f1].valid_until is not None  # closed off across restart


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


async def test_embedding_based_recall_uses_cosine(tmp_path: Path) -> None:
    embedder = _FakeEmbedder(
        {
            "alice loves apples": [1.0, 0.0, 0.0, 0.0],
            "bob hates oranges": [0.0, 1.0, 0.0, 0.0],
            "fruit query": [0.99, 0.1, 0.0, 0.0],
        }
    )
    store = SqliteFactStore(tmp_path / "f.db", embedder=embedder)
    await store.append(
        _fact(subject="alice", predicate="loves", object_="apples")
    )
    await store.append(
        _fact(subject="bob", predicate="hates", object_="oranges")
    )

    out = await store.recall_text("fruit query", limit=1)
    assert len(out) == 1
    assert out[0].subject == "alice"


async def test_embedding_recall_persists_across_instances(tmp_path: Path) -> None:
    embedder = _FakeEmbedder(
        {
            "alice loves apples": [1.0, 0.0, 0.0, 0.0],
            "bob hates oranges": [0.0, 1.0, 0.0, 0.0],
            "fruit": [0.95, 0.05, 0.0, 0.0],
        }
    )
    db = tmp_path / "f.db"
    s1 = SqliteFactStore(db, embedder=embedder)
    await s1.append(
        _fact(subject="alice", predicate="loves", object_="apples")
    )
    await s1.append(
        _fact(subject="bob", predicate="hates", object_="oranges")
    )

    # Restart with same embedder; cosine ranking still works.
    s2 = SqliteFactStore(db, embedder=embedder)
    out = await s2.recall_text("fruit", limit=1)
    assert len(out) == 1
    assert out[0].subject == "alice"
