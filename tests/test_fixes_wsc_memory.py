"""Regression tests for the reviewed memory/vectorstore fixes (WSC).

Covers:

1.  ``InMemoryFactStore.append_many`` cross-tenant supersession guard
2.  Shared anonymous-bucket encoding (``memory._user_key``)
3.  Redis FT schema ``user_id`` TAG + hybrid KNN + FT.ALTER migration
4.  Real vector scores in ``recall_scored`` (sqlite / redis / postgres)
5.  One-shot HashEmbedder-fallback warning
6.  ``FactStore.delete`` / ``count`` + GDPR forget/profile delegation
7.  SQLite WAL / busy_timeout pragmas
8.  Postgres session index + batched ``append_many``
10. Redis batched UNLINK on forget
"""

from __future__ import annotations

import sqlite3
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from loomflow.core.types import Episode, Fact
from loomflow.memory._user_key import (
    ANON_USER_ID,
    decode_legacy_user_id,
    decode_user_id,
    encode_user_id,
    user_id_where_clause,
)
from loomflow.memory.consolidator import Consolidator
from loomflow.memory.embedder import HashEmbedder
from loomflow.memory.facts import InMemoryFactStore
from loomflow.memory.inmemory import InMemoryMemory
from loomflow.memory.postgres import PostgresMemory
from loomflow.memory.postgres_facts import PostgresFactStore
from loomflow.memory.redis import RedisMemory, _escape_tag
from loomflow.memory.redis_facts import RedisFactStore
from loomflow.memory.sqlite import SqliteMemory
from loomflow.memory.sqlite_facts import SqliteFactStore
from loomflow.model.scripted import ScriptedModel, ScriptedTurn

pytestmark = pytest.mark.anyio


def _fact(
    *,
    user_id: str | None = None,
    subject: str = "user",
    predicate: str = "lives_in",
    object_: str = "Tokyo",
    valid_from: datetime | None = None,
    recorded_at: datetime | None = None,
) -> Fact:
    return Fact(
        user_id=user_id,
        subject=subject,
        predicate=predicate,
        object=object_,
        valid_from=valid_from or datetime.now(UTC),
        recorded_at=recorded_at or datetime.now(UTC),
    )


def _ep(
    text_in: str,
    *,
    user_id: str | None = None,
    session_id: str = "s",
    occurred_at: datetime | None = None,
) -> Episode:
    return Episode(
        session_id=session_id,
        user_id=user_id,
        input=text_in,
        output="",
        occurred_at=occurred_at or datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Fix 1 — append_many must not supersede ANOTHER tenant's facts
# ---------------------------------------------------------------------------


async def test_append_many_does_not_cross_supersede_users() -> None:
    store = InMemoryFactStore()
    base = datetime(2026, 1, 1, tzinfo=UTC)
    await store.append_many(
        [
            _fact(user_id="alice", object_="Tokyo", valid_from=base),
            _fact(
                user_id="bob",
                object_="Paris",
                valid_from=base + timedelta(days=1),
            ),
        ]
    )
    [alice] = await store.query(user_id="alice", subject="user")
    [bob] = await store.query(user_id="bob", subject="user")
    assert alice.valid_until is None, "alice's fact was cross-superseded"
    assert bob.valid_until is None


async def test_append_many_still_supersedes_within_same_user() -> None:
    store = InMemoryFactStore(embedder=HashEmbedder(dimensions=8))
    base = datetime(2026, 1, 1, tzinfo=UTC)
    f1 = _fact(user_id="alice", object_="Tokyo", valid_from=base)
    f2 = _fact(
        user_id="alice",
        object_="Paris",
        valid_from=base + timedelta(days=30),
    )
    await store.append_many([f1, f2])
    by_id = {f.id: f for f in await store.all_facts()}
    assert by_id[f1.id].valid_until == base + timedelta(days=30)
    assert by_id[f2.id].valid_until is None


async def test_consolidator_batch_isolated_across_tenants() -> None:
    """End-to-end: the Consolidator routes through ``append_many`` —
    an extracted batch spanning two users must not let one user's
    fact close the other's."""
    tokyo = '[{"subject":"user","predicate":"lives_in","object":"Tokyo"}]'
    paris = '[{"subject":"user","predicate":"lives_in","object":"Paris"}]'
    model = ScriptedModel([ScriptedTurn(text=tokyo), ScriptedTurn(text=paris)])
    store = InMemoryFactStore()
    consolidator = Consolidator(model=model)

    base = datetime(2026, 1, 1, tzinfo=UTC)
    eps = [
        _ep("i live in tokyo", user_id="alice", occurred_at=base),
        _ep(
            "i live in paris",
            user_id="bob",
            occurred_at=base + timedelta(days=1),
        ),
    ]
    await consolidator.consolidate(eps, store=store)

    [alice] = await store.query(user_id="alice", subject="user")
    [bob] = await store.query(user_id="bob", subject="user")
    assert alice.valid_until is None
    assert bob.valid_until is None


# ---------------------------------------------------------------------------
# Fix 2 — shared anonymous-bucket encoding
# ---------------------------------------------------------------------------


def test_encode_user_id_sentinel_and_rejection() -> None:
    assert encode_user_id(None) == ANON_USER_ID
    assert encode_user_id("alice") == "alice"
    with pytest.raises(ValueError, match="reserved"):
        encode_user_id(ANON_USER_ID)


def test_decode_user_id_strict_vs_legacy() -> None:
    assert decode_user_id(ANON_USER_ID) is None
    assert decode_user_id("") == ""  # postgres migrated "" to sentinel
    assert decode_legacy_user_id(ANON_USER_ID) is None
    assert decode_legacy_user_id("") is None  # legacy anon rows
    assert decode_legacy_user_id("alice") == "alice"


def test_user_id_where_clause_shapes() -> None:
    assert user_id_where_clause("alice") == {"user_id": "alice"}
    anon = user_id_where_clause(None)
    assert anon == {"user_id": {"$in": [ANON_USER_ID, ""]}}


# ---------------------------------------------------------------------------
# Fake redis client
# ---------------------------------------------------------------------------


class _FakeRedis:
    """Minimal redis.asyncio-shaped fake with configurable
    ``execute_command`` behaviour + UNLINK/DEL recording."""

    def __init__(
        self,
        *,
        ft_create_error: str | None = None,
        ft_search_result: Any = None,
    ) -> None:
        self.hashes: dict[bytes, dict[bytes, bytes]] = {}
        self.commands: list[tuple[Any, ...]] = []
        self.unlink_calls: list[tuple[Any, ...]] = []
        self._ft_create_error = ft_create_error
        self._ft_search_result = (
            ft_search_result if ft_search_result is not None else [0]
        )

    async def hset(self, key: Any, *, mapping: dict[Any, Any]) -> int:
        kbytes = key if isinstance(key, bytes) else str(key).encode("utf-8")
        bucket = self.hashes.setdefault(kbytes, {})
        for k, v in mapping.items():
            kb = k if isinstance(k, bytes) else str(k).encode("utf-8")
            vb = v if isinstance(v, bytes | bytearray) else str(v).encode("utf-8")
            bucket[kb] = bytes(vb)
        return 1

    async def hgetall(self, key: Any) -> dict[bytes, bytes]:
        kb = key if isinstance(key, bytes) else str(key).encode("utf-8")
        return dict(self.hashes.get(kb, {}))

    async def scan(
        self, *, cursor: int, match: bytes
    ) -> tuple[int, list[bytes]]:
        prefix = match.rstrip(b"*")
        keys = [k for k in self.hashes if k.startswith(prefix)]
        return 0, keys

    async def execute_command(self, *args: Any) -> Any:
        self.commands.append(args)
        if args[0] == "FT.CREATE" and self._ft_create_error:
            raise Exception(self._ft_create_error)  # noqa: TRY002
        if args[0] == "FT.SEARCH":
            return self._ft_search_result
        return b"OK"

    async def unlink(self, *keys: Any) -> int:
        self.unlink_calls.append(keys)
        removed = 0
        for key in keys:
            kb = key if isinstance(key, bytes) else str(key).encode("utf-8")
            if self.hashes.pop(kb, None) is not None:
                removed += 1
        return removed

    async def delete(self, *keys: Any) -> int:
        return await self.unlink(*keys)

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Fix 2/3 — Redis sentinel writes, legacy reads, TAG schema, hybrid KNN
# ---------------------------------------------------------------------------


async def test_redis_fact_store_writes_sentinel_reads_legacy() -> None:
    fake = _FakeRedis()
    store = RedisFactStore(fake, embedder=HashEmbedder(dimensions=4))
    fid = await store.append(_fact(user_id=None, object_="Tokyo"))
    bucket = fake.hashes[f"jeeves:fact:{fid}".encode()]
    assert bucket[b"user_id"] == ANON_USER_ID.encode()

    # Legacy row written by an older version with the "" encoding —
    # must still land in the anonymous bucket on read.
    await fake.hset(
        "jeeves:fact:legacy1",
        mapping={
            b"id": b"legacy1",
            b"user_id": b"",
            b"subject": b"user",
            b"predicate": b"prefers",
            b"object": b"tea",
            b"confidence": b"1.0",
            b"valid_from_ts": b"1700000000.0",
            b"valid_until_ts": b"0",
            b"currently_valid": b"1",
            b"recorded_at_ts": b"1700000000.0",
            b"sources": b"[]",
        },
    )
    anon = await store.query(user_id=None, limit=10)
    assert {f.id for f in anon} == {fid, "legacy1"}
    # Named users never see the anonymous bucket.
    assert await store.query(user_id="alice", limit=10) == []


async def test_redis_episode_write_uses_sentinel_and_legacy_reads() -> None:
    fake = _FakeRedis()
    mem = RedisMemory(
        fake, embedder=HashEmbedder(dimensions=8), use_vector_index=False
    )
    eid = await mem.remember(_ep("hello", user_id=None))
    bucket = fake.hashes[f"jeeves:episode:{eid}".encode()]
    assert bucket[b"user_id"] == ANON_USER_ID.encode()

    # Seed a legacy episode with the "" encoding.
    await fake.hset(
        "jeeves:episode:legacy-ep",
        mapping={
            b"id": b"legacy-ep",
            b"session_id": b"s",
            b"user_id": b"",
            b"occurred_at": b"1700000000.0",
            b"input": b"old data",
            b"output": b"",
        },
    )
    recent = await mem.recall("", limit=10, user_id=None)
    assert {e.id for e in recent} == {eid, "legacy-ep"}


async def test_redis_ft_create_schema_indexes_user_id_tag() -> None:
    fake = _FakeRedis()
    mem = RedisMemory(
        fake, embedder=HashEmbedder(dimensions=8), use_vector_index=True
    )
    await mem.ensure_index()
    create = next(c for c in fake.commands if c[0] == "FT.CREATE")
    schema = list(create[create.index("SCHEMA"):])
    idx = schema.index("user_id")
    assert schema[idx + 1] == "TAG"


async def test_redis_existing_index_migrated_via_ft_alter() -> None:
    fake = _FakeRedis(ft_create_error="Index already exists")
    mem = RedisMemory(
        fake, embedder=HashEmbedder(dimensions=8), use_vector_index=True
    )
    await mem.ensure_index()
    alter = next((c for c in fake.commands if c[0] == "FT.ALTER"), None)
    assert alter is not None
    assert "user_id" in alter and "TAG" in alter
    # Vector mode stays enabled — "already exists" is not a failure.
    assert mem._use_vector_index is True


async def test_redis_knn_query_carries_user_tag_filter() -> None:
    fake = _FakeRedis()
    mem = RedisMemory(
        fake, embedder=HashEmbedder(dimensions=8), use_vector_index=True
    )
    await mem.recall("apples", user_id="alice", limit=3)
    search = next(c for c in fake.commands if c[0] == "FT.SEARCH")
    query_string = search[2]
    assert query_string.startswith("(@user_id:{alice})=>[KNN")

    # Anonymous bucket uses the sentinel tag.
    await mem.recall("apples", user_id=None, limit=3)
    search2 = [c for c in fake.commands if c[0] == "FT.SEARCH"][-1]
    assert f"(@user_id:{{{ANON_USER_ID}}})" in search2[2]


def test_escape_tag_escapes_punctuation() -> None:
    assert _escape_tag("user-1@x.io") == "user\\-1\\@x\\.io"
    assert _escape_tag("plain_id_9") == "plain_id_9"


# ---------------------------------------------------------------------------
# Fix 4 — real vector scores in recall_scored
# ---------------------------------------------------------------------------


async def test_sqlite_recall_scored_returns_real_cosine(
    tmp_path: Path,
) -> None:
    mem = SqliteMemory(
        tmp_path / "m.db", embedder=HashEmbedder(dimensions=32)
    )
    await mem.remember(_ep("apples are red", user_id="alice"))
    await mem.remember(_ep("dogs bark loudly", user_id="alice"))

    matches = await mem.recall_scored(
        "apples are red", user_id="alice", limit=2
    )
    assert len(matches) == 2
    top = matches[0]
    assert top.episode.input == "apples are red"
    # ``score`` is the RRF-fused rank score; the raw cosine rides on
    # ``vector_score`` — an exact text match embeds identically.
    assert top.vector_score == pytest.approx(1.0, abs=1e-6)
    assert top.score > 0.0
    # No fake neutral 1.0s: the fused score differs across ranks.
    assert matches[1].score < top.score


async def test_redis_recall_scored_brute_force_real_cosine() -> None:
    fake = _FakeRedis()
    mem = RedisMemory(
        fake, embedder=HashEmbedder(dimensions=64), use_vector_index=False
    )
    await mem.remember(_ep("apples are red"))
    await mem.remember(_ep("the sky is blue"))

    matches = await mem.recall_scored("apples are red", limit=2)
    assert matches[0].episode.input == "apples are red"
    assert matches[0].vector_score == pytest.approx(1.0, abs=1e-5)
    assert matches[1].score < matches[0].score


async def test_redis_ft_search_scores_decoded_as_similarity() -> None:
    # FT.SEARCH reply carrying a cosine DISTANCE of 0.25 under the
    # ``score`` alias → similarity 0.75 on the decoded pair. The
    # hybrid ``recall_scored`` no longer consumes this path (it scans
    # hashes so it can rank with real embeddings), but ``recall``
    # still orders by it via ``_recall_pairs``.
    reply = [
        1,
        b"jeeves:episode:e1",
        [
            b"session_id", b"s",
            b"user_id", b"alice",
            b"occurred_at", b"1700000000.0",
            b"input", b"hello",
            b"output", b"world",
            b"score", b"0.25",
        ],
    ]
    fake = _FakeRedis(ft_search_result=reply)
    mem = RedisMemory(
        fake, embedder=HashEmbedder(dimensions=8), use_vector_index=True
    )
    pairs = await mem._recall_pairs("hello", 1, None, "alice")
    assert len(pairs) == 1
    episode, score = pairs[0]
    assert episode.id.endswith("e1")
    assert score == pytest.approx(0.75)


async def test_postgres_recall_scored_selects_and_returns_score() -> None:
    embedder = HashEmbedder(dimensions=8)
    query_embedding = await embedder.embed("hello")
    state = _PgState()
    state.next_rows = [
        {
            "id": "e1",
            "session_id": "s",
            "user_id": "alice",
            "occurred_at": datetime(2026, 1, 1, tzinfo=UTC),
            "input": "hello",
            "output": "world",
            "embedding": list(query_embedding),
        }
    ]
    mem = PostgresMemory(pool=_PgPool(state), embedder=embedder)
    matches = await mem.recall_scored("hello", user_id="alice", limit=1)
    sql, _args = state.queried[0]
    # Candidate pool is ANN-ordered by pgvector; scores are computed
    # in process (cosine + BM25 fused via RRF), not neutral 1.0s.
    assert "embedding <=>" in sql
    assert matches[0].episode.id == "e1"
    assert matches[0].vector_score == pytest.approx(1.0, abs=1e-6)
    assert matches[0].bm25_score is not None
    assert matches[0].score > 0.0


# ---------------------------------------------------------------------------
# Fix 5 — one-shot HashEmbedder fallback warning
# ---------------------------------------------------------------------------


def test_hash_embedder_fallback_warns_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import loomflow.memory.embedder as embedder_mod

    monkeypatch.setattr(embedder_mod, "_HASH_FALLBACK_WARNED", False)
    with pytest.warns(UserWarning, match="HashEmbedder"):
        SqliteMemory(tmp_path / "warn1.db")
    # Second construction: flag already set — no repeat warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SqliteMemory(tmp_path / "warn2.db")


def test_no_warning_when_embedder_supplied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import loomflow.memory.embedder as embedder_mod

    monkeypatch.setattr(embedder_mod, "_HASH_FALLBACK_WARNED", False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SqliteMemory(tmp_path / "ok.db", embedder=HashEmbedder(dimensions=8))


# ---------------------------------------------------------------------------
# Fix 6 — FactStore.delete / count + GDPR delegation
# ---------------------------------------------------------------------------


async def test_inmemory_fact_store_delete_and_count() -> None:
    store = InMemoryFactStore(embedder=HashEmbedder(dimensions=4))
    old = datetime.now(UTC) - timedelta(days=100)
    await store.append(
        _fact(user_id="alice", predicate="p1", recorded_at=old)
    )
    await store.append(_fact(user_id="alice", predicate="p2"))
    await store.append(_fact(user_id="bob", predicate="p1"))

    assert await store.count(user_id="alice") == 2
    assert await store.count(user_id="bob") == 1
    assert await store.count(user_id=None) == 0

    cutoff = datetime.now(UTC) - timedelta(days=1)
    assert await store.delete(user_id="alice", before=cutoff) == 1
    assert await store.count(user_id="alice") == 1
    assert await store.delete(user_id="alice") == 1
    assert await store.count(user_id="alice") == 0
    # Embeddings cleaned up alongside facts.
    assert all(
        f.user_id != "alice" for f in await store.all_facts()
    )
    assert await store.count(user_id="bob") == 1


async def test_sqlite_fact_store_delete_and_count(tmp_path: Path) -> None:
    store = SqliteFactStore(tmp_path / "f.db")
    old = datetime.now(UTC) - timedelta(days=100)
    await store.append(
        _fact(user_id="alice", predicate="p1", recorded_at=old)
    )
    await store.append(_fact(user_id="alice", predicate="p2"))
    await store.append(_fact(user_id="bob", predicate="p1"))

    assert await store.count(user_id="alice") == 2
    cutoff = datetime.now(UTC) - timedelta(days=1)
    assert await store.delete(user_id="alice", before=cutoff) == 1
    assert await store.delete(user_id="alice") == 1
    assert await store.count(user_id="alice") == 0
    assert await store.count(user_id="bob") == 1


async def test_redis_fact_store_delete_and_count() -> None:
    fake = _FakeRedis()
    store = RedisFactStore(fake, embedder=HashEmbedder(dimensions=4))
    await store.append(_fact(user_id="alice", predicate="p1"))
    await store.append(_fact(user_id="alice", predicate="p2"))
    await store.append(_fact(user_id="bob", predicate="p1"))

    assert await store.count(user_id="alice") == 2
    deleted = await store.delete(user_id="alice")
    assert deleted == 2
    # Batched removal — one UNLINK carrying both keys.
    assert len(fake.unlink_calls) == 1
    assert len(fake.unlink_calls[0]) == 2
    assert await store.count(user_id="alice") == 0
    assert await store.count(user_id="bob") == 1


async def test_inmemory_forget_does_not_overcount_deleteless_store() -> None:
    """A fact store WITHOUT ``delete`` used to inflate the forget
    count (``deleted += 1`` even when nothing was removed). Now the
    count only reflects rows actually deleted."""

    class _DeletelessStore:
        async def append(self, fact: Fact) -> str:
            return fact.id

        async def query(self, **kwargs: Any) -> list[Fact]:
            return [_fact(user_id="alice"), _fact(user_id="alice")]

        async def recall_text(self, query: str, **kwargs: Any) -> list[Fact]:
            return []

        async def all_facts(self) -> list[Fact]:
            return []

        async def aclose(self) -> None:
            return None

    mem = InMemoryMemory(fact_store=_DeletelessStore())  # type: ignore[arg-type]
    await mem.remember(
        Episode(session_id="s", user_id="alice", input="x", output="y")
    )
    deleted = await mem.forget(user_id="alice")
    assert deleted == 1  # the episode only — no phantom fact deletions


async def test_inmemory_forget_and_profile_use_fact_store_surface() -> None:
    mem = InMemoryMemory()
    await mem.remember(
        Episode(session_id="s", user_id="alice", input="x", output="y")
    )
    await mem.facts.append(_fact(user_id="alice"))
    await mem.facts.append(_fact(user_id="bob"))

    profile = await mem.profile(user_id="alice")
    assert profile.fact_count == 1

    deleted = await mem.forget(user_id="alice")
    assert deleted == 2  # 1 episode + 1 fact, exactly
    assert await mem.facts.count(user_id="alice") == 0
    assert await mem.facts.count(user_id="bob") == 1


async def test_redis_forget_unlinks_episodes_in_one_batch() -> None:
    fake = _FakeRedis()
    mem = RedisMemory(
        fake, embedder=HashEmbedder(dimensions=8), use_vector_index=False
    )
    await mem.remember(_ep("one", user_id="alice"))
    await mem.remember(_ep("two", user_id="alice"))
    await mem.remember(_ep("keep", user_id="bob"))

    deleted = await mem.forget(user_id="alice")
    assert deleted == 2
    assert len(fake.unlink_calls) == 1
    assert len(fake.unlink_calls[0]) == 2
    remaining = await mem.recall("", limit=10, user_id="bob")
    assert [e.input for e in remaining] == ["keep"]


# ---------------------------------------------------------------------------
# Fix 7 — SQLite pragmas
# ---------------------------------------------------------------------------


def test_sqlite_memory_enables_wal(tmp_path: Path) -> None:
    db = tmp_path / "wal.db"
    SqliteMemory(db, embedder=HashEmbedder(dimensions=8))
    # WAL persists in the database file once set.
    with sqlite3.connect(db) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"


def test_sqlite_fact_store_enables_wal(tmp_path: Path) -> None:
    db = tmp_path / "facts_wal.db"
    SqliteFactStore(db)
    with sqlite3.connect(db) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"


# ---------------------------------------------------------------------------
# Fix 8 — Postgres session index, batched append_many, GDPR SQL
# ---------------------------------------------------------------------------


class _PgState:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.queried: list[tuple[str, tuple[Any, ...]]] = []
        self.next_rows: list[Any] = []
        self.fetchrow_result: Any = None
        self.execute_result: str = "DELETE 2"
        self.acquire_count = 0
        self.transactions_opened = 0


class _PgTransaction:
    def __init__(self, state: _PgState) -> None:
        self._state = state

    async def __aenter__(self) -> _PgTransaction:
        self._state.transactions_opened += 1
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None


class _PgConn:
    def __init__(self, state: _PgState) -> None:
        self._state = state

    async def execute(self, sql: str, *args: Any) -> str:
        self._state.executed.append((sql, args))
        return self._state.execute_result

    async def executemany(self, sql: str, rows: Any) -> None:
        self._state.executed.append((sql, tuple(rows)))

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self._state.queried.append((sql, args))
        return self._state.next_rows

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        self._state.queried.append((sql, args))
        return self._state.fetchrow_result

    def transaction(self) -> _PgTransaction:
        return _PgTransaction(self._state)


class _PgAcquire:
    def __init__(self, state: _PgState) -> None:
        self._state = state

    async def __aenter__(self) -> _PgConn:
        self._state.acquire_count += 1
        return _PgConn(self._state)

    async def __aexit__(self, *_: Any) -> None:
        return None


class _PgPool:
    def __init__(self, state: _PgState) -> None:
        self._state = state

    def acquire(self) -> _PgAcquire:
        return _PgAcquire(self._state)


class _CountingEmbedder(HashEmbedder):
    def __init__(self) -> None:
        super().__init__(dimensions=8)
        self.embed_calls = 0
        self.embed_batch_calls = 0

    async def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        return await HashEmbedder.embed(self, text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Zero-arg ``super()`` inside a comprehension resolves its
        # implicit ``__class__`` cell inconsistently across CPython
        # versions (fails on 3.11's genexpr scope). Call the base
        # method explicitly so the fake is interpreter-independent.
        self.embed_batch_calls += 1
        return [await HashEmbedder.embed(self, t) for t in texts]


def test_postgres_schema_includes_session_index() -> None:
    mem = PostgresMemory(pool=None, embedder=HashEmbedder(dimensions=8))
    sql = "\n".join(mem.schema_sql())
    assert "episodes_session_idx" in sql
    assert "(namespace, session_id, occurred_at DESC)" in sql


async def test_postgres_facts_append_many_batches_embeds_and_txn() -> None:
    state = _PgState()
    embedder = _CountingEmbedder()
    store = PostgresFactStore(pool=_PgPool(state), embedder=embedder)
    base = datetime(2026, 1, 1, tzinfo=UTC)

    ids = await store.append_many(
        [
            _fact(user_id="alice", object_="Tokyo", valid_from=base),
            _fact(
                user_id="alice",
                object_="Paris",
                valid_from=base + timedelta(days=1),
            ),
            _fact(user_id="bob", object_="Berlin", valid_from=base),
        ]
    )
    assert len(ids) == 3
    # ONE batched embed call, ONE pooled acquire, ONE transaction.
    assert embedder.embed_batch_calls == 1
    assert embedder.embed_calls == 0
    assert state.acquire_count == 1
    assert state.transactions_opened == 1
    # Per fact: supersede UPDATE precedes its INSERT (intra-batch
    # supersession preserved).
    sqls = [sql for sql, _ in state.executed]
    assert [s.split()[0] for s in sqls] == [
        "UPDATE", "INSERT", "UPDATE", "INSERT", "UPDATE", "INSERT",
    ]


async def test_postgres_facts_delete_and_count_sql() -> None:
    state = _PgState()
    state.execute_result = "DELETE 4"
    state.fetchrow_result = {"c": 7}
    store = PostgresFactStore(pool=_PgPool(state))

    cutoff = datetime(2026, 1, 1, tzinfo=UTC)
    deleted = await store.delete(user_id="alice", before=cutoff)
    assert deleted == 4
    sql, args = state.executed[0]
    assert sql.startswith("DELETE FROM facts")
    assert "user_id IS NOT DISTINCT FROM $1" in sql
    assert "recorded_at < $2" in sql
    assert args == ("alice", cutoff)

    count = await store.count(user_id="alice")
    assert count == 7
    csql, cargs = state.queried[0]
    assert "COUNT(*)" in csql
    assert cargs == ("alice",)


async def test_postgres_memory_forget_delegates_to_fact_store() -> None:
    state = _PgState()
    state.execute_result = "DELETE 2"

    class _StubFacts:
        def __init__(self) -> None:
            self.delete_calls: list[dict[str, Any]] = []

        async def delete(
            self,
            *,
            user_id: str | None = None,
            before: datetime | None = None,
        ) -> int:
            self.delete_calls.append({"user_id": user_id, "before": before})
            return 3

    stub = _StubFacts()
    mem = PostgresMemory(
        pool=_PgPool(state),
        embedder=HashEmbedder(dimensions=8),
        fact_store=stub,
    )
    deleted = await mem.forget(user_id="alice")
    assert deleted == 5  # 2 episodes (parsed) + 3 facts (delegated)
    assert stub.delete_calls == [{"user_id": "alice", "before": None}]
    # No raw DELETE FROM facts issued by the memory backend itself.
    assert not any(
        "DELETE FROM facts" in sql for sql, _ in state.executed
    )


async def test_sqlite_memory_forget_delegates_to_custom_fact_store(
    tmp_path: Path,
) -> None:
    """``fact_store=`` can point at a DIFFERENT file — forget must go
    through the store, not raw-SQL the memory's own db."""
    facts_db = tmp_path / "facts.db"
    store = SqliteFactStore(facts_db)
    mem = SqliteMemory(
        tmp_path / "mem.db",
        embedder=HashEmbedder(dimensions=8),
        fact_store=store,
    )
    await mem.remember(
        Episode(session_id="s", user_id="alice", input="x", output="y")
    )
    await store.append(_fact(user_id="alice"))

    deleted = await mem.forget(user_id="alice")
    assert deleted == 2
    assert await store.count(user_id="alice") == 0
