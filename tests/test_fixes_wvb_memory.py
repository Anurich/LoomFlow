"""Memory regression tests for the reviewed fixes (WVB).

3.  Redis vector-mode recall — anonymous recall unions the
    brute-force scan (legacy empty-TAG rows aren't indexed by
    RediSearch), and the KNN over-fetches when a ``time_range``
    post-filter will thin the results.
4.  Redis fact store — per-SCAN-page pipelined HGETALLs (with a
    sequential fallback for fakes without pipeline support).
5.  ``_cosine`` copies deleted — all memory backends share
    ``memory._embedding_util.cosine``.
6.  ``hybrid_rank_episodes`` — shared BM25+cosine+RRF ranking tail.
11. Postgres ``remember`` — ``ON CONFLICT DO UPDATE`` so a
    re-``remember`` with the same id refreshes the row (SQLite
    ``INSERT OR REPLACE`` parity).
12. Working blocks bounded — VectorMemory / ChromaMemory /
    RedisMemory hold blocks in the same BoundedDict (same defaults)
    as InMemoryMemory.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from loomflow.core._eviction import BoundedDict
from loomflow.core.types import Episode
from loomflow.memory._embedding_util import cosine as shared_cosine
from loomflow.memory._hybrid import hybrid_rank_episodes
from loomflow.memory.embedder import HashEmbedder
from loomflow.memory.redis import RedisMemory
from loomflow.memory.redis_facts import RedisFactStore
from loomflow.memory.vector import VectorMemory

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Fake redis client (mirrors tests/test_redis_memory.py)
# ---------------------------------------------------------------------------


class _FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[bytes, dict[bytes, bytes]] = {}
        self.commands: list[tuple[Any, ...]] = []

    async def hset(self, key: Any, *, mapping: dict[Any, Any]) -> int:
        kbytes = key if isinstance(key, bytes) else str(key).encode()
        bucket = self.hashes.setdefault(kbytes, {})
        for k, v in mapping.items():
            kb = k if isinstance(k, bytes) else str(k).encode()
            vb = v if isinstance(v, bytes | bytearray) else str(v).encode()
            bucket[kb] = bytes(vb)
        return 1

    async def hgetall(self, key: bytes) -> dict[bytes, bytes]:
        return dict(self.hashes.get(key, {}))

    async def scan(
        self, *, cursor: int, match: bytes
    ) -> tuple[int, list[bytes]]:
        prefix = match.rstrip(b"*")
        return 0, [k for k in self.hashes if k.startswith(prefix)]

    async def execute_command(self, *args: Any) -> Any:
        self.commands.append(args)
        # FT.CREATE "succeeds"; FT.SEARCH returns an empty-ish reply
        # (not a list), which decodes to zero index hits.
        return b"OK"

    async def aclose(self) -> None:
        return None


def _ep(text_in: str, user_id: str | None = None) -> Episode:
    return Episode(
        session_id="s",
        input=text_in,
        output="",
        user_id=user_id,
        occurred_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Fix 3a — anonymous recall in vector mode unions the brute-force scan
# ---------------------------------------------------------------------------


async def test_redis_vector_mode_anon_recall_falls_back_to_scan() -> None:
    """With the index path returning nothing for the anonymous bucket
    (legacy empty-TAG rows aren't indexed), recall(user_id=None) must
    still surface the episodes via the brute-force union."""
    fake = _FakeRedis()
    mem = RedisMemory(
        fake,
        embedder=HashEmbedder(dimensions=64),
        use_vector_index=True,  # index path active
    )
    await mem.remember(_ep("apples are red"))
    await mem.remember(_ep("the sky is blue"))

    out = await mem.recall("apples are red", limit=2)
    assert [e.input for e in out][0] == "apples are red"
    assert len(out) == 2


async def test_redis_vector_mode_named_user_does_not_scan_union() -> None:
    """Named-user recall keeps the pure index path (the TAG filter
    covers it) — no brute-force union for non-anonymous callers."""
    fake = _FakeRedis()
    mem = RedisMemory(
        fake, embedder=HashEmbedder(dimensions=32), use_vector_index=True
    )
    await mem.remember(_ep("alice likes tea", user_id="alice"))
    out = await mem.recall("tea", limit=2, user_id="alice")
    # The fake's FT.SEARCH returns no hits and there's no union for
    # named users, so the result is empty — the point is only that
    # the code path doesn't blow up and doesn't leak other buckets.
    assert out == []


# ---------------------------------------------------------------------------
# Fix 3b — KNN over-fetches when time_range is post-filtered
# ---------------------------------------------------------------------------


def _ft_search_commands(fake: _FakeRedis) -> list[tuple[Any, ...]]:
    return [c for c in fake.commands if c and c[0] == "FT.SEARCH"]


async def test_redis_knn_overfetches_with_time_range() -> None:
    fake = _FakeRedis()
    mem = RedisMemory(
        fake, embedder=HashEmbedder(dimensions=32), use_vector_index=True
    )
    await mem.remember(_ep("alpha", user_id="alice"))

    now = datetime.now(UTC)
    await mem.recall(
        "alpha",
        limit=2,
        user_id="alice",
        time_range=(now - timedelta(days=1), now + timedelta(days=1)),
    )
    (cmd,) = _ft_search_commands(fake)
    assert "KNN 16" in cmd[2]  # limit * 8
    assert "16" in [str(a) for a in cmd]  # LIMIT 0 16

    fake.commands.clear()
    await mem.recall("alpha", limit=2, user_id="alice")
    (cmd,) = _ft_search_commands(fake)
    assert "KNN 2" in cmd[2]  # exact limit without time_range


# ---------------------------------------------------------------------------
# Fix 4 — RedisFactStore pipelines HGETALL per SCAN page
# ---------------------------------------------------------------------------


class _FakePipeline:
    def __init__(self, client: _PipelinedFakeRedis) -> None:
        self._client = client
        self._keys: list[bytes] = []

    def hgetall(self, key: bytes) -> None:
        self._keys.append(key)

    async def execute(self) -> list[dict[bytes, bytes]]:
        return [
            dict(self._client.hashes.get(k, {})) for k in self._keys
        ]

    async def __aenter__(self) -> _FakePipeline:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _PipelinedFakeRedis(_FakeRedis):
    def __init__(self) -> None:
        super().__init__()
        self.pipeline_calls = 0
        self.direct_hgetall_calls = 0

    async def hgetall(self, key: bytes) -> dict[bytes, bytes]:
        self.direct_hgetall_calls += 1
        return await super().hgetall(key)

    def pipeline(self, transaction: bool = True) -> _FakePipeline:
        self.pipeline_calls += 1
        return _FakePipeline(self)


async def test_redis_facts_scan_uses_pipeline() -> None:
    from loomflow.core.types import Fact

    fake = _PipelinedFakeRedis()
    store = RedisFactStore(fake, embedder=HashEmbedder(dimensions=16))
    for i in range(3):
        await store.append(
            Fact(subject=f"user{i}", predicate="likes", object="tea")
        )
    facts = await store.all_facts()
    assert len(facts) == 3
    assert fake.pipeline_calls > 0
    assert fake.direct_hgetall_calls == 0  # all reads went via pipeline


async def test_redis_facts_scan_sequential_fallback() -> None:
    """Fakes without ``pipeline`` still work (sequential path)."""
    from loomflow.core.types import Fact

    fake = _FakeRedis()  # no ``pipeline`` attribute
    store = RedisFactStore(fake, embedder=HashEmbedder(dimensions=16))
    await store.append(
        Fact(subject="user", predicate="likes", object="tea")
    )
    assert len(await store.all_facts()) == 1


# ---------------------------------------------------------------------------
# Fix 5 — single shared cosine
# ---------------------------------------------------------------------------


def test_memory_backends_share_one_cosine() -> None:
    from loomflow.memory import (
        _hybrid,
        facts,
        redis_facts,
        sqlite_facts,
        vector,
    )
    from loomflow.memory import redis as redis_mod
    from loomflow.memory import sqlite as sqlite_mod

    assert vector._cosine is shared_cosine
    assert facts._cosine is shared_cosine
    assert redis_facts._cosine is shared_cosine
    assert sqlite_facts._cosine is shared_cosine
    assert _hybrid.cosine is shared_cosine
    # The private copies in redis.py / sqlite.py are gone.
    assert not hasattr(redis_mod, "_cosine")
    assert not hasattr(sqlite_mod, "_cosine")


def test_shared_cosine_keeps_safest_semantics() -> None:
    # Length mismatch and zero vectors return 0.0 (never raise/nan).
    assert shared_cosine([1.0, 2.0], [1.0]) == 0.0
    assert shared_cosine([], [1.0]) == 0.0
    assert shared_cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert shared_cosine([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Fix 6 — shared hybrid ranking tail
# ---------------------------------------------------------------------------


def _episode_at(
    text: str, minutes_ago: int, embedding: list[float] | None = None
) -> Episode:
    return Episode(
        session_id="s",
        input=text,
        output="",
        occurred_at=datetime.now(UTC) - timedelta(minutes=minutes_ago),
        embedding=embedding,
    )


def test_hybrid_rank_episodes_empty_candidates() -> None:
    assert (
        hybrid_rank_episodes(
            [], query="q", query_embedding=None, alpha=0.5, limit=5
        )
        == []
    )


def test_hybrid_rank_episodes_bm25_only_when_no_embedding() -> None:
    candidates = [
        _episode_at("docker container networking", 5),
        _episode_at("gardening tips for spring", 1),
    ]
    matches = hybrid_rank_episodes(
        candidates,
        query="docker networking",
        query_embedding=None,
        alpha=0.5,
        limit=5,
    )
    assert matches[0].episode.input == "docker container networking"
    assert matches[0].bm25_score is not None
    assert matches[0].vector_score is None


def test_hybrid_rank_episodes_uses_parallel_embeddings() -> None:
    candidates = [
        _episode_at("first", 5),
        _episode_at("second", 1),
    ]
    matches = hybrid_rank_episodes(
        candidates,
        query="unrelated words entirely",
        query_embedding=[1.0, 0.0],
        alpha=1.0,  # vector-dominant
        limit=2,
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    assert matches[0].episode.input == "first"
    assert matches[0].vector_score == pytest.approx(1.0)


def test_hybrid_rank_episodes_no_signal_falls_back_to_recency() -> None:
    candidates = [
        _episode_at("alpha", 10, embedding=[0.0, 1.0]),
        _episode_at("beta", 1, embedding=[0.0, 1.0]),
    ]
    matches = hybrid_rank_episodes(
        candidates,
        query="zzz qqq",  # no lexical overlap
        query_embedding=[-1.0, 0.0],  # non-positive cosine everywhere
        alpha=0.5,
        limit=5,
    )
    assert [m.episode.input for m in matches] == ["beta", "alpha"]
    assert all(m.score == 0.0 for m in matches)


async def test_vector_memory_recall_scored_still_hybrid() -> None:
    """Sanity: the refactored VectorMemory path still returns both
    component scores (behavior-preserving refactor)."""
    mem = VectorMemory(embedder=HashEmbedder(dimensions=64))
    await mem.remember(_ep("docker container networking"))
    matches = await mem.recall_scored("docker container networking")
    assert matches
    assert matches[0].bm25_score is not None
    assert matches[0].vector_score is not None and matches[0].vector_score > 0


# ---------------------------------------------------------------------------
# Fix 11 — Postgres remember upserts (SQLite REPLACE parity)
# ---------------------------------------------------------------------------


class _PgFakeConn:
    def __init__(self, store: _PgFakeStore) -> None:
        self._store = store

    async def execute(self, sql: str, *args: Any) -> str:
        self._store.executed.append((sql, args))
        return "OK"

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        return []


class _PgFakeStore:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []


class _PgFakeAcquire:
    def __init__(self, conn: _PgFakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _PgFakeConn:
        return self._conn

    async def __aexit__(self, *_: Any) -> None:
        return None


class _PgFakePool:
    def __init__(self, store: _PgFakeStore) -> None:
        self._store = store

    def acquire(self) -> _PgFakeAcquire:
        return _PgFakeAcquire(_PgFakeConn(self._store))


async def test_postgres_remember_upserts_on_conflict() -> None:
    from loomflow.memory.postgres import PostgresMemory

    store = _PgFakeStore()
    mem = PostgresMemory(
        pool=_PgFakePool(store), embedder=HashEmbedder(dimensions=32)
    )
    await mem.remember(Episode(session_id="s", input="a", output="b"))
    sql, _ = store.executed[0]
    assert "ON CONFLICT (id) DO UPDATE" in sql
    assert "DO NOTHING" not in sql
    for col in ("input", "output", "embedding", "occurred_at"):
        assert f"{col} = EXCLUDED.{col}" in sql


# ---------------------------------------------------------------------------
# Fix 12 — bounded working-block dicts
# ---------------------------------------------------------------------------


def _assert_bounded_blocks(mem: Any) -> None:
    assert isinstance(mem._blocks, BoundedDict)
    assert mem._blocks.max_keys == 100_000
    assert mem._blocks.ttl_seconds == 24 * 3600


async def _assert_block_behaviour(mem: Any) -> None:
    await mem.update_block("prefs", "dark mode", user_id="alice")
    await mem.append_block("prefs", " + compact", user_id="alice")
    await mem.update_block("prefs", "bob prefs", user_id="bob")
    alice = await mem.working(user_id="alice")
    bob = await mem.working(user_id="bob")
    anon = await mem.working(user_id=None)
    assert [b.content for b in alice] == ["dark mode + compact"]
    assert [b.content for b in bob] == ["bob prefs"]
    assert anon == []


async def test_vector_memory_blocks_bounded() -> None:
    mem = VectorMemory(embedder=HashEmbedder(dimensions=8))
    _assert_bounded_blocks(mem)
    await _assert_block_behaviour(mem)
    # snapshot() keeps the flat "<user_id>::<name>" key shape.
    snap = mem.snapshot()
    assert "alice::prefs" in snap["blocks"]


async def test_redis_memory_blocks_bounded() -> None:
    mem = RedisMemory(
        _FakeRedis(),
        embedder=HashEmbedder(dimensions=8),
        use_vector_index=False,
    )
    _assert_bounded_blocks(mem)
    await _assert_block_behaviour(mem)


async def test_chroma_memory_blocks_bounded() -> None:
    from loomflow.memory.chroma import ChromaMemory

    # Blocks never touch the client (collection init is lazy), so a
    # bare object stands in — no chromadb needed for this test.
    mem = ChromaMemory(object(), embedder=HashEmbedder(dimensions=8))
    _assert_bounded_blocks(mem)
    await _assert_block_behaviour(mem)
