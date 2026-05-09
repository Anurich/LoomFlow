"""PostgresMemory tests.

The schema-DDL tests run anywhere — they assert on the SQL strings the
backend would emit. The end-to-end test uses a fake asyncpg-shaped pool
so we don't need a live database.

Real integration tests are gated on ``JEEVES_TEST_PG_DSN`` env var; we
skip them when that's not set so CI without a Postgres can still run.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from loomflow.core.types import Episode
from loomflow.memory.embedder import HashEmbedder
from loomflow.memory.postgres import (
    _ANON_USER_ID,
    PostgresMemory,
    _decode_user_id,
    _encode_user_id,
)

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# DDL snapshot — runs anywhere, no database needed
# ---------------------------------------------------------------------------


def test_schema_sql_includes_pgvector_and_hnsw_index() -> None:
    mem = PostgresMemory(pool=None, embedder=HashEmbedder(dimensions=384))
    statements = mem.schema_sql()
    joined = "\n".join(statements)
    assert "CREATE EXTENSION IF NOT EXISTS vector" in joined
    assert "vector(384)" in joined
    assert "hnsw" in joined
    assert "vector_cosine_ops" in joined


def test_schema_sql_dimensions_track_embedder() -> None:
    mem = PostgresMemory(pool=None, embedder=HashEmbedder(dimensions=128))
    assert any("vector(128)" in s for s in mem.schema_sql())


# ---------------------------------------------------------------------------
# M10.2 — anonymous-bucket sentinel (no more empty-string hack)
# ---------------------------------------------------------------------------


def test_encode_user_id_maps_none_to_sentinel() -> None:
    assert _encode_user_id(None) == _ANON_USER_ID
    assert _encode_user_id("alice") == "alice"
    # Empty string is now a valid (if odd) user_id, NOT silently
    # the anonymous bucket.
    assert _encode_user_id("") == ""


def test_encode_user_id_rejects_sentinel_collision() -> None:
    """Callers must not be allowed to pass the reserved sentinel
    as their user_id — that would let one user impersonate the
    anonymous bucket on the wire."""
    with pytest.raises(ValueError, match="reserved"):
        _encode_user_id(_ANON_USER_ID)


def test_decode_user_id_round_trips() -> None:
    assert _decode_user_id(_ANON_USER_ID) is None
    assert _decode_user_id("alice") == "alice"
    assert _decode_user_id("") == ""


def test_schema_uses_sentinel_default_and_migrates_legacy_rows() -> None:
    """Schema DDL must:

    * Use the sentinel as the column default (not ``''``).
    * Include the migration UPDATE that rewrites legacy ``''`` rows.
    * Set the column DEFAULT to the sentinel via ALTER (idempotent).
    """
    mem = PostgresMemory(pool=None, embedder=HashEmbedder(dimensions=64))
    sql = "\n".join(mem.schema_sql())
    assert _ANON_USER_ID in sql
    assert f"DEFAULT '{_ANON_USER_ID}'" in sql
    # The migration step is what fixes deployments that lived on
    # 0.9.x with empty-string anonymous buckets.
    assert (
        f"UPDATE memory_blocks SET user_id = '{_ANON_USER_ID}'"
        in sql
    )
    assert "WHERE user_id = ''" in sql


async def test_update_block_writes_sentinel_for_anonymous() -> None:
    """``user_id=None`` lands as the sentinel on the wire — never
    the empty string."""
    store = _FakeStore()
    pool = _FakePool(store)
    mem = PostgresMemory(pool=pool, embedder=HashEmbedder(dimensions=32))
    await mem.update_block("prefs", "anon prefs", user_id=None)
    sql, args = store.executed[0]
    assert "INSERT INTO memory_blocks" in sql
    assert args[1] == _ANON_USER_ID  # never ""


async def test_update_block_writes_real_user_id_unchanged() -> None:
    store = _FakeStore()
    pool = _FakePool(store)
    mem = PostgresMemory(pool=pool, embedder=HashEmbedder(dimensions=32))
    await mem.update_block("prefs", "alice prefs", user_id="alice")
    _, args = store.executed[0]
    assert args[1] == "alice"


async def test_update_block_rejects_sentinel_as_real_user_id() -> None:
    """A real call passing the sentinel must raise — defense
    against impersonation of the anonymous bucket."""
    store = _FakeStore()
    pool = _FakePool(store)
    mem = PostgresMemory(pool=pool, embedder=HashEmbedder(dimensions=32))
    with pytest.raises(ValueError, match="reserved"):
        await mem.update_block("prefs", "x", user_id=_ANON_USER_ID)


async def test_working_query_filters_by_sentinel_for_anonymous() -> None:
    """``working(user_id=None)`` must filter on the sentinel value
    on the wire, otherwise it'd return zero rows under the new
    schema where anonymous rows live as the sentinel."""
    store = _FakeStore()
    pool = _FakePool(store)
    mem = PostgresMemory(pool=pool, embedder=HashEmbedder(dimensions=32))
    store.next_rows = []
    await mem.working(user_id=None)
    _, args = store.queried[0]
    assert args[1] == _ANON_USER_ID


# ---------------------------------------------------------------------------
# Fake asyncpg pool — for offline integration tests
# ---------------------------------------------------------------------------


class _FakeConn:
    def __init__(self, store: _FakeStore) -> None:
        self._store = store

    async def execute(self, sql: str, *args: Any) -> str:
        self._store.executed.append((sql, args))
        return "OK"

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self._store.queried.append((sql, args))
        return self._store.next_rows


class _FakeStore:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.queried: list[tuple[str, tuple[Any, ...]]] = []
        self.next_rows: list[Any] = []


class _FakeAcquire:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConn:
        return self._conn

    async def __aexit__(self, *_: Any) -> None:
        return None


class _FakePool:
    def __init__(self, store: _FakeStore) -> None:
        self._store = store

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(_FakeConn(self._store))


async def test_remember_inserts_with_namespace_and_embedding() -> None:
    store = _FakeStore()
    pool = _FakePool(store)
    mem = PostgresMemory(pool=pool, embedder=HashEmbedder(dimensions=64))

    await mem.remember(
        Episode(
            session_id="s",
            input="hello",
            output="world",
        )
    )

    assert len(store.executed) == 1
    sql, args = store.executed[0]
    assert "INSERT INTO episodes" in sql
    # Args: id, namespace, session_id, user_id, occurred_at, input, output, embedding
    assert args[1] == "default"
    assert args[2] == "s"
    assert args[3] is None  # user_id (anonymous bucket)
    assert args[5] == "hello"
    assert args[6] == "world"
    assert isinstance(args[7], list)
    assert len(args[7]) == 64


async def test_recall_uses_pgvector_distance_operator() -> None:
    store = _FakeStore()
    pool = _FakePool(store)
    mem = PostgresMemory(pool=pool, embedder=HashEmbedder(dimensions=32))

    base = datetime.now(UTC)
    store.next_rows = [
        {
            "id": "ep_1",
            "session_id": "s",
            "occurred_at": base,
            "input": "hi",
            "output": "hello",
            "embedding": [0.0] * 32,
        }
    ]

    out = await mem.recall("hi there", limit=3)

    assert len(out) == 1
    assert out[0].id == "ep_1"
    sql, args = store.queried[0]
    assert "embedding <=> $5" in sql  # cosine distance ordering
    # Args: namespace, user_id, lo, hi, query_embedding, limit
    assert args[0] == "default"
    assert args[1] is None  # user_id (anonymous bucket)
    assert args[2] is None
    assert args[3] is None
    assert isinstance(args[4], list)
    assert args[5] == 3


async def test_recall_recent_when_query_is_empty() -> None:
    store = _FakeStore()
    pool = _FakePool(store)
    mem = PostgresMemory(pool=pool, embedder=HashEmbedder(dimensions=32))

    base = datetime.now(UTC)
    store.next_rows = [
        {
            "id": "ep_1",
            "session_id": "s",
            "occurred_at": base,
            "input": "x",
            "output": "y",
            "embedding": [0.0] * 32,
        }
    ]
    await mem.recall("   ", limit=2)
    sql, _ = store.queried[0]
    assert "ORDER BY occurred_at DESC" in sql


async def test_recall_with_time_range_passes_bounds() -> None:
    store = _FakeStore()
    pool = _FakePool(store)
    mem = PostgresMemory(pool=pool, embedder=HashEmbedder(dimensions=32))

    base = datetime.now(UTC)
    window = (base - timedelta(hours=2), base)
    store.next_rows = []
    await mem.recall("anything", limit=2, time_range=window)
    _, args = store.queried[0]
    # Args: namespace, user_id, lo, hi, query_embedding, limit
    assert args[2] == window[0]
    assert args[3] == window[1]


# ---------------------------------------------------------------------------
# Live integration — only runs with a real Postgres
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("JEEVES_TEST_PG_DSN"),
    reason="JEEVES_TEST_PG_DSN env var not set",
)
async def test_live_postgres_remember_and_recall() -> None:  # pragma: no cover
    dsn = os.environ["JEEVES_TEST_PG_DSN"]
    mem = await PostgresMemory.connect(dsn, embedder=HashEmbedder())
    try:
        await mem.init_schema()
        eid = await mem.remember(
            Episode(session_id="s-live", input="alpha", output="beta")
        )
        out = await mem.recall("alpha", limit=1)
        assert any(ep.id == eid for ep in out)
    finally:
        await mem.aclose()
