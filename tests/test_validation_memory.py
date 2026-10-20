"""Memory and RAG behave like the references they cite.

Bi-temporal facts followed arrival order instead of valid time (a
backfilled historical fact became "current" and closed the real current
one before it began), across all five fact stores. Postgres runs need
``JEEVES_TEST_PG_DSN``; the others run in-process.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from loomflow.core.types import Fact
from loomflow.memory.facts import InMemoryFactStore, plan_supersession
from loomflow.memory.sqlite_facts import SqliteFactStore

pytestmark = pytest.mark.anyio

Y2020 = datetime(2020, 1, 1, tzinfo=UTC)
Y2022 = datetime(2022, 1, 1, tzinfo=UTC)
Y2024 = datetime(2024, 1, 1, tzinfo=UTC)
NOW = datetime(2026, 6, 1, tzinfo=UTC)


def _lives_in(city: str, since: datetime, until: datetime | None = None) -> Fact:
    return Fact(
        user_id="alice", subject="alice", predicate="lives_in", object=city,
        valid_from=since, valid_until=until,
    )


async def _chroma(tmp: Path) -> Any:
    pytest.importorskip("chromadb")
    from loomflow.memory.chroma_facts import ChromaFactStore

    return ChromaFactStore.ephemeral(collection_name=f"facts_{uuid.uuid4().hex}")


async def _redis(tmp: Path) -> Any:
    from loomflow.memory.redis_facts import RedisFactStore
    from tests.test_redis_facts import _FakeRedis

    return RedisFactStore(_FakeRedis())


async def _postgres(tmp: Path) -> Any:
    dsn = os.environ.get("JEEVES_TEST_PG_DSN")
    if not dsn:
        pytest.skip("JEEVES_TEST_PG_DSN not set")
    asyncpg = pytest.importorskip("asyncpg")
    from loomflow.memory.postgres_facts import PostgresFactStore

    # A schema of its own, so the shared ``facts`` table of other
    # live tests (created with a different vector size) isn't touched.
    schema = f"loomtest_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(dsn)
    await admin.execute(f"CREATE SCHEMA {schema}")
    await admin.close()
    pool = await asyncpg.create_pool(dsn, server_settings={"search_path": f"{schema},public"})
    store = PostgresFactStore(pool=pool)
    await store.init_schema()
    return store


async def _inmemory(tmp: Path) -> Any:
    return InMemoryFactStore()


async def _sqlite(tmp: Path) -> Any:
    return SqliteFactStore(tmp / "facts.db")


STORES: dict[str, Callable[[Path], Awaitable[Any]]] = {
    "inmemory": _inmemory,
    "sqlite": _sqlite,
    "chroma": _chroma,
    "redis": _redis,
    "postgres": _postgres,
}


async def _current(store: Any) -> list[str]:
    facts = await store.query(subject="alice", predicate="lives_in", valid_at=NOW, user_id="alice")
    return [f.object for f in facts]


async def _all(store: Any) -> dict[str, Fact]:
    facts = await store.query(subject="alice", predicate="lives_in", user_id="alice", limit=50)
    return {f.object: f for f in facts}


@pytest.mark.parametrize("backend", sorted(STORES))
async def test_supersession_follows_valid_time_not_arrival_order(
    backend: str, tmp_path: Path
) -> None:
    store = await STORES[backend](tmp_path)
    # Learned in order: current first, the history afterwards.
    await store.append(_lives_in("Berlin", Y2024))
    await store.append(_lives_in("Paris", Y2020))
    assert await _current(store) == ["Berlin"]
    facts = await _all(store)
    assert facts["Berlin"].valid_until is None
    assert facts["Paris"].valid_until == Y2024  # the backfill ends where Berlin begins
    # A claim in between closes the earlier one and is closed by the later.
    await store.append(_lives_in("Rome", Y2022))
    facts = await _all(store)
    assert facts["Paris"].valid_until == Y2022
    assert facts["Rome"].valid_until == Y2024
    assert facts["Berlin"].valid_until is None
    for fact in facts.values():
        assert fact.valid_until is None or fact.valid_until >= fact.valid_from


def test_plan_supersession_never_closes_before_opening() -> None:
    current = _lives_in("Berlin", Y2024)
    closed_history = _lives_in("Paris", Y2020, until=Y2022)
    closes, until = plan_supersession(_lives_in("Rome", Y2022), [current, closed_history])
    # Paris already ended when Rome began (no overlap); Rome ends at Berlin.
    assert closes == {} and until == Y2024


async def test_naive_datetimes_are_utc_everywhere(tmp_path: Path) -> None:
    for store in (InMemoryFactStore(), SqliteFactStore(tmp_path / "f.db")):
        await store.append(_lives_in("Berlin", Y2024))
        naive_now = NOW.replace(tzinfo=None)
        found = await store.query(subject="alice", valid_at=naive_now, user_id="alice")
        assert [f.object for f in found] == ["Berlin"]
    naive = Fact(subject="s", predicate="p", object="o", valid_from=datetime(2024, 1, 1))
    assert naive.valid_from.tzinfo is UTC
