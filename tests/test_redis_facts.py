"""RedisFactStore tests using a fake redis.asyncio-shaped client."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from loomflow.core.types import Fact
from loomflow.memory.embedder import HashEmbedder
from loomflow.memory.redis_facts import RedisFactStore

pytestmark = pytest.mark.anyio


def _fact(
    *,
    subject: str = "user",
    predicate: str = "name_is",
    object_: str = "Alice",
    valid_from: datetime | None = None,
    sources: list[str] | None = None,
) -> Fact:
    return Fact(
        subject=subject,
        predicate=predicate,
        object=object_,
        valid_from=valid_from or datetime.now(UTC),
        recorded_at=datetime.now(UTC),
        sources=sources or [],
    )


# ---------------------------------------------------------------------------
# Fake Redis client (re-uses the shape from test_redis_memory.py)
# ---------------------------------------------------------------------------


class _FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[bytes, dict[bytes, bytes]] = {}

    async def hset(self, key: Any, *, mapping: dict[Any, Any]) -> int:
        kbytes = key if isinstance(key, bytes) else str(key).encode("utf-8")
        bucket = self.hashes.setdefault(kbytes, {})
        for k, v in mapping.items():
            kb = k if isinstance(k, bytes) else str(k).encode("utf-8")
            vb = v if isinstance(v, bytes | bytearray) else str(v).encode("utf-8")
            bucket[kb] = bytes(vb)
        return 1

    async def hgetall(self, key: bytes) -> dict[bytes, bytes]:
        kb = key if isinstance(key, bytes) else str(key).encode("utf-8")
        return dict(self.hashes.get(kb, {}))

    async def scan(
        self, *, cursor: int, match: bytes
    ) -> tuple[int, list[bytes]]:
        prefix = match.rstrip(b"*")
        keys = [k for k in self.hashes if k.startswith(prefix)]
        return 0, keys

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


async def test_append_writes_hash_with_expected_fields() -> None:
    fake = _FakeRedis()
    store = RedisFactStore(fake)
    fid = await store.append(_fact())

    key = f"jeeves:fact:{fid}".encode()
    assert key in fake.hashes
    bucket = fake.hashes[key]
    assert bucket[b"id"] == fid.encode()
    assert bucket[b"subject"] == b"user"
    assert bucket[b"object"] == b"Alice"
    assert bucket[b"currently_valid"] == b"1"


async def test_query_filters_by_subject_and_predicate() -> None:
    fake = _FakeRedis()
    store = RedisFactStore(fake)

    await store.append(
        _fact(subject="alice", predicate="lives_in", object_="Tokyo")
    )
    await store.append(
        _fact(subject="alice", predicate="lives_in", object_="Paris")
    )
    await store.append(
        _fact(subject="bob", predicate="lives_in", object_="Tokyo")
    )

    alice = await store.query(subject="alice")
    assert {f.object for f in alice} == {"Tokyo", "Paris"}

    in_tokyo = await store.query(predicate="lives_in", object_="Tokyo")
    assert {f.subject for f in in_tokyo} == {"alice", "bob"}


async def test_sources_round_trip_through_json() -> None:
    fake = _FakeRedis()
    store = RedisFactStore(fake)
    await store.append(_fact(sources=["ep_1", "ep_2"]))
    out = await store.query(subject="user")
    assert out[0].sources == ["ep_1", "ep_2"]


# ---------------------------------------------------------------------------
# Bi-temporal supersession
# ---------------------------------------------------------------------------


async def test_supersession_closes_off_prior_fact() -> None:
    fake = _FakeRedis()
    store = RedisFactStore(fake)
    base = datetime(2026, 1, 1, tzinfo=UTC)

    f1 = await store.append(
        _fact(predicate="lives_in", object_="Tokyo", valid_from=base)
    )
    f2 = await store.append(
        _fact(
            predicate="lives_in",
            object_="Paris",
            valid_from=base + timedelta(days=30),
        )
    )

    by_id = {f.id: f for f in await store.all_facts()}
    assert by_id[f1].valid_until is not None
    assert by_id[f2].valid_until is None


async def test_supersession_skips_same_object_fact() -> None:
    fake = _FakeRedis()
    store = RedisFactStore(fake)
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


async def test_query_at_specific_time() -> None:
    fake = _FakeRedis()
    store = RedisFactStore(fake)
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

    on_feb = base + timedelta(days=45)
    facts = await store.query(predicate="lives_in", valid_at=on_feb)
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


async def test_recall_text_uses_cosine_with_fake_embedder() -> None:
    embedder = _FakeEmbedder(
        {
            "alice loves apples": [1.0, 0.0, 0.0, 0.0],
            "bob hates oranges": [0.0, 1.0, 0.0, 0.0],
            "fruit": [0.99, 0.1, 0.0, 0.0],
        }
    )
    fake = _FakeRedis()
    store = RedisFactStore(fake, embedder=embedder)

    await store.append(
        _fact(subject="alice", predicate="loves", object_="apples")
    )
    await store.append(
        _fact(subject="bob", predicate="hates", object_="oranges")
    )

    out = await store.recall_text("fruit", limit=1)
    assert len(out) == 1
    assert out[0].subject == "alice"


# ---------------------------------------------------------------------------
# Live integration — gated on env var
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("JEEVES_TEST_REDIS_URL"),
    reason="JEEVES_TEST_REDIS_URL env var not set",
)
async def test_live_redis_fact_store_roundtrip() -> None:  # pragma: no cover
    url = os.environ["JEEVES_TEST_REDIS_URL"]
    store = await RedisFactStore.connect(url, embedder=HashEmbedder())
    try:
        fid = await store.append(_fact(object_="live test"))
        out = await store.query(subject="user")
        assert any(f.id == fid for f in out)
    finally:
        await store.aclose()
