"""RedisMemory tests using a fake redis.asyncio-shaped client.

Live integration is gated on ``JEEVES_TEST_REDIS_URL``; we skip it
when that's not set so CI without Redis still runs.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import pytest

from loomflow.core.types import Episode
from loomflow.memory.embedder import HashEmbedder
from loomflow.memory.redis import RedisMemory, _pack_float32, _unpack_float32

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Float32 encoding round-trip
# ---------------------------------------------------------------------------


def test_pack_unpack_float32_roundtrip() -> None:
    values = [0.0, 1.0, -1.0, 3.14, 0.5]
    packed = _pack_float32(values)
    assert len(packed) == 4 * len(values)
    unpacked = _unpack_float32(packed)
    for a, b in zip(values, unpacked, strict=True):
        assert abs(a - b) < 1e-6


# ---------------------------------------------------------------------------
# Fake redis client — supports brute-force path (use_vector_index=False)
# ---------------------------------------------------------------------------


class _FakeRedis:
    """Minimal fake mimicking redis.asyncio's surface for our paths."""

    def __init__(self) -> None:
        self.hashes: dict[bytes, dict[bytes, bytes]] = {}
        self.commands: list[tuple[Any, ...]] = []

    async def hset(self, key: str, *, mapping: dict[str, Any]) -> int:
        kbytes = key.encode("utf-8")
        bucket = self.hashes.setdefault(kbytes, {})
        for k, v in mapping.items():
            kb = k.encode("utf-8") if isinstance(k, str) else k
            vb = v if isinstance(v, bytes | bytearray) else str(v).encode("utf-8")
            bucket[kb] = bytes(vb)
        return 1

    async def hgetall(self, key: bytes) -> dict[bytes, bytes]:
        return dict(self.hashes.get(key, {}))

    async def scan(
        self, *, cursor: int, match: bytes
    ) -> tuple[int, list[bytes]]:
        # Walk the dict once; we don't paginate.
        prefix = match.rstrip(b"*")
        keys = [k for k in self.hashes if k.startswith(prefix)]
        return 0, keys

    async def execute_command(self, *args: Any) -> Any:
        self.commands.append(args)
        # Pretend FT.CREATE always succeeds.
        return b"OK"

    async def aclose(self) -> None:
        return None


def _ep(text_in: str, text_out: str = "") -> Episode:
    """Default output="" makes ``_embedding_text`` == input — so a recall
    query with the same text produces an exact embedding match. The
    HashEmbedder is deterministic but not semantic, so any pair of
    different texts hashes to ~uncorrelated vectors; tests that care
    about ranking need an exact-text match to score 1.0."""
    return Episode(
        session_id="s",
        input=text_in,
        output=text_out,
        occurred_at=datetime.now(UTC),
    )


async def test_remember_writes_hash_with_packed_embedding() -> None:
    fake = _FakeRedis()
    mem = RedisMemory(
        fake,
        embedder=HashEmbedder(dimensions=32),
        use_vector_index=False,
    )
    eid = await mem.remember(_ep("alpha", "beta"))

    key = f"jeeves:episode:{eid}".encode()
    assert key in fake.hashes
    bucket = fake.hashes[key]
    assert bucket[b"id"] == eid.encode()
    assert bucket[b"input"] == b"alpha"
    assert bucket[b"output"] == b"beta"
    # Embedding is 32 floats * 4 bytes
    assert len(bucket[b"embedding"]) == 32 * 4


async def test_brute_force_recall_returns_best_match_first() -> None:
    fake = _FakeRedis()
    mem = RedisMemory(
        fake,
        embedder=HashEmbedder(dimensions=64),
        use_vector_index=False,
    )
    # Empty output -> embedded text equals input, so the query
    # "apples are red" hashes to the same vector as the stored
    # episode and scores exactly 1.0 (the others score ~0).
    await mem.remember(_ep("apples are red"))
    await mem.remember(_ep("the sky is blue"))
    await mem.remember(_ep("dogs bark loudly"))

    out = await mem.recall("apples are red", limit=1)
    assert len(out) == 1
    assert out[0].input == "apples are red"


async def test_blank_query_falls_back_to_recency() -> None:
    fake = _FakeRedis()
    mem = RedisMemory(
        fake,
        embedder=HashEmbedder(dimensions=32),
        use_vector_index=False,
    )
    e1 = await mem.remember(_ep("first", "x"))
    e2 = await mem.remember(_ep("second", "y"))

    out = await mem.recall("", limit=2)
    ids = [e.id for e in out]
    # Both episodes are present; order is by recency.
    assert e1 in ids
    assert e2 in ids


async def test_ensure_index_invokes_ft_create_in_vector_mode() -> None:
    fake = _FakeRedis()
    mem = RedisMemory(
        fake,
        embedder=HashEmbedder(dimensions=8),
        use_vector_index=True,
    )
    await mem.ensure_index()
    assert any(
        cmd and cmd[0] == "FT.CREATE" for cmd in fake.commands
    )


# ---------------------------------------------------------------------------
# Live integration — only runs with a real Redis
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("JEEVES_TEST_REDIS_URL"),
    reason="JEEVES_TEST_REDIS_URL env var not set",
)
async def test_live_redis_remember_and_recall() -> None:  # pragma: no cover
    url = os.environ["JEEVES_TEST_REDIS_URL"]
    mem = await RedisMemory.connect(url, embedder=HashEmbedder())
    try:
        eid = await mem.remember(_ep("live alpha", "live beta"))
        out = await mem.recall("live alpha", limit=1)
        assert any(ep.id == eid for ep in out)
    finally:
        await mem.aclose()
