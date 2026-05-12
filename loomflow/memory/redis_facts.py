"""Redis-backed bi-temporal fact store.

Each fact lives in a Redis hash at ``{prefix}{id}`` (default prefix
``jeeves:fact:``). Fields:

* ``id`` / ``subject`` / ``predicate`` / ``object`` — strings
* ``confidence`` — string-encoded float
* ``valid_from_ts`` / ``recorded_at_ts`` — string-encoded floats
* ``valid_until_ts`` — string-encoded float (``"0"`` when still valid)
* ``currently_valid`` — ``b"1"`` / ``b"0"`` flag (mirror of
  ``valid_until_ts == 0``)
* ``sources`` — JSON-encoded list of episode ids
* ``embedding`` — float32 BLOB (only present when an embedder is
  configured)

Supersession is a brute-force scan: ``SCAN`` for all fact keys, find
those with matching subject + predicate that are currently valid and
have a different object, and ``HSET`` each to flip
``currently_valid=False`` + stamp ``valid_until_ts``. RediSearch with
HNSW + numeric/tag indexes is a follow-up.
"""

from __future__ import annotations

import json
import math
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import anyio

from ..core.protocols import Embedder
from ..core.types import Fact, _normalize_predicate
from ._embedding_util import pack_float32, unpack_float32
from .embedder import HashEmbedder

DEFAULT_KEY_PREFIX = "jeeves:fact:"


class RedisFactStore:
    """Bi-temporal fact store over plain Redis hashes."""

    def __init__(
        self,
        client: Any,
        *,
        embedder: Embedder | None = None,
        key_prefix: str = DEFAULT_KEY_PREFIX,
    ) -> None:
        self._client = client
        self._embedder: Embedder = (
            embedder if embedder is not None else HashEmbedder()
        )
        self._key_prefix = key_prefix
        self._lock = anyio.Lock()

    @classmethod
    async def connect(
        cls,
        url: str = "redis://localhost:6379/0",
        *,
        embedder: Embedder | None = None,
        key_prefix: str = DEFAULT_KEY_PREFIX,
    ) -> RedisFactStore:
        try:
            from redis.asyncio import (  # type: ignore[import-not-found, import-untyped]
                from_url,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "redis is not installed. "
                "Install with: pip install redis"
            ) from exc
        client = from_url(url, decode_responses=False)
        return cls(client, embedder=embedder, key_prefix=key_prefix)

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    async def aclose(self) -> None:
        if self._client is not None and hasattr(self._client, "aclose"):
            await self._client.aclose()

    # ---- mutation --------------------------------------------------------

    async def append(self, fact: Fact) -> str:
        triple = _triple_text(fact)
        embedding = await self._embedder.embed(triple)

        async with self._lock:
            await self._supersede(fact)
            await self._write_fact(fact, embedding)
        return fact.id

    async def _supersede(self, fact: Fact) -> None:
        ts = str(fact.valid_from.timestamp()).encode("utf-8")
        async for key, data in self._scan_facts():
            # Namespace-scoped supersession: alice's facts never
            # invalidate bob's. Anonymous bucket (None / empty) is
            # its own namespace.
            other_user = _decode_field(data.get(b"user_id", b"")) or None
            if other_user != fact.user_id:
                continue
            if _decode_field(data.get(b"subject", b"")) != fact.subject:
                continue
            if _decode_field(data.get(b"predicate", b"")) != fact.predicate:
                continue
            if data.get(b"currently_valid", b"0") != b"1":
                continue
            if _decode_field(data.get(b"object", b"")) == fact.object:
                continue
            await self._client.hset(
                key,
                mapping={
                    b"currently_valid": b"0",
                    b"valid_until_ts": ts,
                },
            )

    async def _write_fact(
        self, fact: Fact, embedding: list[float]
    ) -> None:
        key = self._key_for(fact.id)
        valid_until_ts = (
            str(fact.valid_until.timestamp()).encode("utf-8")
            if fact.valid_until is not None
            else b"0"
        )
        mapping = {
            b"id": fact.id.encode("utf-8"),
            # Persist ``user_id`` so recall queries can filter by
            # namespace partition. Empty bytes for the anonymous
            # bucket; round-trip back to ``None`` on read.
            b"user_id": (fact.user_id or "").encode("utf-8"),
            b"subject": fact.subject.encode("utf-8"),
            b"predicate": fact.predicate.encode("utf-8"),
            b"object": fact.object.encode("utf-8"),
            b"confidence": str(fact.confidence).encode("utf-8"),
            b"valid_from_ts": str(fact.valid_from.timestamp()).encode("utf-8"),
            b"valid_until_ts": valid_until_ts,
            b"currently_valid": b"1" if fact.valid_until is None else b"0",
            b"recorded_at_ts": str(fact.recorded_at.timestamp()).encode("utf-8"),
            b"sources": json.dumps(list(fact.sources)).encode("utf-8"),
            b"embedding": pack_float32(embedding),
        }
        await self._client.hset(key, mapping=mapping)

    # ---- queries ---------------------------------------------------------

    async def query(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object_: str | None = None,
        valid_at: datetime | None = None,
        limit: int = 10,
        user_id: str | None = None,
    ) -> list[Fact]:
        valid_at_ts = valid_at.timestamp() if valid_at is not None else None
        results: list[Fact] = []
        async for _key, data in self._scan_facts():
            fact = _hash_to_fact(data)
            if fact is None:
                continue
            # Hard namespace partition by ``user_id``.
            if fact.user_id != user_id:
                continue
            if subject is not None and fact.subject != subject:
                continue
            if predicate is not None and fact.predicate != _normalize_predicate(predicate):
                continue
            if object_ is not None and fact.object != object_:
                continue
            if valid_at_ts is not None and not _is_valid_at(fact, valid_at_ts):
                continue
            results.append(fact)
        results.sort(key=lambda f: f.recorded_at, reverse=True)
        return results[:limit]

    async def recall_text(
        self,
        query: str,
        *,
        limit: int = 5,
        valid_at: datetime | None = None,
        user_id: str | None = None,
    ) -> list[Fact]:
        query_embedding = await self._embedder.embed(query)
        valid_at_ts = valid_at.timestamp() if valid_at is not None else None

        scored: list[tuple[float, Fact]] = []
        async for _key, data in self._scan_facts():
            fact = _hash_to_fact(data)
            if fact is None:
                continue
            # Hard namespace partition by ``user_id``.
            if fact.user_id != user_id:
                continue
            if valid_at_ts is not None and not _is_valid_at(fact, valid_at_ts):
                continue
            blob = data.get(b"embedding")
            if not isinstance(blob, bytes | bytearray) or not blob:
                continue
            stored = unpack_float32(bytes(blob))
            scored.append((_cosine(query_embedding, stored), fact))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [f for _, f in scored[:limit]]

    async def all_facts(self) -> list[Fact]:
        out: list[Fact] = []
        async for _key, data in self._scan_facts():
            fact = _hash_to_fact(data)
            if fact is not None:
                out.append(fact)
        out.sort(key=lambda f: f.recorded_at, reverse=True)
        return out

    # ---- scanning helpers ------------------------------------------------

    def _key_for(self, fact_id: str) -> bytes:
        return f"{self._key_prefix}{fact_id}".encode()

    async def _scan_facts(
        self,
    ) -> AsyncIterator[tuple[bytes, dict[bytes, Any]]]:
        cursor: int = 0
        match = f"{self._key_prefix}*".encode()
        while True:
            cursor, keys = await self._client.scan(
                cursor=cursor, match=match
            )
            for key in keys:
                data = await self._client.hgetall(key)
                if data:
                    yield key, _normalize_keys(data)
            if cursor == 0:
                break


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _triple_text(fact: Fact) -> str:
    return f"{fact.subject} {fact.predicate} {fact.object}"


def _normalize_keys(data: dict[Any, Any]) -> dict[bytes, Any]:
    """Coerce all keys to ``bytes`` so we can index uniformly."""
    out: dict[bytes, Any] = {}
    for k, v in data.items():
        if isinstance(k, bytes):
            out[k] = v
        else:
            out[str(k).encode("utf-8")] = v
    return out


def _decode_field(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _hash_to_fact(data: dict[bytes, Any]) -> Fact | None:
    eid = _decode_field(data.get(b"id", b""))
    if not eid:
        return None
    sources_raw = _decode_field(data.get(b"sources", "[]"))
    try:
        sources = list(json.loads(sources_raw))
    except json.JSONDecodeError:
        sources = []
    try:
        valid_from_ts = float(_decode_field(data.get(b"valid_from_ts", "0")))
        recorded_at_ts = float(
            _decode_field(data.get(b"recorded_at_ts", "0"))
        )
        valid_until_ts = float(
            _decode_field(data.get(b"valid_until_ts", "0"))
        )
    except ValueError:
        return None
    valid_until = (
        datetime.fromtimestamp(valid_until_ts, tz=UTC)
        if valid_until_ts > 0 and data.get(b"currently_valid") != b"1"
        else None
    )
    try:
        confidence = float(_decode_field(data.get(b"confidence", "1.0")))
    except ValueError:
        confidence = 1.0
    user_id_raw = _decode_field(data.get(b"user_id", b""))
    return Fact(
        id=eid,
        user_id=user_id_raw or None,
        subject=_decode_field(data.get(b"subject", b"")),
        predicate=_decode_field(data.get(b"predicate", b"")),
        object=_decode_field(data.get(b"object", b"")),
        confidence=confidence,
        valid_from=datetime.fromtimestamp(valid_from_ts, tz=UTC),
        valid_until=valid_until,
        recorded_at=datetime.fromtimestamp(recorded_at_ts, tz=UTC),
        sources=sources,
    )


def _is_valid_at(fact: Fact, when_ts: float) -> bool:
    if when_ts < fact.valid_from.timestamp():
        return False
    if fact.valid_until is None:
        return True
    return when_ts < fact.valid_until.timestamp()


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))
