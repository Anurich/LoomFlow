"""Redis-backed :class:`Memory`.

Two flavours, picked at construction time:

* **vector mode** (default when RediSearch is available) — episodes
  are stored as Redis hashes; a RediSearch ``FT.CREATE`` index with
  ``HNSW`` provides cosine-similarity recall.
* **brute-force mode** (when RediSearch isn't available, e.g. plain
  Redis) — episodes still go to hashes but recall scans every
  episode in process. Fine for small corpora; switch to the vector
  mode (RedisStack) for production scale.

Both modes use the ``redis.asyncio`` client. Working blocks live in
process memory; the redundancy of putting them in Redis isn't worth
the extra round-trip for the small payloads we have.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import anyio

from ..core.errors import MemoryStoreError
from ..core.protocols import Embedder
from ..core.types import (
    Episode,
    Fact,
    MemoryBlock,
    MemoryExport,
    MemoryProfile,
    Message,
    Role,
)
from ._embedding_util import pack_float32, unpack_float32
from .embedder import HashEmbedder

DEFAULT_KEY_PREFIX = "jeeves:episode:"
DEFAULT_INDEX_NAME = "jeeves_idx"


class RedisMemory:
    """Redis-backed :class:`Memory`. Use :meth:`connect` to construct."""

    def __init__(
        self,
        client: Any,
        *,
        embedder: Embedder | None = None,
        key_prefix: str = DEFAULT_KEY_PREFIX,
        index_name: str = DEFAULT_INDEX_NAME,
        use_vector_index: bool = True,
        fact_store: Any | None = None,
    ) -> None:
        self._client = client
        self._embedder: Embedder = embedder if embedder is not None else HashEmbedder()
        self._key_prefix = key_prefix
        self._index_name = index_name
        self._use_vector_index = use_vector_index
        self._index_ready = False
        # Working blocks partition by user_id; key is (user_id, name).
        self._blocks: dict[tuple[str | None, str], MemoryBlock] = {}
        self._lock = anyio.Lock()
        # The Agent loop's fact-recall hook. ``None`` by default —
        # construct an explicit :class:`RedisFactStore` (or pass
        # ``with_facts=True`` to :meth:`connect`) to attach one.
        self.facts: Any | None = fact_store

    # ---- factory ---------------------------------------------------------

    @classmethod
    async def connect(
        cls,
        url: str = "redis://localhost:6379/0",
        *,
        embedder: Embedder | None = None,
        key_prefix: str = DEFAULT_KEY_PREFIX,
        index_name: str = DEFAULT_INDEX_NAME,
        use_vector_index: bool = True,
        with_facts: bool = False,
        fact_key_prefix: str = "jeeves:fact:",
    ) -> RedisMemory:
        """Open an async Redis connection.

        ``with_facts=True`` attaches a :class:`RedisFactStore` sharing
        the same client; facts go to ``{fact_key_prefix}*`` keys so
        they don't collide with episode keys.
        """
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
        instance = cls(
            client,
            embedder=embedder,
            key_prefix=key_prefix,
            index_name=index_name,
            use_vector_index=use_vector_index,
        )
        if with_facts:
            from .redis_facts import RedisFactStore

            instance.facts = RedisFactStore(
                client,
                embedder=instance._embedder,
                key_prefix=fact_key_prefix,
            )
        return instance

    async def aclose(self) -> None:
        if self._client is not None and hasattr(self._client, "aclose"):
            await self._client.aclose()

    # ---- index management -----------------------------------------------

    async def ensure_index(self) -> None:
        """Create the RediSearch HNSW index, if not already present.

        Skipped silently when ``use_vector_index=False`` or when
        RediSearch isn't available on the server.
        """
        if self._index_ready or not self._use_vector_index:
            self._index_ready = True
            return
        try:
            await self._client.execute_command(
                "FT.CREATE",
                self._index_name,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                self._key_prefix,
                "SCHEMA",
                "session_id",
                "TAG",
                "occurred_at",
                "NUMERIC",
                "input",
                "TEXT",
                "output",
                "TEXT",
                "embedding",
                "VECTOR",
                "HNSW",
                "6",
                "TYPE",
                "FLOAT32",
                "DIM",
                str(self._embedder.dimensions),
                "DISTANCE_METRIC",
                "COSINE",
            )
        except Exception as exc:  # noqa: BLE001
            # ``Index already exists`` is OK; otherwise fall back to
            # brute-force recall.
            msg = str(exc).lower()
            if "already exists" not in msg:
                self._use_vector_index = False
        self._index_ready = True

    # ---- working blocks --------------------------------------------------

    async def working(
        self, *, user_id: str | None = None
    ) -> list[MemoryBlock]:
        async with self._lock:
            scoped = [
                b for (uid, _name), b in self._blocks.items() if uid == user_id
            ]
        return sorted(scoped, key=lambda b: b.pinned_order)

    async def update_block(
        self, name: str, content: str, *, user_id: str | None = None
    ) -> None:
        key = (user_id, name)
        async with self._lock:
            existing = self._blocks.get(key)
            user_count = sum(
                1 for (uid, _) in self._blocks if uid == user_id
            )
            self._blocks[key] = MemoryBlock(
                name=name,
                content=content,
                pinned_order=existing.pinned_order if existing else user_count,
            )

    async def append_block(
        self, name: str, content: str, *, user_id: str | None = None
    ) -> None:
        key = (user_id, name)
        async with self._lock:
            existing = self._blocks.get(key)
            if existing is None:
                user_count = sum(
                    1 for (uid, _) in self._blocks if uid == user_id
                )
                self._blocks[key] = MemoryBlock(
                    name=name,
                    content=content,
                    pinned_order=user_count,
                )
            else:
                self._blocks[key] = MemoryBlock(
                    name=name,
                    content=existing.content + content,
                    pinned_order=existing.pinned_order,
                )

    # ---- episodes --------------------------------------------------------

    async def remember(self, episode: Episode) -> str:
        if episode.embedding is None:
            text = "\n".join(p for p in (episode.input, episode.output) if p)
            embedding = await self._embedder.embed(text)
            episode = episode.model_copy(update={"embedding": embedding})

        await self.ensure_index()

        embedding_bytes = _pack_float32(episode.embedding or [])
        key = self._key_for(episode.id)
        mapping = {
            "id": episode.id.encode("utf-8"),
            "session_id": episode.session_id.encode("utf-8"),
            # Persist ``user_id`` so recall queries can filter
            # by namespace partition. Encoded as the empty bytestring
            # for ``None`` so we can round-trip it (Redis doesn't
            # natively distinguish missing fields from empty values).
            "user_id": (episode.user_id or "").encode("utf-8"),
            "occurred_at": str(episode.occurred_at.timestamp()).encode("utf-8"),
            "input": episode.input.encode("utf-8"),
            "output": episode.output.encode("utf-8"),
            "embedding": embedding_bytes,
        }
        await self._client.hset(key, mapping=mapping)
        return episode.id

    async def recall(
        self,
        query: str,
        *,
        kind: str = "episodic",
        limit: int = 5,
        time_range: tuple[datetime, datetime] | None = None,
        user_id: str | None = None,
    ) -> list[Episode]:
        await self.ensure_index()

        if not query.strip():
            return await self._recall_recent(limit, time_range, user_id)

        query_embedding = await self._embedder.embed(query)

        # Over-fetch when filtering so we have enough candidates after
        # the namespace partition is applied. The vector index lacks
        # native ``user_id`` faceting today; this is a post-filter.
        fetch_limit = limit * 8 if user_id is not None else limit

        if self._use_vector_index:
            episodes = await self._recall_via_index(query_embedding, fetch_limit)
        else:
            episodes = await self._recall_brute_force(query_embedding, fetch_limit)

        # Hard namespace partition by ``user_id``.
        episodes = [e for e in episodes if e.user_id == user_id]
        if time_range is not None:
            lo, hi = time_range
            episodes = [e for e in episodes if lo <= e.occurred_at <= hi]
        return episodes[:limit]

    async def _recall_via_index(
        self, query_embedding: list[float], limit: int
    ) -> list[Episode]:
        params = [
            "FT.SEARCH",
            self._index_name,
            f"*=>[KNN {limit} @embedding $vec AS score]",
            "PARAMS",
            "2",
            "vec",
            _pack_float32(query_embedding),
            "SORTBY",
            "score",
            "RETURN",
            "6",
            "session_id",
            "user_id",
            "occurred_at",
            "input",
            "output",
            "score",
            "DIALECT",
            "2",
            "LIMIT",
            "0",
            str(limit),
        ]
        try:
            result = await self._client.execute_command(*params)
        except Exception as exc:  # noqa: BLE001
            raise MemoryStoreError(f"RediSearch KNN query failed: {exc}") from exc
        return _decode_ft_search(result)

    async def _recall_brute_force(
        self, query_embedding: list[float], limit: int
    ) -> list[Episode]:
        episodes = await self._scan_all_episodes()
        scored: list[tuple[float, Episode]] = []
        for ep in episodes:
            if ep.embedding is None:
                continue
            scored.append((_cosine(query_embedding, ep.embedding), ep))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [ep for _, ep in scored[:limit]]

    async def _recall_recent(
        self,
        limit: int,
        time_range: tuple[datetime, datetime] | None,
        user_id: str | None,
    ) -> list[Episode]:
        episodes = await self._scan_all_episodes()
        episodes = [e for e in episodes if e.user_id == user_id]
        if time_range is not None:
            lo, hi = time_range
            episodes = [e for e in episodes if lo <= e.occurred_at <= hi]
        episodes.sort(key=lambda e: e.occurred_at, reverse=True)
        return episodes[:limit]

    async def _scan_all_episodes(self) -> list[Episode]:
        cursor: int = 0
        match = f"{self._key_prefix}*".encode()
        episodes: list[Episode] = []
        while True:
            cursor, keys = await self._client.scan(cursor=cursor, match=match)
            for key in keys:
                data = await self._client.hgetall(key)
                if not data:
                    continue
                ep = _decode_hash(data)
                if ep is not None:
                    episodes.append(ep)
            if cursor == 0:
                break
        return episodes

    async def recall_facts(
        self,
        query: str,
        *,
        limit: int = 5,
        valid_at: datetime | None = None,
        user_id: str | None = None,
    ) -> list[Fact]:
        if self.facts is None:
            return []
        return list(
            await self.facts.recall_text(
                query, limit=limit, valid_at=valid_at, user_id=user_id
            )
        )

    async def session_messages(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
        limit: int = 20,
    ) -> list[Message]:
        # No native ``WHERE session_id`` index in vanilla Redis Hash;
        # scan all episode keys, post-filter, and slice. Matches the
        # InMemoryMemory + Vector backends' best-effort behaviour for
        # the M2 session-continuity path.
        episodes = await self._scan_all_episodes()
        episodes = [
            e
            for e in episodes
            if e.session_id == session_id and e.user_id == user_id
        ]
        episodes.sort(key=lambda e: e.occurred_at)
        max_episodes = max(1, limit // 2)
        episodes = episodes[-max_episodes:]
        out: list[Message] = []
        for ep in episodes:
            if ep.input:
                out.append(Message(role=Role.USER, content=ep.input))
            if ep.output:
                out.append(Message(role=Role.ASSISTANT, content=ep.output))
        return out

    async def consolidate(self) -> None:
        return None

    # ---- profile / forget / export (GDPR) -------------------------------

    async def profile(
        self, *, user_id: str | None = None
    ) -> MemoryProfile:
        episodes = await self._scan_all_episodes()
        episodes = [e for e in episodes if e.user_id == user_id]
        last_seen: datetime | None = (
            max(e.occurred_at for e in episodes) if episodes else None
        )
        seen: set[str] = set()
        recent_sessions: list[str] = []
        for e in sorted(episodes, key=lambda x: x.occurred_at, reverse=True):
            if e.session_id in seen:
                continue
            seen.add(e.session_id)
            recent_sessions.append(e.session_id)
            if len(recent_sessions) >= 10:
                break
        sample_facts: list[Fact] = []
        fact_count = 0
        if self.facts is not None:
            sample_facts = list(
                await self.facts.query(user_id=user_id, limit=10)
            )
            all_facts = await self.facts.query(user_id=user_id, limit=100_000)
            fact_count = len(all_facts)
        return MemoryProfile(
            user_id=user_id,
            episode_count=len(episodes),
            fact_count=fact_count,
            last_seen=last_seen,
            recent_sessions=recent_sessions,
            sample_facts=sample_facts,
        )

    async def forget(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        before: datetime | None = None,
    ) -> int:
        # Scan all episode keys, filter Python-side, DEL the matches.
        # Faster than maintaining a secondary index for what's
        # typically a low-frequency op.
        episodes = await self._scan_all_episodes()
        deleted = 0
        for ep in episodes:
            if ep.user_id != user_id:
                continue
            if session_id is not None and ep.session_id != session_id:
                continue
            if before is not None and ep.occurred_at >= before:
                continue
            await self._client.delete(self._key_for(ep.id))
            deleted += 1
        # Facts: same scan-and-delete pattern via the FactStore's
        # internals.
        if session_id is None and self.facts is not None:
            facts = await self.facts.query(user_id=user_id, limit=100_000)
            if before is not None:
                facts = [f for f in facts if f.recorded_at < before]
            for f in facts:
                if hasattr(self.facts, "_key_for"):
                    key = self.facts._key_for(f.id)  # type: ignore[attr-defined]
                    await self._client.delete(key)
                    deleted += 1
        return deleted

    async def export(
        self, *, user_id: str | None = None
    ) -> MemoryExport:
        episodes = await self._scan_all_episodes()
        episodes = [e for e in episodes if e.user_id == user_id]
        facts: list[Fact] = []
        if self.facts is not None:
            facts = list(
                await self.facts.query(user_id=user_id, limit=100_000)
            )
        return MemoryExport(
            user_id=user_id,
            episodes=sorted(episodes, key=lambda e: e.occurred_at),
            facts=sorted(facts, key=lambda f: f.recorded_at),
        )

    # ---- key helpers -----------------------------------------------------

    def _key_for(self, episode_id: str) -> str:
        return f"{self._key_prefix}{episode_id}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Re-exports of the shared util so existing callers
# (``from .redis import _pack_float32``) keep working.
_pack_float32 = pack_float32
_unpack_float32 = unpack_float32


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


def _decode_field(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _decode_hash(data: dict[Any, Any]) -> Episode | None:
    """Pull an :class:`Episode` from a Redis HGETALL result."""
    # Keys may come back as bytes; normalise to str.
    norm: dict[str, Any] = {}
    for k, v in data.items():
        key = k.decode("utf-8") if isinstance(k, bytes) else str(k)
        norm[key] = v
    eid = _decode_field(norm.get("id", b""))
    if not eid:
        return None
    occurred_raw = _decode_field(norm.get("occurred_at", "0"))
    try:
        occurred_at = datetime.fromtimestamp(float(occurred_raw), tz=UTC)
    except ValueError:
        occurred_at = datetime.now(UTC)
    embedding_blob = norm.get("embedding")
    if isinstance(embedding_blob, bytes | bytearray):
        embedding: list[float] | None = _unpack_float32(bytes(embedding_blob))
    else:
        embedding = None
    user_id_raw = _decode_field(norm.get("user_id", ""))
    return Episode(
        id=eid,
        session_id=_decode_field(norm.get("session_id", "")),
        user_id=user_id_raw or None,
        occurred_at=occurred_at,
        input=_decode_field(norm.get("input", "")),
        output=_decode_field(norm.get("output", "")),
        embedding=embedding,
    )


def _decode_ft_search(result: Any) -> list[Episode]:
    """Translate a ``FT.SEARCH`` reply into Episodes.

    The reply shape is ``[total, id1, [k1, v1, k2, v2, ...], id2, [...], ...]``.
    """
    if not result or not isinstance(result, list):
        return []
    out: list[Episode] = []
    # First element is the total count.
    body = result[1:]
    for i in range(0, len(body), 2):
        if i + 1 >= len(body):
            break
        kvs = body[i + 1]
        if not isinstance(kvs, list):
            continue
        decoded: dict[str, Any] = {}
        for j in range(0, len(kvs), 2):
            if j + 1 >= len(kvs):
                break
            k = kvs[j]
            v = kvs[j + 1]
            key = k.decode("utf-8") if isinstance(k, bytes) else str(k)
            decoded[key] = v
        # Use the doc id as our episode id.
        doc_id = body[i]
        eid = doc_id.decode("utf-8") if isinstance(doc_id, bytes) else str(doc_id)
        # Strip the prefix if present.
        if ":" in eid:
            eid = eid.split(":", 1)[-1]
        occurred_raw = _decode_field(decoded.get("occurred_at", "0"))
        try:
            occurred_at = datetime.fromtimestamp(float(occurred_raw), tz=UTC)
        except ValueError:
            occurred_at = datetime.now(UTC)
        user_id_raw = _decode_field(decoded.get("user_id", ""))
        out.append(
            Episode(
                id=eid,
                session_id=_decode_field(decoded.get("session_id", "")),
                user_id=user_id_raw or None,
                occurred_at=occurred_at,
                input=_decode_field(decoded.get("input", "")),
                output=_decode_field(decoded.get("output", "")),
            )
        )
    return out


__all__ = [
    "RedisMemory",
    "DEFAULT_INDEX_NAME",
    "DEFAULT_KEY_PREFIX",
]
