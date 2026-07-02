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
    EpisodeMatch,
    Fact,
    MemoryBlock,
    MemoryExport,
    MemoryProfile,
    Message,
    Role,
)
from ._embedding_util import cosine, pack_float32, unpack_float32
from ._user_key import decode_legacy_user_id, encode_user_id
from .embedder import HashEmbedder, warn_hash_embedder_fallback
from .facts import count_facts, delete_facts

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
        if embedder is None:
            warn_hash_embedder_fallback("RedisMemory")
            embedder = HashEmbedder()
        self._embedder: Embedder = embedder
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

        Schema migration: indexes created before ``user_id`` was a
        TAG field are upgraded in place via ``FT.ALTER … SCHEMA ADD``
        — the least invasive option (no index rebuild, no key
        renames). RediSearch only applies a new attribute to
        documents (re)indexed AFTER the ALTER, so episodes written
        before the upgrade won't match the tag filter until they're
        rewritten; the Python-side post-filter in :meth:`recall`
        still keeps results correct, just potentially sparse for old
        data. ``FT.ALTER`` failing (attribute already exists, or an
        old server) is ignored.
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
                # ``user_id`` is indexed as a TAG so KNN queries can
                # apply the tenant partition inside RediSearch (hybrid
                # ``(@user_id:{…})=>[KNN …]``) instead of over-fetching
                # globally and post-filtering in Python.
                "user_id",
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
            else:
                # Pre-existing index — make sure it carries the
                # ``user_id`` TAG field (no-op if it already does).
                try:
                    await self._client.execute_command(
                        "FT.ALTER",
                        self._index_name,
                        "SCHEMA",
                        "ADD",
                        "user_id",
                        "TAG",
                    )
                except Exception:  # noqa: BLE001, S110
                    # Attribute already present / server too old —
                    # keep the vector path; worst case the tag filter
                    # returns nothing and the caller sees fewer rows.
                    pass
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
            # Persist ``user_id`` so recall queries can filter by
            # namespace partition. The anonymous bucket uses the
            # shared sentinel (see ``memory._user_key``); legacy rows
            # encoded it as the empty bytestring and are still
            # decoded correctly on read.
            "user_id": encode_user_id(episode.user_id).encode("utf-8"),
            "occurred_at": str(episode.occurred_at.timestamp()).encode("utf-8"),
            "input": episode.input.encode("utf-8"),
            "output": episode.output.encode("utf-8"),
            "embedding": embedding_bytes,
        }
        # Tool transcript — Redis hash fields are scalar, so we
        # serialise the list of Message objects to a single JSON
        # bytestring under ``tool_transcript_json``. ``None``
        # (default — feature disabled) skips the field entirely so
        # pre-feature episodes stay byte-identical and the wire
        # cost is zero for users without
        # ``persist_tool_transcripts=True``. Decoder round-trips
        # the JSON back into Message objects on read.
        if episode.tool_transcript is not None:
            import json
            mapping["tool_transcript_json"] = json.dumps(
                [msg.model_dump(mode="json") for msg in episode.tool_transcript]
            ).encode("utf-8")
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

        pairs = await self._recall_pairs(query, limit, time_range, user_id)
        return [ep for ep, _ in pairs]

    async def _recall_pairs(
        self,
        query: str,
        limit: int,
        time_range: tuple[datetime, datetime] | None,
        user_id: str | None,
    ) -> list[tuple[Episode, float | None]]:
        """Vector recall returning ``(episode, cosine_similarity)``
        pairs, best-first. The index path pushes the ``user_id``
        partition into RediSearch as a hybrid TAG filter; the Python
        post-filter below stays as a defensive backstop (and does the
        real work on the brute-force path)."""
        query_embedding = await self._embedder.embed(query)

        if self._use_vector_index:
            pairs = await self._recall_via_index(
                query_embedding, limit, user_id
            )
        else:
            # Brute-force scan lacks a native filter; over-fetch so
            # enough candidates survive the partition post-filter.
            fetch_limit = limit * 8 if user_id is not None else limit
            pairs = await self._recall_brute_force(
                query_embedding, fetch_limit
            )

        # Hard namespace partition by ``user_id``.
        pairs = [(e, s) for e, s in pairs if e.user_id == user_id]
        if time_range is not None:
            lo, hi = time_range
            pairs = [
                (e, s) for e, s in pairs if lo <= e.occurred_at <= hi
            ]
        return pairs[:limit]

    async def _recall_via_index(
        self,
        query_embedding: list[float],
        limit: int,
        user_id: str | None,
    ) -> list[tuple[Episode, float | None]]:
        # Hybrid query: tenant TAG filter applied inside RediSearch so
        # the KNN candidates all belong to this user — no more global
        # over-fetch that starves busy tenants at scale. Note:
        # episodes written by pre-sentinel versions carry an empty
        # ``user_id`` tag (which RediSearch doesn't index), so legacy
        # anonymous rows only surface via the brute-force path.
        tag = _escape_tag(encode_user_id(user_id))
        params = [
            "FT.SEARCH",
            self._index_name,
            f"(@user_id:{{{tag}}})=>[KNN {limit} @embedding $vec AS score]",
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
        return _decode_ft_search_pairs(result)

    async def recall_scored(
        self,
        query: str,
        *,
        kind: str = "episodic",
        limit: int = 5,
        time_range: tuple[datetime, datetime] | None = None,
        user_id: str | None = None,
        alpha: float = 0.5,
    ) -> list[EpisodeMatch]:
        """Native hybrid recall: BM25 lexical + cosine vector, fused
        via Reciprocal Rank Fusion. Mirrors
        :meth:`VectorMemory.recall_scored`.

        Candidates come from the hash scan (``_scan_all_episodes``),
        which carries the stored ``embedding`` blob on every episode —
        unlike the RediSearch index path, whose ``FT.SEARCH`` reply
        omits the vector. We then apply the user partition and
        time-range filter and score in process so both component scores
        can be populated. Empty queries fall through to recency with
        neutral ``1.0`` scores; a no-match query falls through to
        recency with ``0.0``.
        """
        from ._hybrid import _BM25, hybrid_rank

        await self.ensure_index()

        if not query.strip():
            recent = await self._recall_recent(limit, time_range, user_id)
            return [EpisodeMatch(episode=e, score=1.0) for e in recent]

        episodes = await self._scan_all_episodes()
        episodes = [e for e in episodes if e.user_id == user_id]
        if time_range is not None:
            lo, hi = time_range
            episodes = [e for e in episodes if lo <= e.occurred_at <= hi]
        candidates = [e for e in episodes if e.embedding is not None]
        if not candidates:
            return []

        # Vector arm — cosine over candidate embeddings; drop
        # non-positive sims so RRF doesn't promote them.
        query_embedding = await self._embedder.embed(query)
        vector_scores: list[tuple[int, float]] = []
        for i, ep in enumerate(candidates):
            assert ep.embedding is not None
            sim = cosine(query_embedding, ep.embedding)
            if sim > 0:
                vector_scores.append((i, sim))
        vector_scores.sort(key=lambda x: x[1], reverse=True)

        # BM25 arm — lexical ranking over the same candidate pool.
        texts = [f"{e.input}\n{e.output}" for e in candidates]
        bm25_ranking = _BM25(texts).rank(query)

        fused = hybrid_rank(
            bm25_ranking=bm25_ranking,
            vector_ranking=vector_scores,
            alpha=alpha,
        )
        if not fused:
            recent = sorted(
                candidates, key=lambda e: e.occurred_at, reverse=True
            )[:limit]
            return [EpisodeMatch(episode=e, score=0.0) for e in recent]
        return [
            EpisodeMatch(
                episode=candidates[idx],
                score=score,
                bm25_score=bm25_score,
                vector_score=vector_score,
            )
            for idx, score, bm25_score, vector_score in fused[:limit]
        ]

    async def _recall_brute_force(
        self, query_embedding: list[float], limit: int
    ) -> list[tuple[Episode, float | None]]:
        episodes = await self._scan_all_episodes()
        scored: list[tuple[float, Episode]] = []
        for ep in episodes:
            if ep.embedding is None:
                continue
            scored.append((_cosine(query_embedding, ep.embedding), ep))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [(ep, s) for s, ep in scored[:limit]]

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
            # Pipeline the per-key HGETALLs — one round-trip per SCAN
            # page instead of one per key. Falls back to sequential
            # calls for clients (fakes) without pipeline support.
            for data in await self._hgetall_many(list(keys)):
                if not data:
                    continue
                ep = _decode_hash(data)
                if ep is not None:
                    episodes.append(ep)
            if cursor == 0:
                break
        return episodes

    async def _hgetall_many(self, keys: list[Any]) -> list[dict[Any, Any]]:
        if not keys:
            return []
        pipeline_factory = getattr(self._client, "pipeline", None)
        if pipeline_factory is None:
            return [await self._client.hgetall(key) for key in keys]
        async with pipeline_factory(transaction=False) as pipe:
            for key in keys:
                pipe.hgetall(key)
            results = await pipe.execute()
        return list(results)

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
            # Splice tool transcript between USER and ASSISTANT so
            # a resumed worker sees its prior tool work. The
            # ``_decode_hash`` helper already round-tripped the
            # ``tool_transcript_json`` field into Message objects.
            if ep.tool_transcript:
                out.extend(ep.tool_transcript)
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
            fact_count = await count_facts(self.facts, user_id=user_id)
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
        # Scan all episode keys, filter Python-side, then remove the
        # matches in ONE batched UNLINK (non-blocking delete) instead
        # of a round-trip per key. Faster than maintaining a
        # secondary index for what's typically a low-frequency op.
        episodes = await self._scan_all_episodes()
        keys: list[str] = []
        for ep in episodes:
            if ep.user_id != user_id:
                continue
            if session_id is not None and ep.session_id != session_id:
                continue
            if before is not None and ep.occurred_at >= before:
                continue
            keys.append(self._key_for(ep.id))
        deleted = await _delete_keys(self._client, keys)
        # Facts: delegate to the FactStore's public ``delete``.
        if session_id is None and self.facts is not None:
            deleted += await delete_facts(
                self.facts, user_id=user_id, before=before
            )
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


def _escape_tag(value: str) -> str:
    """Escape a value for use inside a RediSearch TAG filter
    (``@field:{value}``): every non-alphanumeric character gets a
    backslash so ids with ``-``, ``@``, ``.`` etc. match literally."""
    return "".join(
        ch if (ch.isalnum() or ch == "_") else f"\\{ch}" for ch in value
    )


async def _delete_keys(client: Any, keys: list[str]) -> int:
    """Remove ``keys`` in one batched call. Prefers ``UNLINK``
    (non-blocking, reclaims memory off-thread) and falls back to
    ``DEL`` for servers / fakes without it. Returns the number of
    keys removed."""
    if not keys:
        return 0
    unlink = getattr(client, "unlink", None)
    if unlink is not None:
        result = await unlink(*keys)
    else:
        result = await client.delete(*keys)
    try:
        return int(result)
    except (TypeError, ValueError):
        return len(keys)


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
    # Sentinel (new rows) and legacy "" both decode to the anonymous
    # bucket — see ``memory._user_key``.
    user_id_val = decode_legacy_user_id(_decode_field(norm.get("user_id", "")))
    # Round-trip tool_transcript_json → list[Message]. Missing
    # field (pre-feature episodes / feature disabled) leaves
    # the Episode field at its default ``None`` which preserves
    # legacy session_messages() behavior.
    transcript_raw = norm.get("tool_transcript_json")
    tool_transcript: list[Message] | None = None
    if transcript_raw:
        import json
        decoded = _decode_field(transcript_raw)
        if decoded:
            tool_transcript = [
                Message.model_validate(m) for m in json.loads(decoded)
            ]
    return Episode(
        id=eid,
        session_id=_decode_field(norm.get("session_id", "")),
        user_id=user_id_val,
        occurred_at=occurred_at,
        input=_decode_field(norm.get("input", "")),
        output=_decode_field(norm.get("output", "")),
        embedding=embedding,
        tool_transcript=tool_transcript,
    )


def _decode_ft_search_pairs(
    result: Any,
) -> list[tuple[Episode, float | None]]:
    """Translate a ``FT.SEARCH`` reply into ``(episode, similarity)``
    pairs. The KNN clause returns the cosine *distance* under the
    ``score`` alias; we convert to a similarity via ``1 - distance``
    (``None`` when the field is missing).

    The reply shape is ``[total, id1, [k1, v1, k2, v2, ...], id2, [...], ...]``.
    """
    if not result or not isinstance(result, list):
        return []
    out: list[tuple[Episode, float | None]] = []
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
        user_id_val = decode_legacy_user_id(
            _decode_field(decoded.get("user_id", ""))
        )
        similarity: float | None = None
        score_raw = decoded.get("score")
        if score_raw is not None:
            try:
                similarity = 1.0 - float(_decode_field(score_raw))
            except ValueError:
                similarity = None
        out.append(
            (
                Episode(
                    id=eid,
                    session_id=_decode_field(decoded.get("session_id", "")),
                    user_id=user_id_val,
                    occurred_at=occurred_at,
                    input=_decode_field(decoded.get("input", "")),
                    output=_decode_field(decoded.get("output", "")),
                ),
                similarity,
            )
        )
    return out


def _decode_ft_search(result: Any) -> list[Episode]:
    """Back-compat wrapper: Episodes only, scores dropped."""
    return [ep for ep, _ in _decode_ft_search_pairs(result)]


__all__ = [
    "RedisMemory",
    "DEFAULT_INDEX_NAME",
    "DEFAULT_KEY_PREFIX",
]
