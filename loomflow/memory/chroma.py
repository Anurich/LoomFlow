"""Memory backed by Chroma (local persistent or in-memory client).

Chroma's Python API is sync; we dispatch every blocking call to a
worker thread via :func:`anyio.to_thread.run_sync` so the event loop
stays free.

Working blocks are kept in process memory (small, re-derivable);
episodes go to Chroma. The collection is created lazily on first use
and — if a ``persist_directory`` was supplied — survives process
restarts.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import anyio

from ..core._eviction import BoundedDict
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
from ._user_key import (
    decode_legacy_user_id,
    encode_user_id,
    user_id_where_clause,
)
from .embedder import HashEmbedder, warn_hash_embedder_fallback
from .facts import count_facts, delete_facts
from .inmemory import _DEFAULT_MAX_USERS, _DEFAULT_USER_TTL_SECONDS

DEFAULT_COLLECTION = "jeeves_episodes"


class ChromaMemory:
    """Memory backed by ``chromadb``.

    Construct via :meth:`local` for an on-disk persistent client or
    :meth:`ephemeral` for a process-local in-memory client.
    """

    def __init__(
        self,
        client: Any,
        *,
        embedder: Embedder | None = None,
        collection_name: str = DEFAULT_COLLECTION,
        fact_store: Any | None = None,
    ) -> None:
        self._client = client
        if embedder is None:
            warn_hash_embedder_fallback("ChromaMemory")
            embedder = HashEmbedder()
        self._embedder: Embedder = embedder
        self._collection_name = collection_name
        self._collection: Any | None = None
        # Working blocks partition by ``user_id`` (outer key); the
        # container is bounded (LRU + TTL) with the same defaults as
        # :class:`InMemoryMemory` so a runaway tenant explosion can't
        # grow the in-process dict without limit.
        self._blocks: BoundedDict[str | None, dict[str, MemoryBlock]] = (
            BoundedDict(
                max_keys=_DEFAULT_MAX_USERS,
                ttl_seconds=_DEFAULT_USER_TTL_SECONDS,
            )
        )
        self._lock = anyio.Lock()
        # ``facts`` is the Agent loop's hook for surfacing semantic
        # claims into the model's context. Defaults to ``None`` to
        # avoid creating a second Chroma collection by surprise; pass
        # an explicit :class:`ChromaFactStore` or use
        # :meth:`ChromaMemory.ephemeral` / :meth:`ChromaMemory.local`
        # with ``with_facts=True`` to wire one in.
        self.facts: Any | None = fact_store

    # ---- factory ---------------------------------------------------------

    @classmethod
    def local(
        cls,
        persist_directory: str,
        *,
        embedder: Embedder | None = None,
        collection_name: str = DEFAULT_COLLECTION,
        with_facts: bool = False,
        facts_collection_name: str = "jeeves_facts",
    ) -> ChromaMemory:
        """Persistent on-disk client at ``persist_directory``.

        ``with_facts=True`` attaches a :class:`ChromaFactStore` rooted
        at the same client so facts persist alongside episodes in the
        same on-disk store.
        """
        client = _make_client(persist_directory=persist_directory)
        instance = cls(
            client, embedder=embedder, collection_name=collection_name
        )
        if with_facts:
            from .chroma_facts import ChromaFactStore

            instance.facts = ChromaFactStore(
                client,
                embedder=instance._embedder,
                collection_name=facts_collection_name,
            )
        return instance

    @classmethod
    def ephemeral(
        cls,
        *,
        embedder: Embedder | None = None,
        collection_name: str = DEFAULT_COLLECTION,
        with_facts: bool = False,
        facts_collection_name: str = "jeeves_facts",
    ) -> ChromaMemory:
        """In-memory client (lost on process exit). Great for tests."""
        client = _make_client(persist_directory=None)
        instance = cls(
            client, embedder=embedder, collection_name=collection_name
        )
        if with_facts:
            from .chroma_facts import ChromaFactStore

            instance.facts = ChromaFactStore(
                client,
                embedder=instance._embedder,
                collection_name=facts_collection_name,
            )
        return instance

    # ---- collection lazy-init -------------------------------------------

    async def _get_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        # ``get_or_create_collection`` is sync; dispatch to thread.
        coll = await anyio.to_thread.run_sync(
            lambda: self._client.get_or_create_collection(
                name=self._collection_name
            )
        )
        self._collection = coll
        return coll

    # ---- working blocks --------------------------------------------------

    async def working(
        self, *, user_id: str | None = None
    ) -> list[MemoryBlock]:
        async with self._lock:
            user_blocks = self._blocks.get(user_id, {})
            scoped = list(user_blocks.values())
        return sorted(scoped, key=lambda b: b.pinned_order)

    async def update_block(
        self, name: str, content: str, *, user_id: str | None = None
    ) -> None:
        async with self._lock:
            user_blocks = self._blocks.setdefault(user_id, {})
            existing = user_blocks.get(name)
            user_blocks[name] = MemoryBlock(
                name=name,
                content=content,
                pinned_order=(
                    existing.pinned_order
                    if existing is not None
                    else len(user_blocks)
                ),
            )

    async def append_block(
        self, name: str, content: str, *, user_id: str | None = None
    ) -> None:
        async with self._lock:
            user_blocks = self._blocks.setdefault(user_id, {})
            existing = user_blocks.get(name)
            if existing is None:
                user_blocks[name] = MemoryBlock(
                    name=name,
                    content=content,
                    pinned_order=len(user_blocks),
                )
            else:
                user_blocks[name] = MemoryBlock(
                    name=name,
                    content=existing.content + content,
                    pinned_order=existing.pinned_order,
                )

    # ---- episodes --------------------------------------------------------

    async def remember(self, episode: Episode) -> str:
        if episode.embedding is None:
            text = _embedding_text(episode)
            embedding = await self._embedder.embed(text)
            episode = episode.model_copy(update={"embedding": embedding})

        coll = await self._get_collection()
        document = _embedding_text(episode)
        # Store ``user_id`` as a metadata field so Chroma's ``where``
        # filter can partition recall queries natively. Chroma rejects
        # ``None`` metadata values, so the anonymous bucket uses the
        # shared sentinel (see ``memory._user_key``); legacy rows used
        # the empty string and are still matched/decoded on read.
        metadata = {
            "session_id": episode.session_id,
            "user_id": encode_user_id(episode.user_id),
            "input": episode.input,
            "output": episode.output,
            "occurred_at": episode.occurred_at.isoformat(),
        }
        # Tool transcript — Chroma metadata only accepts scalar
        # values (str/int/float/bool), so we serialise the list of
        # Message objects to a JSON string and key it under
        # ``tool_transcript_json``. ``None`` (default — feature
        # disabled) skips the key entirely so existing rows stay
        # byte-identical and recall queries don't waste bandwidth
        # on an empty field. The decoder round-trips the JSON back
        # into Message objects.
        if episode.tool_transcript is not None:
            import json
            metadata["tool_transcript_json"] = json.dumps(
                [msg.model_dump(mode="json") for msg in episode.tool_transcript]
            )
        embedding = list(episode.embedding) if episode.embedding else []
        await anyio.to_thread.run_sync(
            lambda: coll.upsert(
                ids=[episode.id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata],
            )
        )
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
        coll = await self._get_collection()

        if not query.strip():
            return await self._recall_recent(coll, limit, time_range, user_id)

        query_embedding = list(await self._embedder.embed(query))

        # Hard namespace partition by ``user_id``, pushed into Chroma's
        # native ``where`` filter so we don't waste a round-trip on
        # other users' rows.
        where_filter = user_id_where_clause(user_id)

        result = await anyio.to_thread.run_sync(
            lambda: coll.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_filter,
            )
        )
        episodes = [ep for ep, _ in _decode_query_result_pairs(result)]

        if time_range is not None:
            lo, hi = time_range
            episodes = [e for e in episodes if lo <= e.occurred_at <= hi]
        return episodes

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

        We fetch the user's candidate pool with embeddings via
        ``coll.get`` (the same path :meth:`_recall_recent` uses — it
        carries the stored vectors, unlike ``coll.query`` which we'd
        otherwise have to re-rank by Chroma's distance), then compute
        cosine and BM25 in process so we can populate both component
        scores. Empty queries fall through to recency with neutral
        ``1.0`` scores; a no-match query falls through to recency with
        ``0.0``.
        """
        from ._hybrid import hybrid_rank_episodes

        coll = await self._get_collection()

        if not query.strip():
            recent = await self._recall_recent(
                coll, limit, time_range, user_id
            )
            return [EpisodeMatch(episode=e, score=1.0) for e in recent]

        # Sentinel-aware partition filter (matches legacy ""-encoded
        # rows too) — NOT the raw ``user_id or ""`` form this method
        # used before the shared ``_user_key`` encoding landed.
        where_filter = user_id_where_clause(user_id)
        result = await anyio.to_thread.run_sync(
            lambda: coll.get(
                where=where_filter,
                include=["metadatas", "documents", "embeddings"],
            )
        )
        episodes = _decode_get_result(result)
        if time_range is not None:
            lo, hi = time_range
            episodes = [e for e in episodes if lo <= e.occurred_at <= hi]
        candidates = [e for e in episodes if e.embedding is not None]
        if not candidates:
            return []

        query_embedding = list(await self._embedder.embed(query))
        return hybrid_rank_episodes(
            candidates,
            query=query,
            query_embedding=query_embedding,
            alpha=alpha,
            limit=limit,
        )

    async def _recall_recent(
        self,
        coll: Any,
        limit: int,
        time_range: tuple[datetime, datetime] | None,
        user_id: str | None,
    ) -> list[Episode]:
        where_filter = user_id_where_clause(user_id)
        result = await anyio.to_thread.run_sync(
            lambda: coll.get(
                limit=None,  # we'll sort + slice ourselves
                where=where_filter,
                include=["metadatas", "documents", "embeddings"],
            )
        )
        episodes = _decode_get_result(result)
        if time_range is not None:
            lo, hi = time_range
            episodes = [e for e in episodes if lo <= e.occurred_at <= hi]
        episodes.sort(key=lambda e: e.occurred_at, reverse=True)
        return episodes[:limit]

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
        coll = await self._get_collection()
        # Native ``where`` filter — namespace partition on user_id +
        # session pin.
        where_filter = {
            "$and": [
                user_id_where_clause(user_id),
                {"session_id": session_id},
            ]
        }
        result = await anyio.to_thread.run_sync(
            lambda: coll.get(
                where=where_filter,
                include=["metadatas", "documents", "embeddings"],
            )
        )
        episodes = _decode_get_result(result)
        episodes.sort(key=lambda e: e.occurred_at)
        max_episodes = max(1, limit // 2)
        episodes = episodes[-max_episodes:]
        out: list[Message] = []
        for ep in episodes:
            if ep.input:
                out.append(Message(role=Role.USER, content=ep.input))
            # Splice tool transcript between USER and ASSISTANT so
            # a resumed worker sees its prior tool work. The
            # decoder already round-tripped ``tool_transcript_json``
            # into a list of Message objects on the Episode.
            if ep.tool_transcript:
                out.extend(ep.tool_transcript)
            if ep.output:
                out.append(Message(role=Role.ASSISTANT, content=ep.output))
        return out

    # ---- profile / forget / export (GDPR) -------------------------------

    async def profile(
        self, *, user_id: str | None = None
    ) -> MemoryProfile:
        coll = await self._get_collection()
        where_filter = user_id_where_clause(user_id)
        result = await anyio.to_thread.run_sync(
            lambda: coll.get(
                where=where_filter, include=["metadatas", "documents", "embeddings"]
            )
        )
        episodes = _decode_get_result(result)
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
        coll = await self._get_collection()
        where_filter: dict[str, Any] = user_id_where_clause(user_id)
        # Chroma's where filter doesn't natively support "<" on
        # numeric strings; fetch then post-filter for the time-range
        # case. Session filter we can push down.
        if session_id is not None:
            where_filter = {
                "$and": [
                    user_id_where_clause(user_id),
                    {"session_id": session_id},
                ]
            }
        result = await anyio.to_thread.run_sync(
            lambda: coll.get(where=where_filter, include=["metadatas"])
        )
        ids = list(result.get("ids") or [])
        if before is not None:
            metas = list(result.get("metadatas") or [])
            keep_idx = []
            for i, meta in enumerate(metas):
                if meta is None:
                    continue
                ts = meta.get("occurred_at")
                if isinstance(ts, str):
                    try:
                        if datetime.fromisoformat(ts) < before:
                            keep_idx.append(i)
                    except ValueError:
                        pass
            ids = [ids[i] for i in keep_idx]
        if ids:
            await anyio.to_thread.run_sync(lambda: coll.delete(ids=ids))
        deleted = len(ids)

        # Facts: delegate to the FactStore's public ``delete``.
        if session_id is None and self.facts is not None:
            deleted += await delete_facts(
                self.facts, user_id=user_id, before=before
            )
        return deleted

    async def export(
        self, *, user_id: str | None = None
    ) -> MemoryExport:
        coll = await self._get_collection()
        result = await anyio.to_thread.run_sync(
            lambda: coll.get(
                where=user_id_where_clause(user_id),
                include=["metadatas", "documents", "embeddings"],
            )
        )
        episodes = _decode_get_result(result)
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

    async def consolidate(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(*, persist_directory: str | None) -> Any:
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover — depends on user env
        raise ImportError(
            "chromadb is not installed. "
            "Install with: pip install chromadb"
        ) from exc
    if persist_directory is None:
        return chromadb.EphemeralClient()
    return chromadb.PersistentClient(path=persist_directory)


def _embedding_text(episode: Episode) -> str:
    return "\n".join(p for p in (episode.input, episode.output) if p)


def _parse_occurred(meta: dict[str, Any]) -> datetime:
    raw = meta.get("occurred_at")
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            pass
    return datetime.now(UTC)


def _safe_list(result: dict[str, Any], key: str) -> list[Any]:
    """``result[key]`` may be None, a list, or (for embeddings) a numpy
    array. ``or []`` doesn't work on numpy arrays — they raise
    ``ValueError: The truth value of an array... is ambiguous`` — so we
    use an explicit None check."""
    val = result.get(key)
    return list(val) if val is not None else []


def _decode_query_result_pairs(
    result: dict[str, Any],
) -> list[tuple[Episode, float | None]]:
    """Translate a Chroma ``query()`` result into
    ``(episode, similarity)`` pairs. The similarity is derived from
    Chroma's returned distance via the monotone transform
    ``1 / (1 + distance)`` (``None`` when the query didn't include
    distances)."""
    ids_lists = _safe_list(result, "ids")
    metas_lists = _safe_list(result, "metadatas")
    embeds_lists = _safe_list(result, "embeddings")
    dists_lists = _safe_list(result, "distances")

    ids = list(ids_lists[0]) if ids_lists else []
    metas = list(metas_lists[0]) if metas_lists else []
    embeds = list(embeds_lists[0]) if embeds_lists else []
    dists = list(dists_lists[0]) if dists_lists else []

    episodes = _episodes_from_parallel(ids, metas, embeds)
    out: list[tuple[Episode, float | None]] = []
    for i, ep in enumerate(episodes):
        score: float | None = None
        if i < len(dists) and dists[i] is not None:
            distance = max(0.0, float(dists[i]))
            score = 1.0 / (1.0 + distance)
        out.append((ep, score))
    return out


def _decode_get_result(result: dict[str, Any]) -> list[Episode]:
    """Translate a Chroma ``get()`` result (flat lists) into Episodes."""
    ids = _safe_list(result, "ids")
    metas = _safe_list(result, "metadatas")
    embeds = _safe_list(result, "embeddings")
    return _episodes_from_parallel(ids, metas, embeds)


def _episodes_from_parallel(
    ids: list[Any],
    metas: list[Any],
    embeds: list[Any],
) -> list[Episode]:
    episodes: list[Episode] = []
    for i, eid in enumerate(ids):
        meta = metas[i] if i < len(metas) and metas[i] is not None else {}
        emb = list(embeds[i]) if i < len(embeds) else None
        # Chroma can't store ``None`` — the anonymous bucket is the
        # shared sentinel on the wire (legacy rows: empty string);
        # both decode back to ``None`` here.
        user_id_val = decode_legacy_user_id(str(meta.get("user_id", "")))
        # Round-trip tool_transcript_json → list[Message] if present.
        # Missing key (pre-feature episodes, or feature disabled at
        # write time) leaves the field at its default ``None``.
        transcript_json = meta.get("tool_transcript_json")
        tool_transcript: list[Message] | None = None
        if transcript_json:
            import json
            tool_transcript = [
                Message.model_validate(m)
                for m in json.loads(str(transcript_json))
            ]
        episodes.append(
            Episode(
                id=str(eid),
                session_id=str(meta.get("session_id", "")),
                user_id=user_id_val,
                occurred_at=_parse_occurred(meta),
                input=str(meta.get("input", "")),
                output=str(meta.get("output", "")),
                embedding=emb,
                tool_transcript=tool_transcript,
            )
        )
    return episodes
