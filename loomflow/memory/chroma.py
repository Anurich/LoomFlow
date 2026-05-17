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
from ._hybrid import default_recall_scored
from .embedder import HashEmbedder

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
        self._embedder: Embedder = embedder if embedder is not None else HashEmbedder()
        self._collection_name = collection_name
        self._collection: Any | None = None
        # Working blocks partition by ``user_id``; key is ``(user_id, name)``.
        self._blocks: dict[tuple[str | None, str], MemoryBlock] = {}
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
            text = _embedding_text(episode)
            embedding = await self._embedder.embed(text)
            episode = episode.model_copy(update={"embedding": embedding})

        coll = await self._get_collection()
        document = _embedding_text(episode)
        # Store ``user_id`` as a metadata field so Chroma's ``where``
        # filter can partition recall queries natively. Chroma rejects
        # ``None`` metadata values, so we substitute the empty string
        # for the anonymous bucket and round-trip on read.
        metadata = {
            "session_id": episode.session_id,
            "user_id": episode.user_id or "",
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
        # other users' rows. Empty string is the anonymous bucket.
        where_filter = {"user_id": user_id or ""}

        result = await anyio.to_thread.run_sync(
            lambda: coll.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_filter,
            )
        )
        episodes = _decode_query_result(result)

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
        # Chroma already does vector recall under the hood; for now
        # we wrap its results with neutral scores. A future revision
        # could plumb Chroma's distance scores into ``vector_score``
        # — that needs the underlying ``query`` call to request
        # distances and a transform from L2/IP distance into a
        # comparable cosine similarity. Out of scope for this
        # protocol-evolution shim.
        eps = await self.recall(
            query,
            kind=kind,
            limit=limit,
            time_range=time_range,
            user_id=user_id,
        )
        return default_recall_scored(eps)

    async def _recall_recent(
        self,
        coll: Any,
        limit: int,
        time_range: tuple[datetime, datetime] | None,
        user_id: str | None,
    ) -> list[Episode]:
        where_filter = {"user_id": user_id or ""}
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
        # session pin. Empty-string is the anonymous bucket on disk.
        where_filter = {
            "$and": [
                {"user_id": user_id or ""},
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
        where_filter = {"user_id": user_id or ""}
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
        coll = await self._get_collection()
        where_filter: dict[str, Any] = {"user_id": user_id or ""}
        # Chroma's where filter doesn't natively support "<" on
        # numeric strings; fetch then post-filter for the time-range
        # case. Session filter we can push down.
        if session_id is not None:
            where_filter = {
                "$and": [
                    {"user_id": user_id or ""},
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

        # Facts: rely on the FactStore's ``query`` + per-id deletion.
        if session_id is None and self.facts is not None:
            facts = await self.facts.query(user_id=user_id, limit=100_000)
            if before is not None:
                facts = [f for f in facts if f.recorded_at < before]
            # ChromaFactStore stores facts in its own collection;
            # rely on the public ``query`` + private ``_collection``
            # for delete (no public delete method yet).
            if facts and hasattr(self.facts, "_collection"):
                fact_coll = self.facts._collection  # type: ignore[attr-defined]
                fact_ids = [f.id for f in facts]
                await anyio.to_thread.run_sync(
                    lambda: fact_coll.delete(ids=fact_ids)
                )
                deleted += len(fact_ids)
        return deleted

    async def export(
        self, *, user_id: str | None = None
    ) -> MemoryExport:
        coll = await self._get_collection()
        result = await anyio.to_thread.run_sync(
            lambda: coll.get(
                where={"user_id": user_id or ""},
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


def _decode_query_result(result: dict[str, Any]) -> list[Episode]:
    """Translate a Chroma ``query()`` result into our Episodes."""
    ids_lists = _safe_list(result, "ids")
    metas_lists = _safe_list(result, "metadatas")
    embeds_lists = _safe_list(result, "embeddings")

    ids = list(ids_lists[0]) if ids_lists else []
    metas = list(metas_lists[0]) if metas_lists else []
    embeds = list(embeds_lists[0]) if embeds_lists else []

    return _episodes_from_parallel(ids, metas, embeds)


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
        # Chroma can't store ``None`` so the anonymous bucket is the
        # empty string on the wire; round-trip back to ``None`` here.
        user_id_raw = str(meta.get("user_id", ""))
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
                user_id=user_id_raw or None,
                occurred_at=_parse_occurred(meta),
                input=str(meta.get("input", "")),
                output=str(meta.get("output", "")),
                embedding=emb,
                tool_transcript=tool_transcript,
            )
        )
    return episodes
