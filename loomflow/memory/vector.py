"""In-memory :class:`Memory` with embedding-based semantic recall.

Episodes are stored in a dict keyed by id; on :meth:`remember` we
compute and attach an embedding (if the caller didn't already supply
one). :meth:`recall` embeds the query and ranks all episodes by cosine
similarity, with optional time-range filtering.

This backend doesn't scale past a few thousand episodes — the recall is
O(N) over every episode every call. Past that, switch to
:class:`PostgresMemory` (HNSW index) or :class:`ChromaMemory`.
"""

from __future__ import annotations

from datetime import datetime
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
from ._embedding_util import cosine as _cosine
from .consolidator import Consolidator
from .embedder import HashEmbedder
from .facts import FactStore, InMemoryFactStore, count_facts, delete_facts
from .inmemory import _DEFAULT_MAX_USERS, _DEFAULT_USER_TTL_SECONDS


def _embedding_text(episode: Episode) -> str:
    parts = [episode.input, episode.output]
    return "\n".join(p for p in parts if p)


class VectorMemory:
    """Pure-Python embedding-backed :class:`Memory`."""

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        max_episodes: int | None = None,
        consolidator: Consolidator | None = None,
        fact_store: FactStore | None = None,
    ) -> None:
        self._embedder: Embedder = embedder if embedder is not None else HashEmbedder()
        self._max_episodes = max_episodes
        # Working blocks are partitioned by ``user_id`` (outer key,
        # so eviction drops a user's blocks together); the container
        # is bounded (LRU + TTL) with the same defaults as
        # :class:`InMemoryMemory` so a runaway tenant explosion can't
        # grow the in-process dict without limit.
        self._blocks: BoundedDict[str | None, dict[str, MemoryBlock]] = (
            BoundedDict(
                max_keys=_DEFAULT_MAX_USERS,
                ttl_seconds=_DEFAULT_USER_TTL_SECONDS,
            )
        )
        self._episodes: dict[str, Episode] = {}
        self._consolidator = consolidator
        # Default the fact store's embedder to ours, so the same
        # embedder powers both episode and fact recall.
        self.facts: FactStore = (
            fact_store
            if fact_store is not None
            else InMemoryFactStore(embedder=self._embedder)
        )
        self._consolidated_ids: set[str] = set()
        self._lock = anyio.Lock()

    # ---- introspection (test helper) ------------------------------------

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    def snapshot(self) -> dict[str, Any]:
        return {
            "blocks": {
                f"{uid or ''}::{name}": v.model_dump()
                for uid, user_blocks in self._blocks.items()
                for name, v in user_blocks.items()
            },
            "episodes": {k: v.model_dump() for k, v in self._episodes.items()},
        }

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

        async with self._lock:
            self._episodes[episode.id] = episode
            if (
                self._max_episodes is not None
                and len(self._episodes) > self._max_episodes
            ):
                # FIFO eviction by recorded time.
                oldest = min(
                    self._episodes.values(), key=lambda e: e.occurred_at
                )
                if oldest.id != episode.id:
                    self._episodes.pop(oldest.id, None)
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
        if not query.strip():
            return await self._recall_recent(limit, time_range, user_id)

        query_embedding = await self._embedder.embed(query)

        async with self._lock:
            candidates = [
                e
                for e in self._episodes.values()
                if e.embedding is not None
            ]

        # Hard namespace partition by ``user_id``.
        candidates = [e for e in candidates if e.user_id == user_id]
        if time_range is not None:
            lo, hi = time_range
            candidates = [
                e for e in candidates if lo <= e.occurred_at <= hi
            ]

        scored: list[tuple[float, Episode]] = []
        for ep in candidates:
            assert ep.embedding is not None  # filtered above
            scored.append((_cosine(query_embedding, ep.embedding), ep))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [ep for _, ep in scored[:limit]]

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
        """Hybrid recall: cosine vector similarity + BM25 lexical
        scoring fused via Reciprocal Rank Fusion.

        ``alpha`` ∈ ``[0, 1]`` controls the lexical-vs-vector mix:

        * ``0.0`` — pure BM25 (best for exact-term queries:
          model names, error codes, person names)
        * ``1.0`` — pure vector cosine (best for semantic queries)
        * ``0.5`` (default) — balanced; matches the same default
          used by :meth:`InMemoryVectorStore.search_hybrid`

        Returns :class:`EpisodeMatch` carrying both component
        scores so downstream consumers (rerankers, MMR, score-
        threshold filters) can reason about *why* each result was
        chosen without re-running recall.

        Empty queries fall through to recency ordering with
        neutral scores.
        """
        from ._hybrid import hybrid_rank_episodes

        if not query.strip():
            recent = await self._recall_recent(limit, time_range, user_id)
            return [EpisodeMatch(episode=e, score=1.0) for e in recent]

        async with self._lock:
            candidates = [
                e
                for e in self._episodes.values()
                if e.embedding is not None
            ]
        candidates = [e for e in candidates if e.user_id == user_id]
        if time_range is not None:
            lo, hi = time_range
            candidates = [
                e for e in candidates if lo <= e.occurred_at <= hi
            ]
        if not candidates:
            return []

        query_embedding = await self._embedder.embed(query)
        return hybrid_rank_episodes(
            candidates,
            query=query,
            query_embedding=query_embedding,
            alpha=alpha,
            limit=limit,
        )

    async def _recall_recent(
        self,
        limit: int,
        time_range: tuple[datetime, datetime] | None,
        user_id: str | None,
    ) -> list[Episode]:
        async with self._lock:
            episodes = list(self._episodes.values())
        episodes = [e for e in episodes if e.user_id == user_id]
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
        return await self.facts.recall_text(
            query, limit=limit, valid_at=valid_at, user_id=user_id
        )

    async def session_messages(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
        limit: int = 20,
    ) -> list[Message]:
        async with self._lock:
            episodes = list(self._episodes.values())
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
            # Splice tool transcript between USER and ASSISTANT
            # so a resumed worker sees its prior tool work.
            # ``None`` (default) preserves the legacy compact-pair
            # behavior for users without
            # ``persist_tool_transcripts=True``.
            if ep.tool_transcript:
                out.extend(ep.tool_transcript)
            if ep.output:
                out.append(Message(role=Role.ASSISTANT, content=ep.output))
        return out

    # ---- profile / forget / export (GDPR) -------------------------------

    async def profile(
        self, *, user_id: str | None = None
    ) -> MemoryProfile:
        async with self._lock:
            user_eps = [
                e for e in self._episodes.values() if e.user_id == user_id
            ]
        sample_facts: list[Fact] = []
        fact_count = 0
        if self.facts is not None:
            sample_facts = list(
                await self.facts.query(user_id=user_id, limit=10)
            )
            fact_count = await count_facts(self.facts, user_id=user_id)

        seen: set[str] = set()
        recent_sessions: list[str] = []
        for e in sorted(user_eps, key=lambda x: x.occurred_at, reverse=True):
            if e.session_id in seen:
                continue
            seen.add(e.session_id)
            recent_sessions.append(e.session_id)
            if len(recent_sessions) >= 10:
                break

        last_seen: datetime | None = None
        if user_eps:
            last_seen = max(e.occurred_at for e in user_eps)

        return MemoryProfile(
            user_id=user_id,
            episode_count=len(user_eps),
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
        deleted = 0
        async with self._lock:
            to_delete = []
            for eid, ep in self._episodes.items():
                if ep.user_id != user_id:
                    continue
                if session_id is not None and ep.session_id != session_id:
                    continue
                if before is not None and ep.occurred_at >= before:
                    continue
                to_delete.append(eid)
            for eid in to_delete:
                self._episodes.pop(eid, None)
            deleted += len(to_delete)
        # Facts: delegate to the FactStore's public ``delete`` so the
        # count reflects rows actually removed.
        if session_id is None and self.facts is not None:
            deleted += await delete_facts(
                self.facts, user_id=user_id, before=before
            )
        return deleted

    async def export(
        self, *, user_id: str | None = None
    ) -> MemoryExport:
        async with self._lock:
            episodes = [
                e for e in self._episodes.values() if e.user_id == user_id
            ]
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
        """Process unconsolidated episodes through the configured
        :class:`Consolidator`, appending facts to ``self.facts``.

        No-op when no consolidator is configured.
        """
        if self._consolidator is None:
            return
        async with self._lock:
            pending = [
                ep
                for ep in self._episodes.values()
                if ep.id not in self._consolidated_ids
            ]
        if not pending:
            return
        await self._consolidator.consolidate(pending, store=self.facts)
        async with self._lock:
            for ep in pending:
                self._consolidated_ids.add(ep.id)
