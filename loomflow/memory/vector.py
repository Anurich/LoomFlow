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

import math
from datetime import datetime
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
from .consolidator import Consolidator
from .embedder import HashEmbedder
from .facts import FactStore, InMemoryFactStore


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
        # Working blocks are partitioned by ``user_id``. Storage key
        # is ``(user_id, name)``; ``user_id=None`` is its own bucket.
        self._blocks: dict[tuple[str | None, str], MemoryBlock] = {}
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
                for (uid, name), v in self._blocks.items()
            },
            "episodes": {k: v.model_dump() for k, v in self._episodes.items()},
        }

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
        from ._hybrid import _BM25, hybrid_rank

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

        # Vector arm — cosine over precomputed embeddings.
        query_embedding = await self._embedder.embed(query)
        vector_scores: list[tuple[int, float]] = []
        for i, ep in enumerate(candidates):
            assert ep.embedding is not None
            sim = _cosine(query_embedding, ep.embedding)
            if sim > 0:  # filter out negatives so RRF doesn't promote them
                vector_scores.append((i, sim))
        vector_scores.sort(key=lambda x: x[1], reverse=True)

        # BM25 arm — lexical ranking over the same candidate pool.
        texts = [f"{e.input}\n{e.output}" for e in candidates]
        bm25 = _BM25(texts)
        bm25_ranking = bm25.rank(query)

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
            all_facts = await self.facts.query(user_id=user_id, limit=100_000)
            fact_count = len(all_facts)

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
        if session_id is None and self.facts is not None:
            facts = await self.facts.query(user_id=user_id, limit=100_000)
            if before is not None:
                facts = [f for f in facts if f.recorded_at < before]
            for f in facts:
                if hasattr(self.facts, "_facts"):
                    self.facts._facts.pop(f.id, None)  # type: ignore[attr-defined]
                deleted += 1
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
