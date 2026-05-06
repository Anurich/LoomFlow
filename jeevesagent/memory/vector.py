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
from ..core.types import Episode, Fact, MemoryBlock
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
        self._blocks: dict[str, MemoryBlock] = {}
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
            "blocks": {k: v.model_dump() for k, v in self._blocks.items()},
            "episodes": {k: v.model_dump() for k, v in self._episodes.items()},
        }

    # ---- working blocks --------------------------------------------------

    async def working(self) -> list[MemoryBlock]:
        async with self._lock:
            return sorted(self._blocks.values(), key=lambda b: b.pinned_order)

    async def update_block(self, name: str, content: str) -> None:
        async with self._lock:
            existing = self._blocks.get(name)
            self._blocks[name] = MemoryBlock(
                name=name,
                content=content,
                pinned_order=(
                    existing.pinned_order if existing else len(self._blocks)
                ),
            )

    async def append_block(self, name: str, content: str) -> None:
        async with self._lock:
            existing = self._blocks.get(name)
            if existing is None:
                self._blocks[name] = MemoryBlock(
                    name=name,
                    content=content,
                    pinned_order=len(self._blocks),
                )
            else:
                self._blocks[name] = MemoryBlock(
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
    ) -> list[Episode]:
        if not query.strip():
            return await self._recall_recent(limit, time_range)

        query_embedding = await self._embedder.embed(query)

        async with self._lock:
            candidates = [
                e
                for e in self._episodes.values()
                if e.embedding is not None
            ]

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

    async def _recall_recent(
        self,
        limit: int,
        time_range: tuple[datetime, datetime] | None,
    ) -> list[Episode]:
        async with self._lock:
            episodes = list(self._episodes.values())
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
    ) -> list[Fact]:
        return await self.facts.recall_text(
            query, limit=limit, valid_at=valid_at
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
