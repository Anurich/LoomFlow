"""Dict-backed memory for tests and tiny demos.

Recall is naive: filter episodes by substring + recency. Production
deployments use :class:`PostgresMemory` (Phase 4) which uses pgvector
for semantic search and tracks bi-temporal facts.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any

import anyio

from ..core.types import Episode, MemoryBlock
from .consolidator import Consolidator
from .facts import FactStore, InMemoryFactStore


class InMemoryMemory:
    """Dict-backed implementation of :class:`Memory`."""

    def __init__(
        self,
        *,
        consolidator: Consolidator | None = None,
        fact_store: FactStore | None = None,
    ) -> None:
        self._blocks: dict[str, MemoryBlock] = {}
        self._episodes: dict[str, Episode] = {}
        self._consolidator = consolidator
        self.facts: FactStore = (
            fact_store if fact_store is not None else InMemoryFactStore()
        )
        self._consolidated_ids: set[str] = set()
        # Use anyio.Lock so reads/writes coordinate cleanly under
        # structured concurrency.
        self._lock = anyio.Lock()
        # A second sync lock for the rare case we need to read state
        # without holding the async lock (currently unused).
        self._sync_lock = threading.RLock()

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
                pinned_order=existing.pinned_order if existing else len(self._blocks),
            )

    async def append_block(self, name: str, content: str) -> None:
        async with self._lock:
            existing = self._blocks.get(name)
            if existing is None:
                self._blocks[name] = MemoryBlock(
                    name=name, content=content, pinned_order=len(self._blocks)
                )
            else:
                self._blocks[name] = MemoryBlock(
                    name=name,
                    content=existing.content + content,
                    pinned_order=existing.pinned_order,
                )

    # ---- episodes --------------------------------------------------------

    async def remember(self, episode: Episode) -> str:
        async with self._lock:
            self._episodes[episode.id] = episode
            return episode.id

    async def recall(
        self,
        query: str,
        *,
        kind: str = "episodic",
        limit: int = 5,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list[Episode]:
        if kind == "semantic":
            # No semantic store yet — recency-fallback is acceptable for v0.
            return await self._recall_recent(query, limit, time_range)
        return await self._recall_recent(query, limit, time_range)

    async def _recall_recent(
        self,
        query: str,
        limit: int,
        time_range: tuple[datetime, datetime] | None,
    ) -> list[Episode]:
        async with self._lock:
            episodes = list(self._episodes.values())
        if time_range is not None:
            lo, hi = time_range
            episodes = [e for e in episodes if lo <= e.occurred_at <= hi]
        # Crude relevance: substring match wins; otherwise recency.
        q = query.lower().strip()
        if q:
            matched = [e for e in episodes if q in e.input.lower() or q in e.output.lower()]
            if matched:
                episodes = matched
        episodes.sort(key=lambda e: e.occurred_at, reverse=True)
        return episodes[:limit]

    async def consolidate(self) -> None:
        """Process unconsolidated episodes through the configured
        :class:`Consolidator`, appending facts to ``self.facts``."""
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

    # ---- introspection (test helpers) -----------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "blocks": {k: v.model_dump() for k, v in self._blocks.items()},
            "episodes": {k: v.model_dump() for k, v in self._episodes.items()},
        }
