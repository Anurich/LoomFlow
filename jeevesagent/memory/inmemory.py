"""Dict-backed memory for tests and tiny demos.

Recall is naive: filter episodes by substring + recency. Production
deployments use :class:`PostgresMemory` (Phase 4) which uses pgvector
for semantic search and tracks bi-temporal facts.
"""

from __future__ import annotations

import threading
import warnings
from datetime import datetime
from typing import Any

import anyio

from ..core.context import IsolationWarning
from ..core.types import (
    Episode,
    Fact,
    MemoryBlock,
    MemoryExport,
    MemoryProfile,
    Message,
    Role,
)
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
        # Working blocks are partitioned by ``user_id`` to keep one
        # tenant's pinned context invisible to another. Storage key
        # is ``(user_id, name)``; ``user_id=None`` is its own bucket
        # (anonymous / single-tenant).
        self._blocks: dict[tuple[str | None, str], MemoryBlock] = {}
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
            # Pinned-order is per-user: count blocks already in this
            # user's bucket when assigning a slot to a new block.
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
                    name=name, content=content, pinned_order=user_count
                )
            else:
                self._blocks[key] = MemoryBlock(
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
        user_id: str | None = None,
    ) -> list[Episode]:
        # ``kind`` and the absent semantic store are pre-existing
        # caveats; both modes still go through recency-fallback.
        return await self._recall_recent(query, limit, time_range, user_id)

    async def _recall_recent(
        self,
        query: str,
        limit: int,
        time_range: tuple[datetime, datetime] | None,
        user_id: str | None,
    ) -> list[Episode]:
        async with self._lock:
            episodes = list(self._episodes.values())
        # Footgun protection: if the caller forgot ``user_id`` on a
        # memory that DOES contain named-user data, warn loudly.
        # The query is still safe (we'll only return None-bucket
        # rows below), but the dev has almost certainly left a
        # ``user_id=`` off somewhere and is about to be confused
        # by suspiciously-empty results.
        if user_id is None and any(e.user_id is not None for e in episodes):
            warnings.warn(
                "Memory.recall called without user_id, but the store "
                "contains episodes for one or more named users. The "
                "anonymous bucket is partitioned from named-user "
                "buckets, so this query will only see anonymous "
                "episodes. Did you forget to pass user_id=?",
                IsolationWarning,
                stacklevel=3,
            )
        # Hard namespace partition by ``user_id`` — see Episode docstring.
        # ``None`` filters to ``None``; a string filters to that exact
        # value; the buckets never cross.
        episodes = [e for e in episodes if e.user_id == user_id]
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

    # ---- session continuity ---------------------------------------------

    async def session_messages(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
        limit: int = 20,
    ) -> list[Message]:
        """Return user/assistant pairs from prior runs of this session.

        Materialises each persisted :class:`Episode` for the given
        ``session_id`` (within the ``user_id`` partition) into a
        ``[USER input, ASSISTANT output]`` pair, ordered oldest-first
        and capped at ``limit`` turns total — i.e. up to ``limit / 2``
        Q/A exchanges. Tool-call traces are not replayed; the final
        assistant text per turn is sufficient context for follow-ups.
        """
        async with self._lock:
            episodes = list(self._episodes.values())
        # Hard namespace partition + session match.
        episodes = [
            e
            for e in episodes
            if e.session_id == session_id and e.user_id == user_id
        ]
        episodes.sort(key=lambda e: e.occurred_at)
        # Each episode contributes 2 messages; keep the most-recent
        # ``limit`` total messages (so ``limit // 2`` recent turns).
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
            user_episodes = [
                e for e in self._episodes.values() if e.user_id == user_id
            ]
        # Sample of most-recent facts; the fact store handles the
        # filter for us via ``query``.
        sample = await self.facts.query(user_id=user_id, limit=10)
        fact_count = len(await self.facts.query(user_id=user_id, limit=10_000))

        # Sessions, newest-first, dedup'd.
        seen: set[str] = set()
        recent_sessions: list[str] = []
        for e in sorted(user_episodes, key=lambda x: x.occurred_at, reverse=True):
            if e.session_id in seen:
                continue
            seen.add(e.session_id)
            recent_sessions.append(e.session_id)
            if len(recent_sessions) >= 10:
                break

        last_seen: datetime | None = None
        if user_episodes:
            last_seen = max(e.occurred_at for e in user_episodes)

        return MemoryProfile(
            user_id=user_id,
            episode_count=len(user_episodes),
            fact_count=fact_count,
            last_seen=last_seen,
            recent_sessions=recent_sessions,
            sample_facts=list(sample),
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
            to_delete: list[str] = []
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
                self._consolidated_ids.discard(eid)
            deleted += len(to_delete)

        # Facts are also user-scoped; erase them in the same call
        # unless the caller narrowed by session_id (facts have no
        # session, so a session-scoped forget shouldn't touch them).
        if session_id is None:
            facts_to_delete = await self.facts.query(
                user_id=user_id, limit=100_000
            )
            if before is not None:
                facts_to_delete = [
                    f for f in facts_to_delete
                    if f.recorded_at < before
                ]
            for f in facts_to_delete:
                # InMemoryFactStore exposes _facts; we'd rather use a
                # public delete but the protocol doesn't have one
                # (yet). Best-effort.
                if hasattr(self.facts, "_facts"):
                    self.facts._facts.pop(f.id, None)  # type: ignore[attr-defined]
                    if hasattr(self.facts, "_embeddings"):
                        self.facts._embeddings.pop(f.id, None)  # type: ignore[attr-defined]
                deleted += 1

        # Working blocks aren't user-scoped today (one set per
        # Memory instance); don't touch them.
        return deleted

    async def export(
        self, *, user_id: str | None = None
    ) -> MemoryExport:
        async with self._lock:
            episodes = [
                e for e in self._episodes.values() if e.user_id == user_id
            ]
        facts = await self.facts.query(user_id=user_id, limit=100_000)
        return MemoryExport(
            user_id=user_id,
            episodes=sorted(episodes, key=lambda e: e.occurred_at),
            facts=sorted(facts, key=lambda f: f.recorded_at),
        )

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
        # Block keys are now ``(user_id, name)`` tuples; flatten to
        # a string key for JSON-friendly snapshots so test
        # introspection keeps working without per-tuple keys.
        return {
            "blocks": {
                f"{uid or ''}::{name}": v.model_dump()
                for (uid, name), v in self._blocks.items()
            },
            "episodes": {k: v.model_dump() for k, v in self._episodes.items()},
        }
