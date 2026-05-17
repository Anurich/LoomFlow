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

from ..core._eviction import BoundedDict
from ..core.context import IsolationWarning
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
from ._hybrid import _BM25, hybrid_rank
from .consolidator import Consolidator
from .facts import FactStore, InMemoryFactStore

_DEFAULT_MAX_USERS = 100_000
_DEFAULT_USER_TTL_SECONDS = 24 * 3600  # 24h idle


class InMemoryMemory:
    """Dict-backed implementation of :class:`Memory`.

    Multi-tenant accounting (M10): per-user working-block state is
    held in a bounded LRU + TTL container so a runaway tenant or
    one-shot user_id explosion can't grow the in-process dict
    without limit. Defaults: ``max_users=100_000`` and
    ``user_idle_ttl_seconds=86_400`` (24h). Pass ``None`` to
    disable bounding for single-tenant or fixed-tenant deployments.
    Eviction *drops* a user's working blocks; callers needing
    durable spill-to-disk should use :class:`SqliteMemory` or a
    SQL-backed memory instead.
    """

    def __init__(
        self,
        *,
        consolidator: Consolidator | None = None,
        fact_store: FactStore | None = None,
        max_users: int | None = _DEFAULT_MAX_USERS,
        user_idle_ttl_seconds: float | None = _DEFAULT_USER_TTL_SECONDS,
    ) -> None:
        # Working blocks are partitioned by ``user_id`` to keep one
        # tenant's pinned context invisible to another. Outer key is
        # the user_id (so eviction drops ALL of a user's blocks
        # together — the right unit). Inner dict is name -> block.
        # ``user_id=None`` is its own bucket (anonymous /
        # single-tenant) and isn't subject to TTL eviction unless it
        # also goes idle.
        self._blocks: BoundedDict[str | None, dict[str, MemoryBlock]] = (
            BoundedDict(
                max_keys=max_users,
                ttl_seconds=user_idle_ttl_seconds,
            )
        )
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
            user_blocks = self._blocks.get(user_id, {})
            scoped = list(user_blocks.values())
        return sorted(scoped, key=lambda b: b.pinned_order)

    async def update_block(
        self, name: str, content: str, *, user_id: str | None = None
    ) -> None:
        async with self._lock:
            user_blocks = self._blocks.setdefault(user_id, {})
            existing = user_blocks.get(name)
            # New block's slot is the count BEFORE insertion;
            # update keeps the existing slot so order is stable.
            new_order = (
                existing.pinned_order if existing is not None
                else len(user_blocks)
            )
            user_blocks[name] = MemoryBlock(
                name=name,
                content=content,
                pinned_order=new_order,
            )

    async def append_block(
        self, name: str, content: str, *, user_id: str | None = None
    ) -> None:
        async with self._lock:
            user_blocks = self._blocks.setdefault(user_id, {})
            existing = user_blocks.get(name)
            if existing is None:
                user_blocks[name] = MemoryBlock(
                    name=name, content=content, pinned_order=len(user_blocks)
                )
            else:
                user_blocks[name] = MemoryBlock(
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
        """Hybrid recall via BM25 + Reciprocal Rank Fusion.

        :class:`InMemoryMemory` doesn't carry an embedder (its
        whole point is being zero-config), so the "vector" arm is
        empty and ``alpha`` collapses to BM25-only ranking. The
        result still uses :class:`EpisodeMatch` so downstream code
        can treat all backends uniformly — ``vector_score`` is
        ``None`` to flag that it wasn't computed for this backend.

        For real semantic recall, use :class:`VectorMemory` (which
        ships its own embedder + cosine + BM25 hybrid) or one of
        the persistent backends (Chroma, Postgres, Redis).

        Empty query and zero-match queries fall back to recency
        ordering with neutral scores so the caller always sees the
        most-recent episodes when no lexical signal is present.
        """
        # Reuse the existing partitioning + warning + time-range
        # logic by sharing a single private candidate-pool helper.
        candidates = await self._candidate_episodes(
            query, time_range, user_id
        )
        if not candidates:
            return []

        if not query.strip():
            # No query → recency ordering, neutral scores.
            sorted_recent = sorted(
                candidates, key=lambda e: e.occurred_at, reverse=True
            )[:limit]
            return [
                EpisodeMatch(episode=e, score=1.0) for e in sorted_recent
            ]

        # Build a BM25 index over the candidate pool only — no
        # cross-user contamination because filtering already
        # happened in ``_candidate_episodes``.
        texts = [f"{e.input}\n{e.output}" for e in candidates]
        bm25 = _BM25(texts)
        bm25_ranking = bm25.rank(query)

        # No vector arm in this backend; pass an empty ranking and
        # rely on hybrid_rank to fall through to BM25-only.
        fused = hybrid_rank(
            bm25_ranking=bm25_ranking,
            vector_ranking=[],
            alpha=alpha,
        )
        if not fused:
            # No BM25 hits either — fall back to recency so the
            # caller always gets *something* useful.
            sorted_recent = sorted(
                candidates, key=lambda e: e.occurred_at, reverse=True
            )[:limit]
            return [
                EpisodeMatch(episode=e, score=0.0) for e in sorted_recent
            ]
        return [
            EpisodeMatch(
                episode=candidates[idx],
                score=score,
                bm25_score=bm25_score,
                vector_score=vector_score,
            )
            for idx, score, bm25_score, vector_score in fused[:limit]
        ]

    async def _candidate_episodes(
        self,
        query: str,
        time_range: tuple[datetime, datetime] | None,
        user_id: str | None,
    ) -> list[Episode]:
        """Shared partition + time-range filter so ``recall`` and
        ``recall_scored`` share the user_id / time_range logic
        (including the IsolationWarning footgun-guard) and only
        differ on ranking."""
        async with self._lock:
            episodes = list(self._episodes.values())
        if user_id is None and any(e.user_id is not None for e in episodes):
            warnings.warn(
                "Memory.recall called without user_id, but the store "
                "contains episodes for one or more named users. The "
                "anonymous bucket is partitioned from named-user "
                "buckets, so this query will only see anonymous "
                "episodes. Did you forget to pass user_id=?",
                IsolationWarning,
                stacklevel=4,
            )
        episodes = [e for e in episodes if e.user_id == user_id]
        if time_range is not None:
            lo, hi = time_range
            episodes = [e for e in episodes if lo <= e.occurred_at <= hi]
        return episodes

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
            # Splice tool transcript between USER and ASSISTANT
            # so a resumed worker sees its prior tool work, not
            # just the final reply. ``None`` (default) means the
            # episode was written by an Agent without
            # ``persist_tool_transcripts=True`` — old behavior
            # preserved.
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
        # Flatten the nested ``user_id -> name -> block`` shape into
        # JSON-friendly ``"<user_id>::<name>"`` keys for test
        # introspection.
        flat: dict[str, Any] = {}
        for uid, user_blocks in self._blocks.items():
            for name, v in user_blocks.items():
                flat[f"{uid or ''}::{name}"] = v.model_dump()
        return {
            "blocks": flat,
            "episodes": {k: v.model_dump() for k, v in self._episodes.items()},
        }
