"""SQLite-backed :class:`Memory` — persistent, single-file, no server.

Fills the gap between :class:`InMemoryMemory` (lost on restart) and
network-backed backends (Chroma / Postgres / Redis — real infra). One
``.db`` file holds every memory layer:

* ``working_blocks(name, content, pinned_order, updated_at)`` — the
  in-context blocks
* ``episodes(id, session_id, user_id, occurred_at, input, output,
  embedding)`` — episodic record + optional vector
* ``facts(...)`` via the existing :class:`SqliteFactStore` rooted at
  the same file (separate table; same DB)

What this is good for:

* Single-instance production apps that want persistence without
  running Postgres / Redis.
* Local dev where you want runs to survive ``ctrl-c``.
* CI / integration tests that need real durability without spinning
  up containers.

What this is NOT for:

* Concurrent writers from multiple processes — sqlite serialises
  writes, throughput suffers under contention. Use ``PostgresMemory``
  if you have multiple workers writing to the same memory.
* Vector search at million-row scale — we do brute-force cosine
  ranking in Python because sqlite has no native vector type. Fine
  for tens of thousands of episodes; if you have more, switch to
  Chroma or Postgres+pgvector.

Sync sqlite3 calls are dispatched through ``anyio.to_thread.run_sync``
so the agent loop's structured concurrency stays clean.
"""

from __future__ import annotations

import sqlite3
import threading
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio

from ..core.context import IsolationWarning
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
from .embedder import HashEmbedder
from .facts import FactStore
from .sqlite_facts import SqliteFactStore

__all__ = ["SqliteMemory"]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


_BLOCKS_DDL = """
CREATE TABLE IF NOT EXISTS working_blocks (
    user_id      TEXT,
    name         TEXT NOT NULL,
    content      TEXT NOT NULL,
    pinned_order INTEGER NOT NULL DEFAULT 0,
    updated_at   REAL NOT NULL,
    PRIMARY KEY (user_id, name)
)
"""

# Idempotent migration: pre-M9 schemas had ``name`` as the PRIMARY
# KEY (no user_id column). Adding ``user_id`` mid-flight requires a
# table rebuild because SQLite can't add a PK column. We detect the
# old shape and migrate by copying every row into a temp table that
# uses the new schema, then renaming.
_BLOCKS_MIGRATE_CHECK = (
    "SELECT COUNT(*) FROM pragma_table_info('working_blocks') "
    "WHERE name = 'user_id'"
)

_EPISODES_DDL = """
CREATE TABLE IF NOT EXISTS episodes (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    user_id     TEXT,
    occurred_at REAL NOT NULL,
    input       TEXT NOT NULL,
    output      TEXT NOT NULL,
    embedding   BLOB
)
"""

# Idempotent ALTER for upgrades from a pre-``user_id`` schema. SQLite
# will raise ``OperationalError: duplicate column name`` when the
# column already exists; we suppress that case in :meth:`_init_schema`.
_EPISODES_ADD_USER_ID = "ALTER TABLE episodes ADD COLUMN user_id TEXT"

_EPISODES_USER_SESSION_INDEX = (
    "CREATE INDEX IF NOT EXISTS episodes_user_session_idx "
    "ON episodes (user_id, session_id, occurred_at DESC)"
)
_EPISODES_USER_OCCURRED_INDEX = (
    "CREATE INDEX IF NOT EXISTS episodes_user_occurred_idx "
    "ON episodes (user_id, occurred_at DESC)"
)

# Sidecar table for ``Episode.tool_transcript`` — one row per
# captured tool-call / tool-result message, joined by ``episode_id``.
# Stored separately from ``episodes`` so the hot path (recall +
# session_messages without transcripts) doesn't pay the cost of
# loading large transcript blobs unless asked. ``sequence`` preserves
# insertion order so rehydration replays the messages in the order
# they were emitted. ``message_json`` is the full
# :class:`Message.model_dump_json()` — round-trips losslessly via
# ``Message.model_validate_json``.
#
# CASCADE delete via the ``ON DELETE CASCADE`` clause + a paired
# ``forget()`` DELETE on the episodes side keeps GDPR forget
# semantics intact — the transcript dies with its episode. Foreign
# keys are enabled per-connection in ``_connect()`` (sqlite default
# is OFF).
_TOOL_TRANSCRIPTS_DDL = """
CREATE TABLE IF NOT EXISTS episode_tool_transcripts (
    episode_id   TEXT NOT NULL,
    sequence     INTEGER NOT NULL,
    message_json TEXT NOT NULL,
    PRIMARY KEY (episode_id, sequence),
    FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
)
"""
_TOOL_TRANSCRIPTS_INDEX = (
    "CREATE INDEX IF NOT EXISTS episode_tool_transcripts_episode_idx "
    "ON episode_tool_transcripts (episode_id, sequence)"
)


# ---------------------------------------------------------------------------
# SqliteMemory
# ---------------------------------------------------------------------------


class SqliteMemory:
    """Durable :class:`Memory` rooted at a single sqlite file.

    Construct directly from a path::

        memory = SqliteMemory("./bot.db")
        agent = Agent("...", model="gpt-4.1-mini", memory=memory)

    Or via the resolver::

        agent = Agent("...", model="gpt-4.1-mini", memory="sqlite:./bot.db")

    Pass ``path=":memory:"`` for an ephemeral in-process database
    (lost on close — useful for tests).

    The fact store is auto-attached: the same ``.db`` file holds a
    ``facts`` table managed by :class:`SqliteFactStore`. Pass
    ``with_facts=False`` to skip it; pass an explicit
    ``fact_store=`` to override (e.g. point facts at a different
    sqlite file).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        embedder: Embedder | None = None,
        with_facts: bool = True,
        fact_store: FactStore | None = None,
    ) -> None:
        self._path = Path(path) if str(path) != ":memory:" else Path(":memory:")
        if str(self._path) != ":memory:":
            self._path.parent.mkdir(parents=True, exist_ok=True)
        # Hash embedder is the zero-key default — same convention as
        # the other persistent backends. Real production runs pass an
        # OpenAI / Voyage embedder via the resolver or constructor.
        self._embedder: Embedder = embedder if embedder is not None else HashEmbedder()
        # Coordinate writes — sqlite handles cross-connection locking
        # but the python-side Episode/working state still benefits
        # from a single mutator at a time.
        self._async_lock = anyio.Lock()
        self._sync_lock = threading.RLock()
        self._init_schema()

        # Wire up the fact store. Default: SqliteFactStore at the same
        # path, so every layer of memory lives in one file.
        if fact_store is not None:
            self.facts: FactStore | None = fact_store
        elif with_facts:
            self.facts = SqliteFactStore(self._path, embedder=self._embedder)
        else:
            self.facts = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    # ---- connection management -------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """One connection per call. SQLite connections aren't safe to
        share across the worker threads we hop into via
        ``anyio.to_thread.run_sync``."""
        conn = sqlite3.connect(self._path)
        # Foreign keys are OFF by default in SQLite; enable per-
        # connection so the ``episode_tool_transcripts`` ON DELETE
        # CASCADE actually fires when an episode is deleted via
        # ``forget()``. Cheap pragma; safe on tables without FKs.
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            # Working blocks: migrate from the pre-M9 single-PK
            # ``(name)`` shape to the partitioned ``(user_id, name)``
            # shape if needed. SQLite can't ADD COLUMN to a PK
            # unfortunately — we have to detect the old shape and
            # rebuild the table.
            self._migrate_working_blocks(conn)
            conn.execute(_BLOCKS_DDL)
            conn.execute(_EPISODES_DDL)
            # Best-effort migration for legacy schemas that predate
            # ``user_id`` on episodes.
            try:
                conn.execute(_EPISODES_ADD_USER_ID)
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
            conn.execute(_EPISODES_USER_SESSION_INDEX)
            conn.execute(_EPISODES_USER_OCCURRED_INDEX)
            # Tool-transcript sidecar table — created on first
            # init, idempotent on subsequent inits. Existing DBs
            # without it get the table on next connect; episodes
            # written before this migration simply have no rows
            # in the sidecar and rehydrate without a transcript.
            conn.execute(_TOOL_TRANSCRIPTS_DDL)
            conn.execute(_TOOL_TRANSCRIPTS_INDEX)
            conn.commit()

    def _migrate_working_blocks(self, conn: sqlite3.Connection) -> None:
        """Bring a legacy ``working_blocks`` table to the M9 schema.

        Pre-M9: ``(name PRIMARY KEY, content, pinned_order, updated_at)``.
        M9+:   ``(user_id, name, content, pinned_order, updated_at,
                  PRIMARY KEY (user_id, name))``.

        SQLite forbids ALTER TABLE on a PK, so we detect the old
        shape, copy rows into a fresh table with the new schema,
        and rename. Idempotent: a clean DB or an already-migrated
        DB skips the rebuild.
        """
        existing = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'working_blocks'"
        ).fetchone()
        if existing is None:
            # Fresh DB — nothing to migrate; the CREATE TABLE
            # IF NOT EXISTS in ``_init_schema`` will install the
            # new shape.
            return
        has_user_id = (
            conn.execute(_BLOCKS_MIGRATE_CHECK).fetchone()[0] > 0
        )
        if has_user_id:
            return  # already migrated
        # Rebuild path: legacy ``working_blocks`` with name-as-PK.
        # Copy rows into a fresh new-schema table, then rename.
        conn.execute("ALTER TABLE working_blocks RENAME TO _wb_legacy")
        conn.execute(_BLOCKS_DDL)
        conn.execute(
            "INSERT INTO working_blocks (user_id, name, content, "
            "pinned_order, updated_at) "
            "SELECT NULL, name, content, pinned_order, updated_at "
            "FROM _wb_legacy"
        )
        conn.execute("DROP TABLE _wb_legacy")

    # ---- working blocks --------------------------------------------------

    async def working(
        self, *, user_id: str | None = None
    ) -> list[MemoryBlock]:
        rows = await anyio.to_thread.run_sync(self._working_sync, user_id)
        return [_row_to_block(r) for r in rows]

    def _working_sync(
        self, user_id: str | None
    ) -> list[tuple[Any, ...]]:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT name, content, pinned_order, updated_at "
                "FROM working_blocks "
                "WHERE user_id IS ? "
                "ORDER BY pinned_order ASC",
                (user_id,),
            )
            return cursor.fetchall()

    async def update_block(
        self, name: str, content: str, *, user_id: str | None = None
    ) -> None:
        async with self._async_lock:
            await anyio.to_thread.run_sync(
                self._update_block_sync, name, content, user_id
            )

    def _update_block_sync(
        self, name: str, content: str, user_id: str | None
    ) -> None:
        with self._connect() as conn:
            # Upsert keyed by (user_id, name): keep existing
            # pinned_order if the row exists in this user's bucket,
            # else assign the next slot scoped to that user.
            existing = conn.execute(
                "SELECT pinned_order FROM working_blocks "
                "WHERE user_id IS ? AND name = ?",
                (user_id, name),
            ).fetchone()
            if existing is not None:
                conn.execute(
                    "UPDATE working_blocks SET content = ?, updated_at = ? "
                    "WHERE user_id IS ? AND name = ?",
                    (content, _now_epoch(), user_id, name),
                )
            else:
                count = conn.execute(
                    "SELECT COUNT(*) FROM working_blocks WHERE user_id IS ?",
                    (user_id,),
                ).fetchone()[0]
                conn.execute(
                    "INSERT INTO working_blocks "
                    "(user_id, name, content, pinned_order, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (user_id, name, content, count, _now_epoch()),
                )
            conn.commit()

    async def append_block(
        self, name: str, content: str, *, user_id: str | None = None
    ) -> None:
        async with self._async_lock:
            await anyio.to_thread.run_sync(
                self._append_block_sync, name, content, user_id
            )

    def _append_block_sync(
        self, name: str, content: str, user_id: str | None
    ) -> None:
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT content, pinned_order FROM working_blocks "
                "WHERE user_id IS ? AND name = ?",
                (user_id, name),
            ).fetchone()
            if existing is None:
                count = conn.execute(
                    "SELECT COUNT(*) FROM working_blocks WHERE user_id IS ?",
                    (user_id,),
                ).fetchone()[0]
                conn.execute(
                    "INSERT INTO working_blocks "
                    "(user_id, name, content, pinned_order, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (user_id, name, content, count, _now_epoch()),
                )
            else:
                conn.execute(
                    "UPDATE working_blocks SET content = ?, updated_at = ? "
                    "WHERE user_id IS ? AND name = ?",
                    (existing[0] + content, _now_epoch(), user_id, name),
                )
            conn.commit()

    # ---- episodes --------------------------------------------------------

    async def remember(self, episode: Episode) -> str:
        # Embed outside the DB lock — embedders may make network calls.
        embedding_blob: bytes | None = None
        if episode.embedding is not None:
            embedding_blob = pack_float32(episode.embedding)
        else:
            text = "\n".join(p for p in (episode.input, episode.output) if p)
            if text.strip():
                vector = await self._embedder.embed(text)
                embedding_blob = pack_float32(vector)

        async with self._async_lock:
            await anyio.to_thread.run_sync(
                self._remember_sync, episode, embedding_blob
            )
        return episode.id

    def _remember_sync(
        self, episode: Episode, embedding_blob: bytes | None
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO episodes "
                "(id, session_id, user_id, occurred_at, input, output, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    episode.id,
                    episode.session_id,
                    episode.user_id,
                    _to_epoch(episode.occurred_at),
                    episode.input,
                    episode.output,
                    embedding_blob,
                ),
            )
            # Tool transcript sidecar — replace any prior rows for
            # this episode_id (DELETE then INSERT) so a re-remember
            # on the same id stays consistent. Skipped entirely
            # when ``tool_transcript`` is None (the default — opt-in
            # only via ``Agent(persist_tool_transcripts=True)``),
            # avoiding any per-write cost for users who haven't
            # enabled the feature.
            if episode.tool_transcript is not None:
                conn.execute(
                    "DELETE FROM episode_tool_transcripts "
                    "WHERE episode_id = ?",
                    (episode.id,),
                )
                if episode.tool_transcript:
                    conn.executemany(
                        "INSERT INTO episode_tool_transcripts "
                        "(episode_id, sequence, message_json) "
                        "VALUES (?, ?, ?)",
                        [
                            (
                                episode.id,
                                i,
                                msg.model_dump_json(),
                            )
                            for i, msg in enumerate(
                                episode.tool_transcript
                            )
                        ],
                    )
            conn.commit()

    async def recall(
        self,
        query: str,
        *,
        kind: str = "episodic",
        limit: int = 5,
        time_range: tuple[datetime, datetime] | None = None,
        user_id: str | None = None,
    ) -> list[Episode]:
        # Footgun protection — see ``InMemoryMemory.recall``.
        if user_id is None:
            await self._maybe_warn_isolation()

        if not query.strip():
            return await self._recall_recent(limit, time_range, user_id)

        # Brute-force cosine over every candidate in the user's
        # partition. Time-range filter happens at the SQL layer to
        # cut the candidate set first.
        query_vector = await self._embedder.embed(query)
        rows = await anyio.to_thread.run_sync(
            self._scan_episodes_sync, user_id, time_range
        )

        scored: list[tuple[float, tuple[Any, ...]]] = []
        for row in rows:
            blob = row[6]  # embedding column
            if not blob:
                continue
            try:
                vec = unpack_float32(bytes(blob))
            except (TypeError, ValueError):
                continue
            scored.append((_cosine(query_vector, vec), row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [_row_to_episode(r) for _, r in scored[:limit]]

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

        The candidate pool is the same SQL scan ``recall`` uses (user
        partition + time-range filter applied at the SQL layer);
        ``bm25_score`` and ``vector_score`` ride along on each
        :class:`EpisodeMatch`. Empty queries fall through to recency
        with neutral ``1.0`` scores; a query that matches nothing falls
        through to recency with ``0.0``.
        """
        from ._hybrid import _BM25, hybrid_rank

        # Footgun protection — mirror ``recall``.
        if user_id is None:
            await self._maybe_warn_isolation()

        if not query.strip():
            recent = await self._recall_recent(limit, time_range, user_id)
            return [EpisodeMatch(episode=e, score=1.0) for e in recent]

        rows = await anyio.to_thread.run_sync(
            self._scan_episodes_sync, user_id, time_range
        )
        candidates: list[Episode] = []
        embeddings: list[list[float]] = []
        for row in rows:
            blob = row[6]  # embedding column
            if not blob:
                continue
            try:
                vec = unpack_float32(bytes(blob))
            except (TypeError, ValueError):
                continue
            candidates.append(_row_to_episode(row))
            embeddings.append(vec)
        if not candidates:
            return []

        # Vector arm — cosine over candidate embeddings; drop
        # non-positive sims so RRF doesn't promote them.
        query_vector = await self._embedder.embed(query)
        vector_scores: list[tuple[int, float]] = []
        for i, vec in enumerate(embeddings):
            sim = cosine(query_vector, vec)
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

    async def _recall_recent(
        self,
        limit: int,
        time_range: tuple[datetime, datetime] | None,
        user_id: str | None,
    ) -> list[Episode]:
        rows = await anyio.to_thread.run_sync(
            self._scan_episodes_sync, user_id, time_range
        )
        # ``_scan_episodes_sync`` sorts oldest-first; recency-recall
        # wants newest-first, so reverse here.
        return [_row_to_episode(r) for r in rows[-limit:][::-1]]

    def _scan_episodes_sync(
        self,
        user_id: str | None,
        time_range: tuple[datetime, datetime] | None,
    ) -> list[tuple[Any, ...]]:
        with self._connect() as conn:
            sql = (
                "SELECT id, session_id, user_id, occurred_at, input, "
                "output, embedding FROM episodes WHERE user_id IS ?"
            )
            params: list[Any] = [user_id]
            if time_range is not None:
                sql += " AND occurred_at >= ? AND occurred_at <= ?"
                params.extend(
                    [_to_epoch(time_range[0]), _to_epoch(time_range[1])]
                )
            sql += " ORDER BY occurred_at ASC"
            return conn.execute(sql, params).fetchall()

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
        max_episodes = max(1, limit // 2)
        rows = await anyio.to_thread.run_sync(
            self._session_messages_sync,
            session_id,
            user_id,
            max_episodes,
        )
        episodes = [_row_to_episode(r) for r in rows]
        episodes.sort(key=lambda e: e.occurred_at)
        # Batch-fetch tool transcripts for these episodes in ONE
        # round-trip rather than N follow-up queries. Returns an
        # empty dict when the sidecar table has no rows for any of
        # the episode ids — the typical case for users who haven't
        # enabled ``persist_tool_transcripts=True``.
        episode_ids = [ep.id for ep in episodes]
        transcripts = await anyio.to_thread.run_sync(
            self._fetch_transcripts_sync, episode_ids
        )
        out: list[Message] = []
        for ep in episodes:
            if ep.input:
                out.append(Message(role=Role.USER, content=ep.input))
            # Splice transcript between USER and ASSISTANT so a
            # resumed worker sees its prior tool work.
            ep_transcript = transcripts.get(ep.id, [])
            if ep_transcript:
                out.extend(ep_transcript)
            if ep.output:
                out.append(Message(role=Role.ASSISTANT, content=ep.output))
        return out

    def _session_messages_sync(
        self, session_id: str, user_id: str | None, limit: int
    ) -> list[tuple[Any, ...]]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT id, session_id, user_id, occurred_at, input, "
                "output, embedding FROM episodes "
                "WHERE session_id = ? AND user_id IS ? "
                "ORDER BY occurred_at DESC LIMIT ?",
                (session_id, user_id, limit),
            ).fetchall()

    def _fetch_transcripts_sync(
        self, episode_ids: list[str]
    ) -> dict[str, list[Message]]:
        """Bulk-fetch tool transcripts for the given episode IDs.

        Returns ``{episode_id: [Message, ...]}`` ordered by
        ``sequence``. Episodes without transcript rows simply don't
        appear in the result dict — the caller's ``.get(ep.id, [])``
        handles the missing-key case as "no transcript captured."

        One query for N episodes (parameter-binding via the IN
        clause) rather than N round-trips. SQLite has a default
        max-parameter-count of 999; we don't expect ``limit=20``
        runs to push anywhere close to that.
        """
        if not episode_ids:
            return {}
        with self._connect() as conn:
            placeholders = ",".join("?" * len(episode_ids))
            rows = conn.execute(
                "SELECT episode_id, sequence, message_json "
                "FROM episode_tool_transcripts "
                f"WHERE episode_id IN ({placeholders}) "
                "ORDER BY episode_id, sequence",
                episode_ids,
            ).fetchall()
        result: dict[str, list[Message]] = {}
        for ep_id, _seq, msg_json in rows:
            result.setdefault(ep_id, []).append(
                Message.model_validate_json(msg_json)
            )
        return result

    # ---- profile / forget / export (GDPR) -------------------------------

    async def profile(
        self, *, user_id: str | None = None
    ) -> MemoryProfile:
        episodes = await anyio.to_thread.run_sync(
            self._scan_episodes_sync, user_id, None
        )
        episode_count = len(episodes)
        last_seen: datetime | None = None
        if episodes:
            last_seen = _from_epoch(max(row[3] for row in episodes))

        # Recent sessions, newest-first, dedup'd, capped at 10.
        seen: set[str] = set()
        recent_sessions: list[str] = []
        for row in sorted(episodes, key=lambda r: r[3], reverse=True):
            sid = row[1]
            if sid in seen:
                continue
            seen.add(sid)
            recent_sessions.append(sid)
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
            episode_count=episode_count,
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
        async with self._async_lock:
            deleted = await anyio.to_thread.run_sync(
                self._forget_episodes_sync,
                user_id,
                session_id,
                before,
            )

        # Facts: erase only when not narrowed by session_id (facts have
        # no session). Bi-temporal store has no public delete; drop
        # to the underlying SQL.
        if session_id is None and self.facts is not None:
            deleted += await anyio.to_thread.run_sync(
                self._forget_facts_sync, user_id, before
            )
        return deleted

    def _forget_episodes_sync(
        self,
        user_id: str | None,
        session_id: str | None,
        before: datetime | None,
    ) -> int:
        with self._connect() as conn:
            sql = "DELETE FROM episodes WHERE user_id IS ?"
            params: list[Any] = [user_id]
            if session_id is not None:
                sql += " AND session_id = ?"
                params.append(session_id)
            if before is not None:
                sql += " AND occurred_at < ?"
                params.append(_to_epoch(before))
            cursor = conn.execute(sql, params)
            conn.commit()
            return int(cursor.rowcount or 0)

    def _forget_facts_sync(
        self, user_id: str | None, before: datetime | None
    ) -> int:
        # We share the .db file with SqliteFactStore so we can delete
        # facts directly. The store's own connection isolation
        # guarantees this is safe.
        with self._connect() as conn:
            sql = "DELETE FROM facts WHERE user_id IS ?"
            params: list[Any] = [user_id]
            if before is not None:
                sql += " AND recorded_at < ?"
                params.append(_to_epoch(before))
            cursor = conn.execute(sql, params)
            conn.commit()
            return int(cursor.rowcount or 0)

    async def export(
        self, *, user_id: str | None = None
    ) -> MemoryExport:
        rows = await anyio.to_thread.run_sync(
            self._scan_episodes_sync, user_id, None
        )
        episodes = [_row_to_episode(r) for r in rows]
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
        # No background consolidator wired by default. Users who want
        # auto-fact-extraction will pass one explicitly via the dict
        # form of the resolver (see Agent docs). Method exists to
        # satisfy the protocol.
        return None

    async def aclose(self) -> None:
        # SQLite connections are per-call (see ``_connect``); nothing
        # global to close. Method exists for API symmetry with
        # PostgresMemory / RedisMemory.
        return None

    # ---- footgun ---------------------------------------------------------

    async def _maybe_warn_isolation(self) -> None:
        """Emit an ``IsolationWarning`` when this memory contains
        named-user episodes but a recall is running with
        ``user_id=None``. Same contract as :class:`InMemoryMemory`."""
        has_named = await anyio.to_thread.run_sync(self._has_named_user_sync)
        if has_named:
            warnings.warn(
                "Memory.recall called without user_id, but the store "
                "contains episodes for one or more named users. The "
                "anonymous bucket is partitioned from named-user "
                "buckets, so this query will only see anonymous "
                "episodes. Did you forget to pass user_id=?",
                IsolationWarning,
                stacklevel=4,
            )

    def _has_named_user_sync(self) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM episodes WHERE user_id IS NOT NULL LIMIT 1"
            ).fetchone()
            return row is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_epoch() -> float:
    return datetime.now(UTC).timestamp()


def _to_epoch(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def _from_epoch(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=UTC)


def _row_to_block(row: tuple[Any, ...]) -> MemoryBlock:
    name, content, pinned_order, updated_at = row
    return MemoryBlock(
        name=name,
        content=content,
        pinned_order=pinned_order,
        updated_at=_from_epoch(updated_at),
    )


def _row_to_episode(row: tuple[Any, ...]) -> Episode:
    eid, session_id, user_id, occurred_at, input_, output, embedding = row
    embedding_list: list[float] | None = None
    if embedding is not None:
        try:
            embedding_list = unpack_float32(bytes(embedding))
        except (TypeError, ValueError):
            embedding_list = None
    return Episode(
        id=eid,
        session_id=session_id,
        user_id=user_id,
        occurred_at=_from_epoch(occurred_at),
        input=input_,
        output=output,
        embedding=embedding_list,
    )


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))
