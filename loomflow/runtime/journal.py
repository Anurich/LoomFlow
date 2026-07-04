"""Journal stores for the durable runtime.

A journal records the result of every side-effecting step in a run,
keyed by ``(session_id, step_name)`` — where ``step_name`` as written
by :class:`~loomflow.runtime.journaled.JournaledRuntime` already folds
in a fingerprint of the step's inputs. On replay, the runtime returns
the cached result instead of re-executing the step. This is the
mechanism that makes long-running agents resumable across crashes.

Today's stores:

* :class:`InMemoryJournalStore` — dict-backed; lost on process exit.
  Useful for tests and for runs where you want replay-within-a-run
  semantics but don't need durability across restarts.
* :class:`SqliteJournalStore` — sqlite3 file with two tables; survives
  process restarts. Sync sqlite3 calls dispatched through
  :func:`anyio.to_thread.run_sync` against one long-lived connection
  (WAL mode, ``busy_timeout=5000``) serialised by a thread lock.
* :class:`PostgresJournalStore` — asyncpg-pool-backed; multi-host.

Besides step results, all stores persist :class:`Checkpoint` records —
JSON-serialised transcript snapshots (messages + cumulative usage +
architecture cursor) written between turns so a crashed run can resume
without replaying billed model calls. Checkpoints are stored via
``model_dump_json`` (never pickle) and retention is bounded per
session by each store's ``max_checkpoints_per_session``.

All stores expose :meth:`~JournalStore.prune` so operators can purge
completed sessions (by ``session_id``) or old entries (by ``before``
timestamp) — nothing prunes automatically except the per-session
checkpoint retention cap applied on ``put_checkpoint``.

.. warning:: **Pickle wire format.**

    All stores serialise journal values with :mod:`pickle`.
    ``pickle.loads`` executes arbitrary code embedded in the payload:
    anyone who can WRITE to the journal store (the sqlite file, the
    Postgres tables) can achieve code execution in every process that
    replays from it. Treat the journal store with the same trust level
    as your codebase — never point a runtime at a journal writable by
    untrusted parties. Pickled payloads are also fragile across
    library/Python upgrades: a value pickled under one version of a
    class may fail to unpickle after an upgrade, in which case the
    affected sessions must be pruned rather than resumed.

    TODO: switch the default wire format to JSON via Pydantic
    ``model_dump`` (with a registry/fallback for non-Pydantic values)
    and keep pickle as an opt-in for arbitrary objects.
"""

from __future__ import annotations

import pickle
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import anyio
from pydantic import BaseModel, ConfigDict, Field

from ..core.ids import new_id
from ..core.types import Message, Usage


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class JournalEntry:
    """A single recorded step result with a creation timestamp."""

    value: Any
    created_at: float


class Checkpoint(BaseModel):
    """Durable transcript snapshot taken between turns (G4).

    Unlike journal step entries (pickled step *results* keyed by input
    fingerprint), a checkpoint captures the whole conversational state
    an agent loop needs to resume after a crash: the message
    transcript, cumulative usage, and an architecture-local ``cursor``
    marking where to re-enter. Serialised as JSON via
    ``model_dump_json`` — never pickle — so payloads are inspectable
    and safe to load from an untrusted-writable store.
    """

    model_config = ConfigDict(frozen=True)

    session_id: str
    checkpoint_id: str = Field(default_factory=lambda: new_id("ckpt"))
    turn: int
    messages: list[Message] = Field(default_factory=list)
    cumulative_usage: Usage = Field(default_factory=Usage)
    cursor: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


@dataclass(frozen=True)
class CheckpointMeta:
    """Cheap listing row for a checkpoint — no message payload."""

    checkpoint_id: str
    turn: int
    created_at: datetime


@runtime_checkable
class JournalStore(Protocol):
    """Storage surface for the durable runtime."""

    async def get_step(
        self, session_id: str, step_name: str
    ) -> JournalEntry | None: ...

    async def put_step(
        self, session_id: str, step_name: str, value: Any
    ) -> None: ...

    async def get_stream(
        self, session_id: str, step_name: str
    ) -> list[Any] | None: ...

    async def put_stream(
        self, session_id: str, step_name: str, chunks: list[Any]
    ) -> None: ...

    async def put_checkpoint(self, cp: Checkpoint) -> None:
        """Persist a checkpoint; prune the session's oldest beyond
        the store's retention limit."""
        ...

    async def get_checkpoint(
        self, session_id: str, checkpoint_id: str
    ) -> Checkpoint | None: ...

    async def get_latest_checkpoint(
        self, session_id: str
    ) -> Checkpoint | None: ...

    async def list_checkpoints(
        self, session_id: str, limit: int = 50
    ) -> list[CheckpointMeta]:
        """Newest-first checkpoint metadata (id/turn/created_at only)."""
        ...

    async def prune(
        self,
        before: datetime | None = None,
        session_id: str | None = None,
    ) -> None:
        """Delete journal entries AND checkpoints matching ALL filters.

        ``before`` drops entries created before that instant (naive
        datetimes are interpreted in local time, matching
        ``datetime.timestamp()``); ``session_id`` restricts the purge
        to one session. With no filters, the whole journal is purged.
        """
        ...

    async def aclose(self) -> None: ...


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------


def _checkpoint_order(cp: Checkpoint) -> tuple[float, str]:
    # created_at first; checkpoint_id (prefixed ULID, time-sortable)
    # breaks ties deterministically.
    return (cp.created_at.timestamp(), cp.checkpoint_id)


class InMemoryJournalStore:
    """Dict-backed journal. Process-local; lost on exit.

    ``max_checkpoints_per_session`` bounds checkpoint retention: on
    every :meth:`put_checkpoint` the oldest checkpoints beyond the
    limit are dropped. Values < 1 disable pruning (unbounded).
    """

    def __init__(self, *, max_checkpoints_per_session: int = 20) -> None:
        self._steps: dict[tuple[str, str], JournalEntry] = {}
        self._streams: dict[tuple[str, str], list[Any]] = {}
        self._stream_times: dict[tuple[str, str], float] = {}
        self._checkpoints: dict[str, list[Checkpoint]] = {}
        self._max_checkpoints = max_checkpoints_per_session
        self._lock = anyio.Lock()

    async def get_step(
        self, session_id: str, step_name: str
    ) -> JournalEntry | None:
        async with self._lock:
            return self._steps.get((session_id, step_name))

    async def put_step(
        self, session_id: str, step_name: str, value: Any
    ) -> None:
        async with self._lock:
            self._steps[(session_id, step_name)] = JournalEntry(
                value=value, created_at=time.time()
            )

    async def get_stream(
        self, session_id: str, step_name: str
    ) -> list[Any] | None:
        async with self._lock:
            chunks = self._streams.get((session_id, step_name))
            return list(chunks) if chunks is not None else None

    async def put_stream(
        self, session_id: str, step_name: str, chunks: list[Any]
    ) -> None:
        async with self._lock:
            self._streams[(session_id, step_name)] = list(chunks)
            self._stream_times[(session_id, step_name)] = time.time()

    # ---- checkpoint ops ---------------------------------------------------

    async def put_checkpoint(self, cp: Checkpoint) -> None:
        async with self._lock:
            cps = self._checkpoints.setdefault(cp.session_id, [])
            cps.append(cp)
            cps.sort(key=_checkpoint_order)
            if self._max_checkpoints > 0:
                del cps[: -self._max_checkpoints]

    async def get_checkpoint(
        self, session_id: str, checkpoint_id: str
    ) -> Checkpoint | None:
        async with self._lock:
            for cp in self._checkpoints.get(session_id, []):
                if cp.checkpoint_id == checkpoint_id:
                    return cp
        return None

    async def get_latest_checkpoint(
        self, session_id: str
    ) -> Checkpoint | None:
        async with self._lock:
            cps = self._checkpoints.get(session_id)
            return cps[-1] if cps else None

    async def list_checkpoints(
        self, session_id: str, limit: int = 50
    ) -> list[CheckpointMeta]:
        async with self._lock:
            cps = list(reversed(self._checkpoints.get(session_id, [])))
        return [
            CheckpointMeta(
                checkpoint_id=cp.checkpoint_id,
                turn=cp.turn,
                created_at=cp.created_at,
            )
            for cp in cps[:limit]
        ]

    # ---- gc ----------------------------------------------------------------

    async def prune(
        self,
        before: datetime | None = None,
        session_id: str | None = None,
    ) -> None:
        cutoff = before.timestamp() if before is not None else None

        def _matches(key: tuple[str, str], created_at: float) -> bool:
            if session_id is not None and key[0] != session_id:
                return False
            return not (cutoff is not None and created_at >= cutoff)

        async with self._lock:
            self._steps = {
                k: v
                for k, v in self._steps.items()
                if not _matches(k, v.created_at)
            }
            kept_streams = {
                k: v
                for k, v in self._streams.items()
                if not _matches(k, self._stream_times.get(k, 0.0))
            }
            self._streams = kept_streams
            self._stream_times = {
                k: t
                for k, t in self._stream_times.items()
                if k in kept_streams
            }
            for sid in list(self._checkpoints):
                remaining = [
                    cp
                    for cp in self._checkpoints[sid]
                    if not _matches((sid, ""), cp.created_at.timestamp())
                ]
                if remaining:
                    self._checkpoints[sid] = remaining
                else:
                    del self._checkpoints[sid]

    async def aclose(self) -> None:
        return None

    # ---- introspection (test helpers) -----------------------------------

    def step_keys(self) -> list[tuple[str, str]]:
        return list(self._steps.keys())

    def stream_keys(self) -> list[tuple[str, str]]:
        return list(self._streams.keys())


# ---------------------------------------------------------------------------
# SQLite store
# ---------------------------------------------------------------------------


_STEP_DDL = """
CREATE TABLE IF NOT EXISTS journal_steps (
    session_id TEXT NOT NULL,
    step_name  TEXT NOT NULL,
    value      BLOB NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (session_id, step_name)
)
"""

_STREAM_DDL = """
CREATE TABLE IF NOT EXISTS journal_streams (
    session_id TEXT NOT NULL,
    step_name  TEXT NOT NULL,
    chunks     BLOB NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (session_id, step_name)
)
"""

_CHECKPOINT_DDL = """
CREATE TABLE IF NOT EXISTS checkpoints (
    session_id    TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL PRIMARY KEY,
    turn          INTEGER NOT NULL,
    created_at    REAL NOT NULL,
    payload       TEXT NOT NULL
)
"""

_CHECKPOINT_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS checkpoints_session_created
    ON checkpoints (session_id, created_at DESC)
"""

# Postgres-flavoured DDL for :class:`PostgresJournalStore` below.
# ``BYTEA`` instead of ``BLOB``; ``DOUBLE PRECISION`` instead of ``REAL``.
_PG_STEP_DDL = """
CREATE TABLE IF NOT EXISTS journal_steps (
    session_id TEXT NOT NULL,
    step_name  TEXT NOT NULL,
    value      BYTEA NOT NULL,
    created_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (session_id, step_name)
)
"""

_PG_STREAM_DDL = """
CREATE TABLE IF NOT EXISTS journal_streams (
    session_id TEXT NOT NULL,
    step_name  TEXT NOT NULL,
    chunks     BYTEA NOT NULL,
    created_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (session_id, step_name)
)
"""

_PG_CHECKPOINT_DDL = """
CREATE TABLE IF NOT EXISTS checkpoints (
    session_id    TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL PRIMARY KEY,
    turn          INTEGER NOT NULL,
    created_at    DOUBLE PRECISION NOT NULL,
    payload       JSONB NOT NULL
)
"""

_PG_CHECKPOINT_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS checkpoints_session_created
    ON checkpoints (session_id, created_at DESC)
"""

# Keep only the newest N checkpoints for a session. Placeholder style
# differs per backend, so each store formats its own copy.
_SQLITE_CHECKPOINT_RETENTION_SQL = (
    "DELETE FROM checkpoints WHERE session_id = ? "
    "AND checkpoint_id NOT IN ("
    "SELECT checkpoint_id FROM checkpoints WHERE session_id = ? "
    "ORDER BY created_at DESC, checkpoint_id DESC LIMIT ?)"
)

_PG_CHECKPOINT_RETENTION_SQL = (
    "DELETE FROM checkpoints WHERE session_id = $1 "
    "AND checkpoint_id NOT IN ("
    "SELECT checkpoint_id FROM checkpoints WHERE session_id = $1 "
    "ORDER BY created_at DESC, checkpoint_id DESC LIMIT $2)"
)


class SqliteJournalStore:
    """SQLite-backed journal. Durable across process restarts.

    Holds one long-lived connection opened with
    ``check_same_thread=False`` and configured with ``journal_mode=WAL``,
    ``synchronous=NORMAL`` and ``busy_timeout=5000`` so concurrent runs
    (and concurrent processes) sharing the file coordinate instead of
    failing fast with ``database is locked``. Async entry points hop to
    a worker thread via :func:`anyio.to_thread.run_sync`; a
    :class:`threading.Lock` serialises all use of the shared connection
    because those hops may land on different worker threads.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_checkpoints_per_session: int = 20,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._closed = False
        self._max_checkpoints = max_checkpoints_per_session
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute(_STEP_DDL)
            self._conn.execute(_STREAM_DDL)
            self._conn.execute(_CHECKPOINT_DDL)
            self._conn.execute(_CHECKPOINT_INDEX_DDL)
            self._conn.commit()

    @property
    def path(self) -> Path:
        return self._path

    # ---- step ops --------------------------------------------------------

    async def get_step(
        self, session_id: str, step_name: str
    ) -> JournalEntry | None:
        return await anyio.to_thread.run_sync(
            self._get_step_sync, session_id, step_name
        )

    def _get_step_sync(
        self, session_id: str, step_name: str
    ) -> JournalEntry | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value, created_at FROM journal_steps "
                "WHERE session_id = ? AND step_name = ?",
                (session_id, step_name),
            ).fetchone()
        if row is None:
            return None
        return JournalEntry(value=pickle.loads(row[0]), created_at=row[1])

    async def put_step(
        self, session_id: str, step_name: str, value: Any
    ) -> None:
        await anyio.to_thread.run_sync(
            self._put_step_sync, session_id, step_name, value
        )

    def _put_step_sync(
        self, session_id: str, step_name: str, value: Any
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO journal_steps "
                "(session_id, step_name, value, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, step_name, pickle.dumps(value), time.time()),
            )
            self._conn.commit()

    # ---- stream ops ------------------------------------------------------

    async def get_stream(
        self, session_id: str, step_name: str
    ) -> list[Any] | None:
        return await anyio.to_thread.run_sync(
            self._get_stream_sync, session_id, step_name
        )

    def _get_stream_sync(
        self, session_id: str, step_name: str
    ) -> list[Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT chunks FROM journal_streams "
                "WHERE session_id = ? AND step_name = ?",
                (session_id, step_name),
            ).fetchone()
        if row is None:
            return None
        loaded = pickle.loads(row[0])
        return list(loaded) if loaded is not None else []

    async def put_stream(
        self, session_id: str, step_name: str, chunks: list[Any]
    ) -> None:
        await anyio.to_thread.run_sync(
            self._put_stream_sync, session_id, step_name, list(chunks)
        )

    def _put_stream_sync(
        self, session_id: str, step_name: str, chunks: list[Any]
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO journal_streams "
                "(session_id, step_name, chunks, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, step_name, pickle.dumps(chunks), time.time()),
            )
            self._conn.commit()

    # ---- checkpoint ops ---------------------------------------------------

    async def put_checkpoint(self, cp: Checkpoint) -> None:
        await anyio.to_thread.run_sync(self._put_checkpoint_sync, cp)

    def _put_checkpoint_sync(self, cp: Checkpoint) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO checkpoints "
                "(session_id, checkpoint_id, turn, created_at, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    cp.session_id,
                    cp.checkpoint_id,
                    cp.turn,
                    cp.created_at.timestamp(),
                    cp.model_dump_json(),
                ),
            )
            if self._max_checkpoints > 0:
                self._conn.execute(
                    _SQLITE_CHECKPOINT_RETENTION_SQL,
                    (cp.session_id, cp.session_id, self._max_checkpoints),
                )
            self._conn.commit()

    async def get_checkpoint(
        self, session_id: str, checkpoint_id: str
    ) -> Checkpoint | None:
        return await anyio.to_thread.run_sync(
            self._get_checkpoint_sync, session_id, checkpoint_id
        )

    def _get_checkpoint_sync(
        self, session_id: str, checkpoint_id: str
    ) -> Checkpoint | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM checkpoints "
                "WHERE session_id = ? AND checkpoint_id = ?",
                (session_id, checkpoint_id),
            ).fetchone()
        if row is None:
            return None
        return Checkpoint.model_validate_json(row[0])

    async def get_latest_checkpoint(
        self, session_id: str
    ) -> Checkpoint | None:
        return await anyio.to_thread.run_sync(
            self._get_latest_checkpoint_sync, session_id
        )

    def _get_latest_checkpoint_sync(
        self, session_id: str
    ) -> Checkpoint | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM checkpoints WHERE session_id = ? "
                "ORDER BY created_at DESC, checkpoint_id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return Checkpoint.model_validate_json(row[0])

    async def list_checkpoints(
        self, session_id: str, limit: int = 50
    ) -> list[CheckpointMeta]:
        return await anyio.to_thread.run_sync(
            self._list_checkpoints_sync, session_id, limit
        )

    def _list_checkpoints_sync(
        self, session_id: str, limit: int
    ) -> list[CheckpointMeta]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT checkpoint_id, turn, created_at FROM checkpoints "
                "WHERE session_id = ? "
                "ORDER BY created_at DESC, checkpoint_id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [
            CheckpointMeta(
                checkpoint_id=r[0],
                turn=r[1],
                created_at=datetime.fromtimestamp(r[2], tz=UTC),
            )
            for r in rows
        ]

    # ---- gc ---------------------------------------------------------------

    async def prune(
        self,
        before: datetime | None = None,
        session_id: str | None = None,
    ) -> None:
        """Delete entries matching all given filters (see protocol)."""
        await anyio.to_thread.run_sync(self._prune_sync, before, session_id)

    def _prune_sync(
        self, before: datetime | None, session_id: str | None
    ) -> None:
        clauses: list[str] = []
        params: list[Any] = []
        if before is not None:
            clauses.append("created_at < ?")
            params.append(before.timestamp())
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            self._conn.execute(f"DELETE FROM journal_steps{where}", params)
            self._conn.execute(f"DELETE FROM journal_streams{where}", params)
            self._conn.execute(f"DELETE FROM checkpoints{where}", params)
            self._conn.commit()

    # ---- lifecycle -------------------------------------------------------

    async def aclose(self) -> None:
        await anyio.to_thread.run_sync(self._close_sync)

    def _close_sync(self) -> None:
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True


# ---------------------------------------------------------------------------
# Postgres store (Phase 5 production durable runtime)
# ---------------------------------------------------------------------------


class PostgresJournalStore:
    """Postgres-backed journal. Production-grade durable replay.

    Same shape as :class:`SqliteJournalStore` but uses ``asyncpg`` and
    a Postgres database. Designed for users who already run a Postgres
    instance for the rest of their stack (memory, audit, app state)
    and want their durable-runtime journal to live there too.

    Why not a DBOS adapter?

        DBOS Python's workflow model requires ``@DBOS.workflow()`` and
        ``@DBOS.communicator()`` decorators at module-load time. Our
        ``Runtime.step(name, fn, *args)`` API takes arbitrary
        callables at runtime, which doesn't compose cleanly with
        DBOS's static-decoration model. ``PostgresJournalStore``
        gives the same durability guarantee through our existing
        :class:`JournaledRuntime` architecture, with no decorator
        intrusion on user code.
    """

    def __init__(
        self, pool: Any, *, max_checkpoints_per_session: int = 20
    ) -> None:
        self._pool = pool
        self._max_checkpoints = max_checkpoints_per_session

    @classmethod
    async def connect(
        cls,
        dsn: str,
        *,
        min_size: int = 1,
        max_size: int = 10,
        max_checkpoints_per_session: int = 20,
    ) -> PostgresJournalStore:
        """Open an asyncpg pool and return the store rooted at it."""
        try:
            import asyncpg  # type: ignore[import-not-found, import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "asyncpg is not installed. "
                "Install with: pip install 'loomflow[postgres]'"
            ) from exc
        pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=min_size,
            max_size=max_size,
        )
        return cls(
            pool, max_checkpoints_per_session=max_checkpoints_per_session
        )

    async def aclose(self) -> None:
        if self._pool is not None and hasattr(self._pool, "close"):
            await self._pool.close()

    # ---- schema ---------------------------------------------------------

    @staticmethod
    def schema_sql() -> list[str]:
        """Return the DDL needed to bootstrap this store's schema.

        Idempotent; safe to run on every process start.
        """
        return [
            _PG_STEP_DDL.strip(),
            _PG_STREAM_DDL.strip(),
            _PG_CHECKPOINT_DDL.strip(),
            _PG_CHECKPOINT_INDEX_DDL.strip(),
        ]

    async def init_schema(self) -> None:
        async with self._pool.acquire() as conn:
            for stmt in self.schema_sql():
                await conn.execute(stmt)

    # ---- step ops --------------------------------------------------------

    async def get_step(
        self, session_id: str, step_name: str
    ) -> JournalEntry | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT value, created_at FROM journal_steps "
                "WHERE session_id = $1 AND step_name = $2",
                session_id,
                step_name,
            )
        if row is None:
            return None
        return JournalEntry(
            value=pickle.loads(row["value"]),
            created_at=row["created_at"],
        )

    async def put_step(
        self, session_id: str, step_name: str, value: Any
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO journal_steps "
                "(session_id, step_name, value, created_at) "
                "VALUES ($1, $2, $3, $4) "
                "ON CONFLICT (session_id, step_name) DO UPDATE "
                "SET value = EXCLUDED.value, "
                "    created_at = EXCLUDED.created_at",
                session_id,
                step_name,
                pickle.dumps(value),
                time.time(),
            )

    # ---- stream ops -----------------------------------------------------

    async def get_stream(
        self, session_id: str, step_name: str
    ) -> list[Any] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT chunks FROM journal_streams "
                "WHERE session_id = $1 AND step_name = $2",
                session_id,
                step_name,
            )
        if row is None:
            return None
        loaded = pickle.loads(row["chunks"])
        return list(loaded) if loaded is not None else []

    async def put_stream(
        self, session_id: str, step_name: str, chunks: list[Any]
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO journal_streams "
                "(session_id, step_name, chunks, created_at) "
                "VALUES ($1, $2, $3, $4) "
                "ON CONFLICT (session_id, step_name) DO UPDATE "
                "SET chunks = EXCLUDED.chunks, "
                "    created_at = EXCLUDED.created_at",
                session_id,
                step_name,
                pickle.dumps(list(chunks)),
                time.time(),
            )

    # ---- checkpoint ops ---------------------------------------------------

    async def put_checkpoint(self, cp: Checkpoint) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO checkpoints "
                "(session_id, checkpoint_id, turn, created_at, payload) "
                "VALUES ($1, $2, $3, $4, $5) "
                "ON CONFLICT (checkpoint_id) DO UPDATE "
                "SET turn = EXCLUDED.turn, "
                "    created_at = EXCLUDED.created_at, "
                "    payload = EXCLUDED.payload",
                cp.session_id,
                cp.checkpoint_id,
                cp.turn,
                cp.created_at.timestamp(),
                cp.model_dump_json(),
            )
            if self._max_checkpoints > 0:
                await conn.execute(
                    _PG_CHECKPOINT_RETENTION_SQL,
                    cp.session_id,
                    self._max_checkpoints,
                )

    async def get_checkpoint(
        self, session_id: str, checkpoint_id: str
    ) -> Checkpoint | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT payload FROM checkpoints "
                "WHERE session_id = $1 AND checkpoint_id = $2",
                session_id,
                checkpoint_id,
            )
        if row is None:
            return None
        return Checkpoint.model_validate_json(row["payload"])

    async def get_latest_checkpoint(
        self, session_id: str
    ) -> Checkpoint | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT payload FROM checkpoints WHERE session_id = $1 "
                "ORDER BY created_at DESC, checkpoint_id DESC LIMIT 1",
                session_id,
            )
        if row is None:
            return None
        return Checkpoint.model_validate_json(row["payload"])

    async def list_checkpoints(
        self, session_id: str, limit: int = 50
    ) -> list[CheckpointMeta]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT checkpoint_id, turn, created_at FROM checkpoints "
                "WHERE session_id = $1 "
                "ORDER BY created_at DESC, checkpoint_id DESC LIMIT $2",
                session_id,
                limit,
            )
        return [
            CheckpointMeta(
                checkpoint_id=r["checkpoint_id"],
                turn=r["turn"],
                created_at=datetime.fromtimestamp(r["created_at"], tz=UTC),
            )
            for r in rows
        ]

    # ---- gc ---------------------------------------------------------------

    async def prune(
        self,
        before: datetime | None = None,
        session_id: str | None = None,
    ) -> None:
        """Delete entries matching all given filters (see protocol)."""
        clauses: list[str] = []
        params: list[Any] = []
        if before is not None:
            params.append(before.timestamp())
            clauses.append(f"created_at < ${len(params)}")
        if session_id is not None:
            params.append(session_id)
            clauses.append(f"session_id = ${len(params)}")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        async with self._pool.acquire() as conn:
            await conn.execute(f"DELETE FROM journal_steps{where}", *params)
            await conn.execute(
                f"DELETE FROM journal_streams{where}", *params
            )
            await conn.execute(f"DELETE FROM checkpoints{where}", *params)
