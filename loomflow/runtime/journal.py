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

All stores expose :meth:`~JournalStore.prune` so operators can purge
completed sessions (by ``session_id``) or old entries (by ``before``
timestamp) — nothing prunes automatically.

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
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import anyio


@dataclass(frozen=True)
class JournalEntry:
    """A single recorded step result with a creation timestamp."""

    value: Any
    created_at: float


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

    async def prune(
        self,
        before: datetime | None = None,
        session_id: str | None = None,
    ) -> None:
        """Delete journal entries matching ALL provided filters.

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


class InMemoryJournalStore:
    """Dict-backed journal. Process-local; lost on exit."""

    def __init__(self) -> None:
        self._steps: dict[tuple[str, str], JournalEntry] = {}
        self._streams: dict[tuple[str, str], list[Any]] = {}
        self._stream_times: dict[tuple[str, str], float] = {}
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

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._closed = False
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute(_STEP_DDL)
            self._conn.execute(_STREAM_DDL)
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

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    @classmethod
    async def connect(
        cls,
        dsn: str,
        *,
        min_size: int = 1,
        max_size: int = 10,
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
        return cls(pool)

    async def aclose(self) -> None:
        if self._pool is not None and hasattr(self._pool, "close"):
            await self._pool.close()

    # ---- schema ---------------------------------------------------------

    @staticmethod
    def schema_sql() -> list[str]:
        """Return the DDL needed to bootstrap this store's schema.

        Idempotent; safe to run on every process start.
        """
        return [_PG_STEP_DDL.strip(), _PG_STREAM_DDL.strip()]

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
