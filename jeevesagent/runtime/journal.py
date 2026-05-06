"""Journal stores for the durable runtime.

A journal records the result of every side-effecting step in a run,
keyed by ``(session_id, step_name)``. On replay, the runtime returns
the cached result instead of re-executing the step. This is the
mechanism that makes long-running agents resumable across crashes.

Today's stores:

* :class:`InMemoryJournalStore` — dict-backed; lost on process exit.
  Useful for tests and for runs where you want replay-within-a-run
  semantics but don't need durability across restarts.
* :class:`SqliteJournalStore` — sqlite3 file with two tables; survives
  process restarts. Sync sqlite3 calls dispatched through
  :func:`anyio.to_thread.run_sync`.

Both stores use :mod:`pickle` for value serialization. That's safe in
this context because journals only ever hold values returned by *your
own* trusted code (tools, models, memory backends) — the same code
path that ran them in the first place. Switching to JSON would force
every stored value to be JSON-serialisable, which precludes Pydantic
models and arbitrary tool return values.
"""

from __future__ import annotations

import pickle
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
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

    async def aclose(self) -> None: ...


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------


class InMemoryJournalStore:
    """Dict-backed journal. Process-local; lost on exit."""

    def __init__(self) -> None:
        self._steps: dict[tuple[str, str], JournalEntry] = {}
        self._streams: dict[tuple[str, str], list[Any]] = {}
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


class SqliteJournalStore:
    """SQLite-backed journal. Durable across process restarts."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @property
    def path(self) -> Path:
        return self._path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # Each call creates and closes its own connection. SQLite
        # connections are not safe to share across threads by default,
        # and we hop threads for every async call.
        conn = sqlite3.connect(self._path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(_STEP_DDL)
            conn.execute(_STREAM_DDL)
            conn.commit()

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
        with self._connect() as conn:
            row = conn.execute(
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
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO journal_steps "
                "(session_id, step_name, value, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, step_name, pickle.dumps(value), time.time()),
            )
            conn.commit()

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
        with self._connect() as conn:
            row = conn.execute(
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
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO journal_streams "
                "(session_id, step_name, chunks, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, step_name, pickle.dumps(chunks), time.time()),
            )
            conn.commit()

    # ---- lifecycle -------------------------------------------------------

    async def aclose(self) -> None:
        return None
