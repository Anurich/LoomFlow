"""Checkpoint stores for resumable :class:`~loomflow.Workflow` runs.

Pass one to ``Workflow.run(..., checkpointer=store, session_id=...)``
and the scheduler saves its state after every node that completes; a
later run with the same ``session_id`` (and the same input) picks up
where that one stopped instead of starting over. See
:meth:`loomflow.Workflow.run` for the guarantee.

Two stores ship here:

* :class:`InMemoryCheckpointer` — a dict. Survives a failed run in the
  same process (tests, a retry loop); gone when the process exits.
* :class:`SqliteCheckpointer` — one table in a SQLite file. Survives a
  crash and a restart. It may share the file a
  :class:`~loomflow.runtime.SqliteRuntime` journals to; the table is
  its own.

A store holds one opaque blob per key and implements
:class:`WorkflowCheckpointer` (``load`` / ``save`` / ``delete``), so a
Redis or Postgres one is a few lines.

The agent-level journal (:mod:`loomflow.runtime.journaled`) is not
reused as the store: it has no delete, and it silently skips a value it
can't serialise, where a workflow checkpoint has to fail loudly. Its
wire format is reused: state is encoded with the journal's tagged-JSON
codec, which never imports a module or runs a constructor to decode.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import anyio

from ..runtime import _codec

__all__ = [
    "InMemoryCheckpointer",
    "SqliteCheckpointer",
    "WorkflowCheckpointError",
    "WorkflowCheckpointer",
]

STATE_VERSION = 1


class WorkflowCheckpointError(RuntimeError):
    """A workflow's state couldn't be checkpointed or restored."""


@runtime_checkable
class WorkflowCheckpointer(Protocol):
    """Where a workflow run keeps its state between nodes: one opaque
    blob per key. ``save`` replaces; ``load`` returns ``None`` for a
    key never saved (or deleted)."""

    async def load(self, key: str) -> bytes | None: ...

    async def save(self, key: str, state: bytes) -> None: ...

    async def delete(self, key: str) -> None: ...


class InMemoryCheckpointer:
    """Checkpoints in a dict — for tests and same-process retries."""

    def __init__(self) -> None:
        self._states: dict[str, bytes] = {}

    async def load(self, key: str) -> bytes | None:
        return self._states.get(key)

    async def save(self, key: str, state: bytes) -> None:
        self._states[key] = state

    async def delete(self, key: str) -> None:
        self._states.pop(key, None)

    def keys(self) -> list[str]:
        """The keys with a saved checkpoint (unfinished runs)."""
        return sorted(self._states)


_DDL = (
    "CREATE TABLE IF NOT EXISTS workflow_checkpoints ("
    "key TEXT PRIMARY KEY, state BLOB NOT NULL, updated_at REAL NOT NULL)"
)


class SqliteCheckpointer:
    """Checkpoints in a SQLite file — durable across restarts.

    One long-lived connection (WAL, ``busy_timeout``), used from worker
    threads under a lock, as :class:`~loomflow.runtime.SqliteJournalStore`
    does. Each ``save`` is one committed ``INSERT OR REPLACE``, so a
    crash leaves either the previous checkpoint or the new one.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute(_DDL)
            self._conn.commit()

    @property
    def path(self) -> Path:
        return self._path

    async def load(self, key: str) -> bytes | None:
        return await anyio.to_thread.run_sync(self._load_sync, key)

    def _load_sync(self, key: str) -> bytes | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT state FROM workflow_checkpoints WHERE key = ?", (key,)
            ).fetchone()
        return bytes(row[0]) if row is not None else None

    async def save(self, key: str, state: bytes) -> None:
        await anyio.to_thread.run_sync(self._save_sync, key, state)

    def _save_sync(self, key: str, state: bytes) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO workflow_checkpoints (key, state, updated_at) "
                "VALUES (?, ?, ?)",
                (key, state, time.time()),
            )
            self._conn.commit()

    async def delete(self, key: str) -> None:
        await anyio.to_thread.run_sync(self._delete_sync, key)

    def _delete_sync(self, key: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM workflow_checkpoints WHERE key = ?", (key,))
            self._conn.commit()

    async def aclose(self) -> None:
        await anyio.to_thread.run_sync(self._close_sync)

    def _close_sync(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# Helpers the scheduler uses
# ---------------------------------------------------------------------------


def checkpoint_key(workflow: str, session_id: str, user_id: str | None) -> str:
    """One checkpoint per (workflow, user, session). Hashed so a name
    or id containing the separator can't collide with another's."""
    digest = hashlib.sha256(
        "\x1f".join([workflow, "" if user_id is None else f"u:{user_id}", session_id]).encode()
    ).hexdigest()
    return f"workflow:{workflow}:{digest[:32]}"


def check_value(value: Any, what: str) -> None:
    """Raise :class:`WorkflowCheckpointError` naming ``what`` when
    ``value`` can't be stored."""
    try:
        _codec.dumps(value)
    except (TypeError, ValueError, RecursionError) as exc:
        raise WorkflowCheckpointError(
            f"{what} can't be checkpointed ({exc}). A checkpointed workflow's "
            "values must be JSON types, tuples, sets, bytes, datetimes, enums, "
            "dataclasses or Pydantic models; return one of those, or run "
            "without checkpointer=."
        ) from exc


def fingerprint(value: Any) -> str:
    """A stable digest of an (encodable) value."""
    return hashlib.sha256(_codec.dumps(value)).hexdigest()


def encode_state(state: dict[str, Any]) -> bytes:
    return _codec.dumps({"v": STATE_VERSION, **state})


def decode_state(blob: bytes, *, workflow: str, session_id: str) -> dict[str, Any]:
    try:
        state = _codec.loads(blob)
    except _codec.UndecodableValue as exc:
        raise WorkflowCheckpointError(
            f"the checkpoint of workflow {workflow!r} for session {session_id!r} "
            f"can't be restored ({exc}). If it holds a class that moved or is not "
            "imported yet, import it first; otherwise delete the checkpoint "
            "(Workflow.clear_checkpoint) to start the run over."
        ) from exc
    if not isinstance(state, dict) or state.get("v") != STATE_VERSION:
        raise WorkflowCheckpointError(
            f"the checkpoint of workflow {workflow!r} for session {session_id!r} "
            "was written by a different loomflow version; delete it "
            "(Workflow.clear_checkpoint) to start the run over."
        )
    return state
