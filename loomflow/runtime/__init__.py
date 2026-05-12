"""Durable runtime adapters.

* :class:`InProcRuntime` — no durability; just runs every step.
* :class:`JournaledRuntime` — generic journal-backed runtime that
  caches step results and replays them on a second call to the same
  ``(session_id, step_name)``. Pair with any :class:`JournalStore`.
* :class:`SqliteRuntime` — convenience: ``JournaledRuntime`` rooted
  at a sqlite file. Durable across process restarts.

Future adapters: ``DBOSRuntime`` (Postgres-backed via DBOS workflows),
``TemporalRuntime`` (Temporal cluster).
"""

from .inproc import InProcRuntime, InProcSession
from .journal import (
    InMemoryJournalStore,
    JournalEntry,
    JournalStore,
    PostgresJournalStore,
    SqliteJournalStore,
)
from .journaled import JournaledRuntime, JournaledSession
from .postgres import PostgresRuntime
from .resolver import resolve_runtime
from .sqlite import SqliteRuntime

__all__ = [
    "InMemoryJournalStore",
    "InProcRuntime",
    "InProcSession",
    "JournalEntry",
    "JournalStore",
    "JournaledRuntime",
    "JournaledSession",
    "PostgresJournalStore",
    "PostgresRuntime",
    "SqliteJournalStore",
    "SqliteRuntime",
    "resolve_runtime",
]
