"""Convenience subclass: ``JournaledRuntime`` rooted at a sqlite file.

Usage::

    runtime = SqliteRuntime("./jeeves-journal.db")
    agent = Agent("...", runtime=runtime)

The DB file (and any missing parent directories) is created on first
use. Each ``Agent.run()`` opens its own ``runtime.session(session_id)``
context, so multiple concurrent runs share the same sqlite file
without conflicting on rows.
"""

from __future__ import annotations

from pathlib import Path

from .journal import SqliteJournalStore
from .journaled import JournaledRuntime


class SqliteRuntime(JournaledRuntime):
    """:class:`JournaledRuntime` with a :class:`SqliteJournalStore`."""

    name = "sqlite"

    def __init__(
        self,
        path: str | Path,
        *,
        max_checkpoints_per_session: int = 20,
    ) -> None:
        store = SqliteJournalStore(
            path, max_checkpoints_per_session=max_checkpoints_per_session
        )
        super().__init__(store=store)
        self._path = store.path

    @property
    def path(self) -> Path:
        return self._path
