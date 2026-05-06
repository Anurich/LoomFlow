"""Convenience: ``JournaledRuntime`` rooted at a Postgres pool.

Usage::

    runtime = await PostgresRuntime.connect("postgres://localhost/jeeves")
    await runtime.init_schema()
    agent = Agent("...", model="claude-opus-4-7", runtime=runtime)

The journal lives in two Postgres tables (``journal_steps`` and
``journal_streams``) which :meth:`init_schema` creates idempotently.
Same protocol as :class:`SqliteRuntime`; production-grade durability
when paired with a managed Postgres instance.
"""

from __future__ import annotations

from typing import Any

from .journal import PostgresJournalStore
from .journaled import JournaledRuntime


class PostgresRuntime(JournaledRuntime):
    """:class:`JournaledRuntime` backed by Postgres for cross-host
    durable replay."""

    name = "postgres"

    def __init__(self, pool: Any) -> None:
        store = PostgresJournalStore(pool)
        super().__init__(store=store)
        self._pg_store = store

    @classmethod
    async def connect(
        cls,
        dsn: str,
        *,
        min_size: int = 1,
        max_size: int = 10,
    ) -> PostgresRuntime:
        """Open a fresh asyncpg pool and return the runtime rooted at it."""
        store = await PostgresJournalStore.connect(
            dsn, min_size=min_size, max_size=max_size
        )
        instance = cls.__new__(cls)
        # Bypass ``__init__`` (which creates a new store from a pool)
        # because we already have the connected store.
        JournaledRuntime.__init__(instance, store=store)
        instance._pg_store = store
        return instance

    async def init_schema(self) -> None:
        """Create the journal tables if they don't already exist."""
        await self._pg_store.init_schema()

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        await self._pg_store.aclose()
