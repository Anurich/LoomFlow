"""HNSW scan settings shared by the Postgres stores.

pgvector's HNSW index scan returns at most ``hnsw.ef_search`` rows
(40 by default) and, before 0.8, applied ``WHERE`` filters *after* the
scan — so a query for k > 40 rows, or a per-tenant filtered query,
could return fewer rows than exist. :func:`hnsw_scan` sets, for one
transaction, an ``ef_search`` large enough for the request and, when
the query is filtered and the server's pgvector has it (0.8+),
``hnsw.iterative_scan = strict_order`` so the scan keeps going until
the filter has enough rows, in exact distance order.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

#: Upper bound for ``hnsw.ef_search`` (pgvector's maximum is 1000).
MAX_EF_SEARCH = 1000


@asynccontextmanager
async def hnsw_scan(conn: Any, *, fetch: int, filtered: bool) -> AsyncIterator[None]:
    """Run the enclosed query in a transaction tuned to return ``fetch``
    rows. Connections without ``transaction()`` (test fakes) run as-is."""
    transaction = getattr(conn, "transaction", None)
    if transaction is None:
        yield
        return
    async with transaction():
        await conn.execute(f"SET LOCAL hnsw.ef_search = {min(MAX_EF_SEARCH, max(40, fetch))}")
        if filtered:
            try:
                async with transaction():  # a savepoint: older pgvector lacks the setting
                    await conn.execute("SET LOCAL hnsw.iterative_scan = strict_order")
            except Exception:  # noqa: BLE001 — pgvector < 0.8: post-filtering it is
                pass
        yield
