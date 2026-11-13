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

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

#: Upper bound for ``hnsw.ef_search`` (pgvector's maximum is 1000).
MAX_EF_SEARCH = 1000

#: pgvector's HNSW (and IVFFlat) indexes on a ``vector`` column take at
#: most this many dimensions; wider columns can't be ANN-indexed.
HNSW_MAX_DIMENSIONS = 2000

#: Postgres truncates identifiers to 63 bytes (NAMEDATALEN - 1).
MAX_IDENTIFIER_LENGTH = 63

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*\Z")


def validate_identifier(name: Any, *, what: str = "identifier") -> str:
    """Return ``name`` as the identifier Postgres would store for it.

    Table / schema names are interpolated into SQL text (they can't be
    bind parameters), so only a plain identifier is accepted: letters,
    digits, ``_`` and ``$``, not starting with a digit, at most 63
    characters. Anything else — quotes, spaces, hyphens, semicolons —
    raises :class:`ValueError` at construction, before any SQL exists.

    The result is lower-cased, exactly as Postgres folds an unquoted
    identifier, so a name that worked unquoted keeps pointing at the
    same table now that it is emitted quoted.
    """
    if not isinstance(name, str) or not _IDENT_RE.fullmatch(name):
        raise ValueError(
            f"invalid Postgres {what}: {name!r}. Use a plain identifier "
            "(letters, digits and underscores, not starting with a "
            "digit), optionally schema-qualified as 'schema.table'."
        )
    if len(name) > MAX_IDENTIFIER_LENGTH:
        raise ValueError(
            f"Postgres {what} {name!r} is longer than "
            f"{MAX_IDENTIFIER_LENGTH} characters"
        )
    return name.lower()


def quote_identifier(name: str) -> str:
    """Double-quote one (already validated) identifier."""
    return '"' + name.replace('"', '""') + '"'


def split_table_name(table: Any) -> tuple[str | None, str]:
    """Validate a plain or schema-qualified table name and return
    ``(schema, table)`` (``schema`` is ``None`` when unqualified)."""
    if not isinstance(table, str):
        raise ValueError(f"invalid Postgres table name: {table!r}")
    schema, dot, bare = table.rpartition(".")
    if not dot:
        return None, validate_identifier(table, what="table name")
    return (
        validate_identifier(schema, what="schema name"),
        validate_identifier(bare, what="table name"),
    )


def qualified_name(schema: str | None, name: str) -> str:
    """The quoted, optionally schema-qualified reference for SQL."""
    if schema is None:
        return quote_identifier(name)
    return f"{quote_identifier(schema)}.{quote_identifier(name)}"


def index_name(table: str, suffix: str) -> str:
    """A quoted index name derived from the bare table name. Index
    names are never schema-qualified (an index lives in its table's
    schema). A long table name is shortened so the suffix always
    survives Postgres's 63-character limit — otherwise two indexes
    (or the index and its table) could end up with the same name."""
    room = MAX_IDENTIFIER_LENGTH - len(suffix) - 1
    return quote_identifier(f"{table[:room]}_{suffix}")


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


def validate_table_prefix(prefix: Any) -> str:
    """A table-name prefix: empty, or the start of a plain identifier
    (``"myapp_"``). Lower-cased like every other identifier."""
    if prefix == "":
        return ""
    return validate_identifier(prefix, what="table prefix")


def validate_schema(schema: Any) -> str | None:
    """A schema name (``None`` = the connection's search path)."""
    if schema is None:
        return None
    return validate_identifier(schema, what="schema name")


def table_reference(schema: str | None, prefix: str, name: str) -> str:
    """SQL reference for the table ``prefix + name`` in ``schema``.

    With no schema and no prefix this is the bare historical name
    (``episodes``), byte-for-byte what earlier versions emitted;
    anything configured is validated and quoted.
    """
    full = validate_identifier(prefix + name, what="table name")
    if schema is None and not prefix:
        return full
    return qualified_name(schema, full)


def index_reference(prefix: str, name: str) -> str:
    """SQL name for the index ``prefix + name`` (unqualified: an
    index lives in its table's schema)."""
    if not prefix:
        return name
    return quote_identifier(
        validate_identifier(prefix + name, what="index name")
    )
