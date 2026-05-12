"""SQLite-backed bi-temporal fact store.

Same shape as :class:`InMemoryFactStore` (supersession on append,
``valid_at`` queries, optional embedder) but durable across process
restarts. Sync sqlite3 calls dispatched through
:func:`anyio.to_thread.run_sync`.

Schema:

* ``facts(id, subject, predicate, object, confidence, valid_from,
  valid_until, recorded_at, sources, embedding)`` — timestamps stored
  as unix-epoch floats; ``sources`` as a JSON-encoded array;
  ``embedding`` as a float32 BLOB or NULL.
* Indexes on ``subject`` and ``(subject, predicate)`` for the common
  filter shapes.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio

from ..core.protocols import Embedder
from ..core.types import Fact, _normalize_predicate
from ._embedding_util import pack_float32, unpack_float32

_FACTS_DDL = """
CREATE TABLE IF NOT EXISTS facts (
    id          TEXT PRIMARY KEY,
    user_id     TEXT,
    subject     TEXT NOT NULL,
    predicate   TEXT NOT NULL,
    object      TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 1.0,
    valid_from  REAL NOT NULL,
    valid_until REAL,
    recorded_at REAL NOT NULL,
    sources     TEXT NOT NULL DEFAULT '[]',
    embedding   BLOB
)
"""

# Idempotent ALTER for upgrades from a pre-``user_id`` schema.
_FACTS_ADD_USER_ID = "ALTER TABLE facts ADD COLUMN user_id TEXT"

_FACTS_SUBJECT_INDEX = (
    "CREATE INDEX IF NOT EXISTS facts_subject_idx ON facts (subject)"
)
_FACTS_USER_SUBJECT_PRED_INDEX = (
    "CREATE INDEX IF NOT EXISTS facts_user_subject_predicate_idx "
    "ON facts (user_id, subject, predicate)"
)


class SqliteFactStore:
    """Durable bi-temporal fact store rooted at a sqlite file."""

    def __init__(
        self,
        path: str | Path,
        *,
        embedder: Embedder | None = None,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        self._init_schema()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def embedder(self) -> Embedder | None:
        return self._embedder

    # ---- connection management -------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # New connection per call; SQLite connections aren't safe to
        # share across the worker threads we hop into.
        conn = sqlite3.connect(self._path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(_FACTS_DDL)
            # Best-effort upgrade: add the column if it doesn't exist.
            # ``ALTER TABLE ADD COLUMN`` raises if the column is already
            # present; suppress that case but let real errors propagate.
            try:
                conn.execute(_FACTS_ADD_USER_ID)
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
            conn.execute(_FACTS_SUBJECT_INDEX)
            conn.execute(_FACTS_USER_SUBJECT_PRED_INDEX)
            conn.commit()

    # ---- mutation --------------------------------------------------------

    async def append(self, fact: Fact) -> str:
        """Append a fact, invalidating any superseded predecessors.

        Same supersession rule as :class:`InMemoryFactStore`: if there's
        an existing currently-valid fact with matching subject +
        predicate but different object, set its ``valid_until`` to the
        new fact's ``valid_from``.
        """
        embedding_blob: bytes | None = None
        if self._embedder is not None:
            triple = f"{fact.subject} {fact.predicate} {fact.object}"
            embedding = await self._embedder.embed(triple)
            embedding_blob = pack_float32(embedding)

        await anyio.to_thread.run_sync(
            self._append_sync, fact, embedding_blob
        )
        return fact.id

    def _append_sync(
        self,
        fact: Fact,
        embedding_blob: bytes | None,
    ) -> None:
        with self._connect() as conn:
            # Close off any still-valid superseded predecessors.
            # Namespace-scoped: alice's facts never invalidate bob's.
            # SQLite uses ``IS`` to compare against NULL (since
            # ``=`` returns NULL when either side is NULL).
            conn.execute(
                "UPDATE facts SET valid_until = ? "
                "WHERE user_id IS ? "
                "AND subject = ? AND predicate = ? AND object != ? "
                "AND valid_until IS NULL",
                (
                    _to_epoch(fact.valid_from),
                    fact.user_id,
                    fact.subject,
                    fact.predicate,
                    fact.object,
                ),
            )
            conn.execute(
                "INSERT OR REPLACE INTO facts "
                "(id, user_id, subject, predicate, object, confidence, "
                "valid_from, valid_until, recorded_at, sources, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    fact.id,
                    fact.user_id,
                    fact.subject,
                    fact.predicate,
                    fact.object,
                    fact.confidence,
                    _to_epoch(fact.valid_from),
                    _to_epoch(fact.valid_until)
                    if fact.valid_until is not None
                    else None,
                    _to_epoch(fact.recorded_at),
                    json.dumps(list(fact.sources)),
                    embedding_blob,
                ),
            )
            conn.commit()

    # ---- queries ---------------------------------------------------------

    async def query(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object_: str | None = None,
        valid_at: datetime | None = None,
        limit: int = 10,
        user_id: str | None = None,
    ) -> list[Fact]:
        rows = await anyio.to_thread.run_sync(
            self._query_sync,
            subject,
            predicate,
            object_,
            valid_at,
            limit,
            user_id,
        )
        return [_row_to_fact(r) for r in rows]

    def _query_sync(
        self,
        subject: str | None,
        predicate: str | None,
        object_: str | None,
        valid_at: datetime | None,
        limit: int,
        user_id: str | None,
    ) -> list[tuple[Any, ...]]:
        # Hard namespace partition by ``user_id`` (always in WHERE).
        sql_parts = ["SELECT * FROM facts WHERE user_id IS ?"]
        params: list[Any] = [user_id]
        if subject is not None:
            sql_parts.append("AND subject = ?")
            params.append(subject)
        if predicate is not None:
            sql_parts.append("AND predicate = ?")
            params.append(_normalize_predicate(predicate))
        if object_ is not None:
            sql_parts.append("AND object = ?")
            params.append(object_)
        if valid_at is not None:
            ts = _to_epoch(valid_at)
            sql_parts.append(
                "AND valid_from <= ? "
                "AND (valid_until IS NULL OR ? < valid_until)"
            )
            params.extend([ts, ts])
        sql_parts.append("ORDER BY recorded_at DESC LIMIT ?")
        params.append(limit)

        with self._connect() as conn:
            cursor = conn.execute(" ".join(sql_parts), params)
            return cursor.fetchall()

    async def recall_text(
        self,
        query: str,
        *,
        limit: int = 5,
        valid_at: datetime | None = None,
        user_id: str | None = None,
    ) -> list[Fact]:
        if self._embedder is not None:
            return await self._recall_embedding(query, limit, valid_at, user_id)
        return await self._recall_tokens(query, limit, valid_at, user_id)

    async def _recall_embedding(
        self,
        query: str,
        limit: int,
        valid_at: datetime | None,
        user_id: str | None,
    ) -> list[Fact]:
        assert self._embedder is not None
        query_embedding = await self._embedder.embed(query)
        rows = await anyio.to_thread.run_sync(
            self._scan_for_recall, valid_at, user_id
        )
        scored: list[tuple[float, tuple[Any, ...]]] = []
        for row in rows:
            blob = row[10]  # embedding column (index shifted by user_id)
            if not blob:
                continue
            stored = unpack_float32(bytes(blob))
            scored.append((_cosine(query_embedding, stored), row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [_row_to_fact(r) for _, r in scored[:limit]]

    async def _recall_tokens(
        self,
        query: str,
        limit: int,
        valid_at: datetime | None,
        user_id: str | None,
    ) -> list[Fact]:
        rows = await anyio.to_thread.run_sync(
            self._scan_for_recall, valid_at, user_id
        )
        query_tokens = _tokenize(query)
        if not query_tokens:
            # Recency fallback when query has no useful tokens.
            return [_row_to_fact(r) for r in rows[:limit]]

        scored: list[tuple[int, int, tuple[Any, ...]]] = []
        for row in rows:
            # Columns shifted by user_id at index 1: subject/predicate/
            # object are now indices 2/3/4.
            haystack = f"{row[2]} {row[3]} {row[4]}"
            haystack_tokens = _tokenize(haystack)
            overlap = sum(1 for t in query_tokens if t in haystack_tokens)
            if overlap > 0:
                scored.append((-overlap, len(haystack), row))
        scored.sort()
        return [_row_to_fact(r) for _, _, r in scored[:limit]]

    def _scan_for_recall(
        self, valid_at: datetime | None, user_id: str | None
    ) -> list[tuple[Any, ...]]:
        with self._connect() as conn:
            if valid_at is None:
                cursor = conn.execute(
                    "SELECT * FROM facts WHERE user_id IS ? "
                    "ORDER BY recorded_at DESC",
                    (user_id,),
                )
            else:
                ts = _to_epoch(valid_at)
                cursor = conn.execute(
                    "SELECT * FROM facts "
                    "WHERE user_id IS ? "
                    "AND valid_from <= ? "
                    "AND (valid_until IS NULL OR ? < valid_until) "
                    "ORDER BY recorded_at DESC",
                    (user_id, ts, ts),
                )
            return cursor.fetchall()

    async def all_facts(self) -> list[Fact]:
        rows = await anyio.to_thread.run_sync(self._all_sync)
        return [_row_to_fact(r) for r in rows]

    def _all_sync(self) -> list[tuple[Any, ...]]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM facts ORDER BY recorded_at DESC"
            ).fetchall()

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_epoch(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def _from_epoch(ts: float | None) -> datetime | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=UTC)


def _row_to_fact(row: tuple[Any, ...]) -> Fact:
    # Column layout (after the user_id migration):
    # 0:id 1:user_id 2:subject 3:predicate 4:object 5:confidence
    # 6:valid_from 7:valid_until 8:recorded_at 9:sources 10:embedding
    sources: list[str] = []
    if row[9]:
        try:
            sources = list(json.loads(row[9]))
        except json.JSONDecodeError:
            sources = []
    valid_from = _from_epoch(row[6])
    assert valid_from is not None
    recorded_at = _from_epoch(row[8])
    assert recorded_at is not None
    return Fact(
        id=row[0],
        user_id=row[1],
        subject=row[2],
        predicate=row[3],
        object=row[4],
        confidence=row[5],
        valid_from=valid_from,
        valid_until=_from_epoch(row[7]),
        recorded_at=recorded_at,
        sources=sources,
    )


def _tokenize(text: str) -> set[str]:
    """Same tokenisation as :mod:`memory.facts`.

    Duplicated here rather than imported to avoid a circular import
    between ``facts`` and ``sqlite_facts``.
    """
    out: set[str] = set()
    buf: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                token = "".join(buf)
                if len(token) >= 2 and token not in _STOP_WORDS:
                    out.add(token)
            buf = []
    if buf:
        token = "".join(buf)
        if len(token) >= 2 and token not in _STOP_WORDS:
            out.add(token)
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "with", "from", "into", "this", "that",
        "what", "tell", "you", "are", "is", "be", "of", "to", "in",
        "on", "an", "or", "me", "my", "us", "our", "by", "as", "at",
        "it", "its", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "can",
    }
)
