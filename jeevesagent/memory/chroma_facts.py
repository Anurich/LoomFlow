"""Chroma-backed bi-temporal fact store.

Each fact lives in a Chroma collection as a (id, embedding, document,
metadata) tuple. The metadata carries the bi-temporal fields:

* ``subject`` / ``predicate`` / ``object`` — strings
* ``confidence`` — float
* ``valid_from_ts`` / ``recorded_at_ts`` — unix-epoch floats
* ``valid_until_ts`` — unix-epoch float; ``0.0`` when still valid
* ``currently_valid`` — bool, mirrors ``valid_until_ts == 0`` so we
  can use it directly in Chroma's ``where`` filters
* ``sources`` — JSON-encoded list of episode ids

Supersession is two round-trips: a ``coll.get`` to find the prior
currently-valid facts with matching subject + predicate + different
object, followed by a ``coll.update`` that flips their
``currently_valid`` to false and stamps ``valid_until_ts`` to the new
fact's ``valid_from``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import anyio

from ..core.protocols import Embedder
from ..core.types import Fact
from .embedder import HashEmbedder

DEFAULT_FACTS_COLLECTION = "jeeves_facts"


class ChromaFactStore:
    """Bi-temporal fact store backed by a Chroma collection."""

    def __init__(
        self,
        client: Any,
        *,
        embedder: Embedder | None = None,
        collection_name: str = DEFAULT_FACTS_COLLECTION,
    ) -> None:
        self._client = client
        self._embedder: Embedder = (
            embedder if embedder is not None else HashEmbedder()
        )
        self._collection_name = collection_name
        self._collection: Any | None = None
        self._lock = anyio.Lock()

    # ---- factories -------------------------------------------------------

    @classmethod
    def local(
        cls,
        persist_directory: str,
        *,
        embedder: Embedder | None = None,
        collection_name: str = DEFAULT_FACTS_COLLECTION,
    ) -> ChromaFactStore:
        client = _make_client(persist_directory=persist_directory)
        return cls(
            client,
            embedder=embedder,
            collection_name=collection_name,
        )

    @classmethod
    def ephemeral(
        cls,
        *,
        embedder: Embedder | None = None,
        collection_name: str = DEFAULT_FACTS_COLLECTION,
    ) -> ChromaFactStore:
        client = _make_client(persist_directory=None)
        return cls(
            client,
            embedder=embedder,
            collection_name=collection_name,
        )

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    # ---- collection lazy-init -------------------------------------------

    async def _get_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        coll = await anyio.to_thread.run_sync(
            lambda: self._client.get_or_create_collection(
                name=self._collection_name
            )
        )
        self._collection = coll
        return coll

    # ---- mutation --------------------------------------------------------

    async def append(self, fact: Fact) -> str:
        triple = _triple_text(fact)
        embedding = await self._embedder.embed(triple)

        coll = await self._get_collection()

        async with self._lock:
            # Find currently-valid prior facts with matching subject +
            # predicate that we may need to supersede.
            existing = await anyio.to_thread.run_sync(
                lambda: coll.get(
                    where={
                        "$and": [
                            {"subject": fact.subject},
                            {"predicate": fact.predicate},
                            {"currently_valid": True},
                        ]
                    },
                    include=["metadatas"],
                )
            )

            ids_to_close: list[str] = []
            metas_to_close: list[dict[str, Any]] = []
            for eid, meta in zip(
                existing.get("ids") or [],
                existing.get("metadatas") or [],
                strict=False,
            ):
                meta = dict(meta or {})
                if meta.get("object") == fact.object:
                    continue  # same triple — don't supersede
                meta["currently_valid"] = False
                meta["valid_until_ts"] = fact.valid_from.timestamp()
                ids_to_close.append(eid)
                metas_to_close.append(meta)

            if ids_to_close:
                await anyio.to_thread.run_sync(
                    lambda: coll.update(
                        ids=ids_to_close,
                        metadatas=metas_to_close,
                    )
                )

            metadata = _fact_to_metadata(fact)
            await anyio.to_thread.run_sync(
                lambda: coll.upsert(
                    ids=[fact.id],
                    embeddings=[embedding],
                    documents=[triple],
                    metadatas=[metadata],
                )
            )
        return fact.id

    # ---- queries ---------------------------------------------------------

    async def query(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object_: str | None = None,
        valid_at: datetime | None = None,
        limit: int = 10,
    ) -> list[Fact]:
        coll = await self._get_collection()
        where = _build_where(subject, predicate, object_, valid_at)

        # Chroma's ``get`` accepts ``limit`` only in newer releases;
        # fall back to slicing in Python if it raises.
        def _do_get() -> Any:
            try:
                return coll.get(
                    where=where,
                    limit=limit,
                    include=["metadatas"],
                )
            except TypeError:
                return coll.get(where=where, include=["metadatas"])

        result = await anyio.to_thread.run_sync(_do_get)
        facts = _decode_get(result)
        # Sort by recorded_at desc; tie-break by valid_from desc.
        facts.sort(
            key=lambda f: (f.recorded_at, f.valid_from),
            reverse=True,
        )
        return facts[:limit]

    async def recall_text(
        self,
        query: str,
        *,
        limit: int = 5,
        valid_at: datetime | None = None,
    ) -> list[Fact]:
        coll = await self._get_collection()
        query_embedding = await self._embedder.embed(query)
        where = _build_where(None, None, None, valid_at)

        result = await anyio.to_thread.run_sync(
            lambda: coll.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where,
                include=["metadatas"],
            )
        )
        return _decode_query(result)

    async def all_facts(self) -> list[Fact]:
        coll = await self._get_collection()
        result = await anyio.to_thread.run_sync(
            lambda: coll.get(include=["metadatas"])
        )
        return _decode_get(result)

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(*, persist_directory: str | None) -> Any:
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "chromadb is not installed. "
            "Install with: pip install chromadb"
        ) from exc
    if persist_directory is None:
        return chromadb.EphemeralClient()
    return chromadb.PersistentClient(path=persist_directory)


def _triple_text(fact: Fact) -> str:
    return f"{fact.subject} {fact.predicate} {fact.object}"


def _fact_to_metadata(fact: Fact) -> dict[str, Any]:
    return {
        "subject": fact.subject,
        "predicate": fact.predicate,
        "object": fact.object,
        "confidence": fact.confidence,
        "valid_from_ts": fact.valid_from.timestamp(),
        "valid_until_ts": (
            fact.valid_until.timestamp()
            if fact.valid_until is not None
            else 0.0
        ),
        "currently_valid": fact.valid_until is None,
        "recorded_at_ts": fact.recorded_at.timestamp(),
        "sources": json.dumps(list(fact.sources)),
    }


def _metadata_to_fact(eid: str, meta: dict[str, Any]) -> Fact:
    raw_sources = meta.get("sources", "[]")
    sources: list[str] = []
    if isinstance(raw_sources, str):
        try:
            sources = list(json.loads(raw_sources))
        except json.JSONDecodeError:
            sources = []
    valid_until: datetime | None = None
    until_ts = meta.get("valid_until_ts", 0.0) or 0.0
    if not meta.get("currently_valid", True) and until_ts > 0:
        valid_until = datetime.fromtimestamp(float(until_ts), tz=UTC)
    return Fact(
        id=eid,
        subject=str(meta.get("subject", "")),
        predicate=str(meta.get("predicate", "")),
        object=str(meta.get("object", "")),
        confidence=float(meta.get("confidence", 1.0)),
        valid_from=datetime.fromtimestamp(
            float(meta.get("valid_from_ts", 0.0)), tz=UTC
        ),
        valid_until=valid_until,
        recorded_at=datetime.fromtimestamp(
            float(meta.get("recorded_at_ts", 0.0)), tz=UTC
        ),
        sources=sources,
    )


def _decode_get(result: dict[str, Any]) -> list[Fact]:
    ids = result.get("ids") or []
    metas = result.get("metadatas") or []
    facts: list[Fact] = []
    for i, eid in enumerate(ids):
        meta = metas[i] if i < len(metas) and metas[i] is not None else {}
        facts.append(_metadata_to_fact(str(eid), dict(meta)))
    return facts


def _decode_query(result: dict[str, Any]) -> list[Fact]:
    """``coll.query`` returns nested lists (one per query). We always
    pass a single query, so we look at the first row."""
    ids_lists = result.get("ids") or [[]]
    metas_lists = result.get("metadatas") or [[]]
    ids = ids_lists[0] if ids_lists else []
    metas = metas_lists[0] if metas_lists else []
    facts: list[Fact] = []
    for i, eid in enumerate(ids):
        meta = metas[i] if i < len(metas) and metas[i] is not None else {}
        facts.append(_metadata_to_fact(str(eid), dict(meta)))
    return facts


def _build_where(
    subject: str | None,
    predicate: str | None,
    object_: str | None,
    valid_at: datetime | None,
) -> dict[str, Any] | None:
    """Compose Chroma ``where`` from optional filters. Multiple filters
    fold into a single ``$and``; a single filter is its own clause."""
    clauses: list[dict[str, Any]] = []
    if subject is not None:
        clauses.append({"subject": subject})
    if predicate is not None:
        clauses.append({"predicate": predicate})
    if object_ is not None:
        clauses.append({"object": object_})
    if valid_at is not None:
        ts = valid_at.timestamp()
        clauses.append({"valid_from_ts": {"$lte": ts}})
        clauses.append(
            {
                "$or": [
                    {"currently_valid": True},
                    {"valid_until_ts": {"$gt": ts}},
                ]
            }
        )

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
