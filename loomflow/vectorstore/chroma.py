"""Chroma-backed vector store.

Wraps ``chromadb`` for persistent on-disk or hosted Chroma. Lazy
import — install via ``pip install 'loomflow[vectorstore-chroma]'``.

Embeddings come from our framework's :class:`Embedder` protocol so
swapping embedders works the same across every vector store. We
pass ``None`` as Chroma's ``embedding_function`` and supply
embeddings ourselves at ``add`` time.

Filter operators are translated from our Mongo-style language to
Chroma's native ``where`` syntax (which already speaks Mongo-ish
``$eq`` / ``$in`` / ``$gt`` etc.). Chroma runs the part of a filter
its engine can express — always a superset of the matches — and the
full filter is then re-checked in Python, so the results follow the
same contract as the other stores (see
:class:`~loomflow.vectorstore.base.VectorStore`).
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from functools import partial
from typing import Any

import anyio

from ..core.protocols import Embedder
from ..loader.base import Chunk
from ._filter import COMPARISON_OPERATORS, LOGICAL_OPERATORS, FilterError, evaluate_filter
from ._mmr import rerank_tail
from ._util import (
    dedupe_last,
    embed_all,
    resolve_ids,
    validate_k,
    validate_query_vector,
)
from .base import SearchResult, _chunks_from_texts


def get_or_create_cosine_collection(client: Any, name: str) -> Any:
    """Get or create the collection ``name`` in cosine space.

    ``hnsw:space: cosine`` makes Chroma's distance a true cosine
    distance, so rankings match the other backends. (Chroma defaults
    to L2 when the space is unspecified, which ranks differently for
    embeddings that aren't unit length.) ``get_or_create`` can't change
    the space of a pre-existing collection, so a non-cosine one gets a
    warning; missing metadata means it pre-dates this setting and runs
    on Chroma's L2 default.
    """
    collection = client.get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )
    coll_meta = getattr(collection, "metadata", None) or {}
    space = coll_meta.get("hnsw:space", "l2")
    if space != "cosine":
        warnings.warn(
            f"Chroma collection {name!r} already exists with "
            f"hnsw:space={space!r} (not 'cosine'); its scores won't "
            "follow the cosine contract and rankings can differ from "
            "other backends. Re-index into a new collection to fix.",
            UserWarning,
            stacklevel=3,
        )
    return collection


DEFAULT_COLLECTION = "jeeves_vectors"

# Fallback when a client doesn't report its own limit (Chroma's
# default is 5461 records per call).
_FALLBACK_MAX_BATCH = 5000

# Bookkeeping keys this store writes next to the user's metadata.
# Everything under the prefix is stripped again on read.
_RESERVED_PREFIX = "__loomflow_"
# ``<prefix><key>: True`` marks ``key`` as holding a JSON-encoded
# list / dict (Chroma stores scalars only).
_JSON_MARKER_PREFIX = "__loomflow_json__"
# Chroma rejects an empty metadata dict; this stands in for one.
_EMPTY_SENTINEL = "__loomflow_empty__"
# Rows written before the sentinel was namespaced carry exactly
# ``{"_empty": True}``. A user whose own metadata is that exact dict
# gets this extra key so it isn't mistaken for the legacy sentinel.
_LEGACY_EMPTY_SENTINEL = "_empty"
_USER_EMPTY_MARKER = "__loomflow_user_empty__"

# Set once a default (in-memory client, default collection) store has
# warned about sharing its collection — one warning per process.
_SHARED_DEFAULT_WARNED = False


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _json_marker(key: str) -> dict[str, Any]:
    return {_JSON_MARKER_PREFIX + key: {"$eq": True}}


def _combine(op: str, parts: list[dict[str, Any]]) -> dict[str, Any] | None:
    """``$and`` / ``$or`` of ``parts`` (Chroma requires two or more
    operands, so a single part stands alone)."""
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else {op: parts}


def _typed_groups(values: list[Any]) -> list[list[Any]] | None:
    """Split ``$in`` operands into same-typed lists (Chroma rejects a
    mixed one); ``None`` if any can't go to the engine."""
    groups: dict[type, list[Any]] = {}
    for value in values:
        if value is None or not isinstance(value, str | int | float | bool):
            return None
        groups.setdefault(type(value), []).append(value)
    return list(groups.values())


def _engine_field(
    key: str, condition: Any, loose: list[str]
) -> dict[str, Any] | None:
    """Engine-side superset for one field constraint (``None`` = can't
    be expressed; the Python re-check does all the work). Operators
    the engine had to skip are noted in ``loose``."""
    if isinstance(condition, Mapping) and condition and all(
        k.startswith("$") for k in condition
    ):
        ops = dict(condition)
    elif isinstance(condition, list | tuple):
        ops = {"$in": list(condition)}
    else:
        ops = {"$eq": condition}

    parts: list[dict[str, Any]] = []
    for op, expected in ops.items():
        if op not in COMPARISON_OPERATORS:
            raise FilterError(f"Unknown field operator: {op}")
        if op == "$eq":
            if expected is None:
                loose.append(op)  # "missing or None": not expressible
            elif isinstance(expected, str | int | float | bool):
                # Rows holding a JSON-encoded list match by element,
                # which only the re-check can decide: let them through.
                parts.append(
                    {"$or": [{key: {"$eq": expected}}, _json_marker(key)]}
                )
            else:
                parts.append(_json_marker(key))
        elif op == "$in":
            if not isinstance(expected, list | tuple):
                raise FilterError("$in expects a list")
            groups = _typed_groups(list(expected))
            if groups is None:
                loose.append(op)
                continue
            alternatives = [{key: {"$in": g}} for g in groups]
            alternatives.append(_json_marker(key))
            combined = _combine("$or", alternatives)
            if combined is not None:
                parts.append(combined)
        elif op == "$ne":
            if isinstance(expected, str | int | float | bool):
                parts.append({key: {"$ne": expected}})
            else:
                loose.append(op)
        elif op == "$nin":
            if not isinstance(expected, list | tuple):
                raise FilterError("$nin expects a list")
            groups = _typed_groups(list(expected))
            if groups:
                parts.extend({key: {"$nin": g}} for g in groups)
            else:
                loose.append(op)
        elif op in ("$gt", "$gte", "$lt", "$lte"):
            # Chroma orders numbers only; string ordering is re-checked.
            if _is_number(expected):
                parts.append({key: {op: expected}})
            else:
                loose.append(op)
        else:  # ``$exists``: no engine equivalent
            loose.append(op)
    return _combine("$and", parts)


def _engine_part(
    node: Mapping[str, Any], loose: list[str] | None = None
) -> dict[str, Any] | None:
    """A ``where`` Chroma can run that matches *at least* what ``node``
    matches (``None`` = no constraint): conditions its engine can't
    express are dropped from conjunctions, and a disjunction with one
    is dropped whole, so nothing that should match is filtered out.
    The caller re-checks every returned row against the full filter;
    ``loose`` collects what was dropped, so it knows to over-fetch.
    """
    if loose is None:
        loose = []
    kept: list[dict[str, Any]] = []
    for key, value in node.items():
        if key == "$not":
            if not isinstance(value, Mapping):
                raise FilterError("$not expects a subfilter dict")
            loose.append(key)
            continue
        if key == "$and":
            if not isinstance(value, list):
                raise FilterError("$and expects a list of subfilters")
            kept.extend(
                p for p in (_engine_part(v, loose) for v in value) if p
            )
        elif key == "$or":
            if not isinstance(value, list):
                raise FilterError("$or expects a list of subfilters")
            or_parts = [_engine_part(v, loose) for v in value]
            if or_parts and all(or_parts):
                combined = _combine("$or", [p for p in or_parts if p])
                if combined is not None:
                    kept.append(combined)
            else:
                loose.append(key)
        elif key.startswith("$"):
            raise FilterError(f"Unknown top-level operator: {key}")
        else:
            part = _engine_field(key, value, loose)
            if part is not None:
                kept.append(part)
    return _combine("$and", kept)


# How many times a filtered search widens its candidate pool (x4 each)
# when the Python re-check left fewer than ``k`` rows.
_MAX_WIDENINGS = 4


def _translate_filter(
    filter: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Translate our Mongo-style filter to Chroma's ``where`` shape.

    Chroma already speaks Mongo-ish operators natively, but with two
    quirks: scalar shorthand isn't accepted (must be ``{"$eq": v}``),
    and the top-level ``$and`` is implicit when multiple keys are
    present. We normalize both.
    """
    if not filter:
        return None
    return _xlate_node(filter)


def _xlate_node(node: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in LOGICAL_OPERATORS:
            if key == "$not":
                # Never reached from search: ``_engine_part`` drops
                # ``$not`` (it is post-filtered in Python).
                raise FilterError(
                    "$not isn't supported by Chroma's where filter."
                )
            assert isinstance(value, list)
            out[key] = [_xlate_node(sub) for sub in value]
        elif key.startswith("$"):
            raise FilterError(f"Unknown top-level operator: {key}")
        else:
            out[key] = _xlate_field(value)
    # Chroma expects an explicit $and when there are multiple field
    # constraints at the top level.
    if len(out) > 1 and not any(k in LOGICAL_OPERATORS for k in out):
        return {"$and": [{k: v} for k, v in out.items()]}
    return out


def _xlate_field(condition: Any) -> dict[str, Any]:
    """Normalize a field constraint to operator form."""
    if isinstance(condition, Mapping) and condition and all(
        k.startswith("$") for k in condition
    ):
        for op in condition:
            if op not in COMPARISON_OPERATORS:
                raise FilterError(f"Unknown field operator: {op}")
        return dict(condition)
    if isinstance(condition, list | tuple):
        return {"$in": list(condition)}
    return {"$eq": condition}


def _flatten_metadata(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce a chunk's metadata into what chromadb accepts.

    chromadb stores ONLY scalar metadata values (str / int / float /
    bool). The framework's own chunkers emit non-scalars —
    ``MarkdownChunker`` writes a ``headers`` LIST — so a bare
    ``store.add(chunker.split(...))`` would crash on its own loader's
    output. Scalars pass through untouched (equality / range filters
    on them are unchanged). ``None`` values are dropped — Chroma
    rejects them, and the filter contract treats a missing key and
    ``None`` alike. Anything else is JSON-serialised and its key
    flagged with a ``__loomflow_json__<key>`` marker, so reads decode
    it back (:func:`_restore_metadata`) and filters can match lists
    by element. ``default=str`` guards exotic objects so this can
    never itself raise. Postgres (JSONB) / InMemory / FAISS keep
    dicts natively and never hit this path.
    """
    out: dict[str, Any] = {}
    for key, value in meta.items():
        if isinstance(key, str) and key.startswith(_RESERVED_PREFIX):
            raise ValueError(
                f"metadata key {key!r} is reserved: keys starting with "
                f"{_RESERVED_PREFIX!r} are used by ChromaVectorStore"
            )
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        else:
            out[key] = json.dumps(value, default=str)
            out[_JSON_MARKER_PREFIX + key] = True
    return out


def _stored_metadata(meta: Mapping[str, Any] | None) -> dict[str, Any]:
    """What goes into Chroma for one chunk: flattened metadata, or
    the sentinel when there is nothing to store (Chroma rejects an
    empty dict)."""
    flat = _flatten_metadata(meta or {})
    if not flat:
        return {_EMPTY_SENTINEL: True}
    if flat == {_LEGACY_EMPTY_SENTINEL: True}:
        # The user's own ``{"_empty": True}`` — tell it apart from the
        # legacy sentinel so it round-trips.
        flat[_USER_EMPTY_MARKER] = True
    return flat


def _restore_metadata(stored: Mapping[str, Any] | None) -> dict[str, Any]:
    """Inverse of :func:`_stored_metadata`: drop the bookkeeping keys
    and decode JSON-encoded values back into lists / dicts."""
    meta = dict(stored or {})
    if meta == {_LEGACY_EMPTY_SENTINEL: True}:
        return {}  # sentinel written by older versions
    out: dict[str, Any] = {}
    for key, value in meta.items():
        if key.startswith(_RESERVED_PREFIX):
            continue
        if meta.get(_JSON_MARKER_PREFIX + key) is True and isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        out[key] = value
    return out


class ChromaVectorStore:
    """Vector store backed by ``chromadb``.

    Collections are created with ``{"hnsw:space": "cosine"}`` so
    Chroma's reported distance is cosine distance and search scores
    honour the cross-store cosine contract (``score = 1 - distance``
    in ``[-1, 1]``). LIMITATION: ``get_or_create_collection`` cannot
    change the space of a PRE-EXISTING collection — a collection
    created by an older version of this class kept Chroma's default
    ``l2`` space, and its scores are NOT cosine. We emit a
    ``UserWarning`` when that's cheaply detectable from the
    collection's metadata; re-index into a fresh collection to fix.

    .. warning::

       **Stores share data by collection name.** Every store built
       on the same client (or the same ``persist_directory``) with
       the same ``collection_name`` reads and writes the SAME
       collection — including the default ``"jeeves_vectors"``. In
       particular two ``ChromaVectorStore(embedder)`` instances with
       no ``client`` / ``persist_directory`` share chromadb's
       process-wide in-memory client, so documents added through one
       show up in the other's searches. Give each corpus its own
       ``collection_name``. A default in-memory store that finds
       documents it didn't add emits a ``UserWarning`` (once per
       process).

    ``add`` upserts (an existing id is replaced) and splits large
    inputs into the client's maximum batch size, embedding one batch
    at a time.
    """

    name = "chroma"

    def __init__(
        self,
        embedder: Embedder,
        *,
        collection_name: str = DEFAULT_COLLECTION,
        persist_directory: str | None = None,
        client: Any = None,
    ) -> None:
        if embedder is None:
            raise ValueError("embedder is required")
        self._embedder = embedder
        self._collection_name = collection_name
        # The footgun case: default collection on chromadb's shared
        # in-memory client (see the class docstring).
        self._default_in_memory = (
            client is None
            and persist_directory is None
            and collection_name == DEFAULT_COLLECTION
        )
        self._wrote = False

        try:
            import chromadb  # type: ignore[import-not-found, import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "chromadb is not installed. "
                "Install with: pip install 'loomflow[vectorstore-chroma]'."
            ) from exc

        if client is not None:
            self._client = client
        elif persist_directory is not None:
            self._client = chromadb.PersistentClient(
                path=persist_directory
            )
        else:
            self._client = chromadb.Client()

        self._collection = get_or_create_cosine_collection(
            self._client, collection_name
        )

    async def _warn_if_shared_default(self) -> None:
        """Warn (once per process) when a default in-memory store
        finds documents it didn't add: another default store in this
        process is writing to the same collection."""
        global _SHARED_DEFAULT_WARNED
        if (
            not self._default_in_memory
            or self._wrote
            or _SHARED_DEFAULT_WARNED
        ):
            return
        if int(await anyio.to_thread.run_sync(self._collection.count)) == 0:
            return
        _SHARED_DEFAULT_WARNED = True
        warnings.warn(
            f"ChromaVectorStore: the default collection "
            f"{DEFAULT_COLLECTION!r} on chromadb's shared in-memory "
            "client already holds documents added through another "
            "store, and this store will read and write the same data. "
            "Pass a distinct collection_name= (or persist_directory= / "
            "client=) to keep stores separate.",
            UserWarning,
            stacklevel=4,
        )

    def _max_batch_size(self) -> int:
        """The most records the client takes in one call."""
        getter = getattr(self._client, "get_max_batch_size", None)
        try:
            size = int(getter()) if callable(getter) else 0
        except Exception:  # noqa: BLE001 — a limit we can't read isn't fatal
            size = 0
        return size if size > 0 else _FALLBACK_MAX_BATCH

    # ---------------------------------------------------------------
    # Factory classmethods — explicit kwargs so IDEs autocomplete
    # ---------------------------------------------------------------

    @classmethod
    async def from_chunks(
        cls,
        chunks: list[Chunk],
        *,
        embedder: Embedder,
        ids: list[str] | None = None,
        collection_name: str = "jeeves_vectors",
        persist_directory: str | None = None,
        client: Any = None,
    ) -> ChromaVectorStore:
        """One-shot: construct a ChromaVectorStore + add ``chunks``.

        FACTORY — builds and returns a NEW store. To add to an
        EXISTING store call ``store.add(chunks)`` (or
        ``index_document(path, store)``); calling this in a loop
        creates throwaway stores and drops writes.
        """
        store = cls(
            embedder=embedder,
            collection_name=collection_name,
            persist_directory=persist_directory,
            client=client,
        )
        await store.add(chunks, ids=ids)
        return store

    @classmethod
    async def from_texts(
        cls,
        texts: list[str],
        *,
        embedder: Embedder,
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        collection_name: str = "jeeves_vectors",
        persist_directory: str | None = None,
        client: Any = None,
    ) -> ChromaVectorStore:
        """One-shot: construct a ChromaVectorStore from raw text
        strings (each becomes a :class:`Chunk` with the matching
        metadata dict, or empty if ``metadatas`` is None).

        FACTORY — builds and returns a NEW store. ``embedder`` is
        required here because there is no instance yet to inherit it
        from. To add more to an EXISTING store (whose embedder you
        already gave at construction) call ``store.add(chunks)`` or
        ``index_document(path, store)`` — no embedder repeat there.
        """
        return await cls.from_chunks(
            _chunks_from_texts(texts, metadatas),
            embedder=embedder,
            ids=ids,
            collection_name=collection_name,
            persist_directory=persist_directory,
            client=client,
        )

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    async def add(
        self,
        chunks: list[Chunk],
        ids: list[str] | None = None,
    ) -> list[str]:
        if not chunks:
            return []
        assigned = resolve_ids(ids, len(chunks))
        # An id repeated in one call upserts onto itself: last wins
        # (Chroma itself rejects duplicate ids in one request).
        unique_ids, unique_chunks = dedupe_last(assigned, chunks)
        # Everything that can fail without the embedder runs first:
        # Chroma takes scalar metadata only (lists/dicts → JSON
        # strings, None dropped, empties → a sentinel) and a bad key
        # should not cost an embedding call.
        metadatas = [_stored_metadata(c.metadata) for c in unique_chunks]
        batch_size = await anyio.to_thread.run_sync(self._max_batch_size)
        await self._warn_if_shared_default()

        # One client-sized batch at a time: embed, then upsert. A
        # failure part-way leaves the earlier batches stored and pays
        # only for what was embedded.
        for start in range(0, len(unique_chunks), batch_size):
            stop = start + batch_size
            contents = [c.content for c in unique_chunks[start:stop]]
            vectors = await embed_all(self._embedder, contents)
            for vec in vectors:
                validate_query_vector(vec)
            await anyio.to_thread.run_sync(
                partial(
                    self._collection.upsert,
                    ids=unique_ids[start:stop],
                    embeddings=vectors,
                    documents=contents,
                    metadatas=metadatas[start:stop],
                )
            )
            self._wrote = True
        return assigned

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        await anyio.to_thread.run_sync(
            lambda: self._collection.delete(ids=list(ids))
        )

    async def get_by_ids(self, ids: list[str]) -> list[Chunk]:
        if not ids:
            return []
        result = await anyio.to_thread.run_sync(
            lambda: self._collection.get(ids=list(ids))
        )
        got_ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or [{}] * len(got_ids)
        # Preserve caller order; skip unknowns.
        index = {cid: i for i, cid in enumerate(got_ids)}
        out: list[Chunk] = []
        for cid in ids:
            if cid not in index:
                continue
            i = index[cid]
            out.append(
                Chunk(
                    content=docs[i] or "",
                    metadata=_restore_metadata(metas[i]),
                )
            )
        return out

    async def search(
        self,
        query: str,
        *,
        k: int = 4,
        filter: Mapping[str, Any] | None = None,
        diversity: float | None = None,
    ) -> list[SearchResult]:
        validate_k(k)
        q_vec = await self._embedder.embed(query)
        return await self.search_by_vector(
            q_vec, k=k, filter=filter, diversity=diversity
        )

    async def search_by_vector(
        self,
        vector: list[float],
        *,
        k: int = 4,
        filter: Mapping[str, Any] | None = None,
        diversity: float | None = None,
    ) -> list[SearchResult]:
        validate_k(k)
        validate_query_vector(vector)
        await self._warn_if_shared_default()
        # Chroma runs the part of the filter its engine can express (a
        # superset of the matches); the full filter is re-checked
        # below, under the same rules as every other store.
        loose: list[str] = []
        where = _engine_part(filter, loose) if filter else None
        # When diversity is requested, fetch a wider candidate pool
        # and rerank in-process via MMR.
        want = max(k * 4, 20) if diversity else k
        n_fetch = want
        if loose:
            # The engine filter is looser than the real one: over-fetch.
            n_fetch = max(want, k * 8, 50)

        candidates: list[SearchResult] = []
        cand_vecs: list[list[float]] = []
        for _attempt in range(_MAX_WIDENINGS + 1):
            result = await anyio.to_thread.run_sync(
                partial(
                    self._collection.query,
                    query_embeddings=[vector],
                    n_results=n_fetch,
                    where=where,
                    include=["documents", "metadatas", "distances", "embeddings"],
                )
            )
            ids_batch = result.get("ids") or [[]]
            row_ids = ids_batch[0] if ids_batch else []
            if not row_ids:
                return []
            docs = (result.get("documents") or [[]])[0] or [None] * len(row_ids)
            metas = (result.get("metadatas") or [[]])[0] or [None] * len(row_ids)
            dists = (result.get("distances") or [[]])[0] or [0.0] * len(row_ids)
            embs_batch = result.get("embeddings")
            embs = (
                embs_batch[0]
                if embs_batch is not None and len(embs_batch) > 0
                else None
            )

            candidates, cand_vecs = [], []
            for i, cid in enumerate(row_ids):
                chunk_meta = _restore_metadata(metas[i])
                if filter and not evaluate_filter(filter, chunk_meta):
                    continue
                emb = embs[i] if embs is not None else None
                candidates.append(
                    SearchResult(
                        chunk=Chunk(content=docs[i] or "", metadata=chunk_meta),
                        # The collection's space is cosine (see
                        # __init__), so Chroma's distance is
                        # ``1 - cos`` in [0, 2] and this is a true
                        # cosine similarity in [-1, 1] — no clamp.
                        score=1.0 - float(dists[i]),
                        id=cid,
                    )
                )
                cand_vecs.append(list(emb) if emb is not None else [])
            # Enough survivors, or the engine has nothing more to give.
            if len(candidates) >= want or len(row_ids) < n_fetch:
                break
            n_fetch *= 4

        return rerank_tail(vector, candidates, cand_vecs, k, diversity)

    async def count(self) -> int:
        n = await anyio.to_thread.run_sync(self._collection.count)
        return int(n)
