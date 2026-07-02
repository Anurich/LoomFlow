"""Chroma-backed vector store.

Wraps ``chromadb`` for persistent on-disk or hosted Chroma. Lazy
import — install via ``pip install 'loomflow[vectorstore-chroma]'``.

Embeddings come from our framework's :class:`Embedder` protocol so
swapping embedders works the same across every vector store. We
pass ``None`` as Chroma's ``embedding_function`` and supply
embeddings ourselves at ``add`` time.

Filter operators are translated from our Mongo-style language to
Chroma's native ``where`` syntax (which already speaks Mongo-ish
``$eq`` / ``$in`` / ``$gt`` etc., so the translation is mostly
direct).
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from typing import Any

import anyio

from ..core.protocols import Embedder
from ..loader.base import Chunk
from ._filter import COMPARISON_OPERATORS, LOGICAL_OPERATORS, FilterError
from ._mmr import rerank_tail
from ._util import embed_all, resolve_ids
from .base import SearchResult, _chunks_from_texts


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
                # Chroma 0.5+ doesn't have a plain $not; emulate via
                # negated comparisons isn't always feasible. Raise
                # and ask the caller to invert manually.
                raise FilterError(
                    "$not isn't supported by Chroma. "
                    "Invert the underlying comparison instead."
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
    bool / None). The framework's own chunkers emit non-scalars —
    ``MarkdownChunker`` writes a ``headers`` LIST — so a bare
    ``store.add(chunker.split(...))`` would crash on its own loader's
    output. Scalars pass through untouched (equality / range filters
    on them are unchanged); anything else is JSON-serialised to a
    stable, round-trippable string (``json.loads`` it back on read —
    reads are NOT auto-parsed, since a legitimately string value could
    look like JSON). ``default=str`` guards exotic objects so this can
    never itself raise. Postgres (JSONB) / InMemory / FAISS keep dicts
    natively and never hit this path.
    """
    out: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            out[key] = value
        else:
            out[key] = json.dumps(value, default=str)
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
    """

    name = "chroma"

    def __init__(
        self,
        embedder: Embedder,
        *,
        collection_name: str = "jeeves_vectors",
        persist_directory: str | None = None,
        client: Any = None,
    ) -> None:
        if embedder is None:
            raise ValueError("embedder is required")
        self._embedder = embedder
        self._collection_name = collection_name

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

        # ``hnsw:space: cosine`` makes Chroma's distance a true cosine
        # distance so ``1 - distance`` is the contract's cosine score.
        # (Chroma defaults to L2 when the space is unspecified.)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        # ``get_or_create`` can't change the space of a pre-existing
        # collection; warn when we can see it isn't cosine. Missing
        # metadata means the collection pre-dates this setting and
        # runs on Chroma's L2 default.
        coll_meta = getattr(self._collection, "metadata", None) or {}
        space = coll_meta.get("hnsw:space", "l2")
        if space != "cosine":
            warnings.warn(
                f"Chroma collection {collection_name!r} already exists "
                f"with hnsw:space={space!r} (not 'cosine'); search "
                "scores from this store will not follow the cosine "
                "score contract. Re-index into a new collection to fix.",
                UserWarning,
                stacklevel=2,
            )

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
        vectors = await embed_all(
            self._embedder, [c.content for c in chunks]
        )
        contents = [c.content for c in chunks]
        # Chroma rejects empty-dict metadatas AND non-scalar values;
        # flatten (lists/dicts → JSON strings) + sentinel the empties.
        metadatas = [
            _flatten_metadata(c.metadata) if c.metadata else {"_empty": True}
            for c in chunks
        ]

        await anyio.to_thread.run_sync(
            lambda: self._collection.add(
                ids=assigned,
                embeddings=vectors,
                documents=contents,
                metadatas=metadatas,
            )
        )
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
            meta = dict(metas[i] or {})
            meta.pop("_empty", None)
            out.append(Chunk(content=docs[i] or "", metadata=meta))
        return out

    async def search(
        self,
        query: str,
        *,
        k: int = 4,
        filter: Mapping[str, Any] | None = None,
        diversity: float | None = None,
    ) -> list[SearchResult]:
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
        where = _translate_filter(filter)
        # When diversity is requested, fetch a wider candidate pool
        # and rerank in-process via MMR.
        n_fetch = max(k * 4, 20) if diversity else k

        result = await anyio.to_thread.run_sync(
            lambda: self._collection.query(
                query_embeddings=[vector],
                n_results=n_fetch,
                where=where,
                include=["documents", "metadatas", "distances", "embeddings"],
            )
        )

        ids_batch = result.get("ids", [[]])
        docs_batch = result.get("documents", [[]])
        metas_batch = result.get("metadatas", [[]])
        dists_batch = result.get("distances", [[]])
        embs_batch = result.get("embeddings", [[]])

        if not ids_batch or not ids_batch[0]:
            return []

        candidates: list[SearchResult] = []
        cand_vecs: list[list[float]] = []
        for cid, doc, meta, dist, emb in zip(
            ids_batch[0],
            docs_batch[0],
            metas_batch[0] or [{}] * len(ids_batch[0]),
            dists_batch[0] or [0.0] * len(ids_batch[0]),
            (embs_batch[0] if embs_batch else [None] * len(ids_batch[0])),
            strict=False,
        ):
            # The collection's space is cosine (see __init__), so
            # Chroma's distance is ``1 - cos`` in [0, 2] and this is
            # a true cosine similarity in [-1, 1] — no clamp.
            score = 1.0 - float(dist)
            chunk_meta = dict(meta or {})
            chunk_meta.pop("_empty", None)
            candidates.append(
                SearchResult(
                    chunk=Chunk(
                        content=doc or "",
                        metadata=chunk_meta,
                    ),
                    score=score,
                    id=cid,
                )
            )
            cand_vecs.append(list(emb) if emb is not None else [])

        return rerank_tail(vector, candidates, cand_vecs, k, diversity)

    async def count(self) -> int:
        n = await anyio.to_thread.run_sync(self._collection.count)
        return int(n)
