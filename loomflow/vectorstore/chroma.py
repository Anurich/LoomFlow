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

from collections.abc import Mapping
from typing import Any

import anyio

from ..core.ids import new_id
from ..core.protocols import Embedder
from ..loader.base import Chunk
from ._filter import COMPARISON_OPERATORS, LOGICAL_OPERATORS, FilterError
from ._mmr import mmr_select
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


class ChromaVectorStore:
    """Vector store backed by ``chromadb``."""

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

        self._collection = self._client.get_or_create_collection(
            name=collection_name
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
        """One-shot: construct a ChromaVectorStore + add ``chunks``."""
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
        metadata dict, or empty if ``metadatas`` is None)."""
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
        if ids is not None and len(ids) != len(chunks):
            raise ValueError(
                f"ids length ({len(ids)}) must match chunks "
                f"length ({len(chunks)})"
            )
        try:
            vectors = await self._embedder.embed_batch(
                [c.content for c in chunks]
            )
        except (AttributeError, NotImplementedError):
            vectors = [
                await self._embedder.embed(c.content) for c in chunks
            ]

        assigned = (
            list(ids)
            if ids is not None
            else [new_id("vec") for _ in chunks]
        )
        contents = [c.content for c in chunks]
        # Chroma rejects empty-dict metadatas; supply a sentinel.
        metadatas = [
            {**c.metadata} if c.metadata else {"_empty": True}
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
            score = max(0.0, 1.0 - float(dist))
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

        if diversity is None or diversity <= 0:
            return candidates[:k]

        chosen = mmr_select(
            vector, cand_vecs, k, diversity=diversity
        )
        return [candidates[i] for i in chosen]

    async def count(self) -> int:
        n = await anyio.to_thread.run_sync(self._collection.count)
        return int(n)
