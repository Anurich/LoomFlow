"""FAISS-backed vector store.

In-process ANN over a FAISS index. Fast on large corpora (millions
of vectors). Lazy import — install via
``pip install 'loomflow[vectorstore-faiss]'``.

The index is HNSW by default; pass ``index_factory_string`` to
override (``"Flat"`` for exact, ``"IVF1024,Flat"`` for IVF, see
https://github.com/facebookresearch/faiss/wiki/The-index-factory).

FAISS doesn't natively support metadata filtering — we apply the
``filter`` argument by post-filtering the candidate set after the
ANN search returns. For tight filters with large ``k``, we
internally over-fetch so enough candidates survive the filter.

We also keep a parallel in-process vector list to support MMR
diversity reranking (some FAISS index types can't reconstruct
vectors from the index, so we cache them ourselves).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import anyio

from ..core.ids import new_id
from ..core.protocols import Embedder
from ..loader.base import Chunk
from ._filter import evaluate_filter
from ._mmr import mmr_select
from .base import SearchResult, _chunks_from_texts


class FAISSVectorStore:
    """Vector store backed by ``faiss-cpu``."""

    name = "faiss"

    def __init__(
        self,
        embedder: Embedder,
        *,
        dimension: int | None = None,
        index_factory_string: str = "HNSW32",
        metric: str = "ip",  # "ip" (inner product) or "l2"
    ) -> None:
        if embedder is None:
            raise ValueError("embedder is required")
        self._embedder = embedder
        self._index_factory_string = index_factory_string
        self._metric = metric

        try:
            import faiss  # type: ignore[import-not-found, import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "faiss is not installed. "
                "Install with: pip install 'loomflow[vectorstore-faiss]'."
            ) from exc

        self._faiss = faiss
        self._dimension = dimension
        self._index: Any = None  # built lazily on first add (need dim)
        self._ids: list[str] = []
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []  # parallel cache for MMR
        self._row_for_id: dict[str, int] = {}

    @property
    def embedder(self) -> Embedder:
        return self._embedder

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
        dimension: int | None = None,
        index_factory_string: str = "HNSW32",
        metric: str = "ip",
    ) -> FAISSVectorStore:
        """One-shot: construct a FAISSVectorStore + add ``chunks``.

        FACTORY — builds and returns a NEW store. To add to an
        EXISTING store call ``store.add(chunks)`` (or
        ``index_document(path, store)``); calling this in a loop
        creates throwaway stores and drops writes.
        """
        store = cls(
            embedder=embedder,
            dimension=dimension,
            index_factory_string=index_factory_string,
            metric=metric,
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
        dimension: int | None = None,
        index_factory_string: str = "HNSW32",
        metric: str = "ip",
    ) -> FAISSVectorStore:
        """One-shot: construct a FAISSVectorStore from raw text
        strings (each becomes a :class:`Chunk` with the matching
        metadata dict, or empty if ``metadatas`` is None)."""
        return await cls.from_chunks(
            _chunks_from_texts(texts, metadatas),
            embedder=embedder,
            ids=ids,
            dimension=dimension,
            index_factory_string=index_factory_string,
            metric=metric,
        )

    def _ensure_index(self, dim: int) -> None:
        if self._index is not None:
            return
        if self._dimension is None:
            self._dimension = dim
        elif self._dimension != dim:
            raise ValueError(
                f"embedder produced {dim}-dim vectors; "
                f"store was configured for {self._dimension}"
            )
        metric = (
            self._faiss.METRIC_INNER_PRODUCT
            if self._metric == "ip"
            else self._faiss.METRIC_L2
        )
        self._index = self._faiss.index_factory(
            dim, self._index_factory_string, metric
        )

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

        import numpy as np  # type: ignore[import-not-found, import-untyped]

        arr = np.asarray(vectors, dtype="float32")
        if self._metric == "ip":
            self._faiss.normalize_L2(arr)
        self._ensure_index(arr.shape[1])
        await anyio.to_thread.run_sync(self._index.add, arr)

        # Cache the (already-normalized for IP) vectors for MMR.
        cached_vectors = arr.tolist()
        for i, (cid, chunk) in enumerate(
            zip(assigned, chunks, strict=True)
        ):
            self._row_for_id[cid] = len(self._ids)
            self._ids.append(cid)
            self._chunks.append(chunk)
            self._vectors.append(cached_vectors[i])
        return assigned

    async def delete(self, ids: list[str]) -> None:
        if not ids or self._index is None:
            return
        kill = set(ids)
        keep = [
            (cid, chunk, vec)
            for cid, chunk, vec in zip(
                self._ids, self._chunks, self._vectors, strict=True
            )
            if cid not in kill
        ]

        if not keep:
            self._ids = []
            self._chunks = []
            self._vectors = []
            self._row_for_id = {}
            self._index.reset()
            return

        # Reset and rebuild from the cached vectors (no re-embedding —
        # important when the embedder is paid-API-backed).
        self._index.reset()
        self._ids = []
        self._chunks = []
        self._vectors = []
        self._row_for_id = {}

        # Re-add via the FAISS index without going through the
        # embedder (we already have vectors).
        import numpy as np

        survivor_vecs = [vec for _, _, vec in keep]
        arr = np.asarray(survivor_vecs, dtype="float32")
        self._ensure_index(arr.shape[1])
        await anyio.to_thread.run_sync(self._index.add, arr)
        for cid, chunk, vec in keep:
            self._row_for_id[cid] = len(self._ids)
            self._ids.append(cid)
            self._chunks.append(chunk)
            self._vectors.append(vec)

    async def get_by_ids(self, ids: list[str]) -> list[Chunk]:
        if not ids:
            return []
        return [
            self._chunks[self._row_for_id[cid]]
            for cid in ids
            if cid in self._row_for_id
        ]

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
        if self._index is None or not self._ids:
            return []

        import numpy as np

        # Over-fetch so post-filter + MMR have headroom.
        multiplier = 8 if filter else (4 if diversity else 1)
        n_fetch = max(k * multiplier, 20 if (filter or diversity) else k)
        n_fetch = min(n_fetch, len(self._ids))

        q = np.asarray([vector], dtype="float32")
        if self._metric == "ip":
            self._faiss.normalize_L2(q)

        distances, indices = await anyio.to_thread.run_sync(
            self._index.search, q, n_fetch
        )

        # Build the filtered candidate list.
        candidates: list[SearchResult] = []
        cand_vecs: list[list[float]] = []
        for dist, idx in zip(distances[0], indices[0], strict=True):
            if idx < 0 or idx >= len(self._ids):
                continue
            chunk = self._chunks[idx]
            if not evaluate_filter(filter, chunk.metadata):
                continue
            # Normalise to the cross-store SCORE CONTRACT: cosine
            # similarity, higher-is-better. Vectors are L2-normalised
            # for the ``ip`` metric (see _embed_and_store / search_by_
            # vector), so inner product IS cosine in [-1, 1] — return
            # it directly. For ``l2`` on unit vectors, the squared L2
            # distance d relates to cosine c by d = 2(1 - c), so
            # c = 1 - d/2 recovers the same [-1, 1] cosine score
            # instead of the old, differently-scaled ``1 - d``. Either
            # metric now yields scores directly comparable with
            # Chroma / Postgres / InMemory.
            score = (
                float(dist)
                if self._metric == "ip"
                else 1.0 - float(dist) / 2.0
            )
            candidates.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    id=self._ids[idx],
                )
            )
            cand_vecs.append(self._vectors[idx])

        if diversity is None or diversity <= 0:
            return candidates[:k]

        chosen = mmr_select(
            list(q[0]), cand_vecs, k, diversity=diversity
        )
        return [candidates[i] for i in chosen]

    async def count(self) -> int:
        return len(self._ids)
