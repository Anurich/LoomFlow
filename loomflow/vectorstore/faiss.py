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

Vectors are L2-normalised before indexing for BOTH metrics, so the
cross-store cosine score contract holds: with ``"ip"`` the inner
product of unit vectors IS cosine; with ``"l2"`` the squared L2
distance ``d`` between unit vectors satisfies ``d = 2(1 - cos)``,
so ``1 - d/2`` recovers cosine.

Deletes are deferred: HNSW (the default index) can't remove rows,
so deleted rows are masked out of search results and the index is
compacted (rebuilt from the cached vectors, no re-embedding) only
once deleted rows reach half the indexed total.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import anyio

from ..core.protocols import Embedder
from ..loader.base import Chunk
from ._filter import evaluate_filter
from ._mmr import rerank_tail
from ._util import embed_all, resolve_ids
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
        # Deferred deletes: HNSW can't remove rows in place, so
        # deleted row indices are masked at query time and the index
        # is compacted lazily (see ``delete`` / ``_compact``).
        self._deleted_rows: set[int] = set()

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
        assigned = resolve_ids(ids, len(chunks))
        vectors = await embed_all(
            self._embedder, [c.content for c in chunks]
        )

        import numpy as np  # type: ignore[import-not-found, import-untyped]

        arr = np.asarray(vectors, dtype="float32")
        # Normalise for BOTH metrics — unit vectors are what makes
        # the score formulas cosine (ip: dot == cos; l2 on unit
        # vectors: d == 2(1 - cos)). See the module docstring.
        self._faiss.normalize_L2(arr)
        self._ensure_index(arr.shape[1])
        await anyio.to_thread.run_sync(self._index.add, arr)

        # Cache the (already-normalized) vectors for MMR + rebuilds.
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
        """Deferred delete: mark rows dead (masked from every search /
        ``get_by_ids`` / ``count`` immediately) and compact the index
        only when dead rows reach half the indexed total.

        Rationale: the default HNSW index can't remove vectors in
        place (``remove_ids`` is unsupported for it, with or without
        an ``IndexIDMap2`` wrapper), and the previous implementation
        rebuilt the whole index on EVERY delete. Compaction rebuilds
        from the cached vectors — no re-embedding.
        """
        if not ids or self._index is None:
            return
        for cid in ids:
            row = self._row_for_id.pop(cid, None)
            if row is not None:
                self._deleted_rows.add(row)

        if not self._row_for_id:
            # Everything is gone — cheap full reset.
            self._ids = []
            self._chunks = []
            self._vectors = []
            self._deleted_rows = set()
            self._index.reset()
            return

        if len(self._deleted_rows) * 2 >= len(self._ids):
            await self._compact()

    async def _compact(self) -> None:
        """Rebuild the FAISS index from the cached (already
        normalised) survivor vectors — no re-embedding, important
        when the embedder is paid-API-backed."""
        keep_rows = [
            i
            for i in range(len(self._ids))
            if i not in self._deleted_rows
        ]
        ids = [self._ids[i] for i in keep_rows]
        chunks = [self._chunks[i] for i in keep_rows]
        vectors = [self._vectors[i] for i in keep_rows]

        import numpy as np

        self._index.reset()
        self._ids = ids
        self._chunks = chunks
        self._vectors = vectors
        self._row_for_id = {cid: i for i, cid in enumerate(ids)}
        self._deleted_rows = set()

        arr = np.asarray(vectors, dtype="float32")
        self._ensure_index(arr.shape[1])
        await anyio.to_thread.run_sync(self._index.add, arr)

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
        if self._index is None or not self._row_for_id:
            return []

        import numpy as np

        # Over-fetch so post-filter + MMR have headroom; also fetch
        # past any deferred-deleted rows still in the index.
        multiplier = 8 if filter else (4 if diversity else 1)
        n_fetch = max(k * multiplier, 20 if (filter or diversity) else k)
        n_fetch = min(
            n_fetch + len(self._deleted_rows), len(self._ids)
        )

        q = np.asarray([vector], dtype="float32")
        # Queries are normalised for both metrics, matching the
        # stored vectors (see ``add``) so scores are true cosine.
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
            if idx in self._deleted_rows:
                continue
            chunk = self._chunks[idx]
            if not evaluate_filter(filter, chunk.metadata):
                continue
            # Cross-store SCORE CONTRACT: cosine similarity,
            # higher-is-better, in [-1, 1]. Stored + query vectors
            # are L2-normalised for BOTH metrics (see ``add``), so
            # with ``ip`` the inner product IS cosine, and with
            # ``l2`` the squared distance d between unit vectors
            # satisfies d = 2(1 - c), i.e. c = 1 - d/2. Either
            # metric yields scores directly comparable with
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

        return rerank_tail(
            list(q[0]), candidates, cand_vecs, k, diversity
        )

    async def count(self) -> int:
        return len(self._row_for_id)
