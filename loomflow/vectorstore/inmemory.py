"""In-memory vector store — cosine over a Python list.

Zero dependencies. Default for dev, tests, and small corpora (up
to ~10K chunks before search latency starts to bite). For larger
corpora swap to :class:`FAISSVectorStore` (in-process ANN) or
:class:`ChromaVectorStore` / :class:`PostgresVectorStore`
(persistent).

Beyond the protocol contract this backend additionally supports:

* **Diversity (MMR)** — pass ``diversity=0.3`` to :meth:`search`
  for varied top-k.
* **Hybrid search** — :meth:`search_hybrid` combines BM25 lexical
  scores with vector similarity via Reciprocal Rank Fusion.
* **Persistence** — :meth:`save` / :meth:`load` round-trip the
  store to JSON on disk.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..core.protocols import Embedder
from ..loader.base import Chunk
from ._bm25 import BM25Index, reciprocal_rank_fusion
from ._filter import evaluate_filter
from ._mmr import mmr_select
from ._util import embed_all, resolve_ids
from .base import SearchResult, _chunks_from_texts


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


_PERSIST_VERSION = 1


class InMemoryVectorStore:
    """In-process vector store backed by a Python list."""

    name = "in-memory"

    def __init__(self, embedder: Embedder) -> None:
        if embedder is None:
            raise ValueError("embedder is required")
        self._embedder = embedder
        # Parallel lists. Allocated per-instance so concurrent stores
        # in the same process don't share state.
        self._ids: list[str] = []
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []
        # Norms are cached at add/load time so a query doesn't
        # recompute every stored vector's norm on every search.
        self._norms: list[float] = []
        self._bm25: BM25Index | None = None  # built lazily

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
    ) -> InMemoryVectorStore:
        """One-shot: construct an InMemoryVectorStore + add ``chunks``.

        FACTORY — builds and returns a NEW store. To add to an
        EXISTING store call ``store.add(chunks)`` (or
        ``index_document(path, store)``); calling this in a loop
        creates throwaway stores and drops writes.
        """
        store = cls(embedder=embedder)
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
    ) -> InMemoryVectorStore:
        """One-shot: construct an InMemoryVectorStore from raw text
        strings (each becomes a :class:`Chunk` with the matching
        metadata dict, or empty if ``metadatas`` is None)."""
        return await cls.from_chunks(
            _chunks_from_texts(texts, metadatas),
            embedder=embedder,
            ids=ids,
        )

    # ---------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------

    async def add(
        self,
        chunks: list[Chunk],
        ids: list[str] | None = None,
    ) -> list[str]:
        if not chunks:
            return []
        assigned_ids = resolve_ids(ids, len(chunks))
        vectors = await embed_all(
            self._embedder, [c.content for c in chunks]
        )

        for cid, chunk, vec in zip(
            assigned_ids, chunks, vectors, strict=True
        ):
            self._ids.append(cid)
            self._chunks.append(chunk)
            self._vectors.append(vec)
            self._norms.append(_norm(vec))

        # Invalidate BM25 index — built lazily on next hybrid query.
        self._bm25 = None
        return assigned_ids

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        kill = set(ids)
        keep_indices = [
            i for i, cid in enumerate(self._ids) if cid not in kill
        ]
        self._ids = [self._ids[i] for i in keep_indices]
        self._chunks = [self._chunks[i] for i in keep_indices]
        self._vectors = [self._vectors[i] for i in keep_indices]
        self._norms = [self._norms[i] for i in keep_indices]
        self._bm25 = None

    async def get_by_ids(self, ids: list[str]) -> list[Chunk]:
        if not ids:
            return []
        index = {cid: i for i, cid in enumerate(self._ids)}
        return [
            self._chunks[index[cid]] for cid in ids if cid in index
        ]

    async def count(self) -> int:
        return len(self._ids)

    # ---------------------------------------------------------------
    # Search
    # ---------------------------------------------------------------

    def _cosine_to(
        self, query: list[float], q_norm: float, i: int
    ) -> float:
        """Cosine similarity between ``query`` (with precomputed norm
        ``q_norm``) and stored vector ``i`` (norm cached at add
        time)."""
        denom = q_norm * self._norms[i]
        if denom == 0:
            return 0.0
        dot = sum(
            x * y for x, y in zip(query, self._vectors[i], strict=True)
        )
        return dot / denom

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
        if not self._vectors:
            return []

        # Step 1: cosine-rank everything that survives the filter
        # (stored norms are precomputed at add/load time).
        q_norm = _norm(vector)
        scored: list[tuple[int, float]] = []
        for i in range(len(self._vectors)):
            if not evaluate_filter(filter, self._chunks[i].metadata):
                continue
            scored.append((i, self._cosine_to(vector, q_norm, i)))
        scored.sort(key=lambda x: x[1], reverse=True)

        if diversity is None or diversity <= 0:
            return [
                SearchResult(
                    chunk=self._chunks[i],
                    score=score,
                    id=self._ids[i],
                )
                for i, score in scored[:k]
            ]

        # Step 2: MMR rerank over a candidate pool. Pool is k*4 (or
        # the full filtered set if smaller) — wider pool = better
        # diversity, more compute. k*4 is the LangChain default.
        pool_size = min(len(scored), max(k * 4, 20))
        pool = scored[:pool_size]
        pool_indices = [i for i, _ in pool]
        pool_vecs = [self._vectors[i] for i in pool_indices]
        chosen_in_pool = mmr_select(
            vector, pool_vecs, k, diversity=diversity
        )
        return [
            SearchResult(
                chunk=self._chunks[pool_indices[p]],
                score=pool[p][1],
                id=self._ids[pool_indices[p]],
            )
            for p in chosen_in_pool
        ]

    async def search_hybrid(
        self,
        query: str,
        *,
        k: int = 4,
        filter: Mapping[str, Any] | None = None,
        alpha: float = 0.5,
    ) -> list[SearchResult]:
        """Hybrid lexical (BM25) + vector search via RRF.

        ``alpha`` is in [0, 1]: 0 = pure BM25, 1 = pure vector,
        0.5 = even weighting (RRF default). Both rankings are
        computed independently and fused by Reciprocal Rank Fusion,
        then the top-``k`` survivors are returned.

        Embeddings catch semantic similarity ("automobile" ↔ "car"),
        BM25 catches exact-term hits (model names, error codes,
        person names) — together they outperform either alone on
        most retrieval benchmarks.
        """
        if not self._vectors:
            return []
        alpha = max(0.0, min(1.0, alpha))

        # Build BM25 index lazily (covers the whole corpus including
        # filtered-out items; we apply the filter post-rank).
        if self._bm25 is None:
            self._bm25 = BM25Index()
            self._bm25.add([c.content for c in self._chunks])

        # Rank by vector (cached stored-vector norms).
        q_vec = await self._embedder.embed(query)
        q_norm = _norm(q_vec)
        vector_scored: list[tuple[int, float]] = []
        for i in range(len(self._vectors)):
            if not evaluate_filter(filter, self._chunks[i].metadata):
                continue
            vector_scored.append((i, self._cosine_to(q_vec, q_norm, i)))
        vector_scored.sort(key=lambda x: x[1], reverse=True)

        # Rank by BM25 (apply the same filter).
        bm25_raw = self._bm25.search(query, k=len(self._ids))
        bm25_scored = [
            (i, s)
            for i, s in bm25_raw
            if evaluate_filter(filter, self._chunks[i].metadata)
        ]

        # Fuse. RRF ignores raw score magnitudes so we use ``alpha``
        # by replicating each ranking proportionally — alpha=0.7
        # means "vector ranking counts 70%, BM25 30%".
        rankings: list[list[tuple[int, float]]] = []
        if alpha > 0:
            rankings.append(vector_scored)
        if alpha < 1:
            rankings.append(bm25_scored)
        # Weight by replicating: scale tells RRF how strongly to
        # weight each list. Three buckets cover the common cases.
        if 0 < alpha < 1 and abs(alpha - 0.5) > 0.05:
            extra = vector_scored if alpha > 0.5 else bm25_scored
            weight_replications = max(
                1, int(round(abs(alpha - 0.5) * 8))
            )
            rankings.extend([extra] * weight_replications)

        fused = reciprocal_rank_fusion(rankings)
        top = fused[:k]
        return [
            SearchResult(
                chunk=self._chunks[idx],
                score=score,
                id=self._ids[idx],
            )
            for idx, score in top
        ]

    # ---------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------

    async def save(self, path: str | Path) -> None:
        """Write the full store (chunks + vectors + ids) to a JSON
        file. The embedder is NOT serialized — supply the same
        embedder when calling :meth:`load`."""
        target = Path(path)
        data = {
            "version": _PERSIST_VERSION,
            "dimensions": (
                len(self._vectors[0]) if self._vectors else None
            ),
            "rows": [
                {
                    "id": self._ids[i],
                    "content": self._chunks[i].content,
                    "metadata": self._chunks[i].metadata,
                    "vector": self._vectors[i],
                }
                for i in range(len(self._ids))
            ],
        }
        target.write_text(  # noqa: ASYNC240 — sync I/O is fine here, persistence is rare and small
            # ``default=str`` is the safety net so an exotic metadata
            # value can never crash save(); lists/dicts (e.g. a
            # MarkdownChunker ``headers`` list) serialise natively and
            # round-trip on load().
            json.dumps(data, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    @classmethod
    async def load(
        cls, path: str | Path, *, embedder: Embedder
    ) -> InMemoryVectorStore:
        """Restore a store previously :meth:`save`-d. Pass the same
        embedder kind/dimensions or queries will produce nonsense
        scores."""
        # noqa: ASYNC240 — sync I/O is fine here, called once at startup
        data = json.loads(
            Path(path).read_text(encoding="utf-8")  # noqa: ASYNC240
        )
        if data.get("version") != _PERSIST_VERSION:
            raise ValueError(
                f"Unsupported persist version: "
                f"{data.get('version')!r} (expected {_PERSIST_VERSION})"
            )
        store = cls(embedder=embedder)
        for row in data["rows"]:
            store._ids.append(row["id"])
            store._chunks.append(
                Chunk(
                    content=row["content"],
                    metadata=row["metadata"],
                )
            )
            store._vectors.append(row["vector"])
            store._norms.append(_norm(row["vector"]))
        return store
