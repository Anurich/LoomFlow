"""Vectorstore regression tests for the reviewed fixes (WVB).

1.  Chroma score contract — collections are created with
    ``hnsw:space=cosine`` so ``1 - distance`` IS cosine (clamp
    removed); pre-existing non-cosine collections warn.
2.  FAISS ``l2`` metric — vectors are L2-normalised for BOTH metrics
    so ``1 - d/2`` is a true cosine score on the l2 path too.
7.  Shared ``_util`` helpers — ``embed_all`` fallback, ``resolve_ids``
    validation, single ``cosine``, shared MMR ``rerank_tail``.
8.  InMemory norm caching — norms are cached at add/load and stay
    consistent across delete.
9.  FAISS deferred deletes — deletes mask rows instead of rebuilding
    the index each time; compaction kicks in at the 50% threshold.
10. Postgres — the ``embedding`` column is only fetched when MMR
    (diversity) will actually run.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from loomflow.loader.base import Chunk
from loomflow.memory.embedder import HashEmbedder
from loomflow.vectorstore._mmr import mmr_select, rerank_tail
from loomflow.vectorstore._util import cosine, embed_all, resolve_ids
from loomflow.vectorstore.inmemory import InMemoryVectorStore

pytestmark = pytest.mark.anyio


class _DirectionalEmbedder:
    """Deterministic embedder with NON-unit vectors so tests can tell
    a true cosine score from a distance-derived impostor."""

    name = "directional"
    dimensions = 4

    _TABLE = {
        "east": [1.0, 0.0, 0.0, 0.0],
        "far east": [10.0, 0.0, 0.0, 0.0],  # same direction, big norm
        "north": [0.0, 5.0, 0.0, 0.0],  # orthogonal
        "west": [-3.0, 0.0, 0.0, 0.0],  # opposite direction
    }

    async def embed(self, text: str) -> list[float]:
        return list(self._TABLE[text])

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [list(self._TABLE[t]) for t in texts]


# ---------------------------------------------------------------------------
# Fix 1 — Chroma cosine space
# ---------------------------------------------------------------------------


def _fresh_chroma_client() -> Any:
    chromadb = pytest.importorskip("chromadb")
    return chromadb.EphemeralClient()


async def test_chroma_collection_created_with_cosine_space() -> None:
    from loomflow.vectorstore.chroma import ChromaVectorStore

    store = ChromaVectorStore(
        embedder=HashEmbedder(dimensions=8),
        collection_name=f"wvb_{uuid.uuid4().hex[:12]}",
        client=_fresh_chroma_client(),
    )
    meta = store._collection.metadata or {}
    assert meta.get("hnsw:space") == "cosine"


async def test_chroma_scores_are_true_cosine_without_clamp() -> None:
    from loomflow.vectorstore.chroma import ChromaVectorStore

    store = ChromaVectorStore(
        embedder=_DirectionalEmbedder(),
        collection_name=f"wvb_{uuid.uuid4().hex[:12]}",
        client=_fresh_chroma_client(),
    )
    await store.add(
        [
            Chunk(content="far east"),  # cosine 1.0 despite norm 10
            Chunk(content="north"),  # cosine 0.0
            Chunk(content="west"),  # cosine -1.0 (clamp would hide it)
        ]
    )
    results = await store.search("east", k=3)
    scores = {r.chunk.content: r.score for r in results}
    assert scores["far east"] == pytest.approx(1.0, abs=1e-4)
    assert scores["north"] == pytest.approx(0.0, abs=1e-4)
    # The old ``max(0.0, 1 - dist)`` clamp made this 0.0.
    assert scores["west"] == pytest.approx(-1.0, abs=1e-4)


class _L2Collection:
    """Stands in for a pre-existing collection whose HNSW space was
    fixed at creation time (Chroma can't change it afterwards)."""

    metadata = {"hnsw:space": "l2"}


class _StubClientWithL2Collection:
    def get_or_create_collection(
        self, name: str, metadata: dict[str, Any] | None = None
    ) -> _L2Collection:
        return _L2Collection()  # metadata arg can't change the space


async def test_chroma_existing_non_cosine_collection_warns() -> None:
    pytest.importorskip("chromadb")  # constructor imports it
    from loomflow.vectorstore.chroma import ChromaVectorStore

    with pytest.warns(UserWarning, match="hnsw:space"):
        ChromaVectorStore(
            embedder=HashEmbedder(dimensions=8),
            collection_name=f"wvb_{uuid.uuid4().hex[:12]}",
            client=_StubClientWithL2Collection(),
        )


# ---------------------------------------------------------------------------
# Fix 2 — FAISS l2 normalisation
# ---------------------------------------------------------------------------


async def test_faiss_l2_score_is_cosine_on_non_unit_vectors() -> None:
    pytest.importorskip("faiss")
    from loomflow.vectorstore.faiss import FAISSVectorStore

    store = FAISSVectorStore(
        embedder=_DirectionalEmbedder(),
        index_factory_string="Flat",
        metric="l2",
    )
    await store.add(
        [
            Chunk(content="far east"),
            Chunk(content="north"),
            Chunk(content="west"),
        ]
    )
    results = await store.search("east", k=3)
    scores = {r.chunk.content: r.score for r in results}
    # Without normalisation, |[1,0..] - [10,0..]|^2 = 81 would have
    # produced 1 - 81/2 = -39.5; with unit vectors it's exactly 1.0.
    assert scores["far east"] == pytest.approx(1.0, abs=1e-4)
    assert scores["north"] == pytest.approx(0.0, abs=1e-4)
    assert scores["west"] == pytest.approx(-1.0, abs=1e-4)
    for s in scores.values():
        assert -1.0 - 1e-6 <= s <= 1.0 + 1e-6


async def test_faiss_l2_and_ip_scores_agree() -> None:
    pytest.importorskip("faiss")
    from loomflow.vectorstore.faiss import FAISSVectorStore

    corpus = [Chunk(content="far east"), Chunk(content="north")]
    l2 = FAISSVectorStore(
        embedder=_DirectionalEmbedder(),
        index_factory_string="Flat",
        metric="l2",
    )
    ip = FAISSVectorStore(
        embedder=_DirectionalEmbedder(),
        index_factory_string="Flat",
        metric="ip",
    )
    await l2.add(corpus)
    await ip.add(corpus)
    l2_scores = {
        r.chunk.content: r.score for r in await l2.search("east", k=2)
    }
    ip_scores = {
        r.chunk.content: r.score for r in await ip.search("east", k=2)
    }
    for content, score in ip_scores.items():
        assert l2_scores[content] == pytest.approx(score, abs=1e-4)


# ---------------------------------------------------------------------------
# Fix 9 — FAISS deferred delete + threshold compaction
# ---------------------------------------------------------------------------


async def test_faiss_delete_defers_index_rebuild() -> None:
    pytest.importorskip("faiss")
    from loomflow.vectorstore.faiss import FAISSVectorStore

    store = FAISSVectorStore(
        embedder=HashEmbedder(dimensions=32),
        index_factory_string="Flat",
    )
    ids = await store.add(
        [Chunk(content=f"chunk number {i}") for i in range(4)]
    )

    # First delete (1 of 4 < 50%): masked, NOT compacted.
    await store.delete([ids[0]])
    assert await store.count() == 3
    assert store._index.ntotal == 4  # still holds the dead row
    assert await store.get_by_ids([ids[0]]) == []
    results = await store.search("chunk number 0", k=4)
    assert ids[0] not in {r.id for r in results}
    assert len(results) == 3

    # Second delete (2 of 4 >= 50%): compaction rebuilds the index.
    await store.delete([ids[1]])
    assert await store.count() == 2
    assert store._index.ntotal == 2
    assert store._deleted_rows == set()
    results = await store.search("chunk number 2", k=4)
    assert {r.id for r in results} == {ids[2], ids[3]}


async def test_faiss_delete_all_resets() -> None:
    pytest.importorskip("faiss")
    from loomflow.vectorstore.faiss import FAISSVectorStore

    store = FAISSVectorStore(
        embedder=HashEmbedder(dimensions=16),
        index_factory_string="Flat",
    )
    ids = await store.add([Chunk(content="a"), Chunk(content="b")])
    await store.delete(ids)
    assert await store.count() == 0
    assert await store.search("a", k=2) == []
    # Store is still usable after the reset.
    await store.add([Chunk(content="c")])
    assert await store.count() == 1


# ---------------------------------------------------------------------------
# Fix 7 — shared helpers
# ---------------------------------------------------------------------------


class _NoBatchEmbedder:
    """Embedder without ``embed_batch`` — exercises the fallback."""

    name = "no-batch"
    dimensions = 2

    async def embed(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


class _RefusingBatchEmbedder(_NoBatchEmbedder):
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


async def test_embed_all_uses_batch_when_available() -> None:
    vectors = await embed_all(HashEmbedder(dimensions=8), ["a", "b"])
    assert len(vectors) == 2
    assert all(len(v) == 8 for v in vectors)


async def test_embed_all_falls_back_without_batch() -> None:
    for embedder in (_NoBatchEmbedder(), _RefusingBatchEmbedder()):
        vectors = await embed_all(embedder, ["x", "yy"])
        assert vectors == [[1.0, 1.0], [2.0, 1.0]]


def test_resolve_ids_validates_length() -> None:
    with pytest.raises(ValueError, match="ids length"):
        resolve_ids(["only-one"], 2)
    assert resolve_ids(["a", "b"], 2) == ["a", "b"]
    generated = resolve_ids(None, 3)
    assert len(generated) == 3 and len(set(generated)) == 3


def test_shared_cosine_is_single_source() -> None:
    from loomflow.vectorstore import _mmr

    assert _mmr._cosine is cosine
    assert cosine([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_rerank_tail_plain_topk_when_diversity_off() -> None:
    candidates = ["a", "b", "c"]
    vecs = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
    assert rerank_tail([1.0, 0.0], candidates, vecs, 2, None) == ["a", "b"]
    assert rerank_tail([1.0, 0.0], candidates, vecs, 2, 0.0) == ["a", "b"]


def test_rerank_tail_matches_mmr_select() -> None:
    candidates = ["a", "b", "c"]
    vecs = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
    query = [1.0, 0.0]
    expected = [
        candidates[i]
        for i in mmr_select(query, vecs, 2, diversity=1.0)
    ]
    assert rerank_tail(query, candidates, vecs, 2, 1.0) == expected


# ---------------------------------------------------------------------------
# Fix 8 — InMemory norm caching
# ---------------------------------------------------------------------------


async def test_inmemory_norms_cached_and_scores_correct() -> None:
    store = InMemoryVectorStore(embedder=_DirectionalEmbedder())
    await store.add(
        [
            Chunk(content="far east"),
            Chunk(content="north"),
            Chunk(content="west"),
        ]
    )
    assert len(store._norms) == 3
    assert store._norms[0] == pytest.approx(10.0)
    results = await store.search("east", k=3)
    scores = {r.chunk.content: r.score for r in results}
    assert scores["far east"] == pytest.approx(1.0)
    assert scores["north"] == pytest.approx(0.0)
    assert scores["west"] == pytest.approx(-1.0)


async def test_inmemory_norms_survive_delete_and_load(tmp_path: Any) -> None:
    store = InMemoryVectorStore(embedder=_DirectionalEmbedder())
    ids = await store.add(
        [
            Chunk(content="far east"),
            Chunk(content="north"),
            Chunk(content="west"),
        ]
    )
    await store.delete([ids[1]])
    assert len(store._norms) == len(store._vectors) == 2
    results = await store.search("east", k=3)
    assert [r.chunk.content for r in results] == ["far east", "west"]

    path = tmp_path / "store.json"
    await store.save(path)
    loaded = await InMemoryVectorStore.load(
        path, embedder=_DirectionalEmbedder()
    )
    assert len(loaded._norms) == 2
    reloaded = await loaded.search("east", k=3)
    assert [r.chunk.content for r in reloaded] == ["far east", "west"]
    assert reloaded[0].score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Fix 10 — Postgres only fetches embeddings when MMR runs
# ---------------------------------------------------------------------------


class _RecordingConn:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self._log.append(sql)
        return []

    async def execute(self, sql: str, *args: Any) -> str:
        self._log.append(sql)
        return "OK"


class _RecordingPool:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    async def acquire(self) -> _RecordingConn:
        return _RecordingConn(self._log)

    async def release(self, conn: Any) -> None:
        return None


async def test_postgres_search_skips_embedding_column_without_mmr() -> None:
    from loomflow.vectorstore.postgres import PostgresVectorStore

    store = PostgresVectorStore(
        embedder=HashEmbedder(dimensions=8), dsn="postgres://unused"
    )
    log: list[str] = []
    store._pool_obj = _RecordingPool(log)
    store._initialized = True

    await store.search_by_vector([0.0] * 8, k=3)
    select_clause = log[-1].split("FROM")[0]
    assert "embedding," not in select_clause  # column not fetched
    assert "AS score" in select_clause  # score expr still there

    await store.search_by_vector([0.0] * 8, k=3, diversity=0.5)
    select_clause = log[-1].split("FROM")[0]
    assert "embedding," in select_clause  # MMR needs the vectors
