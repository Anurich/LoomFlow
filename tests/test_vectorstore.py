"""Tests for loomflow.vectorstore.

Covers the in-memory implementation end-to-end (it's the default
and only one with zero deps) plus the protocol contract checks
that every backend is supposed to satisfy. Chroma/Postgres/FAISS
are tested via lazy-import fallbacks: each backend test verifies
the ImportError happens cleanly when the SDK is missing, so they
run on a vanilla install.

Also exercises the cross-backend helpers: Mongo-style filter
operators, MMR diversity reranking, BM25 hybrid search, save/load
persistence, and the from_chunks/from_texts factory classmethods.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loomflow import HashEmbedder
from loomflow.loader.base import Chunk
from loomflow.vectorstore import InMemoryVectorStore, SearchResult
from loomflow.vectorstore._bm25 import BM25Index, reciprocal_rank_fusion
from loomflow.vectorstore._filter import (
    FilterError,
    evaluate_filter,
)
from loomflow.vectorstore._mmr import mmr_select
from loomflow.vectorstore.base import VectorStore, matches_filter

# ---------------------------------------------------------------------------
# matches_filter — the shared filter helper
# ---------------------------------------------------------------------------


def test_matches_filter_none_or_empty_passes() -> None:
    assert matches_filter({"a": 1}, None)
    assert matches_filter({"a": 1}, {})


def test_matches_filter_scalar_equality() -> None:
    assert matches_filter({"source": "a.pdf"}, {"source": "a.pdf"})
    assert not matches_filter({"source": "a.pdf"}, {"source": "b.pdf"})


def test_matches_filter_list_membership() -> None:
    assert matches_filter(
        {"source": "a.pdf"}, {"source": ["a.pdf", "b.pdf"]}
    )
    assert not matches_filter(
        {"source": "c.pdf"}, {"source": ["a.pdf", "b.pdf"]}
    )


def test_matches_filter_missing_key_fails() -> None:
    assert not matches_filter({}, {"source": "a.pdf"})


def test_matches_filter_multi_key_all_must_pass() -> None:
    md = {"source": "a.pdf", "section": "intro"}
    assert matches_filter(md, {"source": "a.pdf", "section": "intro"})
    assert not matches_filter(
        md, {"source": "a.pdf", "section": "outro"}
    )


# ---------------------------------------------------------------------------
# InMemoryVectorStore — full lifecycle
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore(embedder=HashEmbedder(dimensions=128))


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        Chunk(
            content="Router classifies a request and dispatches to a specialist.",
            metadata={"source": "router.md", "topic": "routing"},
        ),
        Chunk(
            content="ReAct interleaves thinking and tool calls in a loop.",
            metadata={"source": "react.md", "topic": "react"},
        ),
        Chunk(
            content="Supervisor coordinates a team of worker agents.",
            metadata={"source": "supervisor.md", "topic": "multi-agent"},
        ),
    ]


def test_inmemory_satisfies_protocol(
    store: InMemoryVectorStore,
) -> None:
    """InMemoryVectorStore must structurally satisfy VectorStore."""
    assert isinstance(store, VectorStore)


def test_requires_embedder() -> None:
    with pytest.raises(ValueError):
        InMemoryVectorStore(embedder=None)  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_add_returns_generated_ids(
    store: InMemoryVectorStore, chunks: list[Chunk]
) -> None:
    ids = await store.add(chunks)
    assert len(ids) == 3
    assert all(isinstance(i, str) and i for i in ids)
    assert len(set(ids)) == 3, "ids must be unique"


@pytest.mark.anyio
async def test_add_with_explicit_ids(
    store: InMemoryVectorStore, chunks: list[Chunk]
) -> None:
    custom = ["c1", "c2", "c3"]
    ids = await store.add(chunks, ids=custom)
    assert ids == custom


@pytest.mark.anyio
async def test_add_id_count_mismatch_raises(
    store: InMemoryVectorStore, chunks: list[Chunk]
) -> None:
    with pytest.raises(ValueError):
        await store.add(chunks, ids=["only-one"])


@pytest.mark.anyio
async def test_add_empty_is_noop(
    store: InMemoryVectorStore,
) -> None:
    assert await store.add([]) == []
    assert await store.count() == 0


@pytest.mark.anyio
async def test_count_reflects_state(
    store: InMemoryVectorStore, chunks: list[Chunk]
) -> None:
    assert await store.count() == 0
    await store.add(chunks)
    assert await store.count() == 3


@pytest.mark.anyio
async def test_search_returns_results(
    store: InMemoryVectorStore, chunks: list[Chunk]
) -> None:
    await store.add(chunks)
    results = await store.search("how does routing work?", k=2)
    assert len(results) == 2
    assert all(isinstance(r, SearchResult) for r in results)
    # Cosine similarity is in [-1, 1]; HashEmbedder vectors are
    # signed so negative scores are valid.
    assert all(
        r.id and r.chunk.content and -1.0 <= r.score <= 1.0
        for r in results
    )


@pytest.mark.anyio
async def test_search_k_caps_result_count(
    store: InMemoryVectorStore, chunks: list[Chunk]
) -> None:
    await store.add(chunks)
    results = await store.search("routing", k=2)
    assert len(results) == 2

    results = await store.search("routing", k=10)
    assert len(results) == 3  # only 3 chunks exist


@pytest.mark.anyio
async def test_search_empty_store_returns_empty(
    store: InMemoryVectorStore,
) -> None:
    assert await store.search("anything", k=5) == []


@pytest.mark.anyio
async def test_search_with_filter(
    store: InMemoryVectorStore, chunks: list[Chunk]
) -> None:
    await store.add(chunks)
    results = await store.search(
        "any topic", k=10, filter={"source": "router.md"}
    )
    assert len(results) == 1
    assert results[0].chunk.metadata["source"] == "router.md"


@pytest.mark.anyio
async def test_search_with_list_filter(
    store: InMemoryVectorStore, chunks: list[Chunk]
) -> None:
    await store.add(chunks)
    results = await store.search(
        "any topic",
        k=10,
        filter={"source": ["router.md", "react.md"]},
    )
    assert len(results) == 2
    sources = {r.chunk.metadata["source"] for r in results}
    assert sources == {"router.md", "react.md"}


@pytest.mark.anyio
async def test_search_filter_misses_returns_empty(
    store: InMemoryVectorStore, chunks: list[Chunk]
) -> None:
    await store.add(chunks)
    results = await store.search(
        "any", k=10, filter={"source": "nonexistent.md"}
    )
    assert results == []


@pytest.mark.anyio
async def test_search_results_sorted_by_score_desc(
    store: InMemoryVectorStore, chunks: list[Chunk]
) -> None:
    await store.add(chunks)
    results = await store.search("Router classifies dispatches", k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.anyio
async def test_search_by_vector_matches_search(
    store: InMemoryVectorStore, chunks: list[Chunk]
) -> None:
    await store.add(chunks)
    embedder = store.embedder
    q_vec = await embedder.embed("multi-agent")
    via_vector = await store.search_by_vector(q_vec, k=3)
    via_query = await store.search("multi-agent", k=3)
    assert [r.id for r in via_vector] == [r.id for r in via_query]


@pytest.mark.anyio
async def test_delete_removes_and_is_idempotent(
    store: InMemoryVectorStore, chunks: list[Chunk]
) -> None:
    ids = await store.add(chunks)
    await store.delete([ids[0]])
    assert await store.count() == 2

    # Delete the same id again — must not raise.
    await store.delete([ids[0]])
    assert await store.count() == 2

    # Delete an unknown id — must not raise.
    await store.delete(["not-a-real-id"])
    assert await store.count() == 2


@pytest.mark.anyio
async def test_delete_empty_list_is_noop(
    store: InMemoryVectorStore, chunks: list[Chunk]
) -> None:
    await store.add(chunks)
    await store.delete([])
    assert await store.count() == 3


@pytest.mark.anyio
async def test_deleted_chunks_dont_appear_in_search(
    store: InMemoryVectorStore, chunks: list[Chunk]
) -> None:
    await store.add(chunks)
    # Find which id corresponds to the router chunk.
    results = await store.search("router classifies", k=1)
    router_id = results[0].id

    await store.delete([router_id])
    results = await store.search("router classifies", k=10)
    assert all(r.id != router_id for r in results)


@pytest.mark.anyio
async def test_two_stores_dont_share_state() -> None:
    """Different instances must isolate their state — guards
    against accidental class-level mutable defaults."""
    s1 = InMemoryVectorStore(embedder=HashEmbedder(dimensions=64))
    s2 = InMemoryVectorStore(embedder=HashEmbedder(dimensions=64))
    await s1.add([Chunk(content="hello", metadata={})])
    assert await s1.count() == 1
    assert await s2.count() == 0


# ---------------------------------------------------------------------------
# Lazy SDK import — Chroma/Postgres/FAISS raise ImportError without their dep
# ---------------------------------------------------------------------------


def test_chroma_import_error_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ChromaVectorStore must surface a clear ImportError naming the
    extras path when ``chromadb`` isn't installed."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "chromadb":
            raise ImportError("no module named chromadb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from loomflow.vectorstore import ChromaVectorStore

    with pytest.raises(ImportError, match="vectorstore-chroma"):
        ChromaVectorStore(embedder=HashEmbedder(dimensions=8))


def test_faiss_import_error_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "faiss":
            raise ImportError("no module named faiss")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from loomflow.vectorstore import FAISSVectorStore

    with pytest.raises(ImportError, match="vectorstore-faiss"):
        FAISSVectorStore(embedder=HashEmbedder(dimensions=8))


@pytest.mark.anyio
async def test_postgres_import_error_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``asyncpg`` is imported lazily inside ``_connect``; the
    constructor doesn't fail, but the first DB call should."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "asyncpg":
            raise ImportError("no module named asyncpg")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from loomflow.vectorstore import PostgresVectorStore

    store = PostgresVectorStore(
        embedder=HashEmbedder(dimensions=8),
        dsn="postgresql://localhost/test",
    )
    with pytest.raises(ImportError, match="vectorstore-postgres"):
        await store.init_schema(8)


# ---------------------------------------------------------------------------
# Mongo-style filter operators — beyond the legacy shorthand
# ---------------------------------------------------------------------------


def test_filter_eq_operator() -> None:
    assert evaluate_filter({"a": {"$eq": 1}}, {"a": 1})
    assert not evaluate_filter({"a": {"$eq": 1}}, {"a": 2})


def test_filter_ne_operator() -> None:
    assert evaluate_filter({"a": {"$ne": 1}}, {"a": 2})
    assert not evaluate_filter({"a": {"$ne": 1}}, {"a": 1})
    # Missing key is "not equal" — $ne matches.
    assert evaluate_filter({"a": {"$ne": 1}}, {})


def test_filter_range_operators() -> None:
    md = {"page": 10}
    assert evaluate_filter({"page": {"$gte": 10}}, md)
    assert evaluate_filter({"page": {"$gt": 5}}, md)
    assert evaluate_filter({"page": {"$lt": 20}}, md)
    assert evaluate_filter({"page": {"$lte": 10}}, md)
    assert not evaluate_filter({"page": {"$gt": 10}}, md)
    assert not evaluate_filter({"page": {"$lt": 10}}, md)


def test_filter_range_on_incomparable_types_returns_false() -> None:
    """Mixed-type metadata shouldn't blow up the evaluator."""
    assert not evaluate_filter({"page": {"$gt": 5}}, {"page": "ten"})


def test_filter_in_and_nin() -> None:
    md = {"tag": "draft"}
    assert evaluate_filter({"tag": {"$in": ["draft", "final"]}}, md)
    assert not evaluate_filter({"tag": {"$in": ["other"]}}, md)
    assert evaluate_filter({"tag": {"$nin": ["other"]}}, md)
    assert not evaluate_filter({"tag": {"$nin": ["draft"]}}, md)


def test_filter_exists_operator() -> None:
    md = {"present": "yes"}
    assert evaluate_filter({"present": {"$exists": True}}, md)
    assert not evaluate_filter({"missing": {"$exists": True}}, md)
    assert evaluate_filter({"missing": {"$exists": False}}, md)
    assert not evaluate_filter({"present": {"$exists": False}}, md)


def test_filter_and_composition() -> None:
    md = {"a": 1, "b": 2}
    assert evaluate_filter(
        {"$and": [{"a": 1}, {"b": 2}]}, md
    )
    assert not evaluate_filter(
        {"$and": [{"a": 1}, {"b": 999}]}, md
    )


def test_filter_or_composition() -> None:
    md = {"a": 1}
    assert evaluate_filter(
        {"$or": [{"a": 1}, {"b": 999}]}, md
    )
    assert evaluate_filter(
        {"$or": [{"a": 999}, {"a": 1}]}, md
    )
    assert not evaluate_filter(
        {"$or": [{"a": 999}, {"b": 999}]}, md
    )


def test_filter_not_operator() -> None:
    md = {"a": 1}
    assert evaluate_filter({"$not": {"a": 999}}, md)
    assert not evaluate_filter({"$not": {"a": 1}}, md)


def test_filter_nested_logical() -> None:
    md = {"page": 10, "author": "alice"}
    expr = {
        "$and": [
            {"page": {"$gte": 5}},
            {"$not": {"author": "bob"}},
        ]
    }
    assert evaluate_filter(expr, md)


def test_filter_unknown_operator_raises() -> None:
    with pytest.raises(FilterError):
        evaluate_filter({"$foo": []}, {})
    with pytest.raises(FilterError):
        evaluate_filter({"a": {"$xyz": 1}}, {"a": 1})


def test_filter_shorthand_still_works() -> None:
    """Backwards compat: scalar / list shorthand from v0.5.0."""
    assert evaluate_filter({"k": "v"}, {"k": "v"})
    assert evaluate_filter({"k": ["a", "b"]}, {"k": "a"})


@pytest.mark.anyio
async def test_search_with_range_filter(
    store: InMemoryVectorStore,
) -> None:
    await store.add([
        Chunk(content="page one", metadata={"page": 1}),
        Chunk(content="page five", metadata={"page": 5}),
        Chunk(content="page ten", metadata={"page": 10}),
    ])
    results = await store.search(
        "page", k=10, filter={"page": {"$gte": 5}}
    )
    assert len(results) == 2
    pages = sorted(r.chunk.metadata["page"] for r in results)
    assert pages == [5, 10]


@pytest.mark.anyio
async def test_search_with_or_filter(
    store: InMemoryVectorStore, chunks: list[Chunk]
) -> None:
    await store.add(chunks)
    results = await store.search(
        "anything",
        k=10,
        filter={
            "$or": [
                {"source": "router.md"},
                {"topic": "react"},
            ]
        },
    )
    assert len(results) == 2


# ---------------------------------------------------------------------------
# MMR diversity reranking
# ---------------------------------------------------------------------------


def test_mmr_select_basic() -> None:
    """MMR with diversity=0 should match top-k by query similarity."""
    query = [1.0, 0.0]
    candidates = [
        [1.0, 0.0],   # score 1.0 — most similar
        [0.9, 0.1],   # score 0.99-ish
        [0.0, 1.0],   # score 0.0 — least similar
    ]
    chosen = mmr_select(query, candidates, k=2, diversity=0.0)
    # Pure relevance: first two by similarity to query.
    assert chosen[0] == 0


def test_mmr_select_full_diversity_picks_spread() -> None:
    query = [1.0, 0.0]
    # Three near-duplicates and one outlier.
    candidates = [
        [1.0, 0.0],
        [0.99, 0.01],
        [0.98, 0.02],
        [0.0, 1.0],   # the outlier
    ]
    chosen = mmr_select(query, candidates, k=2, diversity=1.0)
    # First pick: most similar to query (idx 0). Second pick under
    # max-diversity should be the outlier (idx 3), not another
    # near-duplicate.
    assert chosen[0] == 0
    assert chosen[1] == 3


def test_mmr_select_clamps_diversity() -> None:
    chosen = mmr_select(
        [1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], k=2, diversity=5.0
    )
    assert len(chosen) == 2


def test_mmr_select_handles_empty_pool() -> None:
    assert mmr_select([1.0], [], k=5) == []


def test_mmr_select_caps_k_to_pool_size() -> None:
    chosen = mmr_select([1.0], [[1.0], [0.5]], k=10)
    assert len(chosen) == 2


@pytest.mark.anyio
async def test_search_diversity_reranks(
    store: InMemoryVectorStore,
) -> None:
    """End-to-end: with diversity set, the result ordering must differ
    from plain top-k (the algorithm is doing something). The exact
    membership depends on the embedder and is covered by the
    deterministic mmr_select unit tests above."""
    chunks = [
        Chunk(content=f"chunk number {i}", metadata={"id": f"c{i}"})
        for i in range(8)
    ]
    await store.add(chunks)
    plain = await store.search("chunk", k=4, diversity=None)
    diverse = await store.search("chunk", k=4, diversity=1.0)
    plain_order = [r.id for r in plain]
    diverse_order = [r.id for r in diverse]
    # First pick should match (both algorithms start with the most-
    # similar candidate); subsequent picks should differ.
    assert plain_order[0] == diverse_order[0]
    assert plain_order != diverse_order


@pytest.mark.anyio
async def test_search_diversity_zero_matches_plain_search(
    store: InMemoryVectorStore, chunks: list[Chunk]
) -> None:
    await store.add(chunks)
    plain = await store.search("router", k=3)
    diverse = await store.search("router", k=3, diversity=0.0)
    assert [r.id for r in plain] == [r.id for r in diverse]


# ---------------------------------------------------------------------------
# Hybrid search (BM25 + vector via RRF)
# ---------------------------------------------------------------------------


def test_bm25_index_basic() -> None:
    idx = BM25Index()
    idx.add(["the cat sat on the mat", "dogs run fast", "cats love fish"])
    results = idx.search("cat", k=3)
    # The first doc mentions "cat" once; the third has "cats" (different
    # token under our regex tokenizer). Only doc 0 should match.
    assert len(results) >= 1
    assert results[0][0] == 0


def test_bm25_empty_query_returns_nothing() -> None:
    idx = BM25Index()
    idx.add(["any text"])
    assert idx.search("", k=5) == []


def test_bm25_remove_rebuilds() -> None:
    idx = BM25Index()
    idx.add(["foo", "bar", "baz"])
    idx.remove_by_indices({1})
    # After removal, querying "bar" should miss.
    assert all(score == 0 for _, score in idx.search("bar", k=10))


def test_rrf_combines_rankings() -> None:
    a = [(1, 0.9), (2, 0.8), (3, 0.5)]
    b = [(2, 5.0), (3, 4.0), (1, 1.0)]
    fused = reciprocal_rank_fusion([a, b])
    # Doc 2 is rank 2 in A and rank 1 in B; should rank highest.
    assert fused[0][0] == 2


@pytest.mark.anyio
async def test_search_hybrid_finds_exact_term(
    store: InMemoryVectorStore,
) -> None:
    """Hybrid should rank a chunk that contains the exact query token
    above semantically-related chunks that don't."""
    await store.add([
        Chunk(
            content="The XKCD-42 error code means storage exhausted.",
            metadata={"id": "exact"},
        ),
        Chunk(
            content="A general fault occurred on the device.",
            metadata={"id": "vague"},
        ),
        Chunk(
            content="Persistent memory issues affect performance.",
            metadata={"id": "tangent"},
        ),
    ])
    results = await store.search_hybrid("XKCD-42", k=2, alpha=0.5)
    top_ids = {r.chunk.metadata["id"] for r in results}
    assert "exact" in top_ids


@pytest.mark.anyio
async def test_search_hybrid_alpha_extremes(
    store: InMemoryVectorStore, chunks: list[Chunk]
) -> None:
    """alpha=0 (pure BM25) and alpha=1 (pure vector) both produce
    valid result lists without crashing."""
    await store.add(chunks)
    bm25_only = await store.search_hybrid("router", k=2, alpha=0.0)
    vec_only = await store.search_hybrid("router", k=2, alpha=1.0)
    assert len(bm25_only) >= 1
    assert len(vec_only) >= 1


@pytest.mark.anyio
async def test_search_hybrid_respects_filter(
    store: InMemoryVectorStore, chunks: list[Chunk]
) -> None:
    await store.add(chunks)
    results = await store.search_hybrid(
        "anything",
        k=10,
        filter={"source": "router.md"},
    )
    assert all(
        r.chunk.metadata["source"] == "router.md" for r in results
    )


@pytest.mark.anyio
async def test_search_hybrid_empty_store(
    store: InMemoryVectorStore,
) -> None:
    assert await store.search_hybrid("anything", k=3) == []


# ---------------------------------------------------------------------------
# from_chunks / from_texts factory classmethods
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_from_chunks_one_liner(chunks: list[Chunk]) -> None:
    store = await InMemoryVectorStore.from_chunks(
        chunks, embedder=HashEmbedder(dimensions=64)
    )
    assert await store.count() == 3


@pytest.mark.anyio
async def test_from_texts_default_metadata() -> None:
    store = await InMemoryVectorStore.from_texts(
        ["alpha", "beta"],
        embedder=HashEmbedder(dimensions=32),
    )
    assert await store.count() == 2
    results = await store.search("alpha", k=2)
    # Default empty metadata.
    assert all(r.chunk.metadata == {} for r in results)


@pytest.mark.anyio
async def test_from_texts_with_metadatas() -> None:
    store = await InMemoryVectorStore.from_texts(
        ["a", "b"],
        embedder=HashEmbedder(dimensions=32),
        metadatas=[{"page": 1}, {"page": 2}],
    )
    results = await store.search("a", k=2)
    pages = {r.chunk.metadata["page"] for r in results}
    assert pages == {1, 2}


@pytest.mark.anyio
async def test_from_texts_metadata_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        await InMemoryVectorStore.from_texts(
            ["a", "b", "c"],
            embedder=HashEmbedder(dimensions=8),
            metadatas=[{"x": 1}],
        )


# ---------------------------------------------------------------------------
# get_by_ids
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_by_ids_returns_in_order(
    store: InMemoryVectorStore, chunks: list[Chunk]
) -> None:
    ids = await store.add(chunks)
    fetched = await store.get_by_ids([ids[2], ids[0]])
    assert fetched[0].content == chunks[2].content
    assert fetched[1].content == chunks[0].content


@pytest.mark.anyio
async def test_get_by_ids_skips_unknown(
    store: InMemoryVectorStore, chunks: list[Chunk]
) -> None:
    ids = await store.add(chunks)
    fetched = await store.get_by_ids([ids[0], "not-real", ids[1]])
    assert len(fetched) == 2


@pytest.mark.anyio
async def test_get_by_ids_empty_input(
    store: InMemoryVectorStore,
) -> None:
    assert await store.get_by_ids([]) == []


# ---------------------------------------------------------------------------
# Persistence (save/load)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_save_and_load_round_trip(
    tmp_path: Path, chunks: list[Chunk]
) -> None:
    store = InMemoryVectorStore(embedder=HashEmbedder(dimensions=64))
    await store.add(chunks)
    path = tmp_path / "store.json"
    await store.save(path)
    assert path.exists()

    loaded = await InMemoryVectorStore.load(
        path, embedder=HashEmbedder(dimensions=64)
    )
    assert await loaded.count() == 3
    # Search results should be identical with the same embedder seed.
    original = await store.search("router", k=3)
    restored = await loaded.search("router", k=3)
    assert [r.id for r in original] == [r.id for r in restored]
    assert [r.chunk.content for r in original] == [
        r.chunk.content for r in restored
    ]


@pytest.mark.anyio
async def test_save_empty_store(tmp_path: Path) -> None:
    store = InMemoryVectorStore(embedder=HashEmbedder(dimensions=8))
    path = tmp_path / "empty.json"
    await store.save(path)
    data = json.loads(path.read_text())
    assert data["rows"] == []
    assert data["dimensions"] is None


@pytest.mark.anyio
async def test_load_unsupported_version_raises(
    tmp_path: Path,
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"version": 999, "rows": []}))
    with pytest.raises(ValueError, match="version"):
        await InMemoryVectorStore.load(
            bad, embedder=HashEmbedder(dimensions=8)
        )


# ---------------------------------------------------------------------------
# Persistence — metadata consistency (FIX 2: json.dumps default=str so a
# list/dict metadata never crashes save(), and round-trips natively).
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_save_with_list_metadata_does_not_raise(
    tmp_path: Path,
) -> None:
    # The original crash: a MarkdownChunker ``headers`` list reached
    # json.dumps with a non-default serialiser. save() must not raise.
    store = InMemoryVectorStore(embedder=HashEmbedder(dimensions=8))
    await store.add(
        [Chunk(content="x", metadata={"headers": ["A", "B"]})]
    )
    await store.save(tmp_path / "db.json")


@pytest.mark.anyio
async def test_load_list_metadata_round_trips_as_native_list(
    tmp_path: Path,
) -> None:
    store = InMemoryVectorStore(embedder=HashEmbedder(dimensions=8))
    ids = await store.add(
        [Chunk(content="x", metadata={"headers": ["A", "B"]})]
    )
    path = tmp_path / "db.json"
    await store.save(path)

    loaded = await InMemoryVectorStore.load(
        path, embedder=HashEmbedder(dimensions=8)
    )
    got = await loaded.get_by_ids(ids)
    # NATIVE list, not a stringified one.
    assert got[0].metadata["headers"] == ["A", "B"]


@pytest.mark.anyio
async def test_load_scalar_metadata_survives_with_type(
    tmp_path: Path,
) -> None:
    store = InMemoryVectorStore(embedder=HashEmbedder(dimensions=8))
    ids = await store.add(
        [Chunk(content="x", metadata={"page": 3, "src": "a.md"})]
    )
    path = tmp_path / "db.json"
    await store.save(path)

    loaded = await InMemoryVectorStore.load(
        path, embedder=HashEmbedder(dimensions=8)
    )
    md = (await loaded.get_by_ids(ids))[0].metadata
    assert md["page"] == 3
    assert isinstance(md["page"], int)
    assert md["src"] == "a.md"


@pytest.mark.anyio
async def test_save_non_serialisable_metadata_does_not_raise(
    tmp_path: Path,
) -> None:
    # A set isn't JSON-serialisable; default=str is the safety net.
    store = InMemoryVectorStore(embedder=HashEmbedder(dimensions=8))
    await store.add(
        [Chunk(content="x", metadata={"tags": {"a", "b"}})]
    )
    await store.save(tmp_path / "db.json")


@pytest.mark.anyio
async def test_load_non_serialisable_metadata_comes_back_as_string(
    tmp_path: Path,
) -> None:
    store = InMemoryVectorStore(embedder=HashEmbedder(dimensions=8))
    ids = await store.add(
        [Chunk(content="x", metadata={"tags": {"a", "b"}})]
    )
    path = tmp_path / "db.json"
    await store.save(path)

    loaded = await InMemoryVectorStore.load(
        path, embedder=HashEmbedder(dimensions=8)
    )
    tags = (await loaded.get_by_ids(ids))[0].metadata["tags"]
    assert isinstance(tags, str)


# ---------------------------------------------------------------------------
# FAISS — cross-store cosine SCORE CONTRACT (FIX 1). Gated on the
# vectorstore-faiss extra; the import name is ``faiss``.
# ---------------------------------------------------------------------------

_FAISS_CORPUS = [
    Chunk(content="routers forward packets between networks"),
    Chunk(content="photosynthesis converts sunlight into sugar"),
    Chunk(content="the violin is a stringed musical instrument"),
]


@pytest.mark.anyio
async def test_faiss_ip_top_result_matches_query_and_score_in_bounds() -> (
    None
):
    pytest.importorskip("faiss")
    from loomflow.vectorstore import FAISSVectorStore

    embedder = HashEmbedder(dimensions=128)
    store = FAISSVectorStore(embedder=embedder)  # default metric="ip"
    await store.add(_FAISS_CORPUS)

    query = _FAISS_CORPUS[0].content
    results = await store.search(query, k=3)

    assert results
    assert results[0].chunk.content == query
    # Cosine contract: every score in [-1, 1] (the old raw dot product
    # on non-unit vectors could exceed 1).
    for r in results:
        assert -1.0 <= r.score <= 1.0 + 1e-6


@pytest.mark.anyio
async def test_faiss_ip_ranking_and_scores_match_inmemory() -> None:
    pytest.importorskip("faiss")
    from loomflow.vectorstore import FAISSVectorStore

    # Same embedder kind => identical embeddings => comparable scores.
    faiss_store = FAISSVectorStore(embedder=HashEmbedder(dimensions=128))
    mem_store = InMemoryVectorStore(embedder=HashEmbedder(dimensions=128))
    await faiss_store.add(_FAISS_CORPUS)
    await mem_store.add(_FAISS_CORPUS)

    query = _FAISS_CORPUS[0].content
    faiss_res = await faiss_store.search(query, k=3)
    mem_res = await mem_store.search(query, k=3)

    # Same top-ranked chunk across backends.
    assert faiss_res[0].chunk.content == mem_res[0].chunk.content
    # And comparable top scores (cosine on both sides).
    assert abs(faiss_res[0].score - mem_res[0].score) < 0.05


@pytest.mark.anyio
async def test_faiss_l2_scores_in_bounds_and_higher_is_better() -> None:
    pytest.importorskip("faiss")
    from loomflow.vectorstore import FAISSVectorStore

    store = FAISSVectorStore(
        embedder=HashEmbedder(dimensions=128),
        index_factory_string="Flat",
        metric="l2",
    )
    await store.add(_FAISS_CORPUS)

    query = _FAISS_CORPUS[0].content
    results = await store.search(query, k=3)

    assert results
    for r in results:
        assert -1.0 <= r.score <= 1.0 + 1e-6
    # The exact-match chunk outscores a poor match (higher-is-better).
    by_content = {r.chunk.content: r.score for r in results}
    assert by_content[query] > min(by_content.values())


# ---------------------------------------------------------------------------
# Postgres filter SQL translation (no DB required — pure unit test)
# ---------------------------------------------------------------------------


def test_postgres_filter_translation() -> None:
    from loomflow.vectorstore.postgres import _build_where_sql

    sql, params = _build_where_sql({"source": "x.pdf"}, [])
    assert "metadata->>'source'" in sql
    assert params == ["x.pdf"]


def test_postgres_filter_range_translation() -> None:
    from loomflow.vectorstore.postgres import _build_where_sql

    sql, params = _build_where_sql({"page": {"$gte": 5}}, [])
    assert ">=" in sql
    # Numeric operands compare numerically ("10" < "9" as text!) —
    # the extracted value is cast and the operand binds as a number.
    assert "::numeric" in sql
    assert params == [5]


def test_postgres_filter_logical_translation() -> None:
    from loomflow.vectorstore.postgres import _build_where_sql

    sql, _ = _build_where_sql(
        {"$or": [{"a": 1}, {"b": 2}]}, []
    )
    assert " OR " in sql


def test_postgres_filter_exists_translation() -> None:
    from loomflow.vectorstore.postgres import _build_where_sql

    sql, _ = _build_where_sql({"key": {"$exists": True}}, [])
    assert "metadata ? 'key'" in sql


# ---------------------------------------------------------------------------
# Chroma filter translation (no Chroma required for the pure helper)
# ---------------------------------------------------------------------------


def test_chroma_filter_passthrough_operator_form() -> None:
    from loomflow.vectorstore.chroma import _translate_filter

    out = _translate_filter({"page": {"$gte": 5}})
    assert out == {"page": {"$gte": 5}}


def test_chroma_filter_scalar_shorthand_normalized() -> None:
    from loomflow.vectorstore.chroma import _translate_filter

    out = _translate_filter({"source": "x.pdf"})
    assert out == {"source": {"$eq": "x.pdf"}}


def test_chroma_filter_list_shorthand_normalized() -> None:
    from loomflow.vectorstore.chroma import _translate_filter

    out = _translate_filter({"tag": ["a", "b"]})
    assert out == {"tag": {"$in": ["a", "b"]}}


def test_chroma_filter_multi_field_wraps_in_and() -> None:
    from loomflow.vectorstore.chroma import _translate_filter

    out = _translate_filter({"a": 1, "b": 2})
    assert out is not None
    assert "$and" in out


def test_chroma_filter_not_unsupported() -> None:
    from loomflow.vectorstore.chroma import _translate_filter

    with pytest.raises(FilterError, match="Chroma"):
        _translate_filter({"$not": {"a": 1}})


# ---------------------------------------------------------------------------
# anyio backend selection — match the rest of the test suite
# ---------------------------------------------------------------------------


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
