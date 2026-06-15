"""Tests for two vectorstore fixes:

FIX 1 — ``_flatten_metadata`` (loomflow/vectorstore/chroma.py): coerce
non-scalar chunk metadata (lists/dicts) to JSON strings so chromadb
accepts the output of the framework's own chunkers (e.g.
``MarkdownChunker``'s ``headers`` list).

FIX 2 — ``index_document(path, store, *, chunker=None)``
(loomflow/vectorstore/_ingest.py): load + chunk + add to an EXISTING
store in one call. ADDS rather than building a throwaway store.

``_flatten_metadata`` and ``index_document`` import cleanly without the
``chromadb`` extra (the chromadb import is lazy inside
``ChromaVectorStore.__init__``), so the unit + InMemory tests run on a
vanilla install. The Chroma end-to-end tests gate on
``pytest.importorskip("chromadb")``.
"""

from __future__ import annotations

import json

import pytest

from loomflow import HashEmbedder
from loomflow.loader.base import Chunk
from loomflow.loader.chunking import MarkdownChunker, RecursiveChunker
from loomflow.vectorstore import InMemoryVectorStore, index_document
from loomflow.vectorstore.chroma import _flatten_metadata

pytestmark = pytest.mark.anyio


_MARKDOWN_DOC = """# Intro

Some intro text about retrieval augmented generation and embeddings.

## Background

Background section with more detail about vector databases and search.

## Methods

The methods section describes the chunking and indexing pipeline used.
"""


# ---------------------------------------------------------------------------
# A. _flatten_metadata — unit (no chromadb needed)
# ---------------------------------------------------------------------------


def test_flatten_metadata_passes_str_through() -> None:
    assert _flatten_metadata({"source": "a.md"}) == {"source": "a.md"}


def test_flatten_metadata_passes_int_through() -> None:
    out = _flatten_metadata({"page": 3})
    assert out["page"] == 3
    assert isinstance(out["page"], int)


def test_flatten_metadata_passes_float_through() -> None:
    out = _flatten_metadata({"score": 0.5})
    assert out["score"] == 0.5
    assert isinstance(out["score"], float)


def test_flatten_metadata_passes_bool_through() -> None:
    out = _flatten_metadata({"final": True})
    assert out["final"] is True


def test_flatten_metadata_passes_none_through() -> None:
    assert _flatten_metadata({"parent": None}) == {"parent": None}


def test_flatten_metadata_list_becomes_json_string() -> None:
    out = _flatten_metadata({"headers": ["A", "B"]})
    assert out["headers"] == '["A", "B"]'


def test_flatten_metadata_list_round_trips_via_json() -> None:
    out = _flatten_metadata({"headers": ["A", "B"]})
    assert json.loads(out["headers"]) == ["A", "B"]


def test_flatten_metadata_dict_becomes_json_string() -> None:
    out = _flatten_metadata({"extra": {"k": "v"}})
    assert isinstance(out["extra"], str)


def test_flatten_metadata_dict_round_trips_via_json() -> None:
    out = _flatten_metadata({"extra": {"k": "v"}})
    assert json.loads(out["extra"]) == {"k": "v"}


def test_flatten_metadata_never_raises_on_exotic_value() -> None:
    # A set isn't JSON-serialisable; default=str must guard it.
    out = _flatten_metadata({"tags": {"x", "y"}})
    assert isinstance(out["tags"], str)


# ---------------------------------------------------------------------------
# B. index_document against InMemoryVectorStore (offline, no extras)
# ---------------------------------------------------------------------------


async def test_index_document_returns_ids_matching_count(tmp_path) -> None:
    path = tmp_path / "doc.md"
    path.write_text(_MARKDOWN_DOC, encoding="utf-8")
    store = InMemoryVectorStore(embedder=HashEmbedder())

    ids = await index_document(path, store)

    assert ids
    assert await store.count() == len(ids)


async def test_index_document_adds_rather_than_replaces(tmp_path) -> None:
    # Regression for the from_texts "throwaway store" footgun: a second
    # call must GROW the store, not start fresh.
    path = tmp_path / "doc.md"
    path.write_text(_MARKDOWN_DOC, encoding="utf-8")
    store = InMemoryVectorStore(embedder=HashEmbedder())

    first_ids = await index_document(path, store)
    second_ids = await index_document(path, store)

    assert await store.count() == len(first_ids) + len(second_ids)


async def test_index_document_first_ids_still_retrievable(tmp_path) -> None:
    path = tmp_path / "doc.md"
    path.write_text(_MARKDOWN_DOC, encoding="utf-8")
    store = InMemoryVectorStore(embedder=HashEmbedder())

    first_ids = await index_document(path, store)
    await index_document(path, store)

    retrieved = await store.get_by_ids(first_ids)
    assert len(retrieved) == len(first_ids)


async def test_index_document_honors_explicit_markdown_chunker(
    tmp_path,
) -> None:
    path = tmp_path / "doc.md"
    path.write_text(_MARKDOWN_DOC, encoding="utf-8")
    store = InMemoryVectorStore(embedder=HashEmbedder())
    chunker = MarkdownChunker(chunk_size=200, chunk_overlap=20)

    ids = await index_document(path, store, chunker=chunker)

    # A multi-section doc yields more than one chunk.
    assert len(ids) > 1


async def test_index_document_markdown_chunker_carries_headers(
    tmp_path,
) -> None:
    path = tmp_path / "doc.md"
    path.write_text(_MARKDOWN_DOC, encoding="utf-8")
    store = InMemoryVectorStore(embedder=HashEmbedder())
    chunker = MarkdownChunker(chunk_size=200, chunk_overlap=20)

    ids = await index_document(path, store, chunker=chunker)

    chunks = await store.get_by_ids(ids)
    assert any("headers" in c.metadata for c in chunks)


async def test_index_document_default_chunker_sets_strategy(
    tmp_path,
) -> None:
    # RecursiveChunker (the default) tags each chunk with
    # strategy="recursive".
    path = tmp_path / "doc.md"
    path.write_text(_MARKDOWN_DOC, encoding="utf-8")
    store = InMemoryVectorStore(embedder=HashEmbedder())

    ids = await index_document(path, store)

    chunks = await store.get_by_ids(ids)
    assert chunks
    assert all(c.metadata.get("strategy") == "recursive" for c in chunks)


def test_index_document_default_chunker_is_recursive() -> None:
    # Documents the contract: omitting chunker uses RecursiveChunker.
    assert RecursiveChunker().split(_MARKDOWN_DOC)[0].metadata[
        "strategy"
    ] == "recursive"


# ---------------------------------------------------------------------------
# C. Chroma end-to-end (gated on the chromadb extra)
# ---------------------------------------------------------------------------

pytest.importorskip("chromadb")

import uuid  # noqa: E402

from loomflow.vectorstore import ChromaVectorStore  # noqa: E402


def _chroma_store(tmp_path) -> ChromaVectorStore:
    # UUID collection name so Ephemeral-ish state never leaks between
    # tests sharing a process.
    return ChromaVectorStore(
        embedder=HashEmbedder(),
        collection_name=f"test_{uuid.uuid4().hex}",
        persist_directory=str(tmp_path),
    )


async def test_chroma_add_markdown_chunks_does_not_raise(tmp_path) -> None:
    # The EXACT original crash: a MarkdownChunker chunk carries a
    # ``headers`` list, which chromadb rejected before the flatten fix.
    store = _chroma_store(tmp_path)
    chunks = MarkdownChunker().split(_MARKDOWN_DOC, source="d.md")

    ids = await store.add(chunks)

    assert ids
    assert len(ids) == len(chunks)


async def test_chroma_headers_stored_as_json_string(tmp_path) -> None:
    store = _chroma_store(tmp_path)
    chunks = MarkdownChunker().split(_MARKDOWN_DOC, source="d.md")

    ids = await store.add(chunks)
    got = await store.get_by_ids([ids[0]])

    assert got
    headers = got[0].metadata["headers"]
    assert isinstance(headers, str)
    assert json.loads(headers) == chunks[0].metadata["headers"]


async def test_chroma_scalar_metadata_survives_as_int(tmp_path) -> None:
    store = _chroma_store(tmp_path)
    chunk = Chunk(content="page three content", metadata={"page": 3})

    ids = await store.add([chunk])
    got = await store.get_by_ids(ids)

    assert got[0].metadata["page"] == 3
    assert isinstance(got[0].metadata["page"], int)


async def test_chroma_scalar_metadata_is_filterable(tmp_path) -> None:
    store = _chroma_store(tmp_path)
    await store.add(
        [
            Chunk(content="alpha on page three", metadata={"page": 3}),
            Chunk(content="beta on page seven", metadata={"page": 7}),
        ]
    )

    results = await store.search("page", k=5, filter={"page": 3})

    assert len(results) == 1
    assert results[0].chunk.metadata["page"] == 3


async def test_index_document_against_chroma(tmp_path) -> None:
    doc_path = tmp_path / "doc.md"
    doc_path.write_text(_MARKDOWN_DOC, encoding="utf-8")
    store = _chroma_store(tmp_path)

    ids = await index_document(doc_path, store)

    assert ids
    results = await store.search("vector databases and search", k=3)
    assert results
