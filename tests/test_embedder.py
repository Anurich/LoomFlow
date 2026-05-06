"""Embedder tests."""

from __future__ import annotations

import math
from types import SimpleNamespace as NS
from typing import Any

import pytest

from jeevesagent.memory.embedder import HashEmbedder, OpenAIEmbedder

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# HashEmbedder
# ---------------------------------------------------------------------------


async def test_hash_embedder_is_deterministic() -> None:
    e = HashEmbedder()
    v1 = await e.embed("hello world")
    v2 = await e.embed("hello world")
    assert v1 == v2


async def test_hash_embedder_distinct_text_distinct_vectors() -> None:
    e = HashEmbedder()
    v1 = await e.embed("alpha")
    v2 = await e.embed("beta")
    assert v1 != v2


async def test_hash_embedder_returns_unit_norm_vectors() -> None:
    e = HashEmbedder(dimensions=128)
    v = await e.embed("anything")
    assert len(v) == 128
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-6


async def test_hash_embedder_batch_matches_single() -> None:
    e = HashEmbedder(dimensions=64)
    vs = await e.embed_batch(["a", "b", "c"])
    singles = [await e.embed(t) for t in ["a", "b", "c"]]
    assert vs == singles


def test_hash_embedder_rejects_zero_dimensions() -> None:
    with pytest.raises(ValueError):
        HashEmbedder(dimensions=0)


# ---------------------------------------------------------------------------
# OpenAIEmbedder (with fake client; no real network)
# ---------------------------------------------------------------------------


class _FakeOAIEmbeddings:
    def __init__(self) -> None:
        self.captured_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> Any:
        self.captured_kwargs = kwargs
        # Behavior depends on input shape: str or list[str].
        inp = kwargs["input"]
        if isinstance(inp, str):
            data = [NS(embedding=[1.0, 2.0, 3.0])]
        else:
            data = [NS(embedding=[float(i), float(i + 1)]) for i in range(len(inp))]
        return NS(data=data)


class _FakeOAIClient:
    def __init__(self) -> None:
        self.embeddings = _FakeOAIEmbeddings()


async def test_openai_embedder_default_dimensions() -> None:
    fc = _FakeOAIClient()
    e = OpenAIEmbedder("text-embedding-3-small", client=fc)
    assert e.dimensions == 1536
    assert e.name == "text-embedding-3-small"


async def test_openai_embedder_explicit_dimensions_passes_through() -> None:
    fc = _FakeOAIClient()
    e = OpenAIEmbedder(
        "text-embedding-3-small",
        dimensions=256,
        client=fc,
    )
    await e.embed("hi")
    assert fc.embeddings.captured_kwargs is not None
    assert fc.embeddings.captured_kwargs["dimensions"] == 256


async def test_openai_embedder_embed_returns_list() -> None:
    fc = _FakeOAIClient()
    e = OpenAIEmbedder("text-embedding-3-small", client=fc)
    v = await e.embed("hi")
    assert v == [1.0, 2.0, 3.0]


async def test_openai_embedder_batch_returns_per_input() -> None:
    fc = _FakeOAIClient()
    e = OpenAIEmbedder("text-embedding-3-small", client=fc)
    vs = await e.embed_batch(["a", "b", "c"])
    assert len(vs) == 3
    assert vs[0] == [0.0, 1.0]
    assert vs[1] == [1.0, 2.0]
    assert vs[2] == [2.0, 3.0]


async def test_openai_embedder_batch_empty_returns_empty() -> None:
    fc = _FakeOAIClient()
    e = OpenAIEmbedder("text-embedding-3-small", client=fc)
    assert await e.embed_batch([]) == []


# ---------------------------------------------------------------------------
# VoyageEmbedder (with fake client; no real network)
# ---------------------------------------------------------------------------


class _FakeVoyageClient:
    def __init__(self) -> None:
        self.captured_kwargs: dict[str, Any] | None = None

    async def embed(self, **kwargs: Any) -> Any:
        self.captured_kwargs = kwargs
        n = len(kwargs["texts"])
        return NS(embeddings=[[float(i)] * 4 for i in range(n)])


async def test_voyage_embedder_default_dimensions() -> None:
    from jeevesagent import VoyageEmbedder

    fc = _FakeVoyageClient()
    e = VoyageEmbedder("voyage-3", client=fc)
    assert e.name == "voyage-3"
    assert e.dimensions == 1024


async def test_voyage_embedder_lite_has_smaller_dim() -> None:
    from jeevesagent import VoyageEmbedder

    e = VoyageEmbedder("voyage-3-lite", client=_FakeVoyageClient())
    assert e.dimensions == 512


async def test_voyage_embedder_passes_input_type_to_sdk() -> None:
    from jeevesagent import VoyageEmbedder

    fc = _FakeVoyageClient()
    e = VoyageEmbedder("voyage-3", client=fc, input_type="query")
    await e.embed("hi")
    assert fc.captured_kwargs is not None
    assert fc.captured_kwargs["input_type"] == "query"
    assert fc.captured_kwargs["model"] == "voyage-3"


async def test_voyage_embedder_batch_returns_per_input() -> None:
    from jeevesagent import VoyageEmbedder

    fc = _FakeVoyageClient()
    e = VoyageEmbedder("voyage-3", client=fc)
    vs = await e.embed_batch(["a", "b", "c"])
    assert len(vs) == 3
    assert vs[0] == [0.0, 0.0, 0.0, 0.0]
    assert vs[2] == [2.0, 2.0, 2.0, 2.0]


async def test_voyage_embedder_batch_empty_returns_empty() -> None:
    from jeevesagent import VoyageEmbedder

    e = VoyageEmbedder("voyage-3", client=_FakeVoyageClient())
    assert await e.embed_batch([]) == []


# ---------------------------------------------------------------------------
# CohereEmbedder (with fake client; no real network)
# ---------------------------------------------------------------------------


class _FakeCohereEmbedResponse:
    def __init__(self, n: int) -> None:
        self.embeddings = NS(float=[[float(i)] * 4 for i in range(n)])


class _FakeCohereClient:
    def __init__(self) -> None:
        self.captured_kwargs: dict[str, Any] | None = None

    async def embed(self, **kwargs: Any) -> Any:
        self.captured_kwargs = kwargs
        return _FakeCohereEmbedResponse(len(kwargs["texts"]))


async def test_cohere_embedder_default_dimensions() -> None:
    from jeevesagent import CohereEmbedder

    fc = _FakeCohereClient()
    e = CohereEmbedder("embed-english-v3.0", client=fc)
    assert e.name == "embed-english-v3.0"
    assert e.dimensions == 1024


async def test_cohere_embedder_light_has_smaller_dim() -> None:
    from jeevesagent import CohereEmbedder

    e = CohereEmbedder(
        "embed-english-light-v3.0", client=_FakeCohereClient()
    )
    assert e.dimensions == 384


async def test_cohere_embedder_passes_input_type_and_floats() -> None:
    """v3 models require ``input_type`` and ``embedding_types``."""
    from jeevesagent import CohereEmbedder

    fc = _FakeCohereClient()
    e = CohereEmbedder(
        "embed-english-v3.0", client=fc, input_type="search_query"
    )
    await e.embed("hi")
    assert fc.captured_kwargs is not None
    assert fc.captured_kwargs["input_type"] == "search_query"
    assert fc.captured_kwargs["embedding_types"] == ["float"]


async def test_cohere_embedder_batch_returns_per_input() -> None:
    from jeevesagent import CohereEmbedder

    fc = _FakeCohereClient()
    e = CohereEmbedder("embed-english-v3.0", client=fc)
    vs = await e.embed_batch(["a", "b"])
    assert len(vs) == 2
    assert vs[0] == [0.0, 0.0, 0.0, 0.0]


async def test_cohere_embedder_batch_empty_returns_empty() -> None:
    from jeevesagent import CohereEmbedder

    e = CohereEmbedder(
        "embed-english-v3.0", client=_FakeCohereClient()
    )
    assert await e.embed_batch([]) == []
