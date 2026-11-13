"""Embedders that turn text into vectors.

Two implementations land in this slice:

* :class:`HashEmbedder` — deterministic, zero-dep, SHA256-seeded
  Gaussian sample. Same text → same vector. Perfect for tests, dev,
  and for memory backends that only need *some* vector to enable
  recall without the cost of a real embedding API.
* :class:`OpenAIEmbedder` — wraps OpenAI's
  ``text-embedding-3-{small,large}`` via the official ``openai`` SDK.
  Lazy SDK import inside ``__init__`` so the module loads without
  ``openai`` installed; the import only fires when constructing
  without ``client=``.

Batching
--------

``embed_batch`` on the API-backed embedders never sends one giant
request: the texts are split into provider-safe batches (OpenAI: at
most 2048 inputs and ~300k tokens per request; Voyage: 128 inputs;
Cohere: 96) and the batches run with bounded concurrency
(``max_concurrency``, default 4). Pass ``batch_size=`` to use smaller
batches. Results always come back in input order.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import warnings
from collections.abc import Awaitable, Callable
from typing import Any

import anyio

DEFAULT_HASH_DIMENSIONS = 384

#: How many embedding requests one ``embed_batch`` call runs at once.
DEFAULT_MAX_CONCURRENCY = 4

#: Provider limits on the number of inputs in one embedding request.
OPENAI_MAX_BATCH = 2048
VOYAGE_MAX_BATCH = 128
COHERE_MAX_BATCH = 96

#: OpenAI caps one embeddings request at 300,000 tokens in total. A
#: token is at least one UTF-8 byte, so a byte budget of the same size
#: can never exceed it (and costs only a few extra requests for
#: ordinary text, where a token is ~4 bytes).
OPENAI_MAX_BATCH_BYTES = 300_000
#: Voyage's smallest per-request budget is 120,000 tokens
#: (``voyage-3-large`` / ``voyage-code-3``); same byte-for-token bound.
VOYAGE_MAX_BATCH_BYTES = 120_000


def _resolve_batch_size(batch_size: int | None, provider_max: int) -> int:
    if batch_size is None:
        return provider_max
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    if batch_size > provider_max:
        raise ValueError(
            f"batch_size={batch_size} exceeds the provider limit of "
            f"{provider_max} inputs per request"
        )
    return batch_size


def _resolve_concurrency(max_concurrency: int) -> int:
    if (
        isinstance(max_concurrency, bool)
        or not isinstance(max_concurrency, int)
        or max_concurrency < 1
    ):
        raise ValueError(
            f"max_concurrency must be a positive integer, got {max_concurrency!r}"
        )
    return max_concurrency


def split_batches(
    texts: list[str],
    *,
    batch_size: int,
    max_batch_bytes: int | None = None,
) -> list[list[str]]:
    """Split ``texts`` into consecutive batches of at most
    ``batch_size`` items and (when given) at most ``max_batch_bytes``
    UTF-8 bytes. A single text over the byte budget gets a batch of
    its own — the provider decides whether it is too long."""
    batches: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for text in texts:
        size = len(text.encode("utf-8")) if max_batch_bytes is not None else 0
        if current and (
            len(current) >= batch_size
            or (
                max_batch_bytes is not None
                and current_bytes + size > max_batch_bytes
            )
        ):
            batches.append(current)
            current, current_bytes = [], 0
        current.append(text)
        current_bytes += size
    if current:
        batches.append(current)
    return batches


async def embed_in_batches(
    texts: list[str],
    embed_one_batch: Callable[[list[str]], Awaitable[list[list[float]]]],
    *,
    batch_size: int,
    max_batch_bytes: int | None = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> list[list[float]]:
    """Embed ``texts`` through ``embed_one_batch`` in provider-safe
    batches, at most ``max_concurrency`` requests at a time, and
    return the vectors in input order. A batch that comes back with
    the wrong number of vectors raises instead of mis-pairing them."""
    if not texts:
        return []
    batches = split_batches(
        texts, batch_size=batch_size, max_batch_bytes=max_batch_bytes
    )
    results: list[list[list[float]] | None] = [None] * len(batches)

    async def _run(index: int, batch: list[str]) -> None:
        vectors = await embed_one_batch(batch)
        if len(vectors) != len(batch):
            raise ValueError(
                f"embedding provider returned {len(vectors)} vectors "
                f"for a batch of {len(batch)} texts"
            )
        results[index] = vectors

    if len(batches) == 1:
        await _run(0, batches[0])
    else:
        limiter = anyio.Semaphore(max_concurrency)

        async def _bounded(index: int, batch: list[str]) -> None:
            async with limiter:
                await _run(index, batch)

        try:
            async with anyio.create_task_group() as tg:
                for index, batch in enumerate(batches):
                    tg.start_soon(_bounded, index, batch)
        except BaseExceptionGroup as group:
            # Raise what a single request would have raised (a rate
            # limit, an auth error…), not a group wrapping it, so
            # callers' ``except`` clauses and retries keep working.
            error: BaseException = group
            while isinstance(error, BaseExceptionGroup):
                error = error.exceptions[0]
            raise error from None
    return [vec for batch_vecs in results if batch_vecs is not None for vec in batch_vecs]

# Process-wide flag — the HashEmbedder-fallback warning fires once no
# matter how many persistent backends get constructed without an
# embedder (mirrors the one-shot notice pattern in auto_extract.py).
_HASH_FALLBACK_WARNED = False


def warn_hash_embedder_fallback(backend: str, *, stacklevel: int = 3) -> None:
    """One-shot warning when a persistent / vector-capable memory
    backend silently falls back to :class:`HashEmbedder`.

    The HashEmbedder is deterministic but NOT semantic — recall only
    ranks exact-text matches well, so users who expected "semantic
    memory" from e.g. ``PostgresMemory`` get silently degraded
    results. Warn loudly, once per process. Fires both when a backend
    is constructed without an embedder and when the ``memory=``
    resolver picks the hash embedder because nothing else is
    configured.
    """
    global _HASH_FALLBACK_WARNED
    if _HASH_FALLBACK_WARNED:
        return
    _HASH_FALLBACK_WARNED = True
    warnings.warn(
        f"{backend} was constructed without an embedder and is "
        "falling back to HashEmbedder, which is deterministic but "
        "NOT semantic — vector recall will only match near-identical "
        "text. Pass a real embedder for semantic recall, e.g. "
        "embedder=OpenAIEmbedder() (or VoyageEmbedder / "
        "CohereEmbedder from loomflow.memory), or set "
        "LOOMFLOW_EMBEDDER=hash to choose the hash embedder on "
        "purpose. Note that a store keeps the embedder it was first "
        "written with: switching later needs a fresh store.",
        UserWarning,
        stacklevel=stacklevel,
    )


class HashEmbedder:
    """Deterministic SHA256-seeded unit vectors.

    Each text gets a fresh ``random.Random`` seeded by the SHA256 of
    its UTF-8 bytes, then samples ``dimensions`` Gaussian values and
    L2-normalises the result. Same text always produces the same
    vector; different texts produce well-distributed vectors with
    cosine distances that correlate with literal text equality (not
    semantic similarity).

    Use this in tests (fast, no network) and as a default for
    in-memory backends that need *some* vector but don't need real
    semantic recall.
    """

    def __init__(self, dimensions: int = DEFAULT_HASH_DIMENSIONS) -> None:
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
        self.name: str = f"hash-embedder-{dimensions}"
        self.dimensions: int = dimensions

    async def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        rng = random.Random(digest)
        vec = [rng.gauss(0.0, 1.0) for _ in range(self.dimensions)]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= 0.0:
            return vec
        return [v / norm for v in vec]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Per-text RNG seed makes batch-vs-single equivalent.
        return [await self.embed(t) for t in texts]


class OpenAIEmbedder:
    """Embeddings via OpenAI's ``embeddings.create`` API.

    Dimensions are fixed by the model:

    * ``text-embedding-3-small`` -> 1536
    * ``text-embedding-3-large`` -> 3072
    * ``text-embedding-ada-002`` -> 1536

    Pass ``dimensions=`` only for ``text-embedding-3-*`` models, which
    support the ``dimensions`` parameter for projection.

    ``embed_batch`` splits its input into requests of at most
    ``batch_size`` texts (default and maximum 2048, OpenAI's limit)
    and ~300k tokens, run ``max_concurrency`` at a time.
    """

    _DEFAULT_DIMS: dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        dimensions: int | None = None,
        client: Any | None = None,
        api_key: str | None = None,
        batch_size: int | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        self.name: str = model
        self.dimensions: int = dimensions or self._DEFAULT_DIMS.get(model, 1536)
        self._explicit_dimensions = dimensions
        self.batch_size: int = _resolve_batch_size(batch_size, OPENAI_MAX_BATCH)
        self.max_concurrency: int = _resolve_concurrency(max_concurrency)

        if client is not None:
            self._client = client
        else:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "OpenAI SDK not installed. "
                    "Install with: pip install 'loomflow[openai]'"
                ) from exc
            self._client = AsyncOpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            )

    async def embed(self, text: str) -> list[float]:
        kwargs: dict[str, Any] = {"model": self.name, "input": text}
        if self._explicit_dimensions is not None:
            kwargs["dimensions"] = self._explicit_dimensions
        result = await self._client.embeddings.create(**kwargs)
        embedding = result.data[0].embedding
        return list(embedding)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await embed_in_batches(
            texts,
            self._embed_one_batch,
            batch_size=self.batch_size,
            max_batch_bytes=OPENAI_MAX_BATCH_BYTES,
            max_concurrency=self.max_concurrency,
        )

    async def _embed_one_batch(self, texts: list[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {"model": self.name, "input": texts}
        if self._explicit_dimensions is not None:
            kwargs["dimensions"] = self._explicit_dimensions
        result = await self._client.embeddings.create(**kwargs)
        # OpenAI tags each embedding with the index of its input;
        # sort on it rather than trusting the response order.
        items = list(result.data)
        if all(isinstance(getattr(item, "index", None), int) for item in items):
            items.sort(key=lambda item: item.index)
        return [list(item.embedding) for item in items]


# ---------------------------------------------------------------------------
# Voyage AI
# ---------------------------------------------------------------------------


class VoyageEmbedder:
    """Embeddings via Voyage AI's ``voyageai`` SDK.

    Models and dimensions:

    * ``voyage-3`` / ``voyage-3-large`` / ``voyage-code-3`` -> 1024
    * ``voyage-3-lite`` -> 512

    ``input_type`` controls how Voyage encodes the text:

    * ``"document"`` (default) — for corpus / fact-store entries
    * ``"query"`` — for retrieval queries

    Pass an explicit ``input_type=`` if your embedder is dedicated to
    one role; for the agent loop's mixed use (we embed both stored
    triples and recall queries through the same embedder), the
    ``"document"`` default is the safer choice.
    """

    _DEFAULT_DIMS: dict[str, int] = {
        "voyage-3": 1024,
        "voyage-3-large": 1024,
        "voyage-code-3": 1024,
        "voyage-3-lite": 512,
    }

    def __init__(
        self,
        model: str = "voyage-3",
        *,
        client: Any | None = None,
        api_key: str | None = None,
        input_type: str = "document",
        batch_size: int | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        self.name: str = model
        self.dimensions: int = self._DEFAULT_DIMS.get(model, 1024)
        self._input_type = input_type
        # Voyage accepts up to 1000 texts per request but with a
        # per-model token budget; 128 keeps typical chunks inside it.
        self.batch_size: int = _resolve_batch_size(batch_size, VOYAGE_MAX_BATCH)
        self.max_concurrency: int = _resolve_concurrency(max_concurrency)

        if client is not None:
            self._client = client
        else:
            try:
                import voyageai  # type: ignore[import-not-found, import-untyped]
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "voyageai is not installed. "
                    "Install with: pip install 'loomflow[voyage]'"
                ) from exc
            # voyageai ships py.typed in newer releases but doesn't
            # re-export AsyncClient from its package __init__. The
            # class exists at runtime (defined in voyageai.client_async
            # and bound on the package). Locally without the stubs the
            # whole module is Any and this never fires; CI with the
            # stubs sees attr-defined — hence the inline ignore.
            # Paired with disable_error_code = ["unused-ignore"] in
            # pyproject so the ignore is a no-op in stub-less envs.
            self._client = voyageai.AsyncClient(  # type: ignore[attr-defined]
                api_key=api_key or os.environ.get("VOYAGE_API_KEY"),
            )

    async def embed(self, text: str) -> list[float]:
        result = await self._client.embed(
            texts=[text],
            model=self.name,
            input_type=self._input_type,
        )
        return list(result.embeddings[0])

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await embed_in_batches(
            texts,
            self._embed_one_batch,
            batch_size=self.batch_size,
            max_batch_bytes=VOYAGE_MAX_BATCH_BYTES,
            max_concurrency=self.max_concurrency,
        )

    async def _embed_one_batch(self, texts: list[str]) -> list[list[float]]:
        result = await self._client.embed(
            texts=texts,
            model=self.name,
            input_type=self._input_type,
        )
        return [list(e) for e in result.embeddings]


# ---------------------------------------------------------------------------
# Cohere
# ---------------------------------------------------------------------------


class CohereEmbedder:
    """Embeddings via Cohere's ``cohere`` SDK.

    Models and dimensions:

    * ``embed-english-v3.0`` / ``embed-multilingual-v3.0`` -> 1024
    * ``embed-english-light-v3.0`` / ``embed-multilingual-light-v3.0`` -> 384

    ``input_type`` is required by Cohere v3 models:

    * ``"search_document"`` (default) — corpus / fact-store entries
    * ``"search_query"`` — retrieval queries
    * ``"classification"`` / ``"clustering"`` for non-retrieval uses
    """

    _DEFAULT_DIMS: dict[str, int] = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-multilingual-light-v3.0": 384,
    }

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        *,
        client: Any | None = None,
        api_key: str | None = None,
        input_type: str = "search_document",
        batch_size: int | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        self.name: str = model
        self.dimensions: int = self._DEFAULT_DIMS.get(model, 1024)
        self._input_type = input_type
        # Cohere's embed endpoint takes at most 96 texts per call.
        self.batch_size: int = _resolve_batch_size(batch_size, COHERE_MAX_BATCH)
        self.max_concurrency: int = _resolve_concurrency(max_concurrency)

        if client is not None:
            self._client = client
        else:
            try:
                import cohere  # type: ignore[import-not-found, import-untyped]
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "cohere is not installed. "
                    "Install with: pip install 'loomflow[cohere]'"
                ) from exc
            self._client = cohere.AsyncClient(
                api_key=api_key or os.environ.get("COHERE_API_KEY"),
            )

    async def embed(self, text: str) -> list[float]:
        result = await self._client.embed(
            texts=[text],
            model=self.name,
            input_type=self._input_type,
            embedding_types=["float"],
        )
        return list(result.embeddings.float[0])

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await embed_in_batches(
            texts,
            self._embed_one_batch,
            batch_size=self.batch_size,
            max_concurrency=self.max_concurrency,
        )

    async def _embed_one_batch(self, texts: list[str]) -> list[list[float]]:
        result = await self._client.embed(
            texts=texts,
            model=self.name,
            input_type=self._input_type,
            embedding_types=["float"],
        )
        return [list(e) for e in result.embeddings.float]
