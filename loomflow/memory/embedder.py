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
"""

from __future__ import annotations

import hashlib
import math
import os
import random
from typing import Any

DEFAULT_HASH_DIMENSIONS = 384


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
    ) -> None:
        self.name: str = model
        self.dimensions: int = dimensions or self._DEFAULT_DIMS.get(model, 1536)
        self._explicit_dimensions = dimensions

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
        if not texts:
            return []
        kwargs: dict[str, Any] = {"model": self.name, "input": texts}
        if self._explicit_dimensions is not None:
            kwargs["dimensions"] = self._explicit_dimensions
        result = await self._client.embeddings.create(**kwargs)
        # OpenAI returns data sorted by request order.
        return [list(item.embedding) for item in result.data]


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
    ) -> None:
        self.name: str = model
        self.dimensions: int = self._DEFAULT_DIMS.get(model, 1024)
        self._input_type = input_type

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
        if not texts:
            return []
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
    ) -> None:
        self.name: str = model
        self.dimensions: int = self._DEFAULT_DIMS.get(model, 1024)
        self._input_type = input_type

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
        if not texts:
            return []
        result = await self._client.embed(
            texts=texts,
            model=self.name,
            input_type=self._input_type,
            embedding_types=["float"],
        )
        return [list(e) for e in result.embeddings.float]
