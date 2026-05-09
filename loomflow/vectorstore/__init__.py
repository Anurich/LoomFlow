"""Vector stores for semantic search over :class:`Chunk` /
:class:`Document` objects.

Unified async interface (modeled on LangChain's ``VectorStore`` but
properly async-first and typed against our :class:`Chunk` /
:class:`Document` from :mod:`loomflow.loader`):

* :meth:`add` — embed + store chunks; returns their ids.
* :meth:`delete` — remove by id.
* :meth:`search` — top-k by cosine similarity + metadata filter.
* :meth:`search_by_vector` — same with a precomputed query vector.

Implementations:

* :class:`InMemoryVectorStore` — default; zero-deps; cosine over a
  Python list. Great for dev / tests / small corpora.
* :class:`ChromaVectorStore` — wraps ``chromadb`` for persistent
  on-disk or hosted Chroma. Lazy import.
* :class:`PostgresVectorStore` — wraps ``pgvector`` via ``asyncpg``.
  Production durable. Lazy import.
* :class:`FAISSVectorStore` — wraps ``faiss-cpu`` for fast in-memory
  ANN search over large corpora. Lazy import.

One-liner usage::

    from loomflow import HashEmbedder
    from loomflow.vectorstore import InMemoryVectorStore
    from loomflow.loader import load, MarkdownChunker

    vs = InMemoryVectorStore(embedder=HashEmbedder())

    doc = load("research.pdf")
    chunks = MarkdownChunker().split(doc.content, source=str(doc.metadata["source"]))
    await vs.add(chunks)

    results = await vs.search("what is RAG?", k=5)
    for r in results:
        print(f"{r.score:.3f}: {r.chunk.content[:100]}")

Optional dependencies::

    pip install 'loomflow[vectorstore-chroma]'
    pip install 'loomflow[vectorstore-postgres]'
    pip install 'loomflow[vectorstore-faiss]'
    pip install 'loomflow[vectorstore]'              # all of the above
"""

from .base import SearchResult, VectorStore
from .chroma import ChromaVectorStore
from .faiss import FAISSVectorStore
from .inmemory import InMemoryVectorStore
from .postgres import PostgresVectorStore

__all__ = [
    "ChromaVectorStore",
    "FAISSVectorStore",
    "InMemoryVectorStore",
    "PostgresVectorStore",
    "SearchResult",
    "VectorStore",
]
