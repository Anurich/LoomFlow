"""Memory backends.

The simplest way to pick a backend is to pass a string or URL to
``Agent(memory=...)``::

    Agent(..., memory="inmemory")            # default; lost on restart
    Agent(..., memory="sqlite:./bot.db")     # persistent, no infra
    Agent(..., memory="chroma:./vectors")    # local Chroma
    Agent(..., memory="postgres://...")      # Postgres + pgvector
    Agent(..., memory="redis://...")         # Redis (+ RediSearch)

The :func:`resolve_memory` resolver handles the string parsing; it
also accepts a config dict (``{"backend": "chroma", ...}``) and
already-constructed Memory instances pass through unchanged.

Backends explicitly:

* :class:`InMemoryMemory` — dict-backed, no embeddings. Default.
* :class:`SqliteMemory` — single sqlite file. Persistent, no server.
* :class:`VectorMemory` — in-process with cosine recall. No persistence.
* :class:`ChromaMemory` — Chroma client (local persistent or in-process).
* :class:`PostgresMemory` — Postgres + pgvector + HNSW index.
* :class:`RedisMemory` — Redis (+ optional RediSearch HNSW vector index).

Embedders in :mod:`loomflow.memory.embedder`:

* :class:`HashEmbedder` — deterministic, zero-key, fine for dev / tests.
* :class:`OpenAIEmbedder` / :class:`VoyageEmbedder` /
  :class:`CohereEmbedder` — real semantic embeddings.
"""

from ._hybrid import default_recall_scored
from .auto_extract import AutoExtractMemory
from .chroma import ChromaMemory
from .chroma_facts import ChromaFactStore
from .consolidator import Consolidator
from .embedder import CohereEmbedder, HashEmbedder, OpenAIEmbedder, VoyageEmbedder
from .facts import FactStore, InMemoryFactStore
from .graph import Edge, FactGraph, Path, recall_graph
from .inmemory import InMemoryMemory
from .lazy import LazyMemory
from .postgres import PostgresMemory
from .postgres_facts import PostgresFactStore
from .redis import RedisMemory
from .redis_facts import RedisFactStore
from .resolver import resolve_memory
from .sqlite import SqliteMemory
from .sqlite_facts import SqliteFactStore
from .vector import VectorMemory
from .worker import ConsolidationWorker

__all__ = [
    "AutoExtractMemory",
    "ChromaFactStore",
    "ChromaMemory",
    "CohereEmbedder",
    "ConsolidationWorker",
    "Consolidator",
    "Edge",
    "FactGraph",
    "FactStore",
    "default_recall_scored",
    "HashEmbedder",
    "InMemoryFactStore",
    "InMemoryMemory",
    "LazyMemory",
    "OpenAIEmbedder",
    "Path",
    "PostgresFactStore",
    "PostgresMemory",
    "RedisFactStore",
    "RedisMemory",
    "SqliteFactStore",
    "SqliteMemory",
    "VectorMemory",
    "VoyageEmbedder",
    "recall_graph",
    "resolve_memory",
]
