"""Memory backends.

Pick one of these and pass it to ``Agent(..., memory=...)``:

* :class:`InMemoryMemory` — naive dict-backed, no embeddings. The
  default; great for tests and tiny demos.
* :class:`VectorMemory` — in-memory with embedding-based cosine
  recall. Pure Python, no infrastructure. Scales to a few thousand
  episodes.
* :class:`ChromaMemory` — backed by Chroma (local persistent or
  in-memory client). Lazy ``chromadb`` import.
* :class:`PostgresMemory` — Postgres + pgvector with HNSW index.
  Lazy ``asyncpg`` + ``pgvector`` imports. Production-grade.
* :class:`RedisMemory` — Redis with optional RediSearch HNSW
  vector index, falling back to brute-force when RediSearch isn't
  available. Lazy ``redis`` import.

Embedders live in :mod:`jeevesagent.memory.embedder`:

* :class:`HashEmbedder` — deterministic, zero-dep, perfect for tests.
* :class:`OpenAIEmbedder` — real semantic embeddings via OpenAI.
"""

from .chroma import ChromaMemory
from .chroma_facts import ChromaFactStore
from .consolidator import Consolidator
from .embedder import HashEmbedder, OpenAIEmbedder
from .facts import FactStore, InMemoryFactStore
from .inmemory import InMemoryMemory
from .postgres import PostgresMemory
from .postgres_facts import PostgresFactStore
from .redis import RedisMemory
from .redis_facts import RedisFactStore
from .sqlite_facts import SqliteFactStore
from .vector import VectorMemory

__all__ = [
    "ChromaFactStore",
    "ChromaMemory",
    "Consolidator",
    "FactStore",
    "HashEmbedder",
    "InMemoryFactStore",
    "InMemoryMemory",
    "OpenAIEmbedder",
    "PostgresFactStore",
    "PostgresMemory",
    "RedisFactStore",
    "RedisMemory",
    "SqliteFactStore",
    "VectorMemory",
]
