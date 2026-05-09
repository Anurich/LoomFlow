"""Memory.recall_facts protocol method — every backend implements it.

After slice 19 the agent loop calls ``memory.recall_facts(...)`` directly
instead of duck-typing on ``memory.facts``. Each backend must:

* Forward to ``self.facts.recall_text`` when a fact store is wired.
* Return ``[]`` when no fact store is configured.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from loomflow import Fact, InMemoryMemory
from loomflow.memory import PostgresMemory, RedisMemory, VectorMemory

pytestmark = pytest.mark.anyio


def _fact() -> Fact:
    return Fact(
        subject="user",
        predicate="name_is",
        object="Alice",
        valid_from=datetime.now(UTC),
        recorded_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# In-memory backends always have a fact store
# ---------------------------------------------------------------------------


async def test_inmemory_memory_recall_facts_forwards_to_facts() -> None:
    memory = InMemoryMemory()
    await memory.facts.append(_fact())
    out = await memory.recall_facts("Alice", limit=5)
    assert len(out) == 1
    assert out[0].object == "Alice"


async def test_vector_memory_recall_facts_forwards_to_facts() -> None:
    memory = VectorMemory()
    await memory.facts.append(_fact())
    out = await memory.recall_facts("Alice", limit=5)
    assert len(out) == 1


# ---------------------------------------------------------------------------
# Backends with optional fact stores: empty when not wired
# ---------------------------------------------------------------------------


async def test_postgres_memory_recall_facts_empty_without_facts() -> None:
    """``PostgresMemory(pool=...)`` doesn't create a fact store unless
    ``with_facts=True`` was used. ``recall_facts`` must still work
    (returning ``[]``)."""
    memory = PostgresMemory(pool=None)
    out = await memory.recall_facts("anything")
    assert out == []


async def test_redis_memory_recall_facts_empty_without_facts() -> None:
    memory = RedisMemory(client=None)
    out = await memory.recall_facts("anything")
    assert out == []


# ---------------------------------------------------------------------------
# Chroma — covered by tests/test_chroma_memory.py / test_chroma_facts.py.
# Postgres / Redis live integration is gated on env vars and runs in
# ``test_postgres_facts.py`` / ``test_redis_facts.py``.
# ---------------------------------------------------------------------------
