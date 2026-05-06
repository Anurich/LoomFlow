"""VectorMemory tests — embedding-based recall, ranking, time filters."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from jeevesagent import Agent, VectorMemory
from jeevesagent.core.types import Episode
from jeevesagent.memory.embedder import HashEmbedder

pytestmark = pytest.mark.anyio


def _ep(text_in: str, text_out: str, *, when: datetime | None = None) -> Episode:
    return Episode(
        session_id="s",
        input=text_in,
        output=text_out,
        occurred_at=when or datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


async def test_remember_attaches_embedding_when_missing() -> None:
    mem = VectorMemory()
    eid = await mem.remember(_ep("alpha", "beta"))
    snap = mem.snapshot()
    assert eid in snap["episodes"]
    stored = snap["episodes"][eid]
    assert stored["embedding"] is not None
    assert len(stored["embedding"]) == HashEmbedder().dimensions


async def test_recall_returns_query_result_for_exact_match() -> None:
    mem = VectorMemory()
    # output="" makes _embedding_text equal input, so the query
    # text hashes to exactly the stored vector and scores 1.0.
    await mem.remember(_ep("apples are red", ""))
    await mem.remember(_ep("the sky is blue", ""))
    await mem.remember(_ep("dogs bark loudly", ""))

    out = await mem.recall("apples are red", limit=1)
    assert len(out) == 1
    assert out[0].input == "apples are red"


async def test_empty_query_falls_back_to_recency() -> None:
    base = datetime.now(UTC)
    mem = VectorMemory()
    await mem.remember(_ep("oldest", "x", when=base - timedelta(hours=2)))
    await mem.remember(_ep("middle", "x", when=base - timedelta(hours=1)))
    await mem.remember(_ep("newest", "x", when=base))

    out = await mem.recall("", limit=2)
    assert [e.input for e in out] == ["newest", "middle"]


async def test_time_range_filter_excludes_outside_episodes() -> None:
    base = datetime.now(UTC)
    mem = VectorMemory()
    await mem.remember(_ep("very old", "x", when=base - timedelta(days=10)))
    await mem.remember(_ep("recent", "x", when=base - timedelta(hours=1)))

    window = (base - timedelta(hours=2), base)
    out = await mem.recall("recent", limit=5, time_range=window)
    assert len(out) == 1
    assert out[0].input == "recent"


async def test_max_episodes_evicts_oldest() -> None:
    base = datetime.now(UTC)
    mem = VectorMemory(max_episodes=2)
    await mem.remember(_ep("first", "a", when=base - timedelta(hours=2)))
    await mem.remember(_ep("second", "b", when=base - timedelta(hours=1)))
    await mem.remember(_ep("third", "c", when=base))

    snap = mem.snapshot()
    inputs = {ep["input"] for ep in snap["episodes"].values()}
    assert "first" not in inputs  # evicted
    assert {"second", "third"}.issubset(inputs)


# ---------------------------------------------------------------------------
# Working blocks (parity with InMemoryMemory)
# ---------------------------------------------------------------------------


async def test_update_and_append_block() -> None:
    mem = VectorMemory()
    await mem.update_block("user", "alice")
    blocks = await mem.working()
    assert blocks[0].name == "user"
    assert blocks[0].content == "alice"

    await mem.append_block("user", " (admin)")
    blocks = await mem.working()
    assert blocks[0].content == "alice (admin)"


# ---------------------------------------------------------------------------
# End-to-end with Agent
# ---------------------------------------------------------------------------


async def test_agent_with_vector_memory_persists_and_recalls() -> None:
    mem = VectorMemory()
    agent = Agent("you are helpful", model="echo", memory=mem)

    await agent.run("first prompt about apples")
    await agent.run("second prompt about sky")
    await agent.run("third prompt unrelated")

    # Three episodes were persisted via the runtime.step path.
    snap = mem.snapshot()
    assert len(snap["episodes"]) == 3

    # Recall picks the apple episode for an apple-themed query.
    matches = await mem.recall("first prompt about apples", limit=1)
    assert len(matches) == 1
    assert "apples" in matches[0].input
