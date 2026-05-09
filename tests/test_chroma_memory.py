"""ChromaMemory end-to-end tests using an in-process EphemeralClient.

Skipped automatically if ``chromadb`` isn't installed.
"""

from __future__ import annotations

import pytest

# Skip the whole module if chromadb isn't around.
pytest.importorskip("chromadb")

import uuid  # noqa: E402
from datetime import UTC, datetime, timedelta  # noqa: E402

from loomflow import Agent  # noqa: E402
from loomflow.core.types import Episode  # noqa: E402
from loomflow.memory import ChromaMemory  # noqa: E402

pytestmark = pytest.mark.anyio


def _ep(text_in: str, text_out: str, *, when: datetime | None = None) -> Episode:
    return Episode(
        session_id="s",
        input=text_in,
        output=text_out,
        occurred_at=when or datetime.now(UTC),
    )


def _fresh_mem() -> ChromaMemory:
    """Chroma's ``EphemeralClient`` shares collection state across
    instances inside a single process. Each test gets its own
    UUID-suffixed collection so they stay isolated."""
    return ChromaMemory.ephemeral(collection_name=f"jeeves_test_{uuid.uuid4().hex}")


async def test_chroma_remember_and_recall_roundtrip() -> None:
    mem = _fresh_mem()
    eid = await mem.remember(_ep("apples are red", "fruit"))
    assert eid

    out = await mem.recall("apples are red", limit=1)
    assert len(out) == 1
    assert "apples" in out[0].input


async def test_chroma_recall_recent_when_query_blank() -> None:
    base = datetime.now(UTC)
    mem = _fresh_mem()
    await mem.remember(_ep("alpha", "a", when=base - timedelta(hours=2)))
    await mem.remember(_ep("beta", "b", when=base))

    out = await mem.recall("", limit=2)
    inputs = {e.input for e in out}
    assert {"alpha", "beta"}.issubset(inputs)


async def test_chroma_working_blocks_are_in_process_only() -> None:
    mem = _fresh_mem()
    await mem.update_block("user", "alice")
    blocks = await mem.working()
    assert len(blocks) == 1
    assert blocks[0].content == "alice"


async def test_agent_with_chroma_memory_runs_end_to_end() -> None:
    mem = _fresh_mem()
    agent = Agent("be terse", model="echo", memory=mem)

    r1 = await agent.run("apples?")
    assert r1.output

    r2 = await agent.run("apples?")
    assert r2.output

    out = await mem.recall("apples", limit=2)
    # Both runs persisted, and recall finds them.
    assert len(out) >= 1
