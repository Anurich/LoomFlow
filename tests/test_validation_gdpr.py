"""Right to erasure and access: ``forget`` removes everything it should.

``AutoExtractMemory``'s background fact extraction wrote an erased
user's facts back after ``forget()`` returned, and ``forget`` left the
user's working blocks (which Reflexion lessons and other per-user
notes live in) in place.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from loomflow.core.types import Episode, MemoryBlock
from loomflow.memory import AutoExtractMemory, Consolidator, InMemoryMemory, SqliteMemory
from loomflow.memory.facts import delete_blocks
from loomflow.memory.vector import VectorMemory
from loomflow.model.scripted import ScriptedModel, ScriptedTurn

pytestmark = pytest.mark.anyio


class _SlowModel(ScriptedModel):
    async def stream(self, messages: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(0.2)  # the extraction call is still in flight
        async for chunk in super().stream(messages, **kwargs):
            yield chunk


async def test_forget_stops_background_extraction_from_restoring_facts() -> None:
    extracted = '[{"subject":"alice","predicate":"ssn","object":"123-45-6789","confidence":0.9}]'
    mem = AutoExtractMemory(
        InMemoryMemory(),
        Consolidator(model=_SlowModel([ScriptedTurn(text=extracted)] * 2)),
        background=True,
    )
    await mem.remember(Episode(session_id="s", user_id="alice", input="my ssn", output="ok"))
    await mem.remember(Episode(session_id="s", user_id="bob", input="hi", output="ok"))
    await mem.forget(user_id="alice")
    await mem.aclose()
    assert (await mem.export(user_id="alice")).facts == []
    # Other users' extractions are left to finish.
    assert (await mem.export(user_id="bob")).facts


def _chroma() -> Any:
    chroma = pytest.importorskip("loomflow.memory.chroma")
    pytest.importorskip("chromadb")
    return chroma.ChromaMemory.ephemeral(collection_name=f"gdpr_{uuid.uuid4().hex}")


BACKENDS: dict[str, Callable[[Path], Any]] = {
    "inmemory": lambda tmp: InMemoryMemory(),
    "sqlite": lambda tmp: SqliteMemory(tmp / "m.db"),
    "vector": lambda tmp: VectorMemory(),
    "chroma": lambda tmp: _chroma(),
}


@pytest.mark.parametrize("backend", sorted(BACKENDS))
async def test_forget_user_erases_their_working_blocks(backend: str, tmp_path: Path) -> None:
    mem = BACKENDS[backend](tmp_path)
    await mem.update_block("notes", "alice's account 4242", user_id="alice")
    await mem.update_block("notes", "bob's notes", user_id="bob")

    # A session-scoped forget leaves user-level blocks alone.
    await mem.forget(user_id="alice", session_id="some-session")
    assert [b.content for b in await mem.working(user_id="alice")] == ["alice's account 4242"]

    export = await mem.export(user_id="alice")
    assert [b.content for b in export.blocks] == ["alice's account 4242"]

    deleted = await mem.forget(user_id="alice")
    assert deleted >= 1
    assert await mem.working(user_id="alice") == []
    assert (await mem.export(user_id="alice")).blocks == []
    assert [b.content for b in await mem.working(user_id="bob")] == ["bob's notes"]


def test_delete_blocks_honours_before() -> None:
    now = datetime.now(UTC)
    blocks: dict[str | None, dict[str, MemoryBlock]] = {
        "alice": {
            "old": MemoryBlock(name="old", content="x", updated_at=now - timedelta(days=2)),
            "new": MemoryBlock(name="new", content="y", updated_at=now),
        }
    }
    # A naive cutoff is read as UTC instead of raising.
    cutoff = (now - timedelta(days=1)).replace(tzinfo=None)
    assert delete_blocks(blocks, user_id="alice", before=cutoff) == 1
    assert list(blocks["alice"]) == ["new"]
    assert delete_blocks(blocks, user_id="alice") == 1
    assert "alice" not in blocks
    assert delete_blocks(blocks, user_id="nobody") == 0
