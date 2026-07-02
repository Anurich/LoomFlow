"""Chroma-specific regression tests for the reviewed fixes (WSC).

Skipped automatically when ``chromadb`` isn't installed — same
convention as test_chroma_facts.py.
"""

from __future__ import annotations

import pytest

pytest.importorskip("chromadb")

import uuid  # noqa: E402
from datetime import UTC, datetime  # noqa: E402

from loomflow.core.types import Episode, Fact  # noqa: E402
from loomflow.memory._user_key import ANON_USER_ID  # noqa: E402
from loomflow.memory.chroma import ChromaMemory  # noqa: E402
from loomflow.memory.chroma_facts import ChromaFactStore  # noqa: E402
from loomflow.memory.embedder import HashEmbedder  # noqa: E402

pytestmark = pytest.mark.anyio


def _mem(**kwargs: object) -> ChromaMemory:
    return ChromaMemory.ephemeral(
        embedder=HashEmbedder(dimensions=32),
        collection_name=f"wsc_eps_{uuid.uuid4().hex}",
        facts_collection_name=f"wsc_facts_{uuid.uuid4().hex}",
        **kwargs,  # type: ignore[arg-type]
    )


def _facts_store() -> ChromaFactStore:
    return ChromaFactStore.ephemeral(
        embedder=HashEmbedder(dimensions=16),
        collection_name=f"wsc_facts_{uuid.uuid4().hex}",
    )


def _fact(
    *,
    user_id: str | None = None,
    predicate: str = "lives_in",
    object_: str = "Tokyo",
) -> Fact:
    return Fact(
        user_id=user_id,
        subject="user",
        predicate=predicate,
        object=object_,
        valid_from=datetime.now(UTC),
        recorded_at=datetime.now(UTC),
    )


def _ep(text: str, *, user_id: str | None = None) -> Episode:
    return Episode(
        session_id="s",
        user_id=user_id,
        input=text,
        output="",
        occurred_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Fix 4 — recall_scored plumbs real distances
# ---------------------------------------------------------------------------


async def test_chroma_recall_scored_returns_distance_scores() -> None:
    mem = _mem()
    await mem.remember(_ep("apples are red", user_id="alice"))
    await mem.remember(_ep("dogs bark loudly", user_id="alice"))

    matches = await mem.recall_scored(
        "apples are red", user_id="alice", limit=2
    )
    assert len(matches) == 2
    top = matches[0]
    assert top.episode.input == "apples are red"
    # Exact-text match → distance 0 → similarity 1/(1+0) == 1.0.
    assert top.vector_score == pytest.approx(1.0, abs=1e-4)
    assert matches[1].vector_score is not None
    assert matches[1].vector_score < top.vector_score


async def test_chroma_recall_scored_empty_query_neutral() -> None:
    mem = _mem()
    await mem.remember(_ep("hello", user_id="alice"))
    matches = await mem.recall_scored("", user_id="alice", limit=5)
    assert matches
    assert all(m.score == 1.0 and m.vector_score is None for m in matches)


# ---------------------------------------------------------------------------
# Fix 2 — sentinel writes, legacy "" rows still visible
# ---------------------------------------------------------------------------


async def test_chroma_fact_anonymous_bucket_uses_sentinel() -> None:
    store = _facts_store()
    fid = await store.append(_fact(user_id=None))
    coll = await store._get_collection()
    got = coll.get(ids=[fid], include=["metadatas"])
    assert got["metadatas"][0]["user_id"] == ANON_USER_ID

    # Round-trip: anonymous queries see it, named users don't.
    anon = await store.query(user_id=None)
    assert [f.id for f in anon] == [fid]
    assert await store.query(user_id="alice") == []


async def test_chroma_fact_legacy_empty_string_rows_still_match() -> None:
    store = _facts_store()
    fid = await store.append(_fact(user_id=None, object_="Paris"))
    # Simulate a legacy row written with the old "" encoding.
    coll = await store._get_collection()
    coll.upsert(
        ids=["legacy1"],
        embeddings=[[0.0] * 16],
        documents=["user prefers tea"],
        metadatas=[
            {
                "user_id": "",
                "subject": "user",
                "predicate": "prefers",
                "object": "tea",
                "confidence": 1.0,
                "valid_from_ts": 1700000000.0,
                "valid_until_ts": 0.0,
                "currently_valid": True,
                "recorded_at_ts": 1700000000.0,
                "sources": "[]",
            }
        ],
    )
    anon = await store.query(user_id=None, limit=10)
    assert {f.id for f in anon} == {fid, "legacy1"}
    # Decoded back to the anonymous bucket, not user "".
    assert all(f.user_id is None for f in anon)


async def test_chroma_episode_anonymous_roundtrip() -> None:
    mem = _mem()
    eid = await mem.remember(_ep("anon episode", user_id=None))
    out = await mem.recall("anon episode", user_id=None, limit=1)
    assert [e.id for e in out] == [eid]
    assert out[0].user_id is None
    # Named-user recall never sees the anonymous bucket.
    assert await mem.recall("anon episode", user_id="alice", limit=1) == []


# ---------------------------------------------------------------------------
# Fix 6 — ChromaFactStore.delete / count + forget delegation
# ---------------------------------------------------------------------------


async def test_chroma_fact_store_delete_and_count() -> None:
    store = _facts_store()
    await store.append(_fact(user_id="alice", predicate="p1"))
    await store.append(_fact(user_id="alice", predicate="p2"))
    await store.append(_fact(user_id="bob", predicate="p1"))

    assert await store.count(user_id="alice") == 2
    assert await store.count(user_id="bob") == 1
    assert await store.delete(user_id="alice") == 2
    assert await store.count(user_id="alice") == 0
    assert await store.count(user_id="bob") == 1


async def test_chroma_memory_forget_deletes_facts_via_public_surface() -> None:
    mem = _mem(with_facts=True)
    assert mem.facts is not None
    await mem.remember(_ep("alice episode", user_id="alice"))
    await mem.facts.append(_fact(user_id="alice"))
    await mem.facts.append(_fact(user_id="bob"))

    deleted = await mem.forget(user_id="alice")
    assert deleted == 2  # 1 episode + 1 fact
    assert await mem.facts.count(user_id="alice") == 0
    assert await mem.facts.count(user_id="bob") == 1

    profile = await mem.profile(user_id="bob")
    assert profile.fact_count == 1
