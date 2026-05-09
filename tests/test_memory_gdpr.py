"""profile / forget / export — the M7 GDPR-and-inspection contract.

Cross-backend tests for the three new :class:`Memory` methods. Each
backend that ships in-tree gets the same correctness suite: profile
counts what's there, forget honours the partition + filter args,
export is a faithful round-trip.

Network backends (Postgres, Redis, Chroma's persistent path) test
against in-process fakes / ephemeral instances; live integration
tests run separately.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from loomflow import Episode, Fact, MemoryExport, MemoryProfile
from loomflow.memory import SqliteMemory
from loomflow.memory.facts import InMemoryFactStore
from loomflow.memory.inmemory import InMemoryMemory

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# InMemoryMemory
# ---------------------------------------------------------------------------


async def test_inmemory_profile_returns_counts_and_recent_sessions() -> None:
    mem = InMemoryMemory()
    await mem.remember(
        Episode(session_id="s1", user_id="alice", input="hi", output="hello")
    )
    await mem.remember(
        Episode(
            session_id="s1", user_id="alice",
            input="i love jazz", output="cool",
        )
    )
    await mem.remember(
        Episode(session_id="s2", user_id="alice", input="x", output="y")
    )
    # Bob's data should never bleed into alice's profile.
    await mem.remember(
        Episode(session_id="b1", user_id="bob", input="a", output="b")
    )
    await mem.facts.append(
        Fact(user_id="alice", subject="alice", predicate="likes", object="jazz")
    )

    profile = await mem.profile(user_id="alice")
    assert isinstance(profile, MemoryProfile)
    assert profile.user_id == "alice"
    assert profile.episode_count == 3
    assert profile.fact_count == 1
    # Newest-first dedup; alice has 2 distinct sessions.
    assert profile.recent_sessions[:2] == ["s2", "s1"]
    assert profile.last_seen is not None
    assert any(f.object == "jazz" for f in profile.sample_facts)


async def test_inmemory_profile_empty_user_returns_zero_counts() -> None:
    mem = InMemoryMemory()
    await mem.remember(
        Episode(session_id="s", user_id="alice", input="x", output="y")
    )
    p = await mem.profile(user_id="bob")
    assert p.episode_count == 0
    assert p.fact_count == 0
    assert p.recent_sessions == []
    assert p.last_seen is None


async def test_inmemory_forget_user_removes_episodes_and_facts() -> None:
    mem = InMemoryMemory()
    await mem.remember(
        Episode(session_id="s", user_id="alice", input="x", output="y")
    )
    await mem.facts.append(
        Fact(user_id="alice", subject="alice", predicate="lives_in", object="Berlin")
    )
    # Bob is the control — his data must NOT be touched.
    await mem.remember(
        Episode(session_id="s", user_id="bob", input="a", output="b")
    )
    await mem.facts.append(
        Fact(user_id="bob", subject="bob", predicate="lives_in", object="Tokyo")
    )

    deleted = await mem.forget(user_id="alice")
    assert deleted >= 2  # 1 episode + 1 fact

    alice_after = await mem.profile(user_id="alice")
    bob_after = await mem.profile(user_id="bob")
    assert alice_after.episode_count == 0
    assert alice_after.fact_count == 0
    assert bob_after.episode_count == 1
    assert bob_after.fact_count == 1


async def test_inmemory_forget_session_only_targets_one_thread() -> None:
    """Passing ``session_id`` narrows forget to that conversation,
    leaving the user's other sessions intact."""
    mem = InMemoryMemory()
    await mem.remember(
        Episode(session_id="keep", user_id="alice", input="k", output="v")
    )
    await mem.remember(
        Episode(session_id="drop", user_id="alice", input="x", output="y")
    )
    deleted = await mem.forget(user_id="alice", session_id="drop")
    assert deleted == 1

    profile = await mem.profile(user_id="alice")
    assert profile.episode_count == 1
    assert profile.recent_sessions == ["keep"]


async def test_inmemory_forget_before_keeps_recent_episodes() -> None:
    """The ``before`` filter erases data older than the timestamp,
    leaving more-recent rows intact. Common for retention windows."""
    mem = InMemoryMemory()
    old = datetime.now(UTC) - timedelta(days=180)
    new = datetime.now(UTC) - timedelta(days=1)
    # Manually craft timestamps — Episode.occurred_at default is now.
    await mem.remember(
        Episode(
            session_id="s", user_id="alice", input="old", output="x",
            occurred_at=old,
        )
    )
    await mem.remember(
        Episode(
            session_id="s", user_id="alice", input="new", output="y",
            occurred_at=new,
        )
    )

    cutoff = datetime.now(UTC) - timedelta(days=90)
    deleted = await mem.forget(user_id="alice", before=cutoff)
    assert deleted >= 1

    p = await mem.profile(user_id="alice")
    assert p.episode_count == 1
    # Only the recent one should remain.
    export = await mem.export(user_id="alice")
    assert export.episodes[0].input == "new"


async def test_inmemory_export_round_trips_episodes_and_facts() -> None:
    mem = InMemoryMemory()
    await mem.remember(
        Episode(
            session_id="s", user_id="alice", input="hi", output="hello"
        )
    )
    await mem.facts.append(
        Fact(user_id="alice", subject="alice", predicate="works_at", object="Acme")
    )
    export = await mem.export(user_id="alice")
    assert isinstance(export, MemoryExport)
    assert export.user_id == "alice"
    assert len(export.episodes) == 1
    assert len(export.facts) == 1
    assert export.episodes[0].input == "hi"
    assert export.facts[0].object == "Acme"
    # Serialisable for portability / GDPR DSAR responses.
    blob = export.model_dump_json()
    assert "alice" in blob


async def test_inmemory_export_partition_respected() -> None:
    """Export of one user MUST NOT include another user's data."""
    mem = InMemoryMemory()
    await mem.remember(
        Episode(session_id="s", user_id="alice", input="alice secret", output="!")
    )
    await mem.remember(
        Episode(session_id="s", user_id="bob", input="bob secret", output="!")
    )
    e_alice = await mem.export(user_id="alice")
    e_bob = await mem.export(user_id="bob")

    alice_inputs = {ep.input for ep in e_alice.episodes}
    bob_inputs = {ep.input for ep in e_bob.episodes}
    assert "alice secret" in alice_inputs
    assert "alice secret" not in bob_inputs
    assert "bob secret" in bob_inputs
    assert "bob secret" not in alice_inputs


# ---------------------------------------------------------------------------
# SqliteMemory
# ---------------------------------------------------------------------------


async def test_sqlite_profile_and_forget_persist_across_reopen(
    tmp_path: Path,
) -> None:
    """The whole point of SqliteMemory is durability — verify the
    GDPR ops also survive a close+reopen, both for what should
    remain and what shouldn't."""
    db = tmp_path / "gdpr.db"
    m1 = SqliteMemory(db)
    await m1.remember(
        Episode(session_id="s", user_id="alice", input="x", output="y")
    )
    await m1.remember(
        Episode(session_id="s", user_id="bob", input="a", output="b")
    )
    await m1.facts.append(
        Fact(user_id="alice", subject="alice", predicate="likes", object="pizza")
    )
    deleted = await m1.forget(user_id="alice")
    assert deleted >= 2

    # Re-open. Alice's data should be gone; bob's intact.
    m2 = SqliteMemory(db)
    p_alice = await m2.profile(user_id="alice")
    p_bob = await m2.profile(user_id="bob")
    assert p_alice.episode_count == 0
    assert p_alice.fact_count == 0
    assert p_bob.episode_count == 1


async def test_sqlite_export_returns_full_dump(tmp_path: Path) -> None:
    db = tmp_path / "export.db"
    m = SqliteMemory(db)
    await m.remember(
        Episode(session_id="s", user_id="alice", input="x", output="y")
    )
    await m.facts.append(
        Fact(user_id="alice", subject="alice", predicate="works_at", object="Acme")
    )
    export = await m.export(user_id="alice")
    assert export.episodes
    assert export.facts


# ---------------------------------------------------------------------------
# Anonymous bucket — None is its own user_id, partitioned from named
# users
# ---------------------------------------------------------------------------


async def test_forget_anonymous_does_not_affect_named_users() -> None:
    """``user_id=None`` is the anonymous bucket; forgetting it must
    not touch named users' data."""
    mem = InMemoryMemory(fact_store=InMemoryFactStore())
    await mem.remember(
        Episode(session_id="s", user_id=None, input="anon", output="x")
    )
    await mem.remember(
        Episode(session_id="s", user_id="alice", input="alice", output="y")
    )
    await mem.forget(user_id=None)

    anon = await mem.profile(user_id=None)
    alice = await mem.profile(user_id="alice")
    assert anon.episode_count == 0
    assert alice.episode_count == 1
