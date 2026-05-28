"""``memory=`` resolver tests.

Covers the M6 contract:

* string specs (``inmemory`` / ``sqlite:`` / ``chroma:`` /
  ``postgres://`` / ``redis://``) build the right backend
* dict specs map ``backend`` + per-backend kwargs correctly
* explicit Memory instances pass through unchanged
* unrecognised specs raise :class:`ConfigError`
* Postgres / Redis URL specs return a :class:`LazyMemory` proxy
  (no connection attempt at construction)
* :class:`SqliteMemory` end-to-end: persist, partition, rehydrate
* ``Agent(memory=...)`` accepts every form
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loomflow import Agent, ConfigError, Episode, InMemoryMemory, resolve_memory
from loomflow.memory import LazyMemory, SqliteMemory
from loomflow.memory.embedder import HashEmbedder
from loomflow.memory.resolver import _default_embedder

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# String form
# ---------------------------------------------------------------------------


def test_resolver_none_returns_inmemory_default() -> None:
    m = resolve_memory(None)
    assert isinstance(m, InMemoryMemory)


def test_resolver_inmemory_string() -> None:
    m = resolve_memory("inmemory")
    assert isinstance(m, InMemoryMemory)


def test_resolver_sqlite_bare_returns_ephemeral() -> None:
    m = resolve_memory("sqlite")
    assert isinstance(m, SqliteMemory)
    assert str(m.path) == ":memory:"


def test_resolver_sqlite_with_path(tmp_path: Path) -> None:
    db = tmp_path / "bot.db"
    m = resolve_memory(f"sqlite:{db}")
    assert isinstance(m, SqliteMemory)
    assert m.path == db


def test_resolver_sqlite_attaches_fact_store_by_default(tmp_path: Path) -> None:
    """Default for the resolver path is ``with_facts=True``: the
    semantic-recall layer is on out of the box."""
    db = tmp_path / "bot.db"
    m = resolve_memory(f"sqlite:{db}")
    assert isinstance(m, SqliteMemory)
    assert m.facts is not None


def test_resolver_postgres_url_returns_lazy_proxy() -> None:
    """Postgres URLs return a :class:`LazyMemory` — no async connect
    happens at construction. The proxy holds the DSN as its
    description so error messages still tell users which backend
    failed if the connection later breaks."""
    m = resolve_memory("postgres://user:pw@no-such-host/db")
    assert isinstance(m, LazyMemory)
    assert m.is_ready is False
    assert "no-such-host" in m.description


def test_resolver_postgresql_alias_works() -> None:
    m = resolve_memory("postgresql://user:pw@no-such-host/db")
    assert isinstance(m, LazyMemory)


def test_resolver_redis_url_returns_lazy_proxy() -> None:
    m = resolve_memory("redis://no-such-host:6379/0")
    assert isinstance(m, LazyMemory)
    assert m.is_ready is False


def test_resolver_rediss_alias_works() -> None:
    m = resolve_memory("rediss://no-such-host:6380/0")
    assert isinstance(m, LazyMemory)


def test_resolver_empty_string_raises() -> None:
    with pytest.raises(ConfigError, match="empty string"):
        resolve_memory("")


def test_resolver_unknown_scheme_raises() -> None:
    with pytest.raises(ConfigError, match="unrecognised string spec"):
        resolve_memory("mongodb://anywhere")


# ---------------------------------------------------------------------------
# Dict form
# ---------------------------------------------------------------------------


def test_resolver_dict_inmemory() -> None:
    m = resolve_memory({"backend": "inmemory"})
    assert isinstance(m, InMemoryMemory)


def test_resolver_dict_sqlite_with_path(tmp_path: Path) -> None:
    db = tmp_path / "via_dict.db"
    m = resolve_memory({"backend": "sqlite", "path": str(db)})
    assert isinstance(m, SqliteMemory)
    assert m.path == db


def test_resolver_dict_with_facts_false_skips_fact_store(
    tmp_path: Path,
) -> None:
    """Dict-form callers can disable the auto-attached fact store."""
    db = tmp_path / "no_facts.db"
    m = resolve_memory(
        {"backend": "sqlite", "path": str(db), "with_facts": False}
    )
    assert isinstance(m, SqliteMemory)
    assert m.facts is None


def test_resolver_dict_accepts_type_alias() -> None:
    """``type`` is an accepted alias for ``backend`` so YAML/TOML
    configs that use either convention both work."""
    m = resolve_memory({"type": "inmemory"})
    assert isinstance(m, InMemoryMemory)


def test_resolver_dict_missing_backend_raises() -> None:
    with pytest.raises(ConfigError, match="must include 'backend'"):
        resolve_memory({"path": "./mem.db"})  # no backend key


def test_resolver_dict_unknown_backend_raises() -> None:
    with pytest.raises(ConfigError, match="not recognised"):
        resolve_memory({"backend": "vespa"})


def test_resolver_dict_postgres_requires_url() -> None:
    with pytest.raises(ConfigError, match="requires 'url'"):
        resolve_memory({"backend": "postgres"})


def test_resolver_dict_postgres_returns_lazy() -> None:
    m = resolve_memory({"backend": "postgres", "url": "postgres://h/d"})
    assert isinstance(m, LazyMemory)


def test_resolver_dict_embedder_string() -> None:
    """Embedder string aliases resolve cleanly. ``hash`` is the
    safest to test because it requires no API key."""
    m = resolve_memory({"backend": "sqlite", "embedder": "hash"})
    assert isinstance(m, SqliteMemory)
    assert isinstance(m.embedder, HashEmbedder)


def test_resolver_dict_unknown_embedder_string_raises() -> None:
    with pytest.raises(ConfigError, match="embedder spec"):
        resolve_memory({"backend": "inmemory", "embedder": "word2vec"})


# ---------------------------------------------------------------------------
# Instance pass-through (Tier 3)
# ---------------------------------------------------------------------------


def test_resolver_passes_existing_memory_instance_through() -> None:
    """Already-constructed Memory instances are returned unchanged
    so today's ``memory=ChromaMemory.local(...)`` call sites keep
    working without modification."""
    existing = InMemoryMemory()
    assert resolve_memory(existing) is existing


# ---------------------------------------------------------------------------
# Agent integration — the resolver runs through Agent.__init__
# ---------------------------------------------------------------------------


def test_agent_accepts_string_memory_spec(tmp_path: Path) -> None:
    db = tmp_path / "agent.db"
    agent = Agent("hi", model="echo", memory=f"sqlite:{db}")
    assert isinstance(agent.memory, SqliteMemory)


def test_agent_accepts_dict_memory_spec() -> None:
    agent = Agent("hi", model="echo", memory={"backend": "inmemory"})
    assert isinstance(agent.memory, InMemoryMemory)


def test_agent_default_memory_is_inmemory() -> None:
    agent = Agent("hi", model="echo")
    assert isinstance(agent.memory, InMemoryMemory)


def test_agent_accepts_lazy_postgres_url_without_connecting() -> None:
    """Construction must stay synchronous and must NOT try to open a
    Postgres pool — even when the DSN points nowhere."""
    agent = Agent(
        "hi",
        model="echo",
        memory="postgres://user:pw@no-such-host:9999/db",
    )
    assert isinstance(agent.memory, LazyMemory)
    assert agent.memory.is_ready is False


# ---------------------------------------------------------------------------
# SqliteMemory — end-to-end backend behaviour
# ---------------------------------------------------------------------------


async def test_sqlite_memory_persists_across_reopen(tmp_path: Path) -> None:
    """The whole point of SqliteMemory is durability. Write episodes
    in one instance, close, reopen against the same file, and
    ``session_messages`` still returns them."""
    db = tmp_path / "persistent.db"
    m1 = SqliteMemory(db)
    await m1.remember(
        Episode(session_id="s1", user_id="alice", input="hi", output="hello")
    )
    await m1.remember(
        Episode(
            session_id="s1", user_id="alice",
            input="favourite colour is teal", output="noted",
        )
    )

    # Re-open against the same file — fresh instance, same data.
    m2 = SqliteMemory(db)
    msgs = await m2.session_messages("s1", user_id="alice")
    contents = [msg.content for msg in msgs]
    assert "hi" in contents
    assert "favourite colour is teal" in contents


async def test_sqlite_memory_partitions_by_user_id(tmp_path: Path) -> None:
    """Hard partition contract — alice's recall never sees bob's
    episodes, even though they live in the same .db file."""
    db = tmp_path / "shared.db"
    m = SqliteMemory(db)
    await m.remember(
        Episode(session_id="sA", user_id="alice", input="a", output="x")
    )
    await m.remember(
        Episode(session_id="sB", user_id="bob", input="b", output="y")
    )

    alice = await m.recall("", user_id="alice")
    bob = await m.recall("", user_id="bob")
    assert len(alice) == 1
    assert alice[0].input == "a"
    assert len(bob) == 1
    assert bob[0].input == "b"


async def test_sqlite_memory_session_messages_respects_session_id(
    tmp_path: Path,
) -> None:
    db = tmp_path / "sessions.db"
    m = SqliteMemory(db)
    # Two sessions for alice.
    await m.remember(Episode(session_id="a", user_id="alice", input="x", output="x'"))
    await m.remember(Episode(session_id="b", user_id="alice", input="y", output="y'"))

    sa = await m.session_messages("a", user_id="alice")
    sb = await m.session_messages("b", user_id="alice")
    assert [msg.content for msg in sa] == ["x", "x'"]
    assert [msg.content for msg in sb] == ["y", "y'"]


async def test_sqlite_memory_working_blocks_persist(tmp_path: Path) -> None:
    db = tmp_path / "blocks.db"
    m1 = SqliteMemory(db)
    await m1.update_block("preferences", "dark mode")
    await m1.append_block("preferences", "; sans-serif")

    m2 = SqliteMemory(db)
    blocks = await m2.working()
    assert len(blocks) == 1
    assert blocks[0].name == "preferences"
    assert blocks[0].content == "dark mode; sans-serif"


# ---------------------------------------------------------------------------
# LazyMemory — connection deferred
# ---------------------------------------------------------------------------


async def test_lazy_memory_does_not_connect_on_construction() -> None:
    """Constructing a LazyMemory must not call its builder. The
    connection only opens when a protocol method is awaited."""
    builder_calls = []

    async def builder() -> InMemoryMemory:
        builder_calls.append(1)
        return InMemoryMemory()

    lazy = LazyMemory(builder, description="test")
    assert builder_calls == []
    assert lazy.is_ready is False


async def test_lazy_memory_resolves_once_then_caches() -> None:
    builder_calls = []

    async def builder() -> InMemoryMemory:
        builder_calls.append(1)
        return InMemoryMemory()

    lazy = LazyMemory(builder, description="test")
    await lazy.working()  # first use: builds
    await lazy.working()  # cached
    await lazy.working()  # cached
    assert builder_calls == [1]
    assert lazy.is_ready is True


async def test_lazy_memory_wraps_builder_exception_in_memory_store_error(
) -> None:
    """When the builder raises (bad DSN, network error, etc.), the
    framework normalises it into :class:`MemoryStoreError` so callers
    don't have to catch backend-specific exceptions."""
    from loomflow.core import MemoryStoreError

    async def bad_builder() -> InMemoryMemory:
        raise RuntimeError("no DNS resolution")

    lazy = LazyMemory(bad_builder, description="postgres://no-such")
    with pytest.raises(MemoryStoreError, match="postgres://no-such"):
        await lazy.working()


def test_default_embedder_env_override_forces_hash(monkeypatch) -> None:
    # LOOMFLOW_EMBEDDER wins even when OPENAI_API_KEY is present —
    # the opt-out that stops cross-provider OpenAI calls on a
    # Claude/Gemini/local-driven run.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LOOMFLOW_EMBEDDER", "hash")
    assert isinstance(_default_embedder(), HashEmbedder)


def test_default_embedder_no_override_uses_hash_without_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LOOMFLOW_EMBEDDER", raising=False)
    assert isinstance(_default_embedder(), HashEmbedder)
