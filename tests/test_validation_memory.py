"""Memory and RAG behave like the references they cite.

Bi-temporal facts followed arrival order instead of valid time (a
backfilled historical fact became "current" and closed the real current
one before it began), across all five fact stores. Postgres runs need
``JEEVES_TEST_PG_DSN``; the others run in-process.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from loomflow.core.types import Fact
from loomflow.memory.facts import InMemoryFactStore, plan_supersession
from loomflow.memory.sqlite_facts import SqliteFactStore

pytestmark = pytest.mark.anyio

Y2020 = datetime(2020, 1, 1, tzinfo=UTC)
Y2022 = datetime(2022, 1, 1, tzinfo=UTC)
Y2024 = datetime(2024, 1, 1, tzinfo=UTC)
NOW = datetime(2026, 6, 1, tzinfo=UTC)


def _lives_in(city: str, since: datetime, until: datetime | None = None) -> Fact:
    return Fact(
        user_id="alice", subject="alice", predicate="lives_in", object=city,
        valid_from=since, valid_until=until,
    )


async def _chroma(tmp: Path) -> Any:
    pytest.importorskip("chromadb")
    from loomflow.memory.chroma_facts import ChromaFactStore

    return ChromaFactStore.ephemeral(collection_name=f"facts_{uuid.uuid4().hex}")


async def _redis(tmp: Path) -> Any:
    from loomflow.memory.redis_facts import RedisFactStore
    from tests.test_redis_facts import _FakeRedis

    return RedisFactStore(_FakeRedis())


async def _postgres(tmp: Path) -> Any:
    dsn = os.environ.get("JEEVES_TEST_PG_DSN")
    if not dsn:
        pytest.skip("JEEVES_TEST_PG_DSN not set")
    asyncpg = pytest.importorskip("asyncpg")
    from loomflow.memory.postgres_facts import PostgresFactStore

    # A schema of its own, so the shared ``facts`` table of other
    # live tests (created with a different vector size) isn't touched.
    schema = f"loomtest_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(dsn)
    await admin.execute(f"CREATE SCHEMA {schema}")
    await admin.close()
    pool = await asyncpg.create_pool(dsn, server_settings={"search_path": f"{schema},public"})
    store = PostgresFactStore(pool=pool)
    await store.init_schema()
    return store


async def _inmemory(tmp: Path) -> Any:
    return InMemoryFactStore()


async def _sqlite(tmp: Path) -> Any:
    return SqliteFactStore(tmp / "facts.db")


STORES: dict[str, Callable[[Path], Awaitable[Any]]] = {
    "inmemory": _inmemory,
    "sqlite": _sqlite,
    "chroma": _chroma,
    "redis": _redis,
    "postgres": _postgres,
}


async def _current(store: Any) -> list[str]:
    facts = await store.query(subject="alice", predicate="lives_in", valid_at=NOW, user_id="alice")
    return [f.object for f in facts]


async def _all(store: Any) -> dict[str, Fact]:
    facts = await store.query(subject="alice", predicate="lives_in", user_id="alice", limit=50)
    return {f.object: f for f in facts}


@pytest.mark.parametrize("backend", sorted(STORES))
async def test_supersession_follows_valid_time_not_arrival_order(
    backend: str, tmp_path: Path
) -> None:
    store = await STORES[backend](tmp_path)
    # Learned in order: current first, the history afterwards.
    await store.append(_lives_in("Berlin", Y2024))
    await store.append(_lives_in("Paris", Y2020))
    assert await _current(store) == ["Berlin"]
    facts = await _all(store)
    assert facts["Berlin"].valid_until is None
    assert facts["Paris"].valid_until == Y2024  # the backfill ends where Berlin begins
    # A claim in between closes the earlier one and is closed by the later.
    await store.append(_lives_in("Rome", Y2022))
    facts = await _all(store)
    assert facts["Paris"].valid_until == Y2022
    assert facts["Rome"].valid_until == Y2024
    assert facts["Berlin"].valid_until is None
    for fact in facts.values():
        assert fact.valid_until is None or fact.valid_until >= fact.valid_from


def test_plan_supersession_never_closes_before_opening() -> None:
    current = _lives_in("Berlin", Y2024)
    closed_history = _lives_in("Paris", Y2020, until=Y2022)
    closes, until = plan_supersession(_lives_in("Rome", Y2022), [current, closed_history])
    # Paris already ended when Rome began (no overlap); Rome ends at Berlin.
    assert closes == {} and until == Y2024


async def test_naive_datetimes_are_utc_everywhere(tmp_path: Path) -> None:
    for store in (InMemoryFactStore(), SqliteFactStore(tmp_path / "f.db")):
        await store.append(_lives_in("Berlin", Y2024))
        naive_now = NOW.replace(tzinfo=None)
        found = await store.query(subject="alice", valid_at=naive_now, user_id="alice")
        assert [f.object for f in found] == ["Berlin"]
    naive = Fact(subject="s", predicate="p", object="o", valid_from=datetime(2024, 1, 1))
    assert naive.valid_from.tzinfo is UTC



class _SearchableRedis:
    """A fake redis that answers FT.SEARCH like RediSearch (doc id = key)."""

    def __init__(self) -> None:
        self.h: dict[bytes, dict[bytes, Any]] = {}

    async def hset(self, key: Any, mapping: dict[Any, Any]) -> None:
        k = key.encode() if isinstance(key, str) else key
        self.h.setdefault(k, {}).update(
            {(f.encode() if isinstance(f, str) else f): v for f, v in mapping.items()}
        )

    async def hgetall(self, key: bytes) -> dict[bytes, Any]:
        return dict(self.h.get(key, {}))

    async def scan(self, cursor: int = 0, match: bytes = b"*") -> tuple[int, list[bytes]]:
        prefix = match.rstrip(b"*")
        return 0, [k for k in self.h if k.startswith(prefix)]

    async def execute_command(self, *args: Any) -> Any:
        import re

        from loomflow.memory._embedding_util import cosine, unpack_float32

        if args[0] == "FT.CREATE":
            return b"OK"
        query = args[2]
        tag = re.search(r"@user_id:\{(.*?)\}\)", query).group(1).replace("\\", "")  # type: ignore[union-attr]
        k = int(re.search(r"KNN (\d+)", query).group(1))  # type: ignore[union-attr]
        vec = unpack_float32(args[6])
        rows = sorted(
            (1 - cosine(vec, unpack_float32(d[b"embedding"])), key, d)
            for key, d in self.h.items()
            if key.startswith(b"jeeves:episode:") and d.get(b"user_id", b"").decode() == tag
        )[:k]
        out: list[Any] = [len(rows)]
        for dist, key, d in rows:
            out += [key, [b"session_id", d[b"session_id"], b"user_id", d[b"user_id"],
                          b"occurred_at", d[b"occurred_at"], b"input", d[b"input"],
                          b"output", d[b"output"], b"score", str(dist).encode()]]
        return out


async def test_redis_recall_returns_stored_ids_once() -> None:
    from loomflow.core.types import Episode
    from loomflow.memory import HashEmbedder, RedisMemory

    memory = RedisMemory(_SearchableRedis(), embedder=HashEmbedder())
    ids = {
        await memory.remember(Episode(session_id="s", input=f"note {i}", output=""))
        for i in range(3)
    }
    recalled = [e.id for e in await memory.recall("note 1", limit=5)]
    # Ids like "episode:ep_..." (prefix cut at the first colon) used to
    # come back, so the anonymous-bucket union listed every hit twice.
    assert sorted(recalled) == sorted(ids)


async def test_pgvector_filtered_and_large_k_searches_return_every_match() -> None:
    dsn = os.environ.get("JEEVES_TEST_PG_DSN")
    if not dsn:
        pytest.skip("JEEVES_TEST_PG_DSN not set")
    pytest.importorskip("asyncpg")
    from loomflow.loader.base import Chunk
    from loomflow.memory import HashEmbedder
    from loomflow.vectorstore import InMemoryVectorStore
    from loomflow.vectorstore.postgres import PostgresVectorStore

    embedder = HashEmbedder(dimensions=64)
    table = f"vec_{uuid.uuid4().hex[:8]}"
    store = PostgresVectorStore(embedder=embedder, dsn=dsn, table=table)
    await store.init_schema(64)
    chunks = [
        Chunk(content=f"document number {i} about topic {i % 17}",
              metadata={"tenant": "rare" if i % 100 == 0 else "common", **({"tag": "x"} if i % 3 == 0 else {})})
        for i in range(600)
    ]
    await store.add(chunks)
    try:
        # HNSW scans stopped at ef_search (40) candidates, so a rare
        # tenant's rows behind them were lost and k > 40 was capped.
        rare = await store.search("document about topic", k=6, filter={"tenant": "rare"})
        assert len(rare) == 6
        many = await store.search("document about topic", k=80)
        assert len(many) == 80
        # $ne / $not treat a missing key as "not equal", like every store.
        reference = await InMemoryVectorStore.from_chunks(chunks[:30], embedder=embedder)
        small = PostgresVectorStore(embedder=embedder, dsn=dsn, table=table + "_s")
        await small.init_schema(64)
        await small.add(chunks[:30])
        for flt in ({"tag": {"$ne": "x"}}, {"$not": {"tag": "x"}}, {"tag": {"$exists": False}}):
            got = {r.chunk.content for r in await small.search("doc", k=30, filter=flt)}
            want = {r.chunk.content for r in await reference.search("doc", k=30, filter=flt)}
            assert got == want, flt
    finally:
        import asyncpg

        conn = await asyncpg.connect(dsn)
        await conn.execute(f"DROP TABLE IF EXISTS {table}, {table}_s")
        await conn.close()


async def test_postgres_memory_url_creates_its_schema() -> None:
    dsn = os.environ.get("JEEVES_TEST_PG_DSN")
    if not dsn:
        pytest.skip("JEEVES_TEST_PG_DSN not set")
    asyncpg = pytest.importorskip("asyncpg")
    pytest.importorskip("pgvector")
    from loomflow import Agent

    db = f"loomurl_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(dsn)
    await admin.execute(f"CREATE DATABASE {db}")
    await admin.close()
    url = dsn.rsplit("/", 1)[0] + "/" + db
    try:
        agent = Agent("x", model="echo", memory=url)
        await agent.run("remember the lighthouse", user_id="alice")
        found = await agent.memory.recall("lighthouse", user_id="alice")
        assert found and "lighthouse" in found[0].input
        await agent.memory.aclose()  # type: ignore[attr-defined]
    finally:
        admin = await asyncpg.connect(dsn)
        await admin.execute(f"DROP DATABASE IF EXISTS {db} WITH (FORCE)")
        await admin.close()


class _SeedCapture:
    name = "capture"

    def __init__(self) -> None:
        self.system: list[str] = []

    async def complete(self, messages: Any, **kwargs: Any) -> Any:
        from loomflow.core.types import Usage

        self.system = [m.content for m in messages if m.role == "system"]
        return "ok", [], Usage(), "end_turn"


async def test_seed_injection_keeps_cross_session_hits_and_ranks_facts_fairly() -> None:
    from datetime import timedelta

    from loomflow import Agent
    from loomflow.agent.api import Tuning
    from loomflow.core.types import Episode
    from loomflow.memory import HashEmbedder, VectorMemory

    embedder = HashEmbedder()
    memory = VectorMemory(embedder=embedder)
    query = "what is my locker code"
    q = await embedder.embed(query)
    other = await embedder.embed("zzz")
    near = [0.9 * a + 0.1 * b for a, b in zip(q, other, strict=True)]
    await memory.remember(Episode(session_id="B", user_id="u", input=query,
                                  output="locker code is 4711", embedding=near,
                                  occurred_at=NOW - timedelta(days=2)))
    # This session's own near-duplicate turns rank above it.
    for i in range(3):
        await memory.remember(Episode(session_id="A", user_id="u", input=query,
                                      output=f"I don't know ({i})", embedding=q))
    facts = [
        Fact(user_id="u", subject="u", predicate=predicate, object=obj)
        for predicate, obj in (("likes", "tea"), ("owns", "a bike"), ("visited", "Rome"))
    ]
    for fact in facts:
        await memory.facts.append(fact)
    from loomflow.memory._injection import estimate_tokens

    episode = (await memory.recall(query, user_id="u", limit=4))[-1]
    assert "4711" in episode.output
    # Room for the best fact and the best episode, not for all facts.
    tight = max(estimate_tokens(f"- {f.format()}") for f in facts) + estimate_tokens(
        f"- {episode.format()}"
    ) + 1

    for budget in (None, tight):
        model = _SeedCapture()
        agent = Agent("helper", model=model, memory=memory, auto_extract=False,
                      tuning=Tuning(memory_token_budget=budget))
        await agent.run(query, user_id="u", session_id="A")
        recalled = "\n".join(model.system)
        assert "4711" in recalled, budget  # the cross-session answer survives
        if budget is not None:
            # Facts no longer crowd out the best episode.
            assert sum(recalled.count(p) for p in ("likes", "owns", "visited")) < 3


def test_hybrid_alpha_extremes_drop_the_other_ranking() -> None:
    from loomflow.memory._hybrid import hybrid_rank

    bm25 = [(0, 2.0)]
    vector = [(1, 0.9)]
    assert [i for i, *_ in hybrid_rank(bm25_ranking=bm25, vector_ranking=vector, alpha=0.0)] == [0]
    assert [i for i, *_ in hybrid_rank(bm25_ranking=bm25, vector_ranking=vector, alpha=1.0)] == [1]


async def test_chroma_time_range_and_fact_limits_and_cosine_space() -> None:
    pytest.importorskip("chromadb")
    from datetime import timedelta

    from loomflow.core.types import Episode
    from loomflow.memory import HashEmbedder
    from loomflow.memory.chroma import ChromaMemory
    from loomflow.memory.chroma_facts import ChromaFactStore

    memory = ChromaMemory.ephemeral(collection_name=f"ep_{uuid.uuid4().hex}", embedder=HashEmbedder())
    old = NOW - timedelta(days=30)
    await memory.remember(Episode(session_id="s", input="deploy plan v0", output="", occurred_at=old))
    for i in range(5):  # newer, more similar episodes fill a plain top-k
        await memory.remember(Episode(session_id="s", input=f"deploy plan v{i + 1}", output=""))
    window = (old - timedelta(days=1), old + timedelta(days=1))
    found = await memory.recall("deploy plan", limit=3, time_range=window)
    assert [e.input for e in found] == ["deploy plan v0"]
    coll = await memory._get_collection()
    assert (coll.metadata or {}).get("hnsw:space") == "cosine"

    facts = ChromaFactStore.ephemeral(collection_name=f"f_{uuid.uuid4().hex}")
    for i in range(12):
        await facts.append(Fact(subject=f"s{i}", predicate="p", object="o",
                                recorded_at=NOW + timedelta(seconds=i)))
    newest = await facts.query(limit=3)
    assert [f.subject for f in newest] == ["s11", "s10", "s9"]


async def test_non_string_user_ids_are_refused(tmp_path: Path) -> None:
    from loomflow.memory import SqliteMemory

    memory = SqliteMemory(tmp_path / "m.db")
    assert await memory.recall("", limit=0) == []
    with pytest.raises(TypeError, match="user_id must be a string"):
        from loomflow.memory._user_key import encode_user_id

        encode_user_id({"$ne": "mallory"})  # type: ignore[arg-type]
    from loomflow import Agent

    with pytest.raises(TypeError, match="user_id must be a string"):
        await Agent("x", model="echo").run("hi", user_id={"$ne": "mallory"})  # type: ignore[arg-type]


async def test_sqlite_forget_overwrites_the_deleted_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sqlite3

    from loomflow.core.types import Episode
    from loomflow.memory import SqliteMemory

    real_connect = sqlite3.connect

    def default_off(*args: Any, **kwargs: Any) -> Any:  # SQLite's compile-time default
        conn = real_connect(*args, **kwargs)
        conn.execute("PRAGMA secure_delete = OFF")
        return conn

    monkeypatch.setattr(sqlite3, "connect", default_off)
    path = tmp_path / "m.db"
    memory = SqliteMemory(path)
    secret = "SSN-123-45-6789-" * 20
    await memory.remember(Episode(session_id="s", user_id="alice", input=secret, output=""))
    await memory.forget(user_id="alice")
    files = [f for f in tmp_path.iterdir() if f.name.startswith("m.db")]  # noqa: ASYNC240
    data = b"".join(f.read_bytes() for f in files)  # noqa: ASYNC240
    assert b"SSN-123-45-6789" not in data


async def test_chroma_supports_exists_and_not_like_the_other_stores() -> None:
    chromadb = pytest.importorskip("chromadb")
    from loomflow.loader.base import Chunk
    from loomflow.memory import HashEmbedder
    from loomflow.vectorstore import ChromaVectorStore, InMemoryVectorStore

    chunks = [Chunk("a doc", {"tag": "x"}), Chunk("b doc", {"tag": "y"}), Chunk("c doc", {"other": 1})]
    embedder = HashEmbedder(64)
    reference = await InMemoryVectorStore.from_chunks(chunks, embedder=embedder)
    chroma = await ChromaVectorStore.from_chunks(
        chunks, embedder=embedder, client=chromadb.EphemeralClient(),
        collection_name=f"v{uuid.uuid4().hex[:8]}",
    )
    for flt in (
        {"tag": {"$exists": True}},
        {"$not": {"tag": "x"}},
        {"$or": [{"tag": "y"}, {"other": {"$exists": True}}]},
        {"$and": [{"tag": {"$exists": True}}, {"tag": {"$ne": "x"}}]},
    ):
        got = sorted(r.chunk.content for r in await chroma.search("doc", k=5, filter=flt))
        want = sorted(r.chunk.content for r in await reference.search("doc", k=5, filter=flt))
        assert got == want, flt
