"""Regression tests for the memory backends: embedding batches,
embedder identity, per-user bounds, working-block limits, Postgres
table naming and the SQLite anonymous-block uniqueness fix.

Postgres is exercised through a fake pool (no database).
"""

from __future__ import annotations

import logging
import sqlite3
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import anyio
import pytest

from loomflow.core.errors import ConfigError
from loomflow.core.types import Episode
from loomflow.memory import (
    InMemoryMemory,
    PostgresMemory,
    SqliteMemory,
    VectorMemory,
)
from loomflow.memory import embedder as embedder_mod
from loomflow.memory._bounds import DEFAULT_MAX_BLOCK_CHARS, EpisodeIndex
from loomflow.memory._embedding_util import EmbedderMismatchError, cosine
from loomflow.memory.embedder import (
    CohereEmbedder,
    HashEmbedder,
    OpenAIEmbedder,
    VoyageEmbedder,
    split_batches,
)
from loomflow.memory.postgres_facts import PostgresFactStore
from loomflow.memory.resolver import resolve_memory

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _ep(i: int, user: str | None, text: str = "hello", session: str = "s") -> Episode:
    return Episode(
        id=f"{user}-{i}",
        session_id=session,
        user_id=user,
        occurred_at=NOW + timedelta(seconds=i),
        input=text,
        output="out",
    )


# ---------------------------------------------------------------------------
# 2. Embedding batches
# ---------------------------------------------------------------------------


class _OpenAIEmbeddings:
    def __init__(self) -> None:
        self.sizes: list[int] = []
        self.active = 0
        self.peak = 0

    async def create(self, **kwargs: Any) -> Any:
        texts = kwargs["input"]
        assert isinstance(texts, list)
        self.sizes.append(len(texts))
        self.active += 1
        self.peak = max(self.peak, self.active)
        await anyio.sleep(0.01)
        self.active -= 1

        class _Item:
            def __init__(self, index: int, text: str) -> None:
                self.index = index
                self.embedding = [float(text.removeprefix("c")), 0.0]

        class _Result:
            # Deliberately out of order: the embedder must sort on index.
            data = [_Item(i, t) for i, t in enumerate(texts)][::-1]

        return _Result()


class _OpenAIClient:
    def __init__(self) -> None:
        self.embeddings = _OpenAIEmbeddings()


async def test_openai_embed_batch_splits_into_provider_sized_requests() -> None:
    client = _OpenAIClient()
    embedder = OpenAIEmbedder(client=client)
    vectors = await embedder.embed_batch([f"c{i}" for i in range(5000)])
    assert sorted(client.embeddings.sizes, reverse=True) == [2048, 2048, 904]
    assert max(client.embeddings.sizes) <= 2048
    # Order preserved across batches and within an out-of-order reply.
    assert [v[0] for v in vectors] == [float(i) for i in range(5000)]


async def test_openai_embed_batch_bounds_concurrency_and_honours_batch_size() -> None:
    client = _OpenAIClient()
    embedder = OpenAIEmbedder(client=client, batch_size=10, max_concurrency=3)
    vectors = await embedder.embed_batch([f"c{i}" for i in range(95)])
    assert client.embeddings.sizes.count(10) == 9 and 5 in client.embeddings.sizes
    assert 1 < client.embeddings.peak <= 3
    assert [v[0] for v in vectors] == [float(i) for i in range(95)]


async def test_openai_embed_batch_guards_request_size() -> None:
    client = _OpenAIClient()

    class _Wide(OpenAIEmbedder):
        async def _embed_one_batch(self, texts: list[str]) -> list[list[float]]:
            client.embeddings.sizes.append(len(texts))
            assert sum(len(t.encode()) for t in texts) <= 300_000
            return [[0.0] for _ in texts]

    # 100 texts of 10 kB = 1 MB: must not go out as one request.
    await _Wide(client=client).embed_batch(["x" * 10_000] * 100)
    assert len(client.embeddings.sizes) >= 4
    assert sum(client.embeddings.sizes) == 100


def test_split_batches_respects_count_and_bytes() -> None:
    assert [len(b) for b in split_batches(["a"] * 7, batch_size=3)] == [3, 3, 1]
    sized = split_batches(["é" * 10] * 5, batch_size=100, max_batch_bytes=45)
    assert [len(b) for b in sized] == [2, 2, 1]  # 20 bytes each
    # One oversized text still goes out (alone).
    assert split_batches(["x" * 100], batch_size=5, max_batch_bytes=10) == [["x" * 100]]


def test_batch_size_is_validated_against_the_provider_limit() -> None:
    with pytest.raises(ValueError, match="provider limit of 2048"):
        OpenAIEmbedder(client=object(), batch_size=4096)
    with pytest.raises(ValueError, match="provider limit of 96"):
        CohereEmbedder(client=object(), batch_size=97)
    with pytest.raises(ValueError, match="positive integer"):
        VoyageEmbedder(client=object(), batch_size=0)
    with pytest.raises(ValueError, match="max_concurrency"):
        OpenAIEmbedder(client=object(), max_concurrency=0)


async def test_cohere_and_voyage_batches_respect_their_limits() -> None:
    cohere_sizes: list[int] = []
    voyage_sizes: list[int] = []

    class _Cohere:
        async def embed(self, **kwargs: Any) -> Any:
            texts = kwargs["texts"]
            cohere_sizes.append(len(texts))

            class _Embeddings:
                float = [[float(t)] for t in texts]

            class _Result:
                embeddings = _Embeddings()

            return _Result()

    class _Voyage:
        async def embed(self, **kwargs: Any) -> Any:
            texts = kwargs["texts"]
            voyage_sizes.append(len(texts))

            class _Result:
                embeddings = [[float(t)] for t in texts]

            return _Result()

    texts = [str(i) for i in range(300)]
    cohere = await CohereEmbedder(client=_Cohere()).embed_batch(texts)
    voyage = await VoyageEmbedder(client=_Voyage()).embed_batch(texts)
    assert max(cohere_sizes) <= 96 and sum(cohere_sizes) == 300
    assert max(voyage_sizes) <= 128 and sum(voyage_sizes) == 300
    assert [v[0] for v in cohere] == [float(i) for i in range(300)]
    assert [v[0] for v in voyage] == [float(i) for i in range(300)]
    assert await CohereEmbedder(client=_Cohere()).embed_batch([]) == []


async def test_short_provider_batch_raises() -> None:
    class _Short(OpenAIEmbedder):
        async def _embed_one_batch(self, texts: list[str]) -> list[list[float]]:
            return [[0.0]] * (len(texts) - 1)

    with pytest.raises(ValueError, match="2 vectors for a batch of 3"):
        await _Short(client=object()).embed_batch(["a", "b", "c"])


async def test_failed_batch_raises_the_providers_own_error() -> None:
    class _RateLimited(Exception):
        pass

    class _Flaky(OpenAIEmbedder):
        async def _embed_one_batch(self, texts: list[str]) -> list[list[float]]:
            if "c7" in texts:
                raise _RateLimited("slow down")
            await anyio.sleep(0.01)
            return [[0.0] for _ in texts]

    embedder = _Flaky(client=object(), batch_size=5)
    with pytest.raises(_RateLimited, match="slow down"):  # not an ExceptionGroup
        await embedder.embed_batch([f"c{i}" for i in range(40)])


# ---------------------------------------------------------------------------
# 3. Embedder identity
# ---------------------------------------------------------------------------


def test_cosine_raises_on_dimension_mismatch() -> None:
    with pytest.raises(EmbedderMismatchError, match="different embedder"):
        cosine([1.0, 0.0], [1.0, 0.0, 0.0])
    assert cosine([], [1.0]) == 0.0
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


async def test_sqlite_refuses_a_different_embedder(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    memory = SqliteMemory(path, embedder=HashEmbedder(8))
    for i in range(3):
        await memory.remember(_ep(i, "alice", f"topic number {i}"))

    with pytest.raises(EmbedderMismatchError) as excinfo:
        SqliteMemory(path, embedder=HashEmbedder(16))
    message = str(excinfo.value)
    assert "hash-embedder-8" in message and "hash-embedder-16" in message
    assert "original embedder" in message and "fresh store" in message

    # Same dimension, different model: still a different vector space.
    class _OtherModel(HashEmbedder):
        def __init__(self) -> None:
            super().__init__(8)
            self.name = "other-model"

    with pytest.raises(EmbedderMismatchError, match="other-model"):
        SqliteMemory(path, embedder=_OtherModel())

    # The original embedder keeps working.
    again = SqliteMemory(path, embedder=HashEmbedder(8))
    assert len(await again.recall("topic number 1", user_id="alice")) == 3


async def test_sqlite_legacy_database_records_embedder_on_first_use(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.db"
    memory = SqliteMemory(path, embedder=HashEmbedder(8))
    await memory.remember(_ep(0, "alice", "old data"))
    with sqlite3.connect(path) as conn:  # as a pre-record database
        conn.execute("DROP TABLE loomflow_meta")

    reopened = SqliteMemory(path, embedder=HashEmbedder(8))  # opens fine
    assert len(await reopened.recall("old data", user_id="alice")) == 1
    with sqlite3.connect(path) as conn:
        recorded = dict(conn.execute("SELECT key, value FROM loomflow_meta"))
    assert recorded == {
        "embedder_name": "hash-embedder-8",
        "embedder_dimensions": "8",
    }

    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE loomflow_meta")
    with pytest.raises(EmbedderMismatchError, match="8 dimensions"):
        SqliteMemory(path, embedder=HashEmbedder(16))


async def test_sqlite_empty_database_accepts_any_embedder(tmp_path: Path) -> None:
    path = tmp_path / "empty.db"
    SqliteMemory(path, embedder=HashEmbedder(8))  # nothing written
    memory = SqliteMemory(path, embedder=HashEmbedder(16))
    await memory.remember(_ep(0, "alice", "now embedded"))
    with pytest.raises(EmbedderMismatchError):
        SqliteMemory(path, embedder=HashEmbedder(8))


def test_resolver_warns_when_it_defaults_to_the_hash_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LOOMFLOW_EMBEDDER", raising=False)
    monkeypatch.setattr(embedder_mod, "_HASH_FALLBACK_WARNED", False)
    with pytest.warns(UserWarning, match="NOT semantic"):
        resolve_memory(f"sqlite:{tmp_path / 'a.db'}")

    # Choosing it on purpose is not a silent fallback.
    for explicit in (
        {"backend": "sqlite", "path": str(tmp_path / "b.db"), "embedder": "hash"},
        "inmemory",
        {"backend": "inmemory"},
    ):
        monkeypatch.setattr(embedder_mod, "_HASH_FALLBACK_WARNED", False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            resolve_memory(explicit)

    monkeypatch.setattr(embedder_mod, "_HASH_FALLBACK_WARNED", False)
    monkeypatch.setenv("LOOMFLOW_EMBEDDER", "hash")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resolve_memory(f"sqlite:{tmp_path / 'c.db'}")


# ---------------------------------------------------------------------------
# 5. Per-user eviction and indexing
# ---------------------------------------------------------------------------


async def test_vector_memory_cap_is_per_user() -> None:
    memory = VectorMemory(embedder=HashEmbedder(8), max_episodes=5)
    for i in range(3):
        await memory.remember(_ep(i, "alice", f"alice secret {i}"))
    for i in range(10, 20):
        await memory.remember(_ep(i, "bob", f"bob chatter {i}"))

    alice = await memory.recall("", user_id="alice", limit=10)
    bob = await memory.recall("", user_id="bob", limit=10)
    assert len(alice) == 3  # bob's writes evicted none of alice's
    assert [e.id for e in bob] == [f"bob-{i}" for i in range(19, 14, -1)]
    assert len(memory._episodes) == 8


async def test_vector_memory_eviction_keeps_the_new_episode() -> None:
    memory = VectorMemory(embedder=HashEmbedder(8), max_episodes=2)
    await memory.remember(_ep(5, "u"))
    await memory.remember(_ep(6, "u"))
    await memory.remember(_ep(1, "u"))  # older than both, but just written
    kept = {e.id for e in await memory.recall("", user_id="u", limit=10)}
    assert kept == {"u-1", "u-6"}


async def test_inmemory_memory_has_an_optional_per_user_cap() -> None:
    memory = InMemoryMemory(max_episodes=3)
    for i in range(5):
        await memory.remember(_ep(i, "alice"))
    for i in range(2):
        await memory.remember(_ep(i, None))
    assert [e.id for e in await memory.recall("", user_id="alice", limit=10)] == [
        "alice-4",
        "alice-3",
        "alice-2",
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert len(await memory.recall("", limit=10)) == 2
    assert await InMemoryMemory().recall("", limit=3) == []  # uncapped default
    with pytest.raises(ValueError, match="max_episodes"):
        InMemoryMemory(max_episodes=0)


async def test_inmemory_recall_only_touches_the_asking_users_episodes() -> None:
    memory = InMemoryMemory()
    for i in range(300):
        await memory.remember(_ep(i, f"u{i % 100}"))

    seen: list[str | None] = []
    real = EpisodeIndex.for_user

    def _spy(self: EpisodeIndex, user_id: str | None) -> list[Episode]:
        result = real(self, user_id)
        seen.append(user_id)
        assert len(result) == 3  # one tenant's rows, not all 300
        return result

    EpisodeIndex.for_user = _spy  # type: ignore[method-assign]
    try:
        assert len(await memory.recall("hello", user_id="u1", limit=5)) == 3
        assert len(await memory.recall_scored("hello", user_id="u1", limit=5)) == 3
        assert len(await memory.session_messages("s", user_id="u1")) == 6
    finally:
        EpisodeIndex.for_user = real  # type: ignore[method-assign]
    assert seen == ["u1", "u1", "u1"]


def test_episode_index_eviction_does_not_scan() -> None:
    index = EpisodeIndex(max_per_user=100)
    for i in range(5000):
        evicted = index.put(_ep(i, "busy"))
        assert evicted == ([f"busy-{i - 100}"] if i >= 100 else [])
    assert len(index) == 100
    # Stale heap entries are swept rather than accumulating.
    assert len(index._heaps["busy"]) <= 2 * 100 + 16 + 1
    # Re-remembering an id keeps one live entry and its position.
    index.put(_ep(4950, "busy").model_copy(update={"input": "edited"}))
    assert len(index) == 100
    assert index.episodes["busy-4950"].input == "edited"
    # Moving an id to another user removes it from the first.
    index.put(_ep(4950, "busy").model_copy(update={"user_id": "other"}))
    assert len(index.for_user("busy")) == 99
    assert [e.id for e in index.for_user("other")] == ["busy-4950"]


async def test_forget_and_export_stay_user_scoped() -> None:
    for memory in (InMemoryMemory(), VectorMemory(embedder=HashEmbedder(8))):
        for i in range(3):
            await memory.remember(_ep(i, "alice"))
            await memory.remember(_ep(i, "bob"))
        assert await memory.forget(user_id="alice") == 3
        assert await memory.recall("", user_id="alice", limit=5) == []
        assert len((await memory.export(user_id="bob")).episodes) == 3
        assert (await memory.profile(user_id="bob")).episode_count == 3


# ---------------------------------------------------------------------------
# 5. append_block is bounded
# ---------------------------------------------------------------------------


async def test_append_block_trims_oldest_first(tmp_path: Path) -> None:
    memories: list[Any] = [
        InMemoryMemory(),
        VectorMemory(embedder=HashEmbedder(8)),
        SqliteMemory(tmp_path / "blocks.db", embedder=HashEmbedder(8)),
    ]
    for memory in memories:
        for marker in "abc":
            await memory.append_block("notes", marker * 15_000, user_id="u")
        content = (await memory.working(user_id="u"))[0].content
        assert len(content) == DEFAULT_MAX_BLOCK_CHARS == 20_000
        assert content == "b" * 5_000 + "c" * 15_000  # oldest text went first


async def test_append_block_limit_is_configurable(tmp_path: Path) -> None:
    small = InMemoryMemory(max_block_chars=10)
    await small.append_block("n", "0123456789abcdef")
    assert (await small.working())[0].content == "6789abcdef"

    unlimited = SqliteMemory(
        tmp_path / "u.db", embedder=HashEmbedder(8), max_block_chars=None
    )
    for _ in range(3):
        await unlimited.append_block("n", "x" * 15_000)
    assert len((await unlimited.working())[0].content) == 45_000
    with pytest.raises(ValueError, match="max_block_chars"):
        InMemoryMemory(max_block_chars=0)


# ---------------------------------------------------------------------------
# recall(limit <= 0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("limit", [0, -1])
async def test_recall_rejects_non_positive_limits(tmp_path: Path, limit: int) -> None:
    memories: list[Any] = [
        InMemoryMemory(),
        VectorMemory(embedder=HashEmbedder(8)),
        SqliteMemory(tmp_path / "l.db", embedder=HashEmbedder(8)),
    ]
    for memory in memories:
        for i in range(5):
            await memory.remember(_ep(i, "bob", f"bob chatter {i}"))
        for query in ("", "bob"):
            with pytest.raises(ValueError, match="limit must be"):
                await memory.recall(query, user_id="bob", limit=limit)
            with pytest.raises(ValueError, match="limit must be"):
                await memory.recall_scored(query, user_id="bob", limit=limit)


# ---------------------------------------------------------------------------
# 5. SQLite recall no longer loads every row
# ---------------------------------------------------------------------------


async def test_sqlite_recall_results_unchanged_and_scan_is_narrow(
    tmp_path: Path,
) -> None:
    embedder = HashEmbedder(8)
    memory = SqliteMemory(tmp_path / "scan.db", embedder=embedder)
    episodes = [_ep(i, "alice", f"topic number {i}") for i in range(30)]
    for episode in episodes:
        await memory.remember(episode)
    await memory.remember(_ep(99, "bob", "topic number 3"))

    # Reference ranking: brute-force cosine in insertion order.
    query = await embedder.embed("topic number 3")
    scored = []
    for episode in episodes:
        vector = await embedder.embed(f"{episode.input}\n{episode.output}")
        scored.append((cosine(query, vector), episode.id))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    expected = [eid for _, eid in scored[:4]]

    full_scans: list[Any] = []
    real_scan = memory._scan_episodes_sync

    def _spy(user_id: str | None, time_range: Any) -> list[tuple[Any, ...]]:
        full_scans.append(user_id)
        return real_scan(user_id, time_range)

    memory._scan_episodes_sync = _spy  # type: ignore[method-assign]
    got = await memory.recall("topic number 3", user_id="alice", limit=4)
    assert [e.id for e in got] == expected
    assert all(e.embedding is not None and e.input for e in got)

    recent = await memory.recall("", user_id="alice", limit=3)
    assert [e.id for e in recent] == ["alice-29", "alice-28", "alice-27"]
    window = (NOW + timedelta(seconds=5), NOW + timedelta(seconds=9))
    ranged = await memory.recall("", user_id="alice", limit=2, time_range=window)
    assert [e.id for e in ranged] == ["alice-9", "alice-8"]

    profile = await memory.profile(user_id="alice")
    assert profile.episode_count == 30
    assert profile.last_seen == NOW + timedelta(seconds=29)
    assert profile.recent_sessions == ["s"]
    # None of the above needed the every-column scan of the partition.
    assert full_scans == []


async def test_chroma_recall_recent_fetches_embeddings_for_winners_only() -> None:
    pytest.importorskip("chromadb")
    import uuid

    from loomflow.memory.chroma import ChromaMemory

    memory = ChromaMemory.ephemeral(
        embedder=HashEmbedder(8), collection_name=f"rag_{uuid.uuid4().hex}"
    )
    for i in range(12):
        await memory.remember(_ep(i, "alice", f"note {i}"))
    await memory.remember(_ep(50, "bob", "other tenant"))

    coll = await memory._get_collection()
    calls: list[dict[str, Any]] = []
    real_get = coll.get

    class _Spy:
        def __getattr__(self, name: str) -> Any:
            return getattr(coll, name)

        def get(self, **kwargs: Any) -> Any:
            result = real_get(**kwargs)
            calls.append({"include": kwargs.get("include"), "rows": len(result["ids"])})
            return result

    memory._collection = _Spy()
    recent = await memory.recall("", user_id="alice", limit=3)
    assert [e.id for e in recent] == ["alice-11", "alice-10", "alice-9"]
    assert all(e.embedding is not None and len(e.embedding) == 8 for e in recent)
    with_embeddings = [c for c in calls if "embeddings" in (c["include"] or [])]
    assert [c["rows"] for c in with_embeddings] == [3]  # not all 12
    with pytest.raises(ValueError, match="limit must be"):
        await memory.recall("", user_id="alice", limit=0)


# ---------------------------------------------------------------------------
# 12. SQLite anonymous working blocks are unique
# ---------------------------------------------------------------------------


async def test_sqlite_anonymous_blocks_are_deduplicated_and_unique(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wb.db"
    SqliteMemory(path, embedder=HashEmbedder(8))
    with sqlite3.connect(path) as conn:
        # A database from before the fix: no unique index, duplicates.
        conn.execute("DROP INDEX working_blocks_anon_name_uq")
        conn.execute("INSERT INTO working_blocks VALUES (NULL, 'persona', 'old', 0, 10)")
        conn.execute("INSERT INTO working_blocks VALUES (NULL, 'persona', 'new', 0, 20)")
        conn.execute("INSERT INTO working_blocks VALUES (NULL, 'notes', 'n', 1, 5)")
        conn.execute("INSERT INTO working_blocks VALUES ('amy', 'persona', 'amy', 0, 1)")

    memory = SqliteMemory(path, embedder=HashEmbedder(8))  # migrates
    blocks = {b.name: b.content for b in await memory.working()}
    assert blocks == {"persona": "new", "notes": "n"}  # newest row survived
    assert [b.content for b in await memory.working(user_id="amy")] == ["amy"]

    with sqlite3.connect(path) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO working_blocks VALUES (NULL, 'persona', 'dup', 0, 30)")

    await memory.update_block("persona", "updated")
    await memory.append_block("persona", "!")
    assert {b.name: b.content for b in await memory.working()}["persona"] == "updated!"
    SqliteMemory(path, embedder=HashEmbedder(8))  # idempotent


async def test_sqlite_two_instances_cannot_duplicate_an_anonymous_block(
    tmp_path: Path,
) -> None:
    path = tmp_path / "race.db"
    first = SqliteMemory(path, embedder=HashEmbedder(8))
    second = SqliteMemory(path, embedder=HashEmbedder(8))
    async with anyio.create_task_group() as tg:
        for i in range(10):
            tg.start_soon(first.append_block, "log", f"a{i};")
            tg.start_soon(second.append_block, "log", f"b{i};")
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            "SELECT content FROM working_blocks WHERE user_id IS NULL AND name = 'log'"
        ).fetchall()
    assert len(rows) == 1
    assert sorted(filter(None, rows[0][0].split(";"))) == sorted(
        [f"a{i}" for i in range(10)] + [f"b{i}" for i in range(10)]
    )


# ---------------------------------------------------------------------------
# 6 / 7. PostgresMemory table names, foreign tables, HNSW cap
# ---------------------------------------------------------------------------


class _PgConn:
    def __init__(self, log: list[tuple[str, tuple[Any, ...]]], columns: dict[str, list[str]]):
        self._log = log
        self._columns = columns

    async def execute(self, sql: str, *args: Any) -> str:
        self._log.append((" ".join(sql.split()), args))
        return "OK"

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        if "information_schema.columns" in sql:
            return [{"column_name": c} for c in self._columns.get(args[1], [])]
        self._log.append((" ".join(sql.split()), args))
        return []


class _PgAcquire:
    def __init__(self, conn: _PgConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _PgConn:
        return self._conn

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _PgPool:
    def __init__(self, columns: dict[str, list[str]] | None = None) -> None:
        self.log: list[tuple[str, tuple[Any, ...]]] = []
        self._columns = columns or {}

    def acquire(self) -> _PgAcquire:
        return _PgAcquire(_PgConn(self.log, self._columns))

    @property
    def sql(self) -> list[str]:
        return [sql for sql, _ in self.log]


def test_postgres_memory_default_table_names_are_unchanged() -> None:
    memory = PostgresMemory(pool=None, embedder=HashEmbedder(8))
    ddl = "\n".join(memory.schema_sql())
    assert "CREATE TABLE IF NOT EXISTS episodes (" in ddl
    assert "CREATE TABLE IF NOT EXISTS memory_blocks (" in ddl
    assert "CREATE TABLE IF NOT EXISTS episode_tool_transcripts (" in ddl
    assert "episodes_embedding_idx ON episodes USING hnsw" in ddl
    assert "CREATE SCHEMA" not in ddl
    facts = "\n".join(PostgresFactStore(None, embedder=HashEmbedder(8)).schema_sql())
    assert "CREATE TABLE IF NOT EXISTS facts (" in facts


async def test_postgres_memory_table_prefix_and_schema() -> None:
    pool = _PgPool()
    memory = PostgresMemory(
        pool, embedder=HashEmbedder(8), table_prefix="Loom_", schema="agent"
    )
    memory.facts = PostgresFactStore(
        pool, embedder=HashEmbedder(8), table_prefix="Loom_", schema="agent"
    )
    await memory.init_schema()
    await memory.remember(_ep(0, "alice"))
    await memory.recall("hello", user_id="alice")
    await memory.recall("", user_id="alice")
    await memory.append_block("notes", "x", user_id="alice")
    await memory.update_block("notes", "y", user_id="alice")
    await memory.working(user_id="alice")
    await memory.session_messages("s", user_id="alice")
    await memory.forget(user_id="alice")

    ddl = "\n".join(pool.sql)
    assert 'CREATE SCHEMA IF NOT EXISTS "agent";' in ddl
    for table in ("loom_episodes", "loom_memory_blocks", "loom_episode_tool_transcripts", "loom_facts"):
        assert f'CREATE TABLE IF NOT EXISTS "agent"."{table}" (' in ddl
    assert 'REFERENCES "agent"."loom_episodes"(id)' in ddl
    assert '"loom_episodes_embedding_idx" ON "agent"."loom_episodes" USING hnsw' in ddl
    # No statement touches the unprefixed tables.
    for sql in pool.sql:
        for bare in (" episodes", " memory_blocks", " facts", " episode_tool_transcripts"):
            assert bare + " " not in sql and bare + "(" not in sql, sql


@pytest.mark.parametrize(
    "kwargs",
    [
        {"table_prefix": "bad-prefix_"},
        {"table_prefix": 'x"; DROP TABLE users; --'},
        {"schema": "my schema"},
        {"schema": "a.b"},
        {"table_prefix": "p" * 40},
    ],
)
def test_postgres_memory_rejects_unsafe_identifiers(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="invalid Postgres|longer than"):
        PostgresMemory(pool=None, embedder=HashEmbedder(8), **kwargs)
    with pytest.raises(ValueError, match="invalid Postgres|longer than"):
        PostgresFactStore(None, embedder=HashEmbedder(8), **kwargs)


def test_resolver_validates_postgres_identifiers_eagerly() -> None:
    with pytest.raises(ConfigError, match="invalid Postgres"):
        resolve_memory(
            {
                "backend": "postgres",
                "url": "postgres://x/y",
                "embedder": "hash",
                "table_prefix": "no good",
            }
        )


async def test_postgres_memory_refuses_a_foreign_episodes_table() -> None:
    # The application's own table: TV episodes, not ours.
    pool = _PgPool({"episodes": ["id", "show_id", "title", "aired_on"]})
    memory = PostgresMemory(pool, embedder=HashEmbedder(8))
    with pytest.raises(ConfigError) as excinfo:
        await memory.init_schema()
    message = str(excinfo.value)
    assert "'episodes'" in message and "table_prefix" in message and "schema" in message
    assert pool.sql == []  # nothing was created or altered

    # With a prefix the two coexist.
    prefixed = PostgresMemory(pool, embedder=HashEmbedder(8), table_prefix="loomflow_")
    await prefixed.init_schema()
    assert any('"loomflow_episodes"' in sql for sql in pool.sql)
    assert not any("ALTER TABLE episodes" in sql for sql in pool.sql)


async def test_postgres_memory_upgrades_its_own_older_table() -> None:
    # A pre-``user_id`` loomflow table is ours: the ALTER must still run.
    pool = _PgPool(
        {
            "episodes": [
                "id", "namespace", "session_id", "occurred_at",
                "input", "output", "embedding",
            ],
            "memory_blocks": ["namespace", "name", "content", "pinned_order", "updated_at"],
        }
    )
    await PostgresMemory(pool, embedder=HashEmbedder(8)).init_schema()
    assert any(
        sql.startswith("ALTER TABLE episodes ADD COLUMN IF NOT EXISTS user_id")
        for sql in pool.sql
    )


async def test_postgres_memory_skips_hnsw_above_2000_dimensions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pool = _PgPool()
    memory = PostgresMemory(pool, embedder=HashEmbedder(3072))
    memory.facts = PostgresFactStore(pool, embedder=HashEmbedder(3072))
    with caplog.at_level(logging.WARNING):
        await memory.init_schema()
    ddl = "\n".join(pool.sql)
    assert "vector(3072)" in ddl
    assert "hnsw" not in ddl
    assert "episodes_namespace_idx" in ddl  # the other indexes are still made
    assert caplog.text.count("2000") >= 2  # memory + fact store both say so

    small = "\n".join(PostgresMemory(None, embedder=HashEmbedder(2000)).schema_sql())
    assert "USING hnsw" in small


async def test_postgres_append_block_is_bounded_in_sql() -> None:
    pool = _PgPool()
    await PostgresMemory(pool, embedder=HashEmbedder(8)).append_block("n", "abc")
    sql, args = pool.log[-1]
    assert "RIGHT(existing.content || EXCLUDED.content, $5)" in sql
    assert args[-1] == DEFAULT_MAX_BLOCK_CHARS

    await PostgresMemory(
        pool, embedder=HashEmbedder(8), max_block_chars=None
    ).append_block("n", "abc")
    sql, args = pool.log[-1]
    assert "RIGHT(" not in sql and "existing.content || EXCLUDED.content" in sql
    assert args[-1] == "abc"


async def test_postgres_recall_rejects_non_positive_limits() -> None:
    pool = _PgPool()
    memory = PostgresMemory(pool, embedder=HashEmbedder(8))
    with pytest.raises(ValueError, match="limit must be"):
        await memory.recall("q", limit=0)
    with pytest.raises(ValueError, match="limit must be"):
        await memory.recall_scored("", limit=-1)
    assert pool.sql == []
