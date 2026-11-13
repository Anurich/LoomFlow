"""Regression tests for the vector-store contract.

One interface, one behaviour: upsert on re-add, ``k <= 0`` raises,
finite query vectors, the same filter semantics everywhere, safe SQL
identifiers, provider-sized embedding batches, and the FAISS add lock.
Postgres is exercised through a fake connection (no database).
"""

from __future__ import annotations

import json
import logging
import sys
import types
import warnings
from pathlib import Path
from typing import Any

import anyio
import pytest

from loomflow.loader.base import Chunk
from loomflow.memory.embedder import HashEmbedder
from loomflow.vectorstore import InMemoryVectorStore, PostgresVectorStore
from loomflow.vectorstore._filter import evaluate_filter
from loomflow.vectorstore._util import embed_all
from loomflow.vectorstore.postgres import _build_where_sql

pytestmark = pytest.mark.anyio


def _names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def _chunks(*texts: str, **metadata: Any) -> list[Chunk]:
    return [Chunk(content=t, metadata=dict(metadata)) for t in texts]


# ---------------------------------------------------------------------------
# Fake asyncpg pool
# ---------------------------------------------------------------------------


class _Conn:
    def __init__(self, sqls: list[str]) -> None:
        self._sqls = sqls

    async def execute(self, sql: str, *args: Any) -> str:
        self._sqls.append(" ".join(sql.split()))
        return "OK"

    async def executemany(self, sql: str, rows: Any) -> None:
        self._sqls.append(" ".join(sql.split()))

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self._sqls.append(" ".join(sql.split()))
        return []

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any]:
        self._sqls.append(" ".join(sql.split()))
        return {"n": 0}


class _Pool:
    def __init__(self, sqls: list[str]) -> None:
        self._sqls = sqls

    async def acquire(self) -> _Conn:
        return _Conn(self._sqls)

    async def release(self, conn: _Conn) -> None:
        return None


def _pg(sqls: list[str], embedder: Any = None, **kwargs: Any) -> PostgresVectorStore:
    store = PostgresVectorStore(embedder or HashEmbedder(4), dsn="x", **kwargs)
    store._pool_obj = _Pool(sqls)
    return store


# ---------------------------------------------------------------------------
# 1. Table identifiers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "table",
    [
        "v (id text); DROP TABLE users; --",
        "My-Docs",
        'docs"; DROP TABLE x; --',
        "docs docs",
        "a.b.c",
        "1docs",
        "",
        "x" * 64,
    ],
)
def test_postgres_rejects_unsafe_table_names(table: str) -> None:
    with pytest.raises(ValueError, match="invalid Postgres|longer than"):
        PostgresVectorStore(HashEmbedder(4), dsn="x", table=table)


async def test_postgres_quotes_table_everywhere() -> None:
    sqls: list[str] = []
    store = _pg(sqls, table="docs")
    await store.add(_chunks("a"))
    await store.count()
    await store.delete(["x"])
    await store.get_by_ids(["x"])
    await store.search("a", k=2)
    await store.search_hybrid("a", k=2)
    table_sqls = [s for s in sqls if "docs" in s]
    assert table_sqls
    for sql in table_sqls:
        # Never a bare, unquoted table reference.
        assert '"docs"' in sql or '"docs_' in sql, sql
    assert any('CREATE TABLE IF NOT EXISTS "docs" (' in s for s in sqls)
    assert any('"docs_embedding_hnsw" ON "docs"' in s for s in sqls)
    assert any('"docs_content_tsv_gin" ON "docs"' in s for s in sqls)


async def test_postgres_schema_qualified_table_and_index_names() -> None:
    sqls: list[str] = []
    store = _pg(sqls, table="rag.Docs")
    await store.add(_chunks("a"))
    # Lower-cased like an unquoted identifier, schema and table quoted
    # separately; the derived index name is NOT schema-qualified.
    assert any('CREATE TABLE IF NOT EXISTS "rag"."docs" (' in s for s in sqls)
    assert any(
        'CREATE INDEX IF NOT EXISTS "docs_embedding_hnsw" ON "rag"."docs"' in s
        for s in sqls
    )
    assert not any("rag.docs_embedding_hnsw" in s for s in sqls)


async def test_postgres_long_table_name_keeps_distinct_index_names() -> None:
    sqls: list[str] = []
    table = "t" * 63
    store = _pg(sqls, table=table)
    await store.add(_chunks("a"))
    hnsw = f'"{"t" * 48}_embedding_hnsw"'
    gin = f'"{"t" * 47}_content_tsv_gin"'
    assert any(f"CREATE INDEX IF NOT EXISTS {hnsw} ON" in s for s in sqls)
    assert any(f"CREATE INDEX IF NOT EXISTS {gin} ON" in s for s in sqls)
    assert len(hnsw) - 2 == 63 and len(gin) - 2 == 63


async def test_postgres_default_table_name_is_unchanged() -> None:
    sqls: list[str] = []
    store = _pg(sqls)
    await store.add(_chunks("a"))
    assert any('INSERT INTO "jeeves_vectors"' in s for s in sqls)


# ---------------------------------------------------------------------------
# 7. HNSW dimension cap + constructor ``dimension=``
# ---------------------------------------------------------------------------


async def test_postgres_skips_hnsw_above_2000_dimensions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sqls: list[str] = []
    store = _pg(sqls, embedder=HashEmbedder(3072))
    with caplog.at_level(logging.WARNING, logger="loomflow.vectorstore.postgres"):
        await store.add(_chunks("a"))
    assert any("vector(3072)" in s for s in sqls)
    assert not any("USING hnsw" in s for s in sqls)
    assert any("USING gin" in s for s in sqls)  # the rest still happens
    assert "2000" in caplog.text and "3072" in caplog.text


async def test_postgres_creates_hnsw_at_the_cap() -> None:
    sqls: list[str] = []
    store = _pg(sqls, embedder=HashEmbedder(2000))
    await store.add(_chunks("a"))
    assert any("USING hnsw" in s for s in sqls)


def test_postgres_dimension_must_match_embedder() -> None:
    with pytest.raises(ValueError, match="dimension=999"):
        PostgresVectorStore(HashEmbedder(4), dsn="x", dimension=999)
    PostgresVectorStore(HashEmbedder(4), dsn="x", dimension=4)  # fine


async def test_postgres_dimension_checked_against_actual_vectors() -> None:
    class _NoDims:
        name = "custom"

        async def embed(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0]

        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0, 0.0] for _ in texts]

    sqls: list[str] = []
    store = _pg(sqls, embedder=_NoDims(), dimension=5)
    with pytest.raises(ValueError, match="configured for 5"):
        await store.add(_chunks("a"))
    assert sqls == []  # nothing reached the database


# ---------------------------------------------------------------------------
# 4 / 14. k, limits and query vectors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0, -1, -2, True, 1.5])
async def test_inmemory_rejects_non_positive_k(k: Any) -> None:
    store = await InMemoryVectorStore.from_texts(
        [f"doc {i}" for i in range(6)], embedder=HashEmbedder(16)
    )
    with pytest.raises(ValueError, match="k must be"):
        await store.search("doc 1", k=k)
    with pytest.raises(ValueError, match="k must be"):
        await store.search("doc 1", k=k, diversity=0.5)
    with pytest.raises(ValueError, match="k must be"):
        await store.search_by_vector([0.1] * 16, k=k)
    with pytest.raises(ValueError, match="k must be"):
        await store.search_hybrid("doc 1", k=k)


async def test_postgres_rejects_non_positive_k() -> None:
    sqls: list[str] = []
    store = _pg(sqls)
    for k in (0, -1):
        with pytest.raises(ValueError, match="k must be"):
            await store.search("q", k=k)
        with pytest.raises(ValueError, match="k must be"):
            await store.search_by_vector([0.1] * 4, k=k)
        with pytest.raises(ValueError, match="k must be"):
            await store.search_hybrid("q", k=k)
    assert sqls == []


async def test_k_larger_than_store_returns_everything() -> None:
    store = await InMemoryVectorStore.from_texts(
        ["a", "b"], embedder=HashEmbedder(8)
    )
    assert len(await store.search("a", k=50)) == 2


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
async def test_nan_query_vector_raises(bad: float) -> None:
    store = await InMemoryVectorStore.from_texts(
        ["a", "b"], embedder=HashEmbedder(4)
    )
    with pytest.raises(ValueError, match="NaN or infinite"):
        await store.search_by_vector([bad, 0.0, 0.0, 0.0], k=2)
    sqls: list[str] = []
    with pytest.raises(ValueError, match="NaN or infinite"):
        await _pg(sqls).search_by_vector([bad, 0.0, 0.0, 0.0], k=2)
    assert sqls == []


async def test_inmemory_rejects_vectors_of_another_dimension() -> None:
    store = InMemoryVectorStore(HashEmbedder(8))
    await store.add(_chunks("a"))
    with pytest.raises(ValueError, match="4 dimensions.*8-dimensional"):
        await store.search_by_vector([0.1] * 4, k=1)
    # ... and a changed embedder can't poison the store on add.
    store._embedder = HashEmbedder(16)
    with pytest.raises(ValueError, match="16-dimensional.*8-dimensional"):
        await store.add(_chunks("b"))
    assert await store.count() == 1


# ---------------------------------------------------------------------------
# 4. add upserts
# ---------------------------------------------------------------------------


async def test_inmemory_readding_an_id_upserts() -> None:
    store = await InMemoryVectorStore.from_texts(
        [f"doc {i}" for i in range(6)],
        embedder=HashEmbedder(16),
        ids=[f"i{i}" for i in range(6)],
    )
    await store.add(_chunks("doc 1 v2"), ids=["i1"])
    assert await store.count() == 6
    assert [c.content for c in await store.get_by_ids(["i1"])] == ["doc 1 v2"]
    hits = await store.search("doc 1 v2", k=10)
    assert sum(1 for h in hits if h.id == "i1") == 1
    assert hits[0].id == "i1" and hits[0].chunk.content == "doc 1 v2"
    hybrid = await store.search_hybrid("v2", k=10)
    assert sum(1 for h in hybrid if h.id == "i1") == 1
    await store.delete(["i1"])
    assert await store.count() == 5


async def test_inmemory_duplicate_ids_in_one_call_keep_the_last() -> None:
    store = InMemoryVectorStore(HashEmbedder(8))
    ids = await store.add(_chunks("first", "second"), ids=["x", "x"])
    assert ids == ["x", "x"]
    assert await store.count() == 1
    assert [c.content for c in await store.get_by_ids(["x"])] == ["second"]


# ---------------------------------------------------------------------------
# 14. save() / load()
# ---------------------------------------------------------------------------


async def test_save_is_atomic_and_leaves_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from loomflow.vectorstore import inmemory as inmemory_mod

    path = tmp_path / "store.json"
    store = await InMemoryVectorStore.from_texts(
        ["a", "b"], embedder=HashEmbedder(8)
    )
    await store.save(path)
    good = path.read_text(encoding="utf-8")
    assert _names(tmp_path) == ["store.json"]

    # The new content goes to a temp file and is moved into place in
    # one step: if that step never happens, the old file is untouched
    # (a plain write_text would already have truncated it).
    await store.add(_chunks("c"))
    seen: list[tuple[str, str]] = []

    def _boom(src: Any, dst: Any) -> None:
        seen.append((Path(src).name, Path(dst).name))
        assert json.loads(Path(src).read_text(encoding="utf-8"))["rows"][2]["content"] == "c"
        raise RuntimeError("crash before rename")

    monkeypatch.setattr(inmemory_mod.os, "replace", _boom)
    with pytest.raises(RuntimeError, match="crash before rename"):
        await store.save(path)
    assert len(seen) == 1 and seen[0][1] == "store.json" and seen[0][0] != "store.json"
    assert path.read_text(encoding="utf-8") == good
    assert _names(tmp_path) == ["store.json"]  # temp file removed


async def test_save_does_not_block_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from loomflow.vectorstore import inmemory as inmemory_mod

    loop_thread = threading.get_ident()
    writer_threads: list[int] = []
    real_write = inmemory_mod._write_json_atomic

    def _spy(target: Path, data: Any) -> None:
        writer_threads.append(threading.get_ident())
        real_write(target, data)

    monkeypatch.setattr(inmemory_mod, "_write_json_atomic", _spy)
    store = await InMemoryVectorStore.from_texts(["a"], embedder=HashEmbedder(8))
    await store.save(tmp_path / "s.json")
    assert writer_threads and writer_threads[0] != loop_thread


async def test_load_rejects_a_different_embedder(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    store = await InMemoryVectorStore.from_texts(
        ["doc a", "doc b"], embedder=HashEmbedder(16)
    )
    await store.save(path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["embedder"] == {"name": "hash-embedder-16", "dimensions": 16}

    with pytest.raises(ValueError, match="hash-embedder-16.*hash-embedder-32"):
        await InMemoryVectorStore.load(path, embedder=HashEmbedder(32))

    class _SameDimsOtherModel(HashEmbedder):
        def __init__(self) -> None:
            super().__init__(16)
            self.name = "other-model"

    with pytest.raises(ValueError, match="original embedder"):
        await InMemoryVectorStore.load(path, embedder=_SameDimsOtherModel())

    again = await InMemoryVectorStore.load(path, embedder=HashEmbedder(16))
    assert (await again.search("doc a", k=1))[0].chunk.content == "doc a"


async def test_load_checks_dimensions_of_files_without_embedder_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "old.json"
    store = await InMemoryVectorStore.from_texts(["a"], embedder=HashEmbedder(8))
    await store.save(path)
    legacy = json.loads(path.read_text(encoding="utf-8"))
    del legacy["embedder"]  # as written by earlier versions
    path.write_text(json.dumps(legacy), encoding="utf-8")

    assert await (await InMemoryVectorStore.load(path, embedder=HashEmbedder(8))).count() == 1
    with pytest.raises(ValueError, match="8 dimensions"):
        await InMemoryVectorStore.load(path, embedder=HashEmbedder(4))


# ---------------------------------------------------------------------------
# 4. Filter semantics — the in-process evaluator
# ---------------------------------------------------------------------------


def test_filter_equality_is_type_strict() -> None:
    assert not evaluate_filter({"year": 2020}, {"year": "2020"})
    assert not evaluate_filter({"n": "5"}, {"n": 5})
    assert not evaluate_filter({"flag": 1}, {"flag": True})
    assert not evaluate_filter({"flag": True}, {"flag": 1})
    assert evaluate_filter({"n": 5}, {"n": 5.0})
    assert evaluate_filter({"flag": True}, {"flag": True})
    assert not evaluate_filter({"n": {"$in": ["5"]}}, {"n": 5})
    assert evaluate_filter({"n": {"$ne": "5"}}, {"n": 5})


def test_filter_none_matches_missing_or_none() -> None:
    assert evaluate_filter({"k": None}, {"k": None})
    assert evaluate_filter({"k": None}, {})
    assert not evaluate_filter({"k": None}, {"k": "x"})
    assert evaluate_filter({"k": {"$ne": None}}, {"k": "x"})
    assert not evaluate_filter({"k": {"$ne": None}}, {})
    assert evaluate_filter({"k": {"$in": [None, "a"]}}, {})
    assert evaluate_filter({"k": {"$exists": True}}, {"k": None})
    assert not evaluate_filter({"k": {"$exists": True}}, {})


def test_filter_lists_match_by_element() -> None:
    meta = {"headers": ["Intro", "Setup"]}
    assert evaluate_filter({"headers": "Intro"}, meta)
    assert evaluate_filter({"headers": {"$in": ["Setup", "Other"]}}, meta)
    assert not evaluate_filter({"headers": {"$in": ["Other"]}}, meta)
    assert evaluate_filter({"headers": {"$nin": ["Other"]}}, meta)
    assert not evaluate_filter({"headers": {"$ne": "Intro"}}, meta)
    assert evaluate_filter({"headers": {"$eq": ["Intro", "Setup"]}}, meta)


def test_filter_ordering_is_number_to_number_or_string_to_string() -> None:
    assert evaluate_filter({"n": {"$gt": 1}}, {"n": 2})
    assert evaluate_filter({"s": {"$gt": "a"}}, {"s": "b"})
    assert not evaluate_filter({"s": {"$gt": "b"}}, {"s": "a"})
    assert not evaluate_filter({"n": {"$gt": "1"}}, {"n": 2})
    assert not evaluate_filter({"n": {"$gt": 1}}, {"n": "2"})
    assert not evaluate_filter({"n": {"$gt": 0}}, {"n": True})
    assert not evaluate_filter({"n": {"$gt": 0}}, {})


# ---------------------------------------------------------------------------
# 4. Filter semantics — Postgres translation
# ---------------------------------------------------------------------------


def test_postgres_none_is_not_bound_as_the_string_none() -> None:
    sql, params = _build_where_sql({"k": None}, [])
    assert "None" not in params
    assert params == []
    assert "(metadata->>'k') IS NULL" in sql

    sql, params = _build_where_sql({"k": {"$ne": None}}, [])
    assert "NOT" in sql and "IS NULL" in sql and params == []


def test_postgres_equality_is_jsonb_typed() -> None:
    # A number is bound as a JSON number and a string as a JSON string:
    # 2020 can't equal "2020" (it used to, via text comparison).
    sql, params = _build_where_sql({"year": 2020}, [])
    assert "(metadata->'year') @> $1::jsonb" in sql
    assert params == ["2020"]
    sql, params = _build_where_sql({"n": "5"}, [])
    assert params == ['"5"']
    sql, params = _build_where_sql({"flag": True}, [])
    assert params == ["true"]


def test_postgres_in_matches_typed_values_and_list_elements() -> None:
    sql, params = _build_where_sql({"n": {"$in": [1, "a"]}}, [])
    assert "jsonb_array_elements($1::jsonb)" in sql
    assert "(metadata->'n') @> _in.v" in sql
    assert json.loads(params[0]) == [1, "a"]

    sql, params = _build_where_sql({"n": {"$in": [None, 1]}}, [])
    assert "IS NULL" in sql and json.loads(params[0]) == [1]

    sql, params = _build_where_sql({"n": {"$in": []}}, [])
    assert sql == "WHERE false" and params == []

    sql, _ = _build_where_sql({"n": {"$nin": [1]}}, [])
    assert sql.startswith("WHERE (NOT (EXISTS")


def test_postgres_string_ordering_only_compares_strings() -> None:
    sql, params = _build_where_sql({"n": {"$gt": "9"}}, [])
    assert "jsonb_typeof(metadata->'n') = 'string'" in sql
    assert 'COLLATE "C" > $1::text' in sql
    assert params == ["9"]


def test_postgres_numeric_ordering_only_compares_numbers() -> None:
    sql, params = _build_where_sql({"n": {"$gte": 3}}, [])
    assert "CASE WHEN jsonb_typeof(metadata->'n') = 'number'" in sql
    assert "::numeric >= $1" in sql
    assert params == [3]
    # A bool operand has no ordering.
    sql, params = _build_where_sql({"n": {"$gt": True}}, [])
    assert sql == "WHERE false" and params == []


# ---------------------------------------------------------------------------
# 2. embed_all
# ---------------------------------------------------------------------------


async def test_embed_all_does_not_swallow_attribute_errors() -> None:
    per_item_calls = 0

    class _Buggy:
        name = "buggy"
        dimensions = 1

        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return None.foo  # type: ignore[attr-defined]

        async def embed(self, text: str) -> list[float]:
            nonlocal per_item_calls
            per_item_calls += 1
            return [1.0]

    with pytest.raises(AttributeError, match="foo"):
        await embed_all(_Buggy(), ["a", "b", "c"])
    assert per_item_calls == 0


async def test_embed_all_falls_back_when_embed_batch_is_absent() -> None:
    class _Single:
        name = "single"
        dimensions = 1

        async def embed(self, text: str) -> list[float]:
            return [float(len(text))]

    class _NotImplemented(_Single):
        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            raise NotImplementedError

    assert await embed_all(_Single(), ["a", "bb"]) == [[1.0], [2.0]]  # type: ignore[arg-type]
    assert await embed_all(_NotImplemented(), ["a", "bb"]) == [[1.0], [2.0]]


async def test_embed_all_rejects_a_short_batch() -> None:
    class _Short:
        name = "short"
        dimensions = 1

        async def embed(self, text: str) -> list[float]:
            return [1.0]

        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [[1.0]] * (len(texts) - 1)

    with pytest.raises(ValueError, match="2 vectors for 3 texts"):
        await InMemoryVectorStore(_Short()).add(_chunks("a", "b", "c"))


# ---------------------------------------------------------------------------
# 15. FAISS — concurrent adds keep rows and ids in step (fake faiss)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_faiss(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    np = pytest.importorskip("numpy")
    import threading
    import time

    class _Index:
        def __init__(self, dim: int) -> None:
            self.dim = dim
            self.rows: list[Any] = []
            self._busy = threading.Lock()
            self.overlapping_adds = 0

        def add(self, arr: Any) -> None:
            if not self._busy.acquire(blocking=False):
                self.overlapping_adds += 1
                self._busy.acquire()
            try:
                # The first row of a batch decides how long the "index
                # build" takes, so a later call can finish first.
                time.sleep(0.05 if float(arr[0][0]) > 0.5 else 0.0)
                self.rows.extend(arr.copy())
            finally:
                self._busy.release()

        def search(self, q: Any, n: int) -> tuple[Any, Any]:
            if not self.rows:
                return np.zeros((1, 0), dtype="float32"), np.zeros((1, 0), dtype="int64")
            matrix = np.stack(self.rows)
            sims = matrix @ q[0]
            order = np.argsort(-sims)[:n]
            return sims[order][None, :], order[None, :]

        def reset(self) -> None:
            self.rows = []

    module = types.ModuleType("faiss")
    module.METRIC_INNER_PRODUCT = 0  # type: ignore[attr-defined]
    module.METRIC_L2 = 1  # type: ignore[attr-defined]
    module.index_factory = lambda dim, spec, metric: _Index(dim)  # type: ignore[attr-defined]

    def _normalize(arr: Any) -> None:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr /= norms

    module.normalize_L2 = _normalize  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faiss", module)
    return module


class _AxisEmbedder:
    """``slow`` → axis 0, anything else → axis 1."""

    name = "axis"
    dimensions = 2

    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0] if text.startswith("slow") else [0.0, 1.0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


async def test_faiss_concurrent_adds_keep_ids_aligned(
    fake_faiss: types.ModuleType,
) -> None:
    from loomflow.vectorstore.faiss import FAISSVectorStore

    store = FAISSVectorStore(_AxisEmbedder(), dimension=2)
    await store.add(_chunks("seed fast"), ids=["seed"])  # builds the index

    async with anyio.create_task_group() as tg:
        # The slow add starts first; without the lock the fast one
        # finishes its index.add first and the two swap rows.
        tg.start_soon(store.add, _chunks("slow one"), ["slow"])
        await anyio.sleep(0.01)
        tg.start_soon(store.add, _chunks("fast one"), ["fast"])

    assert store._index.overlapping_adds == 0
    slow_hit = (await store.search("slow query", k=1))[0]
    assert slow_hit.id == "slow" and slow_hit.chunk.content == "slow one"
    assert slow_hit.score == pytest.approx(1.0)
    for row, cid in enumerate(store._ids):
        expected = [1.0, 0.0] if cid == "slow" else [0.0, 1.0]
        assert list(store._index.rows[row]) == pytest.approx(expected)


async def test_faiss_readding_an_id_upserts(fake_faiss: types.ModuleType) -> None:
    from loomflow.vectorstore.faiss import FAISSVectorStore

    store = FAISSVectorStore(_AxisEmbedder(), dimension=2)
    await store.add(_chunks("slow a", "fast b", "fast c"), ids=["a", "b", "c"])
    await store.add(_chunks("slow a v2"), ids=["a"])
    assert await store.count() == 3
    assert [c.content for c in await store.get_by_ids(["a"])] == ["slow a v2"]
    hits = await store.search("slow", k=3)
    assert [h.id for h in hits].count("a") == 1
    assert hits[0].chunk.content == "slow a v2"
    with pytest.raises(ValueError, match="k must be"):
        await store.search("slow", k=0)
    with pytest.raises(ValueError, match="NaN"):
        await store.search_by_vector([float("nan"), 0.0], k=1)


# ---------------------------------------------------------------------------
# Chroma (real in-process client)
# ---------------------------------------------------------------------------


def _chroma(collection: str | None = None, embedder: Any = None) -> Any:
    chromadb = pytest.importorskip("chromadb")
    import uuid

    from loomflow.vectorstore.chroma import ChromaVectorStore

    client = chromadb.EphemeralClient()
    return ChromaVectorStore(
        embedder or HashEmbedder(8),
        client=client,
        collection_name=collection or f"rag_{uuid.uuid4().hex}",
    )


async def test_chroma_readding_an_id_upserts() -> None:
    store = _chroma()
    await store.add(_chunks("v1"), ids=["x"])
    await store.add(_chunks("v2"), ids=["x"])
    assert [c.content for c in await store.get_by_ids(["x"])] == ["v2"]
    assert await store.count() == 1
    # Duplicate ids inside one call: last wins (Chroma itself rejects them).
    await store.add(_chunks("a", "b"), ids=["y", "y"])
    assert [c.content for c in await store.get_by_ids(["y"])] == ["b"]


async def test_chroma_rejects_non_positive_k_with_value_error() -> None:
    store = _chroma()
    await store.add(_chunks("v1"))
    for k in (0, -1):
        with pytest.raises(ValueError, match="k must be"):
            await store.search("v1", k=k)
    assert len(await store.search("v1", k=50)) == 1


async def test_chroma_add_splits_to_the_client_batch_size() -> None:
    class _CountingEmbedder(HashEmbedder):
        def __init__(self) -> None:
            super().__init__(4)
            self.batches: list[int] = []

        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            self.batches.append(len(texts))
            return await super().embed_batch(texts)

    embedder = _CountingEmbedder()
    store = _chroma(embedder=embedder)
    store._client = types.SimpleNamespace(get_max_batch_size=lambda: 7)
    upserts: list[int] = []
    real_upsert = store._collection.upsert

    def _upsert(**kwargs: Any) -> Any:
        upserts.append(len(kwargs["ids"]))
        return real_upsert(**kwargs)

    store._collection = types.SimpleNamespace(
        upsert=_upsert,
        count=store._collection.count,
        get=store._collection.get,
        query=store._collection.query,
    )
    ids = await store.add(_chunks(*[f"d{i}" for i in range(20)]))
    assert len(ids) == 20
    assert upserts == [7, 7, 6]
    assert embedder.batches == [7, 7, 6]  # embedded batch by batch
    assert await store.count() == 20


async def test_chroma_validates_metadata_before_embedding() -> None:
    class _Paid(HashEmbedder):
        calls = 0

        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            type(self).calls += 1
            return await super().embed_batch(texts)

    store = _chroma(embedder=_Paid(4))
    with pytest.raises(ValueError, match="reserved"):
        await store.add([Chunk(content="x", metadata={"__loomflow_x": 1})])
    with pytest.raises(ValueError, match="ids length"):
        await store.add(_chunks("a", "b"), ids=["only-one"])
    assert _Paid.calls == 0


async def test_chroma_metadata_round_trips() -> None:
    store = _chroma()
    metadata = {
        "headers": ["A", "B"],
        "_empty": "user value",
        "nothing": None,
        "page": 3,
        "nested": {"k": [1, 2]},
    }
    await store.add([Chunk(content="t", metadata=dict(metadata))], ids=["1"])
    await store.add([Chunk(content="n", metadata={"n": None})], ids=["2"])
    await store.add([Chunk(content="e", metadata={})], ids=["3"])
    await store.add([Chunk(content="u", metadata={"_empty": True})], ids=["4"])

    got = {cid: (await store.get_by_ids([cid]))[0].metadata for cid in "1234"}
    assert got["1"] == {
        "headers": ["A", "B"],
        "_empty": "user value",
        "page": 3,
        "nested": {"k": [1, 2]},
    }
    assert got["2"] == {}  # None is stored as an absent key
    assert got["3"] == {}
    assert got["4"] == {"_empty": True}
    hit = (await store.search("t", k=1))[0]
    assert hit.chunk.metadata["headers"] == ["A", "B"]


async def test_chroma_reads_legacy_empty_sentinel() -> None:
    store = _chroma()
    vector = await HashEmbedder(8).embed("old")
    store._collection.add(
        ids=["old"],
        embeddings=[vector],
        documents=["old"],
        metadatas=[{"_empty": True}],  # written by earlier versions
    )
    assert (await store.get_by_ids(["old"]))[0].metadata == {}
    assert (await store.search("old", k=1))[0].chunk.metadata == {}


async def test_chroma_filters_follow_the_shared_contract() -> None:
    chroma = _chroma()
    memory = InMemoryVectorStore(HashEmbedder(8))
    rows = [
        ("1", {"headers": ["A", "B"], "year": 2020, "tag": "x"}),
        ("2", {"headers": ["C"], "year": "2020", "tag": "y", "n": None}),
        ("3", {"year": 2021, "flag": True, "tag": "z"}),
        ("4", {}),
    ]
    for store in (chroma, memory):
        await store.add(
            [Chunk(content=f"doc {cid}", metadata=dict(md)) for cid, md in rows],
            ids=[cid for cid, _ in rows],
        )
    filters: list[dict[str, Any]] = [
        {"headers": {"$in": ["A"]}},
        {"headers": "C"},
        {"headers": {"$nin": ["A"]}},
        {"year": 2020},
        {"year": "2020"},
        {"year": {"$in": [2020, "2020"]}},
        {"year": {"$gt": 2020}},
        {"tag": {"$gt": "x"}},
        {"tag": {"$lte": "y"}},
        {"n": None},
        {"tag": {"$ne": None}},
        {"flag": 1},
        {"flag": True},
        {"tag": {"$in": []}},
        {"$or": [{"headers": "A"}, {"tag": {"$gte": "z"}}]},
        {"$and": [{"year": {"$exists": True}}, {"$not": {"tag": "x"}}]},
    ]
    for flt in filters:
        expected = sorted(r.id for r in await memory.search("doc", k=10, filter=flt))
        actual = sorted(r.id for r in await chroma.search("doc", k=10, filter=flt))
        assert actual == expected, flt
    assert sorted(
        r.id for r in await chroma.search("doc", k=10, filter={"headers": {"$in": ["A"]}})
    ) == ["1"]


async def test_chroma_filtered_search_widens_until_k_match() -> None:
    store = _chroma()
    # 80 rows whose JSON-encoded list does NOT contain the wanted tag
    # rank ahead of nothing in particular; only 3 rows match.
    chunks = [
        Chunk(content=f"doc {i}", metadata={"tags": ["other"]}) for i in range(80)
    ] + [Chunk(content=f"hit {i}", metadata={"tags": ["wanted"]}) for i in range(3)]
    await store.add(chunks)
    hits = await store.search("doc 1", k=3, filter={"tags": "wanted"})
    assert sorted(h.chunk.content for h in hits) == ["hit 0", "hit 1", "hit 2"]


# ---------------------------------------------------------------------------
# 10. Default Chroma collection shared between stores
# ---------------------------------------------------------------------------


async def test_default_chroma_stores_warn_when_they_share_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("chromadb")
    from loomflow.vectorstore import chroma as chroma_mod
    from loomflow.vectorstore.chroma import ChromaVectorStore

    first = ChromaVectorStore(HashEmbedder(8))
    second = ChromaVectorStore(HashEmbedder(8))
    _wipe(first)  # clean slate, whatever ran before
    monkeypatch.setattr(chroma_mod, "_SHARED_DEFAULT_WARNED", False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            await first.add(_chunks("private to the first store"))
            await first.search("private", k=1)  # its own data: no warning
        with pytest.warns(UserWarning, match="jeeves_vectors.*collection_name"):
            hits = await second.search("private", k=1)
        assert hits  # still works — compatibility is kept
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            await second.search("private", k=1)  # once per process
    finally:
        _wipe(first)


async def test_named_or_persistent_chroma_stores_do_not_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("chromadb")
    from loomflow.vectorstore import chroma as chroma_mod
    from loomflow.vectorstore.chroma import ChromaVectorStore

    monkeypatch.setattr(chroma_mod, "_SHARED_DEFAULT_WARNED", False)
    first = ChromaVectorStore(HashEmbedder(8), persist_directory=str(tmp_path))
    await first.add(_chunks("persisted"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Reopening persisted data under the default name is the point.
        reopened = ChromaVectorStore(HashEmbedder(8), persist_directory=str(tmp_path))
        assert len(await reopened.search("persisted", k=1)) == 1


def _wipe(store: Any) -> None:
    ids = store._collection.get(include=[])["ids"]
    if ids:
        store._collection.delete(ids=ids)
