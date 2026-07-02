"""Vectorstore regression tests for the reviewed fixes (WSC).

9.  Postgres JSONB filter: numeric operands compare numerically and
    metadata keys are validated before SQL interpolation.
11. ``index_document`` offloads sync load/chunk work to a thread
    (behavioural smoke test — same results as before).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loomflow.memory.embedder import HashEmbedder
from loomflow.vectorstore._filter import FilterError
from loomflow.vectorstore.postgres import _build_where_sql

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Fix 9 — numeric casts
# ---------------------------------------------------------------------------


def test_numeric_operator_casts_to_numeric() -> None:
    sql, params = _build_where_sql({"page": {"$lt": 10}}, [])
    assert "::numeric" in sql
    assert "<" in sql
    assert params == [10]


def test_numeric_operator_binds_floats() -> None:
    sql, params = _build_where_sql({"score": {"$gte": 0.5}}, [])
    assert "::numeric" in sql
    assert params == [0.5]


def test_string_operator_stays_textual() -> None:
    sql, params = _build_where_sql({"source": {"$eq": "a.pdf"}}, [])
    assert "::numeric" not in sql
    assert params == ["a.pdf"]


def test_bool_operand_not_treated_as_numeric() -> None:
    sql, params = _build_where_sql({"flag": {"$eq": True}}, [])
    assert "::numeric" not in sql
    assert params == ["true"]


def test_scalar_numeric_shorthand_casts() -> None:
    sql, params = _build_where_sql({"page": 3}, [])
    assert "::numeric" in sql
    assert params == [3]


def test_numeric_cast_guarded_against_non_numeric_rows() -> None:
    # The cast is wrapped in a regex guard so a non-numeric metadata
    # value can't error the whole query.
    sql, _ = _build_where_sql({"page": {"$gt": 1}}, [])
    assert "~" in sql


# ---------------------------------------------------------------------------
# Fix 9 — key validation (no SQL injection via field names)
# ---------------------------------------------------------------------------


def test_quoted_key_rejected() -> None:
    with pytest.raises(FilterError, match="invalid metadata key"):
        _build_where_sql({"bad'key": 1}, [])


def test_quoted_key_rejected_in_operator_form() -> None:
    with pytest.raises(FilterError, match="invalid metadata key"):
        _build_where_sql({"x') OR ('1'='1": {"$gte": 5}}, [])


def test_normal_keys_accepted() -> None:
    sql, _ = _build_where_sql({"chunk.page-no_1": "x"}, [])
    assert "chunk.page-no_1" in sql


# ---------------------------------------------------------------------------
# Fix 11 — index_document still works with thread-offloaded load/split
# ---------------------------------------------------------------------------


async def test_index_document_offloaded_smoke(tmp_path: Path) -> None:
    pytest.importorskip("loomflow.loader")
    from loomflow.vectorstore import InMemoryVectorStore, index_document

    path = tmp_path / "doc.md"
    path.write_text("# Title\n\nalpha beta gamma\n\ndelta epsilon\n")
    store = InMemoryVectorStore(embedder=HashEmbedder(dimensions=16))
    ids = await index_document(path, store)
    assert ids
    results = await store.search("alpha beta gamma", k=1)
    assert results
    assert "alpha" in results[0].chunk.content
