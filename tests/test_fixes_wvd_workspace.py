"""Regression tests for the WVD review fixes in ``loomflow/workspace/``.

Covers:

* ``LocalDiskWorkspace.attribute_outcome`` walks the notes tree ONCE
  per call (slug -> path map) instead of one ``rglob`` per cited slug.
* ``LocalDiskWorkspace._semantic_scores`` walks the notes tree ONCE
  instead of calling ``_find_note_path`` (a per-author walk) per
  candidate note.
* The scoring / citation helpers are hoisted to ``_common`` and
  shared by both backends — single source of truth, no drift.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from loomflow.workspace.disk import LocalDiskWorkspace

pytestmark = pytest.mark.anyio


class _StubEmbedder:
    """Deterministic char-bucket embedder (no API key needed)."""

    name = "stub-embedder"
    dimensions = 26

    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * 26
        for ch in text.lower():
            if ch.isalpha():
                vec[ord(ch) - ord("a")] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def _count_walks(ws: LocalDiskWorkspace) -> list[int]:
    """Wrap the workspace's ``_walk_note_files`` with a call counter.

    Returns a single-element list that holds the running count.
    """
    counter = [0]
    orig = ws._walk_note_files

    def counting(user_id: str | None) -> list[Path]:
        counter[0] += 1
        return orig(user_id)

    ws._walk_note_files = counting  # type: ignore[method-assign]
    return counter


# ---------------------------------------------------------------------------
# Fix 4a — attribute_outcome: one walk per call, not one per slug
# ---------------------------------------------------------------------------


async def test_attribute_outcome_walks_tree_once(tmp_path: Path) -> None:
    ws = LocalDiskWorkspace(tmp_path / "ws")
    n1 = await ws.write_note(
        author="alice", title="Alpha", body="aa", user_id="u"
    )
    n2 = await ws.write_note(
        author="bob", title="Beta", body="bb", user_id="u"
    )
    n3 = await ws.write_note(
        author="bob", title="Gamma", body="cc", user_id="u"
    )

    walks = _count_walks(ws)
    updated = await ws.attribute_outcome(
        success=True,
        slugs=[n1.slug, n2.slug, n3.slug, "999-does-not-exist"],
        user_id="u",
    )
    assert updated == 3
    # ONE walk builds the slug -> path map for all cited slugs (the
    # index regeneration patches its warm cache, no extra walk).
    assert walks[0] == 1

    # Correctness: every cited note (across BOTH authors) got its
    # citation metadata bumped; the unknown slug was skipped.
    for slug in (n1.slug, n2.slug, n3.slug):
        note = await ws.read_note(slug, user_id="u")
        assert note is not None
        assert note.cited_count == 1
        assert note.success_count == 1
        assert note.last_cited_at is not None


async def test_attribute_outcome_failure_counts_only_citations(
    tmp_path: Path,
) -> None:
    ws = LocalDiskWorkspace(tmp_path / "ws")
    n = await ws.write_note(
        author="alice", title="Alpha", body="aa", user_id="u"
    )
    updated = await ws.attribute_outcome(
        success=False, slugs=[n.slug], user_id="u"
    )
    assert updated == 1
    note = await ws.read_note(n.slug, user_id="u")
    assert note is not None
    assert note.cited_count == 1
    assert note.success_count == 0


# ---------------------------------------------------------------------------
# Fix 4b — _semantic_scores: one walk, no per-candidate _find_note_path
# ---------------------------------------------------------------------------


async def test_semantic_search_walks_tree_once_not_per_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = LocalDiskWorkspace(tmp_path / "ws", embedder=_StubEmbedder())
    await ws.write_note(
        author="agent", title="apple pie recipe",
        body="how to bake apples", user_id="u",
    )
    await ws.write_note(
        author="agent", title="zebra facts",
        body="black and white stripes", user_id="u",
    )
    # Namespaced note — the (author, slug) map must still find it
    # one level deeper (notes/<author>/<ns>/<slug>.md).
    await ws.write_note(
        author="agent", title="apple orchard",
        body="apples on trees", user_id="u", namespace="farming",
    )

    # The search path must not fall back to the per-candidate
    # author-subtree walk.
    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "_find_note_path called from the semantic search path"
        )

    monkeypatch.setattr(ws, "_find_note_path", _boom)
    walks = _count_walks(ws)

    results = await ws.search_notes("apple", user_id="u", mode="semantic")
    # One walk for the candidate scan + one for the embedding map.
    assert walks[0] == 2
    assert len(results) >= 2
    titles = [m.summary.title for m in results]
    assert any("apple" in t for t in titles)
    # Both apple notes (incl. the namespaced one) outrank zebra.
    assert "zebra" not in titles[0]


# ---------------------------------------------------------------------------
# Fix 5 — scoring/citation helpers are shared via _common
# ---------------------------------------------------------------------------


def test_scoring_helpers_are_single_sourced() -> None:
    from loomflow.workspace import _common, disk, inmemory

    assert disk._cosine is inmemory._cosine is _common.cosine_similarity
    assert disk._rrf_fuse is inmemory._rrf_fuse is _common.rrf_fuse
    assert (
        disk._score_semantic
        is inmemory._score_semantic
        is _common.score_semantic
    )
    assert (
        disk._apply_relevance_boost
        is inmemory._apply_relevance_boost
        is _common.apply_relevance_boost
    )
    assert (
        disk._log_citation
        is inmemory._log_citation
        is _common.log_citation
    )
    assert (
        disk._drain_citations
        is inmemory._drain_citations
        is _common.drain_citations
    )
