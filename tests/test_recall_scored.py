"""Hybrid scored recall tests — :meth:`Memory.recall_scored`.

Covers the new :class:`EpisodeMatch`-returning surface added to the
``Memory`` protocol: ``InMemoryMemory`` ships a BM25-only hybrid;
``VectorMemory`` ships full BM25 + cosine + RRF; backends without
native hybrid (Chroma / Sqlite / Postgres / Redis / AutoExtract /
Lazy) delegate to ``recall()`` and wrap with neutral scores via
``default_recall_scored``.

Tests focus on observable contract, not internals — assert on
score-component presence, ordering, and back-compat with ``recall()``.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from loomflow import (
    Episode,
    EpisodeMatch,
    InMemoryMemory,
)
from loomflow.memory import VectorMemory, default_recall_scored
from loomflow.memory.auto_extract import AutoExtractMemory

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# InMemoryMemory — BM25 hybrid; no embedder so vector_score is always None
# ---------------------------------------------------------------------------


async def test_inmemory_recall_scored_returns_episode_match() -> None:
    """Every result is wrapped as an :class:`EpisodeMatch` with at
    least a final ``score``. Backends always populate the wrapper
    even when only one component (here, BM25) is available."""
    mem = InMemoryMemory()
    await mem.remember(
        Episode(
            input="favourite colour question",
            output="my favourite colour is blue",
            user_id="alice", session_id="s1",
            occurred_at=datetime(2026, 5, 1, tzinfo=None),
        )
    )
    matches = await mem.recall_scored("favourite colour", user_id="alice")
    assert matches, "expected at least one match"
    for m in matches:
        assert isinstance(m, EpisodeMatch)
        assert isinstance(m.episode, Episode)
        assert m.score >= 0


async def test_inmemory_recall_scored_ranks_bm25_matches_first() -> None:
    """A query that lexically matches one episode but not another
    should rank the matching episode first, regardless of recency.
    This is the win over the previous substring-then-recency
    behaviour: BM25 catches relevance, not just substring presence."""
    mem = InMemoryMemory()

    # Older episode that LEXICALLY matches the query.
    await mem.remember(
        Episode(
            input="user asked about postgres replication",
            output="explained physical vs logical replication",
            user_id="alice", session_id="s1",
            occurred_at=datetime(2026, 1, 1, tzinfo=None),
        )
    )
    # Newer episode that doesn't match.
    await mem.remember(
        Episode(
            input="weather chat",
            output="discussed the rain",
            user_id="alice", session_id="s1",
            occurred_at=datetime(2026, 5, 10, tzinfo=None),
        )
    )

    matches = await mem.recall_scored(
        "postgres replication", user_id="alice", limit=2
    )
    # First match is the relevant one even though it's older.
    assert "postgres" in matches[0].episode.input.lower()
    assert matches[0].bm25_score is not None
    assert matches[0].bm25_score > 0


async def test_inmemory_recall_scored_no_query_falls_back_to_recency() -> None:
    """An empty query has no lexical signal — fall back to recency
    so the caller always gets a useful answer (last N episodes for
    that user). Scores are neutral (1.0) to flag the fallback."""
    mem = InMemoryMemory()
    await mem.remember(
        Episode(
            input="old", output="x", user_id="alice", session_id="s1",
            occurred_at=datetime(2026, 1, 1, tzinfo=None),
        )
    )
    await mem.remember(
        Episode(
            input="new", output="y", user_id="alice", session_id="s1",
            occurred_at=datetime(2026, 5, 10, tzinfo=None),
        )
    )

    matches = await mem.recall_scored("", user_id="alice", limit=2)
    # Most recent first; both scores neutral.
    assert matches[0].episode.input == "new"
    assert matches[0].score == 1.0
    assert matches[0].bm25_score is None  # no BM25 ran for empty query


async def test_inmemory_recall_scored_no_matches_falls_back_to_recency() -> None:
    """A query with zero BM25 hits should still return SOMETHING
    useful (recency fallback) rather than an empty list — losing
    all results for an off-topic query is worse than 'best guess
    by recency'. Scores are 0.0 to flag this is the no-signal path."""
    mem = InMemoryMemory()
    await mem.remember(
        Episode(
            input="hello", output="world", user_id="alice", session_id="s1",
            occurred_at=datetime(2026, 5, 1, tzinfo=None),
        )
    )
    matches = await mem.recall_scored(
        "completely unrelated query terms", user_id="alice"
    )
    assert len(matches) == 1
    assert matches[0].score == 0.0


async def test_inmemory_recall_scored_respects_user_id_partition() -> None:
    """The new method MUST keep the same hard ``user_id``
    namespace partition as :meth:`recall`. A query for alice must
    never see bob's episodes."""
    mem = InMemoryMemory()
    await mem.remember(
        Episode(
            input="alice's secret", output="42", user_id="alice", session_id="s1",
            occurred_at=datetime(2026, 5, 1, tzinfo=None),
        )
    )
    await mem.remember(
        Episode(
            input="alice's secret", output="not_42", user_id="bob",
            session_id="s2",
            occurred_at=datetime(2026, 5, 1, tzinfo=None),
        )
    )

    alice_matches = await mem.recall_scored(
        "alice's secret", user_id="alice"
    )
    assert all(m.episode.user_id == "alice" for m in alice_matches)
    assert all(m.episode.output == "42" for m in alice_matches)


# ---------------------------------------------------------------------------
# VectorMemory — full BM25 + cosine + RRF
# ---------------------------------------------------------------------------


async def test_vector_recall_scored_populates_both_score_components() -> None:
    """``VectorMemory`` runs both BM25 and cosine arms, so each
    match should have BOTH ``bm25_score`` and ``vector_score``
    populated when the query produces hits in both rankings."""
    mem = VectorMemory()  # uses HashEmbedder by default
    await mem.remember(
        Episode(
            input="docker container networking",
            output="bridge mode is the default",
            user_id="alice", session_id="s1",
        )
    )

    matches = await mem.recall_scored(
        "docker networking", user_id="alice"
    )
    assert matches
    top = matches[0]
    # BM25 ran (lexical overlap on "docker"/"networking").
    assert top.bm25_score is not None and top.bm25_score > 0
    # Vector ran (HashEmbedder produces a deterministic embedding).
    assert top.vector_score is not None


async def test_vector_recall_scored_alpha_zero_collapses_to_bm25() -> None:
    """``alpha=0`` should give pure BM25 ranking. With a query that
    BM25 strongly matches one episode and cosine doesn't, alpha=0
    should rank the BM25-favoured episode first."""
    mem = VectorMemory()
    await mem.remember(
        Episode(
            input="GitHub Actions workflow",
            output="defined in .github/workflows",
            user_id="alice", session_id="s1",
        )
    )
    await mem.remember(
        Episode(
            input="random unrelated chat",
            output="weather is fine today",
            user_id="alice", session_id="s1",
        )
    )
    matches = await mem.recall_scored(
        "GitHub Actions", user_id="alice", alpha=0.0
    )
    assert "GitHub" in matches[0].episode.input


async def test_vector_recall_scored_empty_query_neutral_scores() -> None:
    mem = VectorMemory()
    await mem.remember(
        Episode(
            input="x", output="y", user_id="alice", session_id="s1",
            occurred_at=datetime(2026, 5, 1, tzinfo=None),
        )
    )
    matches = await mem.recall_scored("", user_id="alice")
    assert all(m.score == 1.0 for m in matches)


# ---------------------------------------------------------------------------
# default_recall_scored — fallback helper used by shim backends
# ---------------------------------------------------------------------------


async def test_default_recall_scored_wraps_with_neutral_score() -> None:
    """Backends without native hybrid scoring use this helper to
    satisfy the protocol. Verify the wrapping is faithful and the
    score is the configured neutral default."""
    eps = [
        Episode(
            input=f"e{i}", output=f"o{i}",
            user_id="alice", session_id="s",
        )
        for i in range(3)
    ]
    matches = default_recall_scored(eps)
    assert len(matches) == 3
    for m, e in zip(matches, eps, strict=True):
        assert m.episode is e
        assert m.score == 1.0
        assert m.bm25_score is None
        assert m.vector_score is None


async def test_default_recall_scored_custom_score() -> None:
    eps = [Episode(input="x", output="y", user_id="alice", session_id="s")]
    matches = default_recall_scored(eps, score=0.5)
    assert matches[0].score == 0.5


# ---------------------------------------------------------------------------
# AutoExtractMemory — wrapper passes through to inner's scored recall
# ---------------------------------------------------------------------------


async def test_auto_extract_passes_through_to_inner_scored_recall() -> None:
    """AutoExtractMemory wrapping a backend WITH native scored
    recall (InMemoryMemory) should forward calls and preserve the
    score breakdown — not lose it by wrapping into neutral scores."""
    inner = InMemoryMemory()
    await inner.remember(
        Episode(
            input="kubernetes pod scheduling",
            output="scheduler picks nodes",
            user_id="alice", session_id="s1",
        )
    )
    # AutoExtractMemory needs a model for consolidation; skip auto
    # extract by passing a dummy that's never invoked here.
    from loomflow import EchoModel
    from loomflow.memory import Consolidator

    wrapped = AutoExtractMemory(
        inner, Consolidator(model=EchoModel())
    )
    matches = await wrapped.recall_scored(
        "kubernetes scheduling", user_id="alice"
    )
    assert matches
    # BM25 score travelled through the wrapper unchanged.
    assert matches[0].bm25_score is not None
    assert matches[0].bm25_score > 0
