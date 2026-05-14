"""Citation tracking + outcome attribution + relevance-aware search.

These three pieces close the workspace's self-improvement loop:
notes that get read AND associated with successful outcomes
accumulate ``cited_count`` + ``success_count``, which can be used
to rank future searches via ``boost_relevance=True``.
"""

from __future__ import annotations

import pytest

from loomflow import InMemoryWorkspace, LocalDiskWorkspace
from loomflow.core.context import _ambient_citations_var

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Citation tracking — read_note logs slugs into the ambient set
# ---------------------------------------------------------------------------


async def test_read_note_logs_citation_in_run() -> None:
    ws = InMemoryWorkspace()
    n = await ws.write_note(
        author="agent", title="T", body="b", user_id="u",
    )
    # Outside a run, citations are a no-op (no active set).
    await ws.read_note(n.slug, user_id="u")
    # Inside a run (install the contextvar manually):
    cites: set[str] = set()
    token = _ambient_citations_var.set(cites)
    try:
        await ws.read_note(n.slug, user_id="u")
        assert n.slug in cites
    finally:
        _ambient_citations_var.reset(token)


async def test_read_version_also_logs() -> None:
    ws = InMemoryWorkspace()
    n = await ws.write_note(
        author="agent", title="T", body="v1", user_id="u",
    )
    await ws.update_note(
        author="agent", slug=n.slug, body="v2", user_id="u",
    )
    cites: set[str] = set()
    token = _ambient_citations_var.set(cites)
    try:
        await ws.read_version(
            n.slug, 1, author="agent", user_id="u"
        )
        assert n.slug in cites
    finally:
        _ambient_citations_var.reset(token)


async def test_citation_outside_run_is_no_op() -> None:
    """No contextvar set → reads still work, just don't log."""
    ws = InMemoryWorkspace()
    n = await ws.write_note(
        author="agent", title="T", body="b", user_id="u",
    )
    # No contextvar active.
    result = await ws.read_note(n.slug, user_id="u")
    assert result is not None
    # No exception, no problem.


# ---------------------------------------------------------------------------
# attribute_outcome flow
# ---------------------------------------------------------------------------


async def test_attribute_outcome_increments_counts() -> None:
    ws = InMemoryWorkspace()
    n = await ws.write_note(
        author="agent", title="T", body="b", user_id="u",
    )
    cites: set[str] = set()
    token = _ambient_citations_var.set(cites)
    try:
        await ws.read_note(n.slug, user_id="u")
        updated = await ws.attribute_outcome(
            success=True, user_id="u"
        )
        assert updated == 1
        after = await ws.read_note(n.slug, user_id="u")
        assert after is not None
        assert after.cited_count == 1
        assert after.success_count == 1
        assert after.last_cited_at is not None
    finally:
        _ambient_citations_var.reset(token)


async def test_attribute_outcome_failure_only_increments_cited() -> None:
    """A failed run still counts as a citation, but doesn't
    increment success_count."""
    ws = InMemoryWorkspace()
    n = await ws.write_note(
        author="agent", title="T", body="b", user_id="u",
    )
    cites: set[str] = set()
    token = _ambient_citations_var.set(cites)
    try:
        await ws.read_note(n.slug, user_id="u")
        await ws.attribute_outcome(success=False, user_id="u")
        after = await ws.read_note(n.slug, user_id="u")
        assert after is not None
        assert after.cited_count == 1
        assert after.success_count == 0
    finally:
        _ambient_citations_var.reset(token)


async def test_attribute_outcome_drains_set() -> None:
    """After attribute_outcome, the citation set should be empty
    so subsequent attributions don't double-count."""
    ws = InMemoryWorkspace()
    n = await ws.write_note(
        author="agent", title="T", body="b", user_id="u",
    )
    cites: set[str] = set()
    token = _ambient_citations_var.set(cites)
    try:
        await ws.read_note(n.slug, user_id="u")
        await ws.attribute_outcome(success=True, user_id="u")
        # Second attribution with no new reads → no updates.
        updated_again = await ws.attribute_outcome(
            success=True, user_id="u"
        )
        assert updated_again == 0
    finally:
        _ambient_citations_var.reset(token)


async def test_attribute_outcome_no_run_returns_zero() -> None:
    ws = InMemoryWorkspace()
    # No contextvar active.
    updated = await ws.attribute_outcome(success=True, user_id="u")
    assert updated == 0


async def test_attribute_outcome_disk_persistence() -> None:
    """Disk backend must persist citation counts to frontmatter."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        ws = LocalDiskWorkspace(d)
        n = await ws.write_note(
            author="agent", title="T", body="b", user_id="u",
        )
        cites: set[str] = set()
        token = _ambient_citations_var.set(cites)
        try:
            await ws.read_note(n.slug, user_id="u")
            await ws.attribute_outcome(success=True, user_id="u")
        finally:
            _ambient_citations_var.reset(token)
        # Re-open and confirm persistence.
        ws2 = LocalDiskWorkspace(d)
        after = await ws2.read_note(n.slug, user_id="u")
        assert after is not None
        assert after.cited_count == 1
        assert after.success_count == 1


# ---------------------------------------------------------------------------
# Relevance-aware search
# ---------------------------------------------------------------------------


async def test_boost_relevance_promotes_popular_notes() -> None:
    """A note that's been cited many successful times should
    outrank a never-cited title-match when boost_relevance=True."""
    ws = InMemoryWorkspace()
    # Two notes; both match the query "thing" in body.
    popular = await ws.write_note(
        author="agent", title="A note", body="thing word",
        user_id="u",
    )
    await ws.write_note(
        author="agent", title="B note", body="thing word",
        user_id="u",
    )
    # Cite "popular" multiple times with success.
    for _ in range(5):
        cites: set[str] = set()
        token = _ambient_citations_var.set(cites)
        try:
            await ws.read_note(popular.slug, user_id="u")
            await ws.attribute_outcome(success=True, user_id="u")
        finally:
            _ambient_citations_var.reset(token)
    # Search WITHOUT boost — both score equally (same tier).
    plain = await ws.search_notes(
        "thing", user_id="u", boost_relevance=False
    )
    # Search WITH boost — popular should win.
    boosted = await ws.search_notes(
        "thing", user_id="u", boost_relevance=True
    )
    assert len(boosted) == 2
    assert boosted[0].summary.slug == popular.slug
    # And the boosted score must be higher than the plain score
    # for the same note.
    popular_plain = next(
        m for m in plain if m.summary.slug == popular.slug
    )
    popular_boosted = next(
        m for m in boosted if m.summary.slug == popular.slug
    )
    assert popular_boosted.score > popular_plain.score


async def test_boost_relevance_default_off() -> None:
    """Default behavior preserves v0.9 ranking (no boost)."""
    ws = InMemoryWorkspace()
    a = await ws.write_note(
        author="agent", title="A", body="x", user_id="u",
    )
    b = await ws.write_note(
        author="agent", title="B", body="x", user_id="u",
    )
    # Make `a` very popular.
    for _ in range(10):
        cites: set[str] = set()
        token = _ambient_citations_var.set(cites)
        try:
            await ws.read_note(a.slug, user_id="u")
            await ws.attribute_outcome(success=True, user_id="u")
        finally:
            _ambient_citations_var.reset(token)
    # Without boost, ranking is by base score + updated_at —
    # citation count is NOT factored in.
    results = await ws.search_notes("x", user_id="u")
    assert len(results) == 2
    # Order may vary based on updated_at — what matters is the
    # scores are not multiplied by the relevance boost.
    for m in results:
        assert m.score <= 1.0  # raw BM25 tier scores
    # Note that `b` is still referenced to avoid "unused" warning
    assert b.slug != a.slug


async def test_summary_carries_citation_fields() -> None:
    """NoteSummary mirrors citation fields so list_notes can show
    them without fetching the full body."""
    ws = InMemoryWorkspace()
    n = await ws.write_note(
        author="agent", title="T", body="b", user_id="u",
    )
    cites: set[str] = set()
    token = _ambient_citations_var.set(cites)
    try:
        await ws.read_note(n.slug, user_id="u")
        await ws.attribute_outcome(success=True, user_id="u")
    finally:
        _ambient_citations_var.reset(token)
    listed = await ws.list_notes(user_id="u")
    assert len(listed) == 1
    assert listed[0].cited_count == 1
    assert listed[0].success_count == 1
    assert listed[0].last_cited_at is not None
