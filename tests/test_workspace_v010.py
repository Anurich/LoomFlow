"""Tests for the v0.10.x workspace additions: namespacing, archive,
.history versioning, ask/answer questions, semantic search via
embedder, shape-aware prompt section.

Existing v0.9 tests live in ``tests/test_workspace.py`` and exercise
the back-compat surface. This file covers the new additions only.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from loomflow import (
    InMemoryWorkspace,
    LocalDiskWorkspace,
    Note,
)
from loomflow.workspace.tools import (
    make_workspace_tools,
    workspace_prompt_section,
)

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Namespacing
# ---------------------------------------------------------------------------


async def test_namespace_write_filter() -> None:
    """Notes can be written under a namespace and filtered by it."""
    ws = InMemoryWorkspace()
    await ws.write_note(
        author="agent", title="R1", body="x", user_id="u",
        namespace="research",
    )
    await ws.write_note(
        author="agent", title="O1", body="y", user_id="u",
        namespace="ops",
    )
    research = await ws.list_notes(user_id="u", namespace="research")
    ops = await ws.list_notes(user_id="u", namespace="ops")
    assert len(research) == 1
    assert research[0].namespace == "research"
    assert len(ops) == 1
    assert ops[0].namespace == "ops"


async def test_namespace_default_lists_all() -> None:
    """By default ``list_notes`` ignores namespace — teammates see
    each other's work across namespaces."""
    ws = InMemoryWorkspace()
    await ws.write_note(
        author="a", title="X", body="x", user_id="u",
        namespace="research",
    )
    await ws.write_note(
        author="a", title="Y", body="y", user_id="u", namespace="ops",
    )
    all_notes = await ws.list_notes(user_id="u")
    assert len(all_notes) == 2


async def test_namespace_disk_roundtrip(tmp_path: object) -> None:
    """Disk backend persists namespace through frontmatter."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        ws = LocalDiskWorkspace(d)
        n = await ws.write_note(
            author="agent", title="ns", body="b", user_id="u",
            namespace="research",
        )
        read_back = await ws.read_note(n.slug, user_id="u")
        assert read_back is not None
        assert read_back.namespace == "research"


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------


async def test_archive_excluded_by_default() -> None:
    ws = InMemoryWorkspace()
    n = await ws.write_note(
        author="agent", title="T", body="b", user_id="u",
    )
    await ws.archive_note(author="agent", slug=n.slug, user_id="u")
    listed = await ws.list_notes(user_id="u")
    assert listed == []


async def test_archive_visible_with_include_archived() -> None:
    ws = InMemoryWorkspace()
    n = await ws.write_note(
        author="agent", title="T", body="b", user_id="u",
    )
    await ws.archive_note(author="agent", slug=n.slug, user_id="u")
    listed = await ws.list_notes(user_id="u", include_archived=True)
    assert len(listed) == 1
    assert listed[0].archived_at is not None


async def test_archive_still_readable_by_slug() -> None:
    """Archive hides from listings only — direct reads still work."""
    ws = InMemoryWorkspace()
    n = await ws.write_note(
        author="agent", title="T", body="b", user_id="u",
    )
    await ws.archive_note(author="agent", slug=n.slug, user_id="u")
    read = await ws.read_note(n.slug, user_id="u")
    assert read is not None
    assert read.archived_at is not None


async def test_archive_cross_author_denied() -> None:
    ws = InMemoryWorkspace()
    n = await ws.write_note(
        author="alice", title="T", body="b", user_id="u",
    )
    with pytest.raises(PermissionError):
        await ws.archive_note(author="bob", slug=n.slug, user_id="u")


# ---------------------------------------------------------------------------
# Versioning via .history
# ---------------------------------------------------------------------------


async def test_versions_appended_on_update() -> None:
    ws = InMemoryWorkspace()
    n = await ws.write_note(
        author="agent", title="T", body="v1", user_id="u",
    )
    await ws.update_note(
        author="agent", slug=n.slug, body="v2", user_id="u",
    )
    await ws.update_note(
        author="agent", slug=n.slug, body="v3", user_id="u",
    )
    versions = await ws.list_versions(n.slug, author="agent", user_id="u")
    # Two updates → two history entries (the original + the v2).
    assert len(versions) == 2
    # Live note is v3; versions are v1, v2 in order.
    live = await ws.read_note(n.slug, user_id="u")
    assert live is not None
    assert live.body == "v3"
    v1 = await ws.read_version(
        n.slug, 1, author="agent", user_id="u"
    )
    v2 = await ws.read_version(
        n.slug, 2, author="agent", user_id="u"
    )
    assert v1 is not None and v1.body == "v1"
    assert v2 is not None and v2.body == "v2"


async def test_versions_disk_persistence() -> None:
    """Disk backend writes history to .history/<slug>/NNNN.md."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        ws = LocalDiskWorkspace(d)
        n = await ws.write_note(
            author="agent", title="T", body="v1", user_id="u",
        )
        await ws.update_note(
            author="agent", slug=n.slug, body="v2", user_id="u",
        )
        versions = await ws.list_versions(
            n.slug, author="agent", user_id="u"
        )
        assert len(versions) == 1
        v1 = await ws.read_version(
            n.slug, 1, author="agent", user_id="u"
        )
        assert v1 is not None
        assert v1.body == "v1"


async def test_history_excluded_from_list_notes() -> None:
    """Historical revisions must NOT appear in `list_notes` (would
    silently bloat results)."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        ws = LocalDiskWorkspace(d)
        n = await ws.write_note(
            author="agent", title="T", body="v1", user_id="u",
        )
        await ws.update_note(
            author="agent", slug=n.slug, body="v2", user_id="u",
        )
        await ws.update_note(
            author="agent", slug=n.slug, body="v3", user_id="u",
        )
        # Three writes, but only ONE live note should appear.
        listed = await ws.list_notes(user_id="u")
        assert len(listed) == 1


# ---------------------------------------------------------------------------
# Questions / answers
# ---------------------------------------------------------------------------


async def test_ask_question_writes_kind_question() -> None:
    ws = InMemoryWorkspace()
    n = await ws.write_note(
        author="alice", title="Q", body="q?", user_id="u",
        kind="question", answered=False,
    )
    assert n.kind == "question"
    assert n.answered is False


async def test_mark_answered_cross_author() -> None:
    """The ``mark_answered`` carve-out lets a non-owner flip
    ``answered=True`` — the documented exception to author-owns-
    update enforcement."""
    ws = InMemoryWorkspace()
    q = await ws.write_note(
        author="alice", title="Q", body="q?", user_id="u",
        kind="question", answered=False,
    )
    answer = await ws.write_note(
        author="bob", title="A", body="a", user_id="u",
        kind="finding", parent_slug=q.slug,
    )
    # Bob marks Alice's question answered — should succeed.
    updated = await ws.update_note(
        author="bob",
        slug=q.slug,
        body=q.body,  # body untouched
        user_id="u",
        mark_answered=answer.slug,
    )
    assert updated.answered is True
    assert updated.answered_by == answer.slug
    # Alice's body is still her body (Bob can't rewrite it).
    assert updated.body == "q?"


async def test_update_note_cross_author_without_mark_answered_denied() -> None:
    """The carve-out applies ONLY to mark_answered. Plain
    cross-author updates still raise PermissionError."""
    ws = InMemoryWorkspace()
    n = await ws.write_note(
        author="alice", title="T", body="b", user_id="u",
    )
    with pytest.raises(PermissionError):
        await ws.update_note(
            author="bob", slug=n.slug, body="hijacked", user_id="u",
        )


# ---------------------------------------------------------------------------
# Question tools (opt-in via questions=True)
# ---------------------------------------------------------------------------


def test_question_tools_off_by_default() -> None:
    ws = InMemoryWorkspace()
    tools = make_workspace_tools(ws, author="agent")
    names = {t.name for t in tools}
    assert "ask_question" not in names
    assert "answer_question" not in names
    assert "list_open_questions" not in names


def test_question_tools_opt_in() -> None:
    ws = InMemoryWorkspace()
    tools = make_workspace_tools(ws, author="agent", questions=True)
    names = {t.name for t in tools}
    assert "ask_question" in names
    assert "answer_question" in names
    assert "list_open_questions" in names


def test_archive_tool_default_on() -> None:
    ws = InMemoryWorkspace()
    tools = make_workspace_tools(ws, author="agent")
    names = {t.name for t in tools}
    assert "archive_note" in names


def test_archive_tool_off() -> None:
    ws = InMemoryWorkspace()
    tools = make_workspace_tools(
        ws, author="agent", include_archive=False
    )
    names = {t.name for t in tools}
    assert "archive_note" not in names


# ---------------------------------------------------------------------------
# Shape-aware prompt section
# ---------------------------------------------------------------------------


def test_prompt_team_mode_says_teammates() -> None:
    p = workspace_prompt_section(
        author="alice", teammates=["alice", "bob"],
    )
    assert "teammates" in p.lower()
    assert "bob" in p


def test_prompt_single_agent_says_persistent() -> None:
    """When no teammates, the prompt must NOT confuse the model
    with team language. It should emphasize cross-run persistence
    instead."""
    p = workspace_prompt_section(author="agent")
    assert "persistent" in p.lower()
    assert "across runs" in p.lower()
    # Should not say "teammates" (only said in team mode).
    # The word "team" might still appear in the title — check the
    # actual coordination instructions are absent.
    assert "Your teammates are" not in p


def test_prompt_questions_listed_when_enabled() -> None:
    p = workspace_prompt_section(
        author="agent", questions=True,
    )
    assert "ask_question" in p
    assert "list_open_questions" in p


def test_prompt_questions_absent_when_disabled() -> None:
    p = workspace_prompt_section(author="agent")
    assert "ask_question" not in p


# ---------------------------------------------------------------------------
# Semantic search (with a stub embedder)
# ---------------------------------------------------------------------------


class _StubEmbedder:
    """Deterministic embedder that returns char-bucket vectors —
    enough to make 'apple' and 'apples' more similar than
    'apple' and 'zebra' without needing an API key."""

    name = "stub-embedder"
    dimensions = 26

    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * 26
        for ch in text.lower():
            if ch.isalpha():
                vec[ord(ch) - ord("a")] += 1.0
        # Normalise.
        import math
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


async def test_semantic_search_with_embedder() -> None:
    ws = InMemoryWorkspace(embedder=_StubEmbedder())
    await ws.write_note(
        author="agent", title="apple pie recipe",
        body="how to bake apples", user_id="u",
    )
    await ws.write_note(
        author="agent", title="zebra facts",
        body="black and white stripes", user_id="u",
    )
    # Semantic search for "apple": apple-pie note should rank
    # higher than zebra note.
    results = await ws.search_notes(
        "apple", user_id="u", mode="semantic"
    )
    assert len(results) >= 1
    assert "apple" in results[0].summary.title.lower()


async def test_search_mode_auto_falls_back_to_bm25_without_embedder() -> None:
    """mode=auto on a workspace WITHOUT an embedder should
    silently use BM25 (the v0.9 behavior)."""
    ws = InMemoryWorkspace()  # no embedder
    await ws.write_note(
        author="agent", title="apple", body="x", user_id="u",
    )
    # Pre-fix: would have crashed assuming an embedder. Post-fix:
    # plain BM25 substring match on "apple".
    results = await ws.search_notes(
        "apple", user_id="u", mode="auto"
    )
    assert len(results) == 1


async def test_search_hybrid_mode_with_embedder() -> None:
    ws = InMemoryWorkspace(embedder=_StubEmbedder())
    await ws.write_note(
        author="agent", title="A", body="apple apple", user_id="u",
    )
    await ws.write_note(
        author="agent", title="B", body="banana", user_id="u",
    )
    results = await ws.search_notes(
        "apple", user_id="u", mode="hybrid"
    )
    assert len(results) >= 1


# ---------------------------------------------------------------------------
# Legacy .md file forward-compat (frontmatter without new fields)
# ---------------------------------------------------------------------------


async def test_legacy_frontmatter_loads_clean() -> None:
    """A pre-v0.10 .md file (no namespace/archived_at/answered
    keys in frontmatter) must load without error and have the
    new fields default to None."""
    import tempfile
    from pathlib import Path
    legacy_text = (
        "---\n"
        "slug: 001-legacy\n"
        "title: legacy\n"
        "author: agent\n"
        "kind: finding\n"
        "tags: []\n"
        "created_at: 2025-01-01T00:00:00+00:00\n"
        "updated_at: 2025-01-01T00:00:00+00:00\n"
        "---\n\n"
        "Pre-v0.10 body."
    )
    with tempfile.TemporaryDirectory() as d:
        # Write a legacy file directly into the workspace
        # structure to simulate an upgrade from v0.9.
        root = Path(d)
        notes = root / "u" / "notes" / "agent"
        notes.mkdir(parents=True, exist_ok=True)
        (notes / "001-legacy.md").write_text(legacy_text)
        ws = LocalDiskWorkspace(d)
        listed = await ws.list_notes(user_id="u")
        assert len(listed) == 1
        full = await ws.read_note("001-legacy", user_id="u")
        assert full is not None
        # New fields default to None on legacy files.
        assert full.namespace is None
        assert full.archived_at is None
        assert full.answered is None
        assert full.answered_by is None
        assert full.parent_slug is None


# ---------------------------------------------------------------------------
# NoteVersion is exposed at the right tier
# ---------------------------------------------------------------------------


def test_note_version_in_types_module() -> None:
    """``NoteVersion`` should be importable from the workspace
    package."""
    from loomflow.workspace import NoteVersion  # noqa: F401


def test_note_has_new_fields() -> None:
    """The Note model carries the v0.10 fields with sensible
    defaults so existing code constructing Notes by hand keeps
    working."""
    n = Note(
        slug="x", author="a", title="t", body="b",
        created_at=datetime.now(UTC), updated_at=datetime.now(UTC),
    )
    assert n.namespace is None
    assert n.archived_at is None
    assert n.answered is None
    assert n.answered_by is None
    assert n.parent_slug is None
