"""Regression tests for the workspace write-path review fixes
(WSF2 batch).

Covers:

* ``write_note`` slug-counter computation + file creation now run
  under the write lock — concurrent same-author writes can't pick
  the same slug and clobber each other.
* Parsed notes are cached by (mtime_ns, size) — repeated
  list/search/read calls don't re-parse the whole tree, and
  ``list_open_questions`` no longer costs O(N²) parses.
* ``WORKSPACE.md`` regeneration is incremental but stays correct:
  new notes appear, archived notes disappear.
"""

from __future__ import annotations

from pathlib import Path

import anyio
import pytest

import loomflow.workspace.disk as disk_mod
from loomflow.workspace.disk import LocalDiskWorkspace
from loomflow.workspace.tools import make_workspace_tools

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Concurrent same-author writes — the actual race from the review
# ---------------------------------------------------------------------------


async def test_same_author_concurrent_writes_unique_slugs(
    tmp_path: Path,
) -> None:
    """Two concurrent writes by ONE author used to compute the same
    counter (the rglob scan ran outside any lock) and the second
    os.replace silently clobbered the first note."""
    ws = LocalDiskWorkspace.open(tmp_path / "ws")
    slugs: list[str] = []

    async def writer(i: int) -> None:
        n = await ws.write_note(author="solo", title=f"t{i}", body=str(i))
        slugs.append(n.slug)

    async with anyio.create_task_group() as tg:
        for i in range(12):
            tg.start_soon(writer, i)

    assert len(slugs) == 12
    assert len(set(slugs)) == 12, f"slug collision: {sorted(slugs)}"
    counters = sorted(int(s[:3]) for s in slugs)
    assert counters == list(range(1, 13))
    # All twelve notes must actually exist on disk.
    summaries = await ws.list_notes(limit=100)
    assert len(summaries) == 12


async def test_counter_survives_across_instances(tmp_path: Path) -> None:
    """The in-memory counter cache must not reset numbering when a
    new workspace instance opens the same directory."""
    ws1 = LocalDiskWorkspace.open(tmp_path / "ws")
    await ws1.write_note(author="a", title="one", body="x")
    await ws1.write_note(author="a", title="two", body="x")

    ws2 = LocalDiskWorkspace.open(tmp_path / "ws")
    n3 = await ws2.write_note(author="a", title="three", body="x")
    assert n3.slug.startswith("003-")


# ---------------------------------------------------------------------------
# Parse cache — (path, mtime) keyed
# ---------------------------------------------------------------------------


async def test_repeated_reads_do_not_reparse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = LocalDiskWorkspace.open(tmp_path / "ws")
    for i in range(5):
        await ws.write_note(author="a", title=f"note {i}", body=f"body {i}")

    calls = {"n": 0}
    real_parse = disk_mod.parse_note_file

    def counting_parse(text: str) -> tuple[dict[str, object], str]:
        calls["n"] += 1
        return real_parse(text)

    monkeypatch.setattr(disk_mod, "parse_note_file", counting_parse)

    # First listing may parse every note once (cache warm-up)...
    await ws.list_notes(limit=100)
    warmup = calls["n"]
    assert warmup <= 5
    # ...but repeated listings / searches / reads must parse NOTHING.
    await ws.list_notes(limit=100)
    await ws.search_notes("body")
    await ws.read_note("001-note-0")
    assert calls["n"] == warmup


async def test_cache_invalidates_on_update(tmp_path: Path) -> None:
    ws = LocalDiskWorkspace.open(tmp_path / "ws")
    n = await ws.write_note(author="a", title="target", body="v1")
    # Warm the cache.
    got1 = await ws.read_note(n.slug)
    assert got1 is not None and got1.body == "v1"
    await ws.update_note(author="a", slug=n.slug, body="v2")
    got2 = await ws.read_note(n.slug)
    assert got2 is not None and got2.body == "v2"


async def test_external_file_change_is_picked_up(tmp_path: Path) -> None:
    """A note edited OUTSIDE the workspace instance (different
    process, human with an editor) must not be served stale from
    the cache — the (mtime_ns, size) key changes."""
    ws = LocalDiskWorkspace.open(tmp_path / "ws")
    n = await ws.write_note(author="a", title="ext", body="original")
    got = await ws.read_note(n.slug)  # warm cache
    assert got is not None and got.body == "original"

    path = next((tmp_path / "ws").rglob(f"{n.slug}.md"))
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace("original", "edited-externally"), encoding="utf-8"
    )

    got2 = await ws.read_note(n.slug)
    assert got2 is not None and got2.body == "edited-externally"


# ---------------------------------------------------------------------------
# Incremental index stays correct
# ---------------------------------------------------------------------------


async def test_index_reflects_writes_and_archives(tmp_path: Path) -> None:
    ws = LocalDiskWorkspace.open(tmp_path / "ws")
    keep = await ws.write_note(author="a", title="keeper", body="k")
    gone = await ws.write_note(author="a", title="stale", body="s")

    index_path = tmp_path / "ws" / "_anon" / "WORKSPACE.md"
    text = index_path.read_text(encoding="utf-8")
    assert keep.slug in text
    assert gone.slug in text

    await ws.archive_note(author="a", slug=gone.slug)
    text = index_path.read_text(encoding="utf-8")
    assert keep.slug in text
    assert gone.slug not in text


async def test_index_update_reflected_in_file(tmp_path: Path) -> None:
    ws = LocalDiskWorkspace.open(tmp_path / "ws")
    n = await ws.write_note(author="a", title="first title", body="alpha lede")
    await ws.update_note(author="a", slug=n.slug, body="bravo lede")
    index_path = tmp_path / "ws" / "_anon" / "WORKSPACE.md"
    text = index_path.read_text(encoding="utf-8")
    assert "bravo lede" in text
    assert "alpha lede" not in text


async def test_prune_rebuilds_index(tmp_path: Path) -> None:
    ws = LocalDiskWorkspace.open(tmp_path / "ws")
    n1 = await ws.write_note(author="a", title="doomed", body="x")
    n2 = await ws.write_note(author="a", title="cited", body="x")
    await ws.attribute_outcome(success=True, slugs=[n2.slug])

    result = await ws.prune(min_cited_count=1)
    assert result.notes_deleted == 1
    index_path = tmp_path / "ws" / "_anon" / "WORKSPACE.md"
    text = index_path.read_text(encoding="utf-8")
    assert n2.slug in text
    assert n1.slug not in text


# ---------------------------------------------------------------------------
# list_open_questions — works and benefits from the parse cache
# ---------------------------------------------------------------------------


async def test_list_open_questions_no_reparse_storm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = LocalDiskWorkspace.open(tmp_path / "ws")
    tools = make_workspace_tools(ws, author="asker", questions=True)
    ask = next(t for t in tools if t.name == "ask_question")
    list_open = next(t for t in tools if t.name == "list_open_questions")
    for i in range(4):
        await ask.execute({"title": f"q{i}", "content": f"why {i}?"})

    calls = {"n": 0}
    real_parse = disk_mod.parse_note_file

    def counting_parse(text: str) -> tuple[dict[str, object], str]:
        calls["n"] += 1
        return real_parse(text)

    monkeypatch.setattr(disk_mod, "parse_note_file", counting_parse)

    out1 = await list_open.execute({})
    warmup = calls["n"]
    assert warmup <= 4  # at most one parse per note, not O(N²)
    out2 = await list_open.execute({})
    assert calls["n"] == warmup  # second call: pure cache
    assert out1 == out2
    for i in range(4):
        assert f"q{i}" in out1
