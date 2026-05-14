"""Workspace tests — protocol surface, both backends, multi-tenant
isolation, concurrent writes, Agent + Team integration.

Resolver string/dict forms live in ``test_workspace_resolver.py``;
this file focuses on the storage backends and the wiring.
"""

from __future__ import annotations

from pathlib import Path

import anyio
import pytest

from loomflow import (
    Agent,
    InMemoryWorkspace,
    LocalDiskWorkspace,
    ScriptedModel,
    ScriptedTurn,
    Workspace,
)
from loomflow.core.context import RunContext, set_run_context
from loomflow.core.types import ToolCall
from loomflow.team import Team
from loomflow.workspace import make_workspace_tools

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------


def test_inmemory_satisfies_workspace_protocol() -> None:
    assert isinstance(InMemoryWorkspace(), Workspace)


def test_disk_satisfies_workspace_protocol(tmp_path: Path) -> None:
    ws = LocalDiskWorkspace(tmp_path / "ws")
    assert isinstance(ws, Workspace)


# ---------------------------------------------------------------------------
# InMemoryWorkspace — note lifecycle
# ---------------------------------------------------------------------------


async def test_inmemory_write_and_read_by_slug() -> None:
    ws = InMemoryWorkspace()
    note = await ws.write_note(
        author="researcher",
        title="Population stats",
        body="Tokyo metro: 37 million.",
    )
    assert note.slug.startswith("001-")
    assert note.author == "researcher"
    # Read it back by slug.
    got = await ws.read_note(note.slug)
    assert got is not None
    assert got.title == "Population stats"


async def test_inmemory_read_by_partial_title_case_insensitive() -> None:
    ws = InMemoryWorkspace()
    await ws.write_note(
        author="r", title="Tokyo Population trends", body="foo"
    )
    got = await ws.read_note("population")
    assert got is not None
    assert "Tokyo" in got.title


async def test_inmemory_read_returns_none_when_missing() -> None:
    ws = InMemoryWorkspace()
    assert await ws.read_note("nothing-here") is None


async def test_inmemory_per_author_counter() -> None:
    """Slugs are numbered per author, not globally — author A's
    003-foo coexists with author B's 003-bar without collision."""
    ws = InMemoryWorkspace()
    a1 = await ws.write_note(author="alice", title="One", body="...")
    a2 = await ws.write_note(author="alice", title="Two", body="...")
    b1 = await ws.write_note(author="bob", title="Bee one", body="...")
    assert a1.slug.startswith("001-")
    assert a2.slug.startswith("002-")
    assert b1.slug.startswith("001-")  # bob's first, not third


async def test_inmemory_list_notes_filters_by_author_and_kind() -> None:
    ws = InMemoryWorkspace()
    await ws.write_note(
        author="a", title="A1", body="...", kind="finding"
    )
    await ws.write_note(
        author="a", title="A2", body="...", kind="question"
    )
    await ws.write_note(
        author="b", title="B1", body="...", kind="finding"
    )

    only_a = await ws.list_notes(author="a")
    assert {s.title for s in only_a} == {"A1", "A2"}

    only_findings = await ws.list_notes(kind="finding")
    assert {s.title for s in only_findings} == {"A1", "B1"}

    a_findings = await ws.list_notes(author="a", kind="finding")
    assert [s.title for s in a_findings] == ["A1"]


async def test_inmemory_list_notes_newest_first() -> None:
    ws = InMemoryWorkspace()
    n1 = await ws.write_note(author="a", title="Older", body="x")
    n2 = await ws.write_note(author="a", title="Newer", body="x")
    summaries = await ws.list_notes()
    assert [s.slug for s in summaries] == [n2.slug, n1.slug]


async def test_inmemory_search_ranks_title_over_body() -> None:
    ws = InMemoryWorkspace()
    await ws.write_note(
        author="r", title="Coffee report", body="benchmarks below"
    )
    await ws.write_note(
        author="r", title="Random topic", body="mention coffee here"
    )
    matches = await ws.search_notes("coffee")
    assert len(matches) == 2
    # Title hit ranks higher than body hit.
    assert matches[0].summary.title == "Coffee report"
    assert matches[0].score > matches[1].score


@pytest.mark.parametrize("backend", ["inmemory", "disk"])
async def test_search_multi_word_query_matches(
    backend: str, tmp_path: Path
) -> None:
    """Regression: a multi-word query must match notes that contain
    the terms *separately*. The earlier scorer tested the whole
    query as ONE substring, so ``search_notes("conda env conflict")``
    returned nothing unless that exact phrase appeared verbatim —
    useless for agent recall, which is mostly multi-word queries."""
    ws: Workspace = (
        InMemoryWorkspace()
        if backend == "inmemory"
        else LocalDiskWorkspace(tmp_path)
    )
    await ws.write_note(
        author="r",
        title="Resolving a conda environment conflict",
        body="pinned numpy then recreated the env from scratch",
        user_id="u",
    )
    await ws.write_note(
        author="r",
        title="Unrelated note",
        body="nothing to see here",
        user_id="u",
    )
    # Every term appears (across title + body) but never as a
    # contiguous phrase.
    matches = await ws.search_notes("conda env conflict", user_id="u")
    assert len(matches) == 1
    assert "conda" in matches[0].summary.title.lower()
    # A note matching MORE query terms outranks one matching fewer.
    await ws.write_note(
        author="r",
        title="conda only",
        body="just conda mentioned",
        user_id="u",
    )
    ranked = await ws.search_notes("conda env conflict", user_id="u")
    assert (
        ranked[0].summary.title
        == "Resolving a conda environment conflict"
    )
    assert ranked[0].score > ranked[1].score


async def test_inmemory_update_note() -> None:
    ws = InMemoryWorkspace()
    n = await ws.write_note(author="r", title="t", body="v1")
    updated = await ws.update_note(
        author="r", slug=n.slug, body="v2"
    )
    assert updated.body == "v2"
    assert updated.created_at == n.created_at
    assert updated.updated_at >= n.updated_at


async def test_inmemory_update_note_rejects_wrong_author() -> None:
    ws = InMemoryWorkspace()
    n = await ws.write_note(author="alice", title="t", body="v1")
    with pytest.raises(PermissionError, match="bob"):
        await ws.update_note(author="bob", slug=n.slug, body="hijack")


async def test_inmemory_update_note_raises_when_missing() -> None:
    ws = InMemoryWorkspace()
    with pytest.raises(FileNotFoundError):
        await ws.update_note(author="r", slug="ghost", body="...")


async def test_inmemory_render_index_includes_every_note() -> None:
    ws = InMemoryWorkspace()
    await ws.write_note(author="r", title="Alpha", body="a")
    await ws.write_note(author="r", title="Beta", body="b")
    index = await ws.render_index()
    assert "Alpha" in index
    assert "Beta" in index
    assert "## Contributors" in index


# ---------------------------------------------------------------------------
# Multi-tenant — Alice's notes never appear in Bob's listings
# ---------------------------------------------------------------------------


async def test_inmemory_user_id_partitions_writes_and_reads() -> None:
    ws = InMemoryWorkspace()
    await ws.write_note(
        author="r", title="alice secret", body="...", user_id="alice"
    )
    await ws.write_note(
        author="r", title="bob secret", body="...", user_id="bob"
    )

    alice_view = await ws.list_notes(user_id="alice")
    bob_view = await ws.list_notes(user_id="bob")
    anon_view = await ws.list_notes(user_id=None)

    assert {s.title for s in alice_view} == {"alice secret"}
    assert {s.title for s in bob_view} == {"bob secret"}
    assert anon_view == []  # nothing in the anonymous bucket


async def test_inmemory_partition_isolates_search() -> None:
    ws = InMemoryWorkspace()
    await ws.write_note(
        author="r", title="secret", body="alpha", user_id="alice"
    )
    bob_results = await ws.search_notes("alpha", user_id="bob")
    assert bob_results == []
    alice_results = await ws.search_notes("alpha", user_id="alice")
    assert len(alice_results) == 1


# ---------------------------------------------------------------------------
# LocalDiskWorkspace — same behavioural contract + persistence
# ---------------------------------------------------------------------------


async def test_disk_write_and_read_persists_across_instances(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ws"
    ws1 = LocalDiskWorkspace.open(root)
    n = await ws1.write_note(author="r", title="Persist", body="hello")
    # Open a fresh instance pointed at the same directory.
    ws2 = LocalDiskWorkspace.open(root)
    got = await ws2.read_note(n.slug)
    assert got is not None
    assert got.body == "hello"


async def test_disk_workspace_md_index_is_generated(tmp_path: Path) -> None:
    ws = LocalDiskWorkspace.open(tmp_path / "ws")
    await ws.write_note(author="r", title="Indexed", body="appears")
    index_path = tmp_path / "ws" / "_anon" / "WORKSPACE.md"
    assert index_path.exists()
    text = index_path.read_text()
    assert "Indexed" in text


async def test_disk_per_author_subdirs(tmp_path: Path) -> None:
    ws = LocalDiskWorkspace.open(tmp_path / "ws")
    await ws.write_note(author="alice", title="A", body="x")
    await ws.write_note(author="bob", title="B", body="y")
    notes_root = tmp_path / "ws" / "_anon" / "notes"
    assert (notes_root / "alice").is_dir()
    assert (notes_root / "bob").is_dir()
    # Alice can't accidentally overwrite Bob's notes — separate dirs.
    alice_files = list((notes_root / "alice").glob("*.md"))
    bob_files = list((notes_root / "bob").glob("*.md"))
    assert len(alice_files) == 1
    assert len(bob_files) == 1


async def test_disk_user_id_creates_separate_partitions(
    tmp_path: Path,
) -> None:
    ws = LocalDiskWorkspace.open(tmp_path / "ws")
    await ws.write_note(
        author="r", title="A", body="x", user_id="alice"
    )
    await ws.write_note(
        author="r", title="B", body="y", user_id="bob"
    )
    assert (tmp_path / "ws" / "alice" / "notes").is_dir()
    assert (tmp_path / "ws" / "bob" / "notes").is_dir()
    bob_list = await ws.list_notes(user_id="bob")
    assert [s.title for s in bob_list] == ["B"]


async def test_disk_temp_workspace_cleans_up(tmp_path: Path) -> None:
    ws = LocalDiskWorkspace.temp(prefix="loom-test-", cleanup=True)
    root = ws.root
    await ws.write_note(author="r", title="A", body="x")
    assert root.exists()
    await ws.aclose()
    assert not root.exists()


async def test_disk_temp_workspace_no_cleanup_when_disabled() -> None:
    ws = LocalDiskWorkspace.temp(prefix="loom-test-", cleanup=False)
    root = ws.root
    await ws.write_note(author="r", title="A", body="x")
    try:
        await ws.aclose()
        assert root.exists()
    finally:
        # Manual cleanup so the test doesn't leak.
        import shutil
        shutil.rmtree(root, ignore_errors=True)


async def test_disk_seeds_are_copied(tmp_path: Path) -> None:
    seed_dir = tmp_path / "refs"
    seed_dir.mkdir()
    (seed_dir / "spec.md").write_text("# Spec\n\nGround truth.")

    ws = LocalDiskWorkspace.open(
        tmp_path / "ws",
        seed_paths=[seed_dir],
    )
    seeded = tmp_path / "ws" / "seeds" / "refs" / "spec.md"
    assert seeded.exists()
    assert "Ground truth" in seeded.read_text()
    # Workspace is otherwise unused.
    del ws


# ---------------------------------------------------------------------------
# Concurrent writes — no race when different authors write at once
# ---------------------------------------------------------------------------


async def test_disk_concurrent_writes_no_collision(tmp_path: Path) -> None:
    ws = LocalDiskWorkspace.open(tmp_path / "ws")

    async def writer(author: str, count: int) -> None:
        for i in range(count):
            await ws.write_note(
                author=author,
                title=f"{author}-{i}",
                body=str(i),
            )

    async with anyio.create_task_group() as tg:
        tg.start_soon(writer, "alice", 10)
        tg.start_soon(writer, "bob", 10)
        tg.start_soon(writer, "carol", 10)

    summaries = await ws.list_notes(limit=100)
    assert len(summaries) == 30
    # Each author should have ten unique slugs counted 001..010.
    by_author: dict[str, list[str]] = {}
    for s in summaries:
        by_author.setdefault(s.author, []).append(s.slug)
    for author in ("alice", "bob", "carol"):
        slugs = by_author[author]
        assert len(slugs) == 10
        counters = sorted(int(s[:3]) for s in slugs)
        assert counters == list(range(1, 11))


# ---------------------------------------------------------------------------
# Filesystem-tools bridge (workspace.filesystem_tools)
# ---------------------------------------------------------------------------


def test_disk_filesystem_tools_returns_existing_builtin_tools(
    tmp_path: Path,
) -> None:
    """``workspace.filesystem_tools()`` reuses the builtin file tools
    rooted at the workspace path — no duplicate file-IO subsystem."""
    ws = LocalDiskWorkspace.open(tmp_path / "ws")
    tools = ws.filesystem_tools()
    names = sorted(t.name for t in tools)
    assert names == ["edit", "read", "write"]
    tools_with_bash = ws.filesystem_tools(include_bash=True)
    assert "bash" in {t.name for t in tools_with_bash}


# ---------------------------------------------------------------------------
# Tool factory + Agent integration
# ---------------------------------------------------------------------------


async def test_workspace_tool_factory_attribution() -> None:
    """When ``make_workspace_tools(ws, author='researcher')`` is
    called, every note the tools write is attributed to 'researcher'.
    The author is closure-baked; the agent never types it."""
    ws = InMemoryWorkspace()
    tools = make_workspace_tools(ws, author="researcher")
    note_tool = next(t for t in tools if t.name == "note")
    await note_tool.execute(
        {"title": "Finding", "content": "stuff"}
    )
    summaries = await ws.list_notes()
    assert summaries[0].author == "researcher"


async def test_workspace_tool_picks_up_user_id_from_run_context() -> None:
    """The tool factory uses the ambient :class:`RunContext` for
    ``user_id`` so writes go into the right partition without
    the agent threading the id explicitly."""
    ws = InMemoryWorkspace()
    tools = make_workspace_tools(ws, author="r")
    note_tool = next(t for t in tools if t.name == "note")

    async with set_run_context(RunContext(user_id="alice")):
        await note_tool.execute({"title": "A", "content": "x"})
    async with set_run_context(RunContext(user_id="bob")):
        await note_tool.execute({"title": "B", "content": "y"})

    alice_view = await ws.list_notes(user_id="alice")
    bob_view = await ws.list_notes(user_id="bob")
    assert [s.title for s in alice_view] == ["A"]
    assert [s.title for s in bob_view] == ["B"]


async def test_agent_with_workspace_writes_a_note() -> None:
    """End-to-end: an Agent wired with a workspace can call the
    auto-wired ``note`` tool and the note lands in the workspace."""
    ws = InMemoryWorkspace()
    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        tool="note",
                        args={
                            "title": "From the agent",
                            "content": "wrote this via the tool",
                        },
                    )
                ]
            ),
            ScriptedTurn(text="done."),
        ]
    )
    agent = Agent(
        "you are a researcher",
        model=model,
        workspace=ws.member("researcher"),
    )
    result = await agent.run("share a finding")
    assert "done" in result.output.lower()
    summaries = await ws.list_notes()
    assert summaries[0].title == "From the agent"
    assert summaries[0].author == "researcher"


async def test_agent_with_workspace_kwarg_string_resolves_to_inmemory() -> None:
    """``Agent(workspace='memory')`` resolves via the string form."""
    agent = Agent("x", model="echo", workspace="memory")
    assert isinstance(agent._workspace, InMemoryWorkspace)  # noqa: SLF001


async def test_team_supervisor_threads_workspace_to_workers() -> None:
    """``Team.supervisor(workers=..., workspace=ws)`` stamps each
    worker with its role name AND wires the workspace onto the
    coordinator. The coordinator's run installs the ambient and
    the workers inherit it."""
    ws = InMemoryWorkspace()
    a_model = ScriptedModel([ScriptedTurn(text="alpha done")])
    b_model = ScriptedModel([ScriptedTurn(text="beta done")])
    a = Agent("a worker", model=a_model)
    b = Agent("another worker", model=b_model)
    assert a._workspace is None  # noqa: SLF001 — not wired yet
    team = Team.supervisor(
        workers={"alpha": a, "beta": b},
        model="echo",
        workspace=ws,
    )
    # The Team builder mutates the workers' workspace identity.
    assert a._workspace_name == "alpha"  # noqa: SLF001
    assert b._workspace_name == "beta"  # noqa: SLF001
    assert a._workspace_teammates == ["alpha", "beta"]  # noqa: SLF001
    # The coordinator owns the workspace.
    assert team._workspace is ws  # noqa: SLF001
    assert team._workspace_name == "coordinator"  # noqa: SLF001


async def test_agent_without_workspace_picks_up_ambient_at_run_time() -> None:
    """A nested Agent constructed without ``workspace=`` picks up
    the parent's ambient workspace and gets the notebook tools
    wired on for the duration of that run."""
    ws = InMemoryWorkspace()
    # Worker doesn't know about ws at construction.
    worker_model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        tool="note",
                        args={
                            "title": "Inherited",
                            "content": "the worker found a workspace",
                        },
                    )
                ]
            ),
            ScriptedTurn(text="done."),
        ]
    )
    worker = Agent("a worker", model=worker_model)
    # Manually mutate to mimic what the Team builder does.
    worker._workspace_name = "worker"  # noqa: SLF001

    from loomflow.core.context import _ambient_workspace_var

    token = _ambient_workspace_var.set(ws)
    try:
        await worker.run("write a note")
    finally:
        _ambient_workspace_var.reset(token)

    summaries = await ws.list_notes()
    assert len(summaries) == 1
    assert summaries[0].author == "worker"
