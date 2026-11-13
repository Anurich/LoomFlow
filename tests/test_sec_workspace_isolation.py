"""Security regression tests — workspace tenant isolation.

* A model-controlled ``slug`` must never reach another tenant's
  partition: ``../`` traversal and glob metacharacters match nothing.
* The ``user_id`` → directory mapping is injective (distinct ids never
  share a folder), nobody can name the anonymous bucket or ``seeds``,
  and ids that were already path-safe keep their folder.
* ``.`` / ``..`` are rejected as author / namespace names.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loomflow import Agent
from loomflow.core.types import ToolCall
from loomflow.model.scripted import ScriptedModel, ScriptedTurn
from loomflow.workspace.disk import (
    LocalDiskWorkspace,
    _is_safe_slug,
    _sanitise_author,
    _sanitise_user_id,
)
from loomflow.workspace.inmemory import InMemoryWorkspace

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# 1. slug traversal / glob injection
# ---------------------------------------------------------------------------


async def _victim_and_attacker(ws: LocalDiskWorkspace) -> str:
    victim = await ws.write_note(
        author="agent", title="secret plan", body="VICTIM SECRET", user_id="victim"
    )
    await ws.write_note(author="agent", title="mine", body="x", user_id="attacker")
    return victim.slug


@pytest.mark.parametrize(
    "template",
    [
        "../../../victim/notes/agent/{slug}",
        "../../victim/notes/agent/{slug}",
        "..\\..\\..\\victim\\notes\\agent\\{slug}",
        "**/{slug}",
        "../**/{slug}",
    ],
)
async def test_update_note_cannot_reach_another_tenant(
    tmp_path: Path, template: str
) -> None:
    ws = LocalDiskWorkspace(tmp_path)
    slug = await _victim_and_attacker(ws)
    with pytest.raises(FileNotFoundError):
        await ws.update_note(
            author="agent",
            slug=template.format(slug=slug),
            body="PWNED",
            user_id="attacker",
        )
    victim = await ws.read_note(slug, user_id="victim")
    assert victim is not None and victim.body == "VICTIM SECRET"


async def test_archive_note_cannot_reach_another_tenant(tmp_path: Path) -> None:
    ws = LocalDiskWorkspace(tmp_path)
    slug = await _victim_and_attacker(ws)
    with pytest.raises(FileNotFoundError):
        await ws.archive_note(
            author="agent",
            slug=f"../../../victim/notes/agent/{slug}",
            user_id="attacker",
        )
    listed = await ws.list_notes(user_id="victim")
    assert [s.slug for s in listed] == [slug]


@pytest.mark.parametrize("pattern", ["*", "0*", "???-mine", "[0-9]*", "001-min?"])
async def test_glob_characters_in_slug_match_nothing(
    tmp_path: Path, pattern: str
) -> None:
    ws = LocalDiskWorkspace(tmp_path)
    await _victim_and_attacker(ws)
    with pytest.raises(FileNotFoundError):
        await ws.archive_note(author="agent", slug=pattern, user_id="attacker")
    with pytest.raises(FileNotFoundError):
        await ws.update_note(
            author="agent", slug=pattern, body="x", user_id="attacker"
        )
    # Nothing was archived by the wildcard.
    assert len(await ws.list_notes(user_id="attacker")) == 1


async def test_history_lookups_reject_traversal(tmp_path: Path) -> None:
    ws = LocalDiskWorkspace(tmp_path)
    slug = await _victim_and_attacker(ws)
    await ws.update_note(author="agent", slug=slug, body="v2", user_id="victim")
    assert await ws.read_version(slug, 1, author="agent", user_id="victim") is not None

    evil = f"../../../../victim/notes/agent/.history/{slug}"
    assert await ws.read_version(evil, 1, author="agent", user_id="attacker") is None
    assert await ws.list_versions(evil, author="agent", user_id="attacker") == []
    with pytest.raises(ValueError, match="invalid note slug"):
        ws._history_dir("attacker", "agent", evil)


async def test_symlinked_note_outside_partition_is_not_followed(
    tmp_path: Path,
) -> None:
    ws = LocalDiskWorkspace(tmp_path)
    slug = await _victim_and_attacker(ws)
    victim_file = tmp_path / "victim" / "notes" / "agent" / f"{slug}.md"
    link = tmp_path / "attacker" / "notes" / "agent" / "999-link.md"
    try:
        link.symlink_to(victim_file)
    except OSError:  # pragma: no cover — platforms without symlinks
        pytest.skip("symlinks unavailable")
    assert ws._find_note_path("attacker", "agent", "999-link") is None


async def test_legitimate_update_and_archive_still_work(tmp_path: Path) -> None:
    ws = LocalDiskWorkspace(tmp_path)
    note = await ws.write_note(
        author="agent", title="plan", body="v1", user_id="u", namespace="ns"
    )
    updated = await ws.update_note(
        author="agent", slug=note.slug, body="v2", user_id="u"
    )
    assert updated.body == "v2"
    versions = await ws.list_versions(note.slug, author="agent", user_id="u")
    assert [v.version for v in versions] == [1]
    archived = await ws.archive_note(author="agent", slug=note.slug, user_id="u")
    assert archived.archived_at is not None


async def test_update_note_tool_cannot_cross_tenants(tmp_path: Path) -> None:
    """End to end: the model emits the traversal slug as a tool arg."""
    ws = LocalDiskWorkspace(tmp_path)
    slug = await _victim_and_attacker(ws)
    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        tool="update_note",
                        args={
                            "slug": f"../../../victim/notes/agent/{slug}",
                            "content": "PWNED via tool",
                        },
                    )
                ]
            ),
            ScriptedTurn(text="done"),
        ]
    )
    agent = Agent("hi", model=model, workspace=ws)
    await agent.run("go", user_id="attacker")
    victim = await ws.read_note(slug, user_id="victim")
    assert victim is not None and victim.body == "VICTIM SECRET"


async def test_inmemory_backend_is_keyed_by_exact_user_and_slug() -> None:
    ws = InMemoryWorkspace()
    victim = await ws.write_note(
        author="agent", title="secret", body="VICTIM", user_id="victim"
    )
    for slug in (f"../victim/{victim.slug}", "*", victim.slug):
        with pytest.raises(FileNotFoundError):
            await ws.update_note(
                author="agent", slug=slug, body="PWNED", user_id="attacker"
            )
    assert await ws.read_note(victim.slug, user_id="attacker") is None
    assert await ws.list_notes(user_id="_anon") == []


def test_is_safe_slug() -> None:
    for good in ("001-plan", "003-foo.bar", "012-résumé", "a_b-c"):
        assert _is_safe_slug(good)
    for bad in ("", ".", "..", "a/b", "a\\b", "*", "a?", "[x]", "a\x00b", "../x"):
        assert not _is_safe_slug(bad)


# ---------------------------------------------------------------------------
# 2. injective user_id → directory mapping
# ---------------------------------------------------------------------------


def test_user_id_mapping_is_injective() -> None:
    ids = [
        "alice@example.com",
        "alice#example.com",
        "alice_example.com",
        "alice/example.com",
        "alice%40example.com",
        "alice example.com",
        "_anon",
        "%5Fanon",
        "seeds",
        "%73eeds",
        ".",
        "..",
        "%2E",
        "%2E%2E",
        "a/../b",
        "a_.._b",
        "a\x00b",
        "tenant:1",
        "tenant;1",
        "ünï",
    ]
    mapped = [_sanitise_user_id(i) for i in ids]
    assert len(set(mapped)) == len(ids), sorted(zip(mapped, ids, strict=True))
    for name in mapped:
        assert "/" not in name and "\\" not in name and "\x00" not in name
        assert name not in {"", ".", "..", "_anon", "seeds"}


@pytest.mark.parametrize(
    "user_id",
    ["alice", "alice_example.com", "user-42", "Tenant.7", "a.b-c_d", "ünï", "x..y"],
)
def test_already_safe_ids_keep_their_folder(user_id: str) -> None:
    # No migration for ids that never needed sanitising.
    assert _sanitise_user_id(user_id) == user_id


async def test_lookalike_ids_do_not_share_a_partition(tmp_path: Path) -> None:
    ws = LocalDiskWorkspace(tmp_path)
    await ws.write_note(
        author="agent", title="private", body="ALICE", user_id="alice@example.com"
    )
    for other in ("alice_example.com", "alice#example.com", "alice/example.com"):
        assert await ws.list_notes(user_id=other) == []
        assert await ws.read_note("private", user_id=other) is None
    assert len(await ws.list_notes(user_id="alice@example.com")) == 1


async def test_nobody_can_name_the_anonymous_bucket_or_seeds(tmp_path: Path) -> None:
    ws = LocalDiskWorkspace(tmp_path)
    await ws.write_note(author="agent", title="anon", body="ANON", user_id=None)
    assert await ws.list_notes(user_id="_anon") == []
    assert len(await ws.list_notes(user_id=None)) == 1
    assert len(await ws.list_notes(user_id="")) == 1  # "" is the anonymous bucket
    assert ws._user_root("seeds") != ws.root / "seeds"
    for hostile in ("..", ".", "../..", "a/../../b"):
        root = ws._user_root(hostile)
        assert root.parent == ws.root


# ---------------------------------------------------------------------------
# 2b. author / namespace names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [".", ".."])
def test_sanitise_author_rejects_dot_names(name: str) -> None:
    with pytest.raises(ValueError, match="not allowed"):
        _sanitise_author(name)


async def test_dotdot_author_or_namespace_cannot_escape(tmp_path: Path) -> None:
    ws = LocalDiskWorkspace(tmp_path)
    with pytest.raises(ValueError):
        await ws.write_note(author="..", title="t", body="b", user_id="u")
    with pytest.raises(ValueError):
        await ws.write_note(
            author="agent", title="t", body="b", user_id="u", namespace=".."
        )
    assert _sanitise_author("a/b") == "a_b"
    assert _sanitise_author("") == "agent"
