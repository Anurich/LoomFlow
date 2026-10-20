"""Regression tests for the runtime / security validation findings.

Each test reproduces a finding from the validation pass and fails on
the code before its fix.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio
import pytest

from loomflow.security import FilesystemSandbox
from loomflow.security.audit import (
    GENESIS_HASH,
    FileAuditLog,
    InMemoryAuditLog,
    verify_chain,
    verify_signature,
)
from loomflow.tools import InProcessToolHost, tool
from loomflow.tools.builtin import grep_tool

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# grep / FilesystemSandbox containment
# ---------------------------------------------------------------------------


async def test_grep_skips_symlinks_that_leave_the_workdir(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("TOKEN=hunter2\n")
    work = tmp_path / "work"
    work.mkdir()
    (work / "link.txt").symlink_to(outside / "secret.txt")
    (work / "ok.txt").write_text("TOKEN=public\n")

    out = await grep_tool(work).execute({"pattern": "TOKEN"})

    assert "hunter2" not in out
    assert "public" in out


@tool
async def open_path(path: str) -> str:
    """Open a path."""
    return path


@tool
async def nested(options: dict[str, Any]) -> str:
    """Take nested options."""
    return "ok"


async def _call(sandbox: FilesystemSandbox, name: str, args: dict[str, Any]) -> Any:
    return await sandbox.call(name, args, call_id="c")


@pytest.mark.parametrize(
    "value",
    [
        "file:///etc/passwd",
        "file:///%65tc/passwd",
        "~/.ssh/id_rsa",
        "../outside/secret.txt",
        "a\x00b",
    ],
)
async def test_filesystem_sandbox_refuses_escapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    # With the process inside the root, a file: URL read as a relative
    # path ("file:/etc/passwd" under the cwd) used to pass.
    monkeypatch.chdir(root)
    sandbox = FilesystemSandbox(InProcessToolHost([open_path]), roots=[root])
    result = await _call(sandbox, "open_path", {"path": value})
    assert not result.ok and result.denied, value


async def test_filesystem_sandbox_relative_paths_resolve_against_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative path is checked where workdir-based tools open it —
    under the root — not against the process's working directory."""
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.chdir(tmp_path)
    sandbox = FilesystemSandbox(InProcessToolHost([open_path]), roots=[root])
    result = await _call(sandbox, "open_path", {"path": "sub/file.txt"})
    assert result.ok


async def test_filesystem_sandbox_refuses_arguments_nested_past_scan_depth(
    tmp_path: Path,
) -> None:
    value: Any = {"path": "/etc/passwd"}
    for _ in range(40):
        value = {"options": value}
    sandbox = FilesystemSandbox(InProcessToolHost([nested]), roots=[tmp_path])
    result = await _call(sandbox, "nested", {"options": value})
    assert not result.ok and result.denied


# ---------------------------------------------------------------------------
# Audit log: hash chain, multi-writer, corrupt lines, canonical payloads
# ---------------------------------------------------------------------------

_KEY = "k" * 32


async def _five_entry_log(path: Path) -> list[str]:
    log = FileAuditLog(path, secret=_KEY)
    for i in range(5):
        await log.append(session_id="s", actor="agent", action="tool_call", payload={"i": i})
    return (await anyio.Path(path).read_text()).splitlines()


async def test_audit_verify_detects_deleted_and_reordered_entries(tmp_path: Path) -> None:
    lines = await _five_entry_log(tmp_path / "a.jsonl")
    assert (await FileAuditLog(tmp_path / "a.jsonl", secret=_KEY).verify()).ok

    for name, kept in {
        "deleted": [lines[0], lines[1], lines[3], lines[4]],
        "reordered": lines[::-1],
        "first removed": lines[1:],
    }.items():
        p = tmp_path / f"{name}.jsonl"
        p.write_text("\n".join(kept) + "\n")
        report = await FileAuditLog(p, secret=_KEY).verify()
        assert not report.ok, name


async def test_audit_verify_detects_truncation_against_a_recorded_head(
    tmp_path: Path,
) -> None:
    path = tmp_path / "a.jsonl"
    lines = await _five_entry_log(path)
    head = (await FileAuditLog(path, secret=_KEY).verify()).head
    path.write_text("\n".join(lines[:3]) + "\n")
    report = await FileAuditLog(path, secret=_KEY).verify(expected_head=head)
    assert not report.ok


async def test_audit_verify_reports_corrupt_lines(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    lines = await _five_entry_log(path)
    path.write_text("\n".join([lines[0], lines[1][:-5], *lines[2:]]) + "\n")
    report = await FileAuditLog(path, secret=_KEY).verify()
    assert not report.ok
    assert any(p.line == 2 for p in report.problems)


async def test_audit_append_after_a_torn_last_line_stays_readable(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    lines = await _five_entry_log(path)
    path.write_text("\n".join(lines[:4]) + "\n" + lines[4][:20])  # crash mid-write
    entry = await FileAuditLog(path, secret=_KEY).append(
        session_id="s", actor="a", action="x", payload={}
    )
    assert entry.seq == 5
    assert [e.seq for e in await FileAuditLog(path, secret=_KEY).query()] == [1, 2, 3, 4, 5]


async def test_audit_two_writers_share_one_seq_and_chain(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    a, b = FileAuditLog(path, secret=_KEY), FileAuditLog(path, secret=_KEY)
    for w in (a, b, a, b, a):
        await w.append(session_id="s", actor="x", action="y", payload={})
    report = await FileAuditLog(path, secret=_KEY).verify()
    assert report.ok, report.problems
    assert [e.seq for e in await a.query()] == [1, 2, 3, 4, 5]


def _append_many(path: str, n: int) -> None:
    async def go() -> None:
        log = FileAuditLog(path, secret=_KEY)
        for _ in range(n):
            await log.append(session_id="s", actor="p", action="y", payload={})

    anyio.run(go)


async def test_audit_processes_appending_to_one_file(tmp_path: Path) -> None:
    import multiprocessing

    path = tmp_path / "a.jsonl"
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_append_many, args=(str(path), 25)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    report = await FileAuditLog(path, secret=_KEY).verify()
    assert report.ok, report.problems[:3]
    assert report.entries == 100


async def test_audit_entry_read_back_verifies_with_non_json_payload(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    log = FileAuditLog(path, secret=_KEY)
    written = await log.append(
        session_id="s",
        actor="x",
        action="y",
        payload={"when": datetime(2026, 1, 1, tzinfo=UTC), "tags": {"a"}, "pair": (1, 2)},
    )
    (read,) = await FileAuditLog(path, secret=_KEY).query()
    assert verify_signature(written, _KEY)
    assert verify_signature(read, _KEY)
    assert read.payload == written.payload


async def test_audit_in_memory_log_is_chained() -> None:
    log = InMemoryAuditLog(secret=_KEY)
    for _ in range(3):
        await log.append(session_id="s", actor="x", action="y", payload={})
    entries = await log.all_entries()
    assert entries[0].prev_hash == GENESIS_HASH
    assert entries[1].prev_hash == entries[0].signature
    assert (await log.verify()).ok
    assert not verify_chain([entries[0], entries[2]], _KEY).ok
