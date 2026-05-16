"""Regression tests for the Windows ``charmap`` codec bug.

A remote loom-code user on Windows hit::

    'charmap' codec can't encode character '\\u2265' in position 1210:
    character maps to <undefined>

Cause: ``LocalDiskWorkspace`` previously called ``Path.write_text`` /
``Path.read_text`` without an explicit ``encoding=``, so it picked up
the system locale codec. On Windows that's ``cp1252`` ("charmap"),
which can't represent ``≥`` (U+2265) and many other Unicode
characters models routinely emit (em-dashes, smart quotes, arrows,
emoji).

The fix (loomflow 0.10.11) forces UTF-8 on every read and write in
``loomflow/workspace/disk.py``. These tests would FAIL on Windows
pre-fix; on POSIX hosts UTF-8 is already the default so the
behavioural difference doesn't show up, but the tests still pin the
encoding via a temp-file roundtrip and via a forced-locale-style
proof that ``cp1252.encode`` would have raised on the same content.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loomflow import LocalDiskWorkspace

pytestmark = pytest.mark.anyio


# The exact character from the bug report, plus a few others
# Windows ``cp1252`` can't encode. ``cp1252`` is a 1-byte codec, so
# any code point above U+00FF that isn't in its small special-cases
# table raises ``UnicodeEncodeError``.
#
# Em-dash and angle quotes are NOT in this list — cp1252 happens
# to map them onto its 0x80-0x9F custom block (0x97, 0x8B, 0x9B).
# Inequality signs, arrows, and emoji are the canonical "blow up
# on Windows" set.
_NON_CP1252_CHARS = "≥≤≠←→⇒✓☆🚀"


def test_non_cp1252_chars_are_unencodable_under_cp1252() -> None:
    """Sanity check: the characters from the bug report really would
    fail under cp1252. If this assertion ever breaks (cp1252 gains a
    code point), the regression below becomes weaker but still
    covers the UTF-8-forcing behaviour at the source."""
    for ch in _NON_CP1252_CHARS:
        with pytest.raises(UnicodeEncodeError):
            ch.encode("cp1252")


async def test_write_note_with_non_cp1252_content(tmp_path: Path) -> None:
    """End-to-end: writing a note whose body contains ``≥`` succeeds.

    Mirrors the screenshot from the bug report: a model writes a
    finding-kind note about ``package.json`` whose body includes
    npm version specifiers rendered as ``≥1.0.0``.
    """
    ws = LocalDiskWorkspace(str(tmp_path))
    body = (
        "## File: 'frontend/package.json' (full contents)\n"
        "```json\n"
        '{"dependencies": {"react": "≥18.0.0", "eslint": "≥9.0.0"}}\n'
        "```\n"
    )

    note = await ws.write_note(
        author="explorer",
        title="frontend package.json dependencies and ESLint config",
        body=body,
        kind="finding",
        tags=["frontend", "eslint"],
    )

    # Round-trip: read the note back. If the on-disk write was
    # locale-default and the on-disk read was locale-default, both
    # sides would tend to "agree" on POSIX and silently break on
    # Windows. Forcing UTF-8 on both sides makes the round-trip
    # platform-independent.
    fetched = await ws.read_note(note.slug)
    assert fetched is not None
    assert "≥18.0.0" in fetched.body
    assert "≥9.0.0" in fetched.body


async def test_workspace_index_with_unicode_titles(tmp_path: Path) -> None:
    """``WORKSPACE.md`` regeneration handles Unicode titles.

    The index file is rewritten after every note write — covers a
    different writer than ``write_note`` itself (the
    ``_regenerate_index`` path at the bottom of ``disk.py``).
    """
    ws = LocalDiskWorkspace(str(tmp_path))
    for ch in ["≥", "→", "✓", "🚀"]:
        await ws.write_note(
            author="explorer",
            title=f"Test {ch} title",
            body=f"body with {ch}",
            kind="finding",
        )

    # WORKSPACE.md lives under the user partition (anonymous bucket
    # is an empty-string subdir). Find it anywhere under the root.
    # Sync pathlib here is intentional for a test fixture check
    # (no real I/O concern over blocking the loop in unit tests).
    indexes = list(tmp_path.rglob("WORKSPACE.md"))  # noqa: ASYNC240
    assert indexes, "WORKSPACE.md was not regenerated"
    content = indexes[0].read_text(encoding="utf-8")  # noqa: ASYNC240
    for ch in ["≥", "→", "✓", "🚀"]:
        assert ch in content, f"missing {ch!r} in index"


async def test_history_preserves_unicode(tmp_path: Path) -> None:
    """``.history/<slug>/NNNN.md`` keeps Unicode across versions.

    The history path is a separate writer (``_snapshot_to_history``);
    needs the same UTF-8 fix to round-trip unchanged.
    """
    ws = LocalDiskWorkspace(str(tmp_path))
    initial = await ws.write_note(
        author="explorer",
        title="doc",
        body="v1 ≥1.0.0",
        kind="finding",
    )

    # Update — snapshots v1 into .history/<slug>/0001.md
    await ws.update_note(
        author="explorer",
        slug=initial.slug,
        body="v2 ≥2.0.0",
    )

    versions = await ws.list_versions(initial.slug, author="explorer")
    assert len(versions) >= 1
    v1 = await ws.read_version(initial.slug, 1, author="explorer")
    assert v1 is not None
    assert "≥1.0.0" in v1.body
