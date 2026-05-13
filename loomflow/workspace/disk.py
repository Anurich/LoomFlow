"""Local-disk :class:`Workspace` backend.

Layout per user_id partition (anonymous bucket is the empty string)::

    <root>/<user_id>/
        WORKSPACE.md            ← auto-regenerated index
        seeds/                  ← read-only user-provided refs (optional)
        notes/<author>/<slug>.md
        .loom/audit.jsonl       ← reserved for future audit hook

Slugs are ``<NNN>-<slugified-title>`` where ``NNN`` is a per-author
zero-padded counter that survives across writes within the same run.
Counters live on disk (in the existing file count) so cross-run
reuse keeps numbering monotonic.

Write atomicity:

* Note files are written to a temp file in the same dir and
  ``Path.replace()``'d into place — atomic on POSIX, mostly atomic
  on Windows.
* ``WORKSPACE.md`` is regenerated after every write using the same
  temp-file + replace pattern.

An ``anyio.Lock`` serialises index regeneration so concurrent
writers don't tear it. Note bodies have no shared write target
(each author has their own subdir + unique slug) so they don't
need the lock.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import anyio

from ._common import (
    extract_lede,
    note_from_frontmatter,
    parse_note_file,
    render_note_file,
    render_workspace_index,
    slugify_title,
    summary_from_note,
)
from .types import Note, NoteKind, NoteMatch, NoteSummary, WorkspaceMembership

if TYPE_CHECKING:
    # Only needed for the ``filesystem_tools()`` return type. Import
    # is deferred under TYPE_CHECKING because:
    #   1. ``from __future__ import annotations`` makes every annotation
    #      a string at runtime, so ``Tool`` doesn't need to be resolvable
    #      at import time.
    #   2. Importing it eagerly would pull in the ``tools`` subpackage
    #      (and ``anyio.to_thread`` machinery) just to type one return
    #      annotation. Worth the deferred-import pattern.
    from ..tools.registry import Tool

INDEX_FILENAME = "WORKSPACE.md"
NOTES_DIR = "notes"
SEEDS_DIR = "seeds"
LOOM_META_DIR = ".loom"


class LocalDiskWorkspace:
    """File-backed :class:`Workspace` with a per-user partition.

    Construct via :meth:`__init__` with an absolute path, or via
    one of the classmethod sugar constructors:

    * :meth:`temp` — fresh temp directory; auto-cleaned on
      :meth:`aclose` when ``cleanup=True``.
    * :meth:`open` — open or create at a fixed path; never cleans
      up (the user owns the directory).
    """

    def __init__(
        self,
        root: str | Path,
        *,
        seed_paths: Iterable[str | Path] | None = None,
        cleanup_on_close: bool = False,
    ) -> None:
        self._root = Path(root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._cleanup_on_close = cleanup_on_close
        self._index_lock = anyio.Lock()
        if seed_paths is not None:
            self._copy_seeds(seed_paths)

    # ---- constructors ----------------------------------------------------

    @classmethod
    def temp(
        cls,
        *,
        prefix: str = "loom-workspace-",
        seed_paths: Iterable[str | Path] | None = None,
        cleanup: bool = True,
    ) -> LocalDiskWorkspace:
        """Build a fresh temp-directory workspace.

        ``cleanup=True`` (the default) wipes the directory on
        :meth:`aclose`. Pass ``cleanup=False`` to keep the
        directory for post-run inspection.
        """
        import tempfile

        path = Path(tempfile.mkdtemp(prefix=prefix))
        return cls(
            path,
            seed_paths=seed_paths,
            cleanup_on_close=cleanup,
        )

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        seed_paths: Iterable[str | Path] | None = None,
    ) -> LocalDiskWorkspace:
        """Open or create a workspace at ``path``. Never auto-cleans."""
        return cls(path, seed_paths=seed_paths, cleanup_on_close=False)

    # ---- properties ------------------------------------------------------

    @property
    def root(self) -> Path:
        """Absolute path to the workspace root."""
        return self._root

    # ---- internal layout helpers ----------------------------------------

    def _user_root(self, user_id: str | None) -> Path:
        # Anonymous bucket goes in ``_anon`` so paths are deterministic
        # and don't collide with named users.
        bucket = user_id if user_id else "_anon"
        return self._root / _sanitise_user_id(bucket)

    def _notes_dir(self, user_id: str | None, author: str) -> Path:
        return self._user_root(user_id) / NOTES_DIR / _sanitise_author(author)

    def _index_path(self, user_id: str | None) -> Path:
        return self._user_root(user_id) / INDEX_FILENAME

    # ---- seeds -----------------------------------------------------------

    def _copy_seeds(self, seed_paths: Iterable[str | Path]) -> None:
        """Drop pre-existing reference docs into ``seeds/`` for
        agents to read. Copied at construction time; agents do NOT
        write to ``seeds/`` (it's read-only by convention)."""
        seeds_root = self._root / SEEDS_DIR
        seeds_root.mkdir(parents=True, exist_ok=True)
        for src in seed_paths:
            src_path = Path(src).expanduser().resolve()
            if not src_path.exists():
                continue
            dest = seeds_root / src_path.name
            if src_path.is_dir():
                shutil.copytree(src_path, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dest)

    # ---- write paths -----------------------------------------------------

    async def write_note(
        self,
        *,
        author: str,
        title: str,
        body: str,
        kind: NoteKind = "finding",
        tags: list[str] | None = None,
        user_id: str | None = None,
        run_id: str | None = None,
    ) -> Note:
        notes_dir = self._notes_dir(user_id, author)
        notes_dir.mkdir(parents=True, exist_ok=True)
        slug_frag = slugify_title(title)
        counter = self._next_counter(notes_dir, slug_frag)
        slug = f"{counter:03d}-{slug_frag}"
        now = datetime.now(UTC)
        note = Note(
            slug=slug,
            author=author,
            title=title,
            body=body,
            kind=kind,
            tags=list(tags or []),
            created_at=now,
            updated_at=now,
            user_id=user_id,
            run_id=run_id,
        )
        await self._write_note_atomic(notes_dir / f"{slug}.md", note)
        await self._regenerate_index(user_id)
        return note

    async def update_note(
        self,
        *,
        author: str,
        slug: str,
        body: str,
        tags: list[str] | None = None,
        user_id: str | None = None,
    ) -> Note:
        notes_dir = self._notes_dir(user_id, author)
        note_path = notes_dir / f"{slug}.md"
        if not note_path.exists():
            raise FileNotFoundError(
                f"note {slug!r} not found under author {author!r}"
            )
        existing = await self._load_note(note_path)
        if existing.author != author:
            raise PermissionError(
                f"agent {author!r} cannot update note {slug!r} "
                f"owned by {existing.author!r}"
            )
        updated = existing.model_copy(
            update={
                "body": body,
                "tags": list(tags) if tags is not None else list(existing.tags),
                "updated_at": datetime.now(UTC),
            }
        )
        await self._write_note_atomic(note_path, updated)
        await self._regenerate_index(user_id)
        return updated

    def _next_counter(self, notes_dir: Path, slug_frag: str) -> int:
        """Per-author counter — looks at the existing files to find
        the highest ``NNN-`` prefix and returns the next."""
        del slug_frag  # we count across the whole author dir, not per slug
        counter = 0
        if not notes_dir.exists():
            return 1
        for p in notes_dir.iterdir():
            if not p.is_file() or not p.suffix == ".md":
                continue
            m = re.match(r"^(\d{3,})-", p.stem)
            if m:
                counter = max(counter, int(m.group(1)))
        return counter + 1

    async def _write_note_atomic(self, dest: Path, note: Note) -> None:
        body = render_note_file(note)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        await anyio.to_thread.run_sync(tmp.write_text, body)
        await anyio.to_thread.run_sync(os.replace, str(tmp), str(dest))

    # ---- read paths ------------------------------------------------------

    async def read_note(
        self,
        slug_or_title: str,
        *,
        user_id: str | None = None,
    ) -> Note | None:
        # First, try slug match across every author dir under this user.
        for path in self._walk_note_files(user_id):
            if path.stem == slug_or_title:
                return await self._load_note(path)
        # Then case-insensitive title substring match.
        needle = slug_or_title.lower()
        candidates: list[Note] = []
        for path in self._walk_note_files(user_id):
            note = await self._load_note(path)
            if needle in note.title.lower():
                candidates.append(note)
        if not candidates:
            return None
        candidates.sort(key=lambda n: n.updated_at, reverse=True)
        return candidates[0]

    async def list_notes(
        self,
        *,
        author: str | None = None,
        kind: NoteKind | None = None,
        user_id: str | None = None,
        limit: int = 50,
    ) -> list[NoteSummary]:
        notes: list[Note] = []
        for path in self._walk_note_files(user_id):
            # Cheap author filter via parent-dir name.
            if author is not None:
                if path.parent.name != _sanitise_author(author):
                    continue
            note = await self._load_note(path)
            if kind is not None and note.kind != kind:
                continue
            notes.append(note)
        notes.sort(key=lambda n: n.updated_at, reverse=True)
        return [summary_from_note(n) for n in notes[:limit]]

    async def search_notes(
        self,
        query: str,
        *,
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[NoteMatch]:
        q = query.lower().strip()
        if not q:
            return []
        scored: list[tuple[float, str, Note]] = []
        for path in self._walk_note_files(user_id):
            note = await self._load_note(path)
            title_l = note.title.lower()
            body_l = note.body.lower()
            tag_match = any(q in t.lower() for t in note.tags)
            if q in title_l:
                scored.append((1.0, note.title, note))
            elif tag_match:
                scored.append((0.7, "tags: " + ", ".join(note.tags), note))
            elif q in body_l:
                idx = body_l.find(q)
                start = max(0, idx - 40)
                end = min(len(note.body), idx + len(q) + 60)
                snippet = note.body[start:end].replace("\n", " ").strip()
                if start > 0:
                    snippet = "…" + snippet
                if end < len(note.body):
                    snippet = snippet + "…"
                scored.append((0.5, snippet, note))
        scored.sort(key=lambda t: (t[0], t[2].updated_at), reverse=True)
        out: list[NoteMatch] = []
        for score, snippet, note in scored[:limit]:
            out.append(
                NoteMatch(
                    summary=summary_from_note(note),
                    score=score,
                    snippet=snippet or extract_lede(note.body),
                )
            )
        return out

    def _walk_note_files(self, user_id: str | None) -> list[Path]:
        notes_root = self._user_root(user_id) / NOTES_DIR
        if not notes_root.exists():
            return []
        return [p for p in notes_root.rglob("*.md") if p.is_file()]

    async def _load_note(self, path: Path) -> Note:
        text = await anyio.to_thread.run_sync(path.read_text)
        fm, body = parse_note_file(text)
        return note_from_frontmatter(fm, body)

    # ---- index regeneration ---------------------------------------------

    async def _regenerate_index(self, user_id: str | None) -> None:
        """Atomically rewrite ``WORKSPACE.md`` for one partition."""
        async with self._index_lock:
            summaries = await self.list_notes(user_id=user_id, limit=10_000)
            rendered = render_workspace_index(summaries)
            user_root = self._user_root(user_id)
            user_root.mkdir(parents=True, exist_ok=True)
            dest = user_root / INDEX_FILENAME
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            await anyio.to_thread.run_sync(tmp.write_text, rendered)
            await anyio.to_thread.run_sync(os.replace, str(tmp), str(dest))

    async def render_index(
        self,
        *,
        user_id: str | None = None,
    ) -> str:
        # Re-render from current state (don't trust the on-disk
        # cache); cheap because we already walked the tree in
        # ``list_notes``.
        summaries = await self.list_notes(user_id=user_id, limit=10_000)
        return render_workspace_index(summaries)

    # ---- lifecycle -------------------------------------------------------

    async def aclose(self) -> None:
        if self._cleanup_on_close and self._root.exists():
            await anyio.to_thread.run_sync(
                shutil.rmtree, str(self._root), True
            )

    async def __aenter__(self) -> LocalDiskWorkspace:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    def member(
        self,
        name: str | None = None,
        *,
        teammates: list[str] | None = None,
    ) -> WorkspaceMembership:
        return WorkspaceMembership(
            workspace=self,
            name=name,
            teammates=list(teammates) if teammates else None,
        )

    # ---- bridge to the builtin file tools --------------------------------

    def filesystem_tools(
        self,
        *,
        include_bash: bool = False,
        user_id: str | None = None,
    ) -> list[Tool]:
        """Return the framework's existing :func:`read_tool` /
        :func:`write_tool` / :func:`edit_tool` rooted at this
        workspace's directory.

        Use case: a power-user agent that wants raw file access
        (drop binaries, read pre-seeded reference docs, edit a
        large artifact) alongside the high-level notebook tools.

        ``include_bash=True`` also returns :func:`bash_tool` rooted
        at the workspace — useful when an agent needs to run
        scripts against its own artifacts. Off by default because
        bash is destructive-capable.

        ``user_id`` scopes the workdir to the per-user partition
        so raw file access respects the same multi-tenant isolation
        as the notebook tools. Pass the agent's ``RunContext.user_id``
        (or rely on :func:`get_run_context` from inside a run).
        """
        from ..tools.builtin import (
            bash_tool,
            edit_tool,
            read_tool,
            write_tool,
        )

        scope = self._user_root(user_id)
        scope.mkdir(parents=True, exist_ok=True)
        tools: list[Tool] = [
            read_tool(scope),
            write_tool(scope),
            edit_tool(scope),
        ]
        if include_bash:
            tools.append(bash_tool(scope))
        return tools


# ---------------------------------------------------------------------------
# Sanitisation — keep paths safe for any filesystem
# ---------------------------------------------------------------------------

_PATH_UNSAFE = re.compile(r"[^\w.\-]")


def _sanitise_user_id(value: str) -> str:
    """User-supplied identifiers get path-sanitised before going on
    disk. ``..``  and slashes can't escape the workspace root."""
    cleaned = _PATH_UNSAFE.sub("_", value)
    if not cleaned or cleaned in (".", ".."):
        return "_anon"
    return cleaned


def _sanitise_author(value: str) -> str:
    cleaned = _PATH_UNSAFE.sub("_", value)
    return cleaned or "agent"
