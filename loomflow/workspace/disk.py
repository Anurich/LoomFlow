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
from datetime import UTC, datetime, timedelta
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
from ._common import score_bm25 as _score_bm25
from .types import (
    Note,
    NoteKind,
    NoteMatch,
    NoteSummary,
    NoteVersion,
    PruneResult,
    WorkspaceMembership,
)

if TYPE_CHECKING:
    # Only needed for type annotations. Imports are deferred under
    # TYPE_CHECKING because:
    #   1. ``from __future__ import annotations`` makes every annotation
    #      a string at runtime, so these don't need to be resolvable
    #      at import time.
    #   2. Importing ``Tool`` eagerly would pull in the ``tools``
    #      subpackage just to type one return annotation. Worth
    #      the deferred-import pattern.
    #   3. ``Embedder`` is used only when the optional ``embedder=``
    #      ctor param is wired; lazy-import keeps the workspace
    #      module light.
    from ..core.protocols import Embedder
    from ..tools.registry import Tool

INDEX_FILENAME = "WORKSPACE.md"
NOTES_DIR = "notes"
SEEDS_DIR = "seeds"
LOOM_META_DIR = ".loom"
# Subdir name under each note's parent that holds revision history.
# Excluded from `_walk_note_files` so historical revisions never
# appear in `list_notes` / `search_notes` / index renders.
HISTORY_DIR = ".history"


def _log_citation(slug: str) -> None:
    """Add ``slug`` to the per-run citation set if one is active.

    No-op outside a run (contextvar default is ``None``). Best-
    effort — failures to log a citation must NEVER break a read.
    """
    from ..core.context import _ambient_citations_var
    citations = _ambient_citations_var.get()
    if citations is None:
        return
    try:
        citations.add(slug)
    except Exception:  # noqa: BLE001 — observation, not load-bearing
        pass


def _drain_citations() -> set[str]:
    """Snapshot + clear the per-run citation set, returning the
    slugs that were cited. ``Workspace.attribute_outcome`` calls
    this once at the end of a run."""
    from ..core.context import _ambient_citations_var
    citations = _ambient_citations_var.get()
    if citations is None:
        return set()
    snapshot = set(citations)
    citations.clear()
    return snapshot


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
        embedder: Embedder | None = None,
    ) -> None:
        self._root = Path(root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._cleanup_on_close = cleanup_on_close
        # One lock for index regeneration AND for note writes /
        # history snapshots. Renaming the old ``_index_lock`` to
        # ``_write_lock`` would break subclasses; keep the old
        # name as the canonical write lock. Concurrent updates of
        # the same slug now correctly serialise behind this.
        self._index_lock = anyio.Lock()
        # Optional embedder for semantic search. When set, every
        # write_note also computes + persists an embedding sidecar;
        # search_notes uses cosine similarity (or hybrid RRF) when
        # mode allows. Default ``None`` preserves BM25-only behavior.
        self._embedder = embedder
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

    def _notes_dir(
        self,
        user_id: str | None,
        author: str,
        namespace: str | None = None,
    ) -> Path:
        """Per-author notes dir, optionally scoped to a namespace.

        Path shape:

        * ``namespace is None``: ``<root>/<user>/notes/<author>/`` —
          the v0.9 layout, preserved for back-compat.
        * ``namespace`` set: ``<root>/<user>/notes/<author>/<namespace>/``
          — files under a sub-bucket. Counter for slug numbering is
          still author-global (NOT reset per namespace) so a slug
          uniquely identifies a note within an author across
          namespaces.
        """
        base = self._user_root(user_id) / NOTES_DIR / _sanitise_author(author)
        if namespace:
            return base / _sanitise_author(namespace)
        return base

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
        namespace: str | None = None,
        answered: bool | None = None,
        parent_slug: str | None = None,
    ) -> Note:
        # The author's slug counter is GLOBAL across namespaces —
        # walking the bare author dir (no namespace) gives the max
        # NNN seen anywhere under that author. This keeps slugs
        # globally unique within ``(user_id, author)``, simplifying
        # the in-memory dict key and any future "find by slug
        # without namespace hint" lookups.
        author_root = self._notes_dir(user_id, author, namespace=None)
        author_root.mkdir(parents=True, exist_ok=True)
        ns_dir = self._notes_dir(user_id, author, namespace=namespace)
        ns_dir.mkdir(parents=True, exist_ok=True)
        slug_frag = slugify_title(title)
        counter = self._next_counter_across_namespaces(author_root)
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
            namespace=namespace,
            answered=answered,
            parent_slug=parent_slug,
        )
        await self._write_note_atomic(ns_dir / f"{slug}.md", note)
        await self._maybe_write_embedding(ns_dir / f"{slug}.md", note)
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
        mark_answered: str | None = None,
    ) -> Note:
        note_path = self._find_note_path(user_id, author, slug)
        if note_path is None:
            raise FileNotFoundError(
                f"note {slug!r} not found under author {author!r}"
            )
        existing = await self._load_note(note_path)
        # The ``mark_answered`` cross-author carve-out: any agent
        # can flip ``answered=True`` + ``answered_by=<slug>`` on a
        # question they didn't author. This is the single
        # exception to the "author owns updates" rule and is
        # documented in the protocol. All OTHER fields stay the
        # asker's property.
        if mark_answered is not None and existing.author != author:
            # Cross-author mark — preserve original body + tags,
            # only update the answered flags. Write a history
            # snapshot anyway so the answered transition is
            # auditable.
            async with self._index_lock:
                await self._snapshot_history(note_path, existing)
                updated = existing.model_copy(
                    update={
                        "answered": True,
                        "answered_by": mark_answered,
                        "updated_at": datetime.now(UTC),
                    }
                )
                await self._write_note_atomic(note_path, updated)
                await self._maybe_write_embedding(note_path, updated)
            await self._regenerate_index(user_id)
            return updated
        if existing.author != author:
            raise PermissionError(
                f"agent {author!r} cannot update note {slug!r} "
                f"owned by {existing.author!r}"
            )
        # Normal author update: snapshot prior body before
        # overwriting. The history snapshot is under the index
        # lock so concurrent updates serialise their version
        # numbers correctly.
        async with self._index_lock:
            await self._snapshot_history(note_path, existing)
            update_dict: dict[str, object] = {
                "body": body,
                "tags": list(tags) if tags is not None else list(existing.tags),
                "updated_at": datetime.now(UTC),
            }
            if mark_answered is not None:
                update_dict["answered"] = True
                update_dict["answered_by"] = mark_answered
            updated = existing.model_copy(update=update_dict)
            await self._write_note_atomic(note_path, updated)
            await self._maybe_write_embedding(note_path, updated)
        await self._regenerate_index(user_id)
        return updated

    async def archive_note(
        self,
        *,
        author: str,
        slug: str,
        user_id: str | None = None,
    ) -> Note:
        note_path = self._find_note_path(user_id, author, slug)
        if note_path is None:
            raise FileNotFoundError(
                f"note {slug!r} not found under author {author!r}"
            )
        existing = await self._load_note(note_path)
        if existing.author != author:
            raise PermissionError(
                f"agent {author!r} cannot archive note {slug!r} "
                f"owned by {existing.author!r}"
            )
        archived = existing.model_copy(
            update={
                "archived_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC),
            }
        )
        # No history snapshot for archive — archiving is metadata-
        # only, the body doesn't change.
        await self._write_note_atomic(note_path, archived)
        await self._regenerate_index(user_id)
        return archived

    async def list_versions(
        self,
        slug: str,
        *,
        author: str,
        user_id: str | None = None,
    ) -> list[NoteVersion]:
        history_dir = self._history_dir(user_id, author, slug)
        if not history_dir.exists():
            return []
        out: list[NoteVersion] = []
        for path in sorted(history_dir.iterdir()):
            if not path.is_file() or path.suffix != ".md":
                continue
            try:
                version = int(path.stem)
            except ValueError:
                continue
            text = await anyio.to_thread.run_sync(path.read_text)
            fm, body = parse_note_file(text)
            note = note_from_frontmatter(fm, body)
            out.append(
                NoteVersion(
                    slug=slug,
                    author=author,
                    version=version,
                    created_at=note.updated_at,
                    body_preview=extract_lede(note.body),
                )
            )
        return out

    async def read_version(
        self,
        slug: str,
        version: int,
        *,
        author: str,
        user_id: str | None = None,
    ) -> Note | None:
        history_dir = self._history_dir(user_id, author, slug)
        path = history_dir / f"{version:04d}.md"
        if not path.exists():
            return None
        text = await anyio.to_thread.run_sync(path.read_text)
        fm, body = parse_note_file(text)
        note = note_from_frontmatter(fm, body)
        _log_citation(slug)
        return note

    async def attribute_outcome(
        self,
        *,
        success: bool,
        slugs: list[str] | None = None,
        user_id: str | None = None,
    ) -> int:
        # Explicit slugs (from RunResult.cited_slugs) win — that's
        # the reliable post-run path. Fall back to draining the
        # contextvar only when no slugs were passed (in-run case).
        cited = set(slugs) if slugs is not None else _drain_citations()
        if not cited:
            return 0
        now = datetime.now(UTC)
        updated = 0
        for slug in cited:
            # Find the note across authors — citations don't carry
            # author identity, so scan any author dir.
            path = self._find_note_by_slug_any_author(user_id, slug)
            if path is None:
                continue
            existing = await self._load_note(path)
            patched = existing.model_copy(
                update={
                    "cited_count": existing.cited_count + 1,
                    "success_count": (
                        existing.success_count + 1 if success
                        else existing.success_count
                    ),
                    "last_cited_at": now,
                }
            )
            # Write in-place — citation update is not a "user edit"
            # so we don't snapshot history.
            await self._write_note_atomic(path, patched)
            updated += 1
        if updated:
            await self._regenerate_index(user_id)
        return updated

    def _find_note_by_slug_any_author(
        self, user_id: str | None, slug: str
    ) -> Path | None:
        notes_root = self._user_root(user_id) / NOTES_DIR
        if not notes_root.exists():
            return None
        for p in notes_root.rglob(f"{slug}.md"):
            if not _is_in_meta_dir(p):
                return p
        return None

    async def prune(
        self,
        *,
        older_than: timedelta | None = None,
        min_cited_count: int = 1,
        keep_kinds: list[NoteKind] | None = None,
        keep_last_versions: int | None = None,
        user_id: str | None = None,
    ) -> PruneResult:
        now = datetime.now(UTC)
        keep_kind_set = set(keep_kinds or [])
        notes_deleted = 0
        notes_kept = 0
        versions_deleted = 0
        async with self._index_lock:
            for path in self._walk_note_files(user_id):
                note = await self._load_note(path)
                if _should_prune(
                    note,
                    now=now,
                    older_than=older_than,
                    min_cited_count=min_cited_count,
                    keep_kind_set=keep_kind_set,
                ):
                    # Hard-delete the note file + its embedding
                    # sidecar + its entire history dir.
                    await anyio.to_thread.run_sync(path.unlink, True)
                    sidecar = path.with_suffix(".embedding.json")
                    await anyio.to_thread.run_sync(sidecar.unlink, True)
                    hist = self._history_dir(
                        user_id, note.author, note.slug
                    )
                    if hist.exists():
                        await anyio.to_thread.run_sync(
                            shutil.rmtree, str(hist), True
                        )
                    notes_deleted += 1
                else:
                    notes_kept += 1
                    # Surviving note: optionally trim its history
                    # to the most recent N revisions.
                    if keep_last_versions is not None:
                        versions_deleted += await self._trim_history(
                            user_id, note.author, note.slug,
                            keep_last_versions,
                        )
        if notes_deleted:
            await self._regenerate_index(user_id)
        return PruneResult(
            notes_deleted=notes_deleted,
            versions_deleted=versions_deleted,
            notes_kept=notes_kept,
        )

    async def _trim_history(
        self,
        user_id: str | None,
        author: str,
        slug: str,
        keep_last: int,
    ) -> int:
        """Delete all but the most recent ``keep_last`` revision
        files for one note. Returns the count deleted."""
        history_dir = self._history_dir(user_id, author, slug)
        if not history_dir.exists():
            return 0
        versioned: list[tuple[int, Path]] = []
        for p in history_dir.iterdir():
            if not p.is_file() or p.suffix != ".md":
                continue
            try:
                versioned.append((int(p.stem), p))
            except ValueError:
                continue
        if len(versioned) <= keep_last:
            return 0
        versioned.sort(key=lambda t: t[0])
        to_delete = versioned[: len(versioned) - keep_last]
        for _, p in to_delete:
            await anyio.to_thread.run_sync(p.unlink, True)
        return len(to_delete)

    def _history_dir(
        self, user_id: str | None, author: str, slug: str
    ) -> Path:
        # History lives under the AUTHOR root (not namespaced) so
        # `read_version`/`list_versions` work regardless of which
        # namespace the live note is in. Single source of truth
        # per (user, author, slug).
        return (
            self._notes_dir(user_id, author, namespace=None)
            / HISTORY_DIR
            / slug
        )

    async def _snapshot_history(self, live_path: Path, note: Note) -> None:
        """Copy the live note body to the next version file in
        ``<author>/.history/<slug>/NNNN.md`` before overwriting.

        Counter is monotonic per-slug; zero-padded to 4 digits so
        agents on long-running tasks can hit 1000+ revisions
        without collision. Caller must hold the index lock.
        """
        user_id = note.user_id
        history_dir = self._history_dir(user_id, note.author, note.slug)
        history_dir.mkdir(parents=True, exist_ok=True)
        # Find the highest existing version number; next = +1.
        existing_max = 0
        for p in history_dir.iterdir():
            if not p.is_file() or p.suffix != ".md":
                continue
            try:
                n = int(p.stem)
            except ValueError:
                continue
            existing_max = max(existing_max, n)
        next_version = existing_max + 1
        dest = history_dir / f"{next_version:04d}.md"
        # Copy the live file's CURRENT content to the version slot.
        # We can't render(note) here because the caller might have
        # already mutated `note` — copy the existing file bytes.
        content = await anyio.to_thread.run_sync(live_path.read_text)
        await anyio.to_thread.run_sync(dest.write_text, content)

    def _find_note_path(
        self, user_id: str | None, author: str, slug: str
    ) -> Path | None:
        """Locate a note's live .md file, regardless of namespace.

        Walks the author's notes dir (excluding ``.history`` /
        ``.embeddings``) looking for ``<slug>.md``. Returns the
        first match or None. Cheap because each author has a
        small subtree.
        """
        author_root = self._notes_dir(user_id, author, namespace=None)
        if not author_root.exists():
            return None
        for p in author_root.rglob(f"{slug}.md"):
            if not _is_in_meta_dir(p):
                return p
        return None

    def _next_counter_across_namespaces(self, author_root: Path) -> int:
        """Highest ``NNN-`` prefix seen anywhere under the author
        root, across every namespace. +1. Excludes history dirs."""
        counter = 0
        if not author_root.exists():
            return 1
        for p in author_root.rglob("*.md"):
            if not p.is_file() or _is_in_meta_dir(p):
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

    async def _maybe_write_embedding(
        self, note_path: Path, note: Note
    ) -> None:
        """When an embedder is wired, persist the note's embedding
        as a JSON sidecar next to the .md file. The sidecar carries
        the embedder model name so swapping embedders silently
        invalidates stale vectors (cosine on different model spaces
        is meaningless).
        """
        if self._embedder is None:
            return
        try:
            text = f"{note.title}\n\n{note.body}"
            vector = await self._embedder.embed(text)
            if not vector:
                return
            import json
            model = getattr(self._embedder, "name", "unknown")
            sidecar = note_path.with_suffix(".embedding.json")
            payload = {
                "model": model,
                "dim": len(vector),
                "vector": list(vector),
            }
            await anyio.to_thread.run_sync(
                sidecar.write_text, json.dumps(payload)
            )
        except anyio.get_cancelled_exc_class():
            raise
        except Exception:  # noqa: BLE001 — embedding is best-effort
            # An embedder failure should never break the note
            # write. Fall back to BM25 silently for this note.
            pass

    # ---- read paths ------------------------------------------------------

    async def read_note(
        self,
        slug_or_title: str,
        *,
        user_id: str | None = None,
    ) -> Note | None:
        result: Note | None = None
        # First, try slug match across every author dir under this user.
        for path in self._walk_note_files(user_id):
            if path.stem == slug_or_title:
                result = await self._load_note(path)
                break
        if result is None:
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
            result = candidates[0]
        _log_citation(result.slug)
        return result

    async def list_notes(
        self,
        *,
        author: str | None = None,
        kind: NoteKind | None = None,
        user_id: str | None = None,
        namespace: str | None = None,
        include_archived: bool = False,
        limit: int = 50,
    ) -> list[NoteSummary]:
        notes: list[Note] = []
        for path in self._walk_note_files(user_id):
            note = await self._load_note(path)
            if author is not None and note.author != author:
                continue
            if kind is not None and note.kind != kind:
                continue
            if namespace is not None and note.namespace != namespace:
                continue
            if not include_archived and note.archived_at is not None:
                continue
            notes.append(note)
        notes.sort(key=lambda n: n.updated_at, reverse=True)
        return [summary_from_note(n) for n in notes[:limit]]

    async def search_notes(
        self,
        query: str,
        *,
        user_id: str | None = None,
        namespace: str | None = None,
        include_archived: bool = False,
        mode: str = "auto",
        boost_relevance: bool = False,
        limit: int = 10,
    ) -> list[NoteMatch]:
        q = query.lower().strip()
        if not q:
            return []
        # Decide scoring mode. ``auto`` = hybrid when embedder is
        # wired, BM25 otherwise. ``semantic`` / ``hybrid`` fall back
        # to BM25 silently when no embedder is wired — preserves the
        # backward-compat default-no-embedder behavior.
        has_embedder = self._embedder is not None
        effective_mode = mode
        if effective_mode == "auto":
            effective_mode = "hybrid" if has_embedder else "bm25"
        if effective_mode in ("semantic", "hybrid") and not has_embedder:
            effective_mode = "bm25"
        candidates: list[Note] = []
        for path in self._walk_note_files(user_id):
            note = await self._load_note(path)
            if namespace is not None and note.namespace != namespace:
                continue
            if not include_archived and note.archived_at is not None:
                continue
            candidates.append(note)
        if effective_mode == "bm25":
            results = _score_bm25(q, candidates, limit)
            return (
                _apply_relevance_boost(results, candidates, limit)
                if boost_relevance else results
            )
        # Semantic / hybrid: compute query embedding once.
        assert self._embedder is not None  # narrow for type-checker
        try:
            qvec = await self._embedder.embed(query)
        except anyio.get_cancelled_exc_class():
            raise
        except Exception:  # noqa: BLE001 — fall back to BM25 on embed failure
            results = _score_bm25(q, candidates, limit)
            return (
                _apply_relevance_boost(results, candidates, limit)
                if boost_relevance else results
            )
        if not qvec:
            results = _score_bm25(q, candidates, limit)
            return (
                _apply_relevance_boost(results, candidates, limit)
                if boost_relevance else results
            )
        sem_scores = await self._semantic_scores(qvec, candidates, user_id)
        if effective_mode == "semantic":
            results = _score_semantic(sem_scores, candidates, limit)
            return (
                _apply_relevance_boost(results, candidates, limit)
                if boost_relevance else results
            )
        # Hybrid: reciprocal rank fusion of BM25 + semantic rankings.
        bm25 = _score_bm25(q, candidates, limit=len(candidates))
        results = _rrf_fuse(bm25, sem_scores, candidates, limit)
        return (
            _apply_relevance_boost(results, candidates, limit)
            if boost_relevance else results
        )

    async def _semantic_scores(
        self,
        qvec: list[float],
        notes: list[Note],
        user_id: str | None,
    ) -> dict[str, float]:
        """Load each candidate's persisted embedding and cosine-
        score against ``qvec``. Notes without a sidecar (legacy or
        embed-failure) score zero and effectively fall back to
        BM25 ranking when hybrid mode fuses the results.
        """
        import json
        out: dict[str, float] = {}
        for note in notes:
            path = self._find_note_path(user_id, note.author, note.slug)
            if path is None:
                continue
            sidecar = path.with_suffix(".embedding.json")
            if not sidecar.exists():
                continue
            try:
                text = await anyio.to_thread.run_sync(sidecar.read_text)
                payload = json.loads(text)
                stored_model = payload.get("model")
                this_model = getattr(self._embedder, "name", "unknown")
                if stored_model != this_model:
                    # Stale vector from a different embedder space.
                    continue
                vector = payload.get("vector") or []
                if len(vector) != len(qvec):
                    continue
                out[note.slug] = _cosine(qvec, vector)
            except Exception:  # noqa: BLE001
                continue
        return out

    def _walk_note_files(self, user_id: str | None) -> list[Path]:
        """Walk live note files, EXCLUDING ``.history`` revisions
        and any other meta dirs. Without this filter, every
        revision would appear in ``list_notes`` / ``search_notes``
        as if it were a separate live note — silently breaking
        every existing test that asserts on count.
        """
        notes_root = self._user_root(user_id) / NOTES_DIR
        if not notes_root.exists():
            return []
        return [
            p for p in notes_root.rglob("*.md")
            if p.is_file() and not _is_in_meta_dir(p)
        ]

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


# ---------------------------------------------------------------------------
# Walk filters — exclude meta dirs
# ---------------------------------------------------------------------------


def _should_prune(
    note: Note,
    *,
    now: datetime,
    older_than: timedelta | None,
    min_cited_count: int,
    keep_kind_set: set[NoteKind],
) -> bool:
    """Decide whether a note is GC-eligible. A note is pruned only
    when ALL hold: not a protected kind, cited below the
    threshold, and (if ``older_than`` is set) idle longer than the
    window. Shared by both backends so prune semantics stay
    identical.
    """
    # Protected kind — never pruned.
    if note.kind in keep_kind_set:
        return False
    # Valuable — cited enough times.
    if note.cited_count >= min_cited_count:
        return False
    # Age filter. When ``older_than`` is None, age is not a factor
    # — every note is age-eligible (the caller opted into that).
    if older_than is not None:
        last_activity = note.last_cited_at or note.updated_at
        if now - last_activity < older_than:
            return False
    return True


def _is_in_meta_dir(path: Path) -> bool:
    """``True`` when ``path`` is inside the ``.history`` revision
    store. Used to filter ``rglob("*.md")`` so historical
    revisions never surface as live notes in ``list_notes`` /
    ``search_notes`` / the index.

    NOTE: this checks specifically for the ``.history`` segment,
    NOT "any dot-prefixed part". An earlier version flagged every
    dot-dir — which silently broke any workspace rooted under a
    dot-directory (``.loom/notebook``, ``.claude/workspace``,
    ...), because then every note path contains a dotted segment
    and every note got filtered out of listings.
    """
    return HISTORY_DIR in path.parts


# ---------------------------------------------------------------------------
# Scoring helpers — BM25-ish, semantic cosine, RRF fusion
# ---------------------------------------------------------------------------


def _score_semantic(
    sem_scores: dict[str, float],
    notes: list[Note],
    limit: int,
) -> list[NoteMatch]:
    """Pure cosine ranking. Notes without embeddings score zero
    and drop to the tail; ties broken by ``updated_at``."""
    by_slug = {n.slug: n for n in notes}
    ranked = sorted(
        by_slug.values(),
        key=lambda n: (sem_scores.get(n.slug, 0.0), n.updated_at),
        reverse=True,
    )
    out: list[NoteMatch] = []
    for note in ranked[:limit]:
        score = sem_scores.get(note.slug, 0.0)
        if score <= 0.0:
            continue
        out.append(
            NoteMatch(
                summary=summary_from_note(note),
                score=score,
                snippet=extract_lede(note.body),
            )
        )
    return out


def _rrf_fuse(
    bm25: list[NoteMatch],
    sem_scores: dict[str, float],
    notes: list[Note],
    limit: int,
    k: int = 60,
) -> list[NoteMatch]:
    """Reciprocal rank fusion: combines BM25 ranking and semantic
    ranking by summing ``1/(k + rank)`` across both. Robust to
    score-scale differences between the two methods. ``k=60`` is
    the canonical RRF constant from the original Cormack et al.
    paper; any value in the 30-100 range works.
    """
    rank_bm = {m.summary.slug: i for i, m in enumerate(bm25)}
    sem_ranked = sorted(
        sem_scores.keys(),
        key=lambda s: sem_scores[s],
        reverse=True,
    )
    rank_sem = {slug: i for i, slug in enumerate(sem_ranked)}
    fused: dict[str, float] = {}
    for slug in set(rank_bm) | set(rank_sem):
        score = 0.0
        if slug in rank_bm:
            score += 1.0 / (k + rank_bm[slug])
        if slug in rank_sem:
            score += 1.0 / (k + rank_sem[slug])
        fused[slug] = score
    by_slug = {n.slug: n for n in notes}
    bm_by_slug = {m.summary.slug: m for m in bm25}
    ranked_slugs = sorted(
        fused.keys(), key=lambda s: fused[s], reverse=True
    )
    out: list[NoteMatch] = []
    for slug in ranked_slugs[:limit]:
        note = by_slug.get(slug)
        if note is None:
            continue
        # Prefer the BM25 snippet when available (more informative
        # than the lede); otherwise fall back.
        bm_match = bm_by_slug.get(slug)
        snippet = bm_match.snippet if bm_match else extract_lede(note.body)
        out.append(
            NoteMatch(
                summary=summary_from_note(note),
                score=fused[slug],
                snippet=snippet,
            )
        )
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Returns 0 on degenerate inputs."""
    import math
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _apply_relevance_boost(
    results: list[NoteMatch],
    candidates: list[Note],
    limit: int,
) -> list[NoteMatch]:
    """Multiply each result's score by a relevance boost based on
    citation metadata, then re-sort.

    Formula: ``boost = 1 + log(1 + cited_count) + 2*log(1 +
    success_count)``. Success-citations weighted more than mere
    citations — a note that's been validated through successful
    runs is more trustworthy than one that's been read but never
    associated with success. Log scaling so a runaway-popular
    note doesn't drown out new ones with a single citation.
    """
    import math
    note_by_slug = {n.slug: n for n in candidates}
    boosted: list[NoteMatch] = []
    for m in results:
        n = note_by_slug.get(m.summary.slug)
        if n is None:
            boosted.append(m)
            continue
        boost = (
            1.0
            + math.log(1 + n.cited_count)
            + 2.0 * math.log(1 + n.success_count)
        )
        boosted.append(
            NoteMatch(
                summary=m.summary,
                score=m.score * boost,
                snippet=m.snippet,
            )
        )
    boosted.sort(key=lambda m: m.score, reverse=True)
    return boosted[:limit]
