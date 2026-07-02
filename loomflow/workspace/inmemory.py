"""In-memory :class:`Workspace` backend.

Zero-dependency, no persistence, no filesystem. Useful for tests,
ephemeral coordination inside a single process, and as a reference
implementation for what the Protocol promises behaviourally.

Same multi-tenant partition rules as the disk backend — notes from
different ``user_id`` runs are never visible to each other.

Parity with the disk backend (v0.10.x): namespacing, archive,
revision history, semantic search (via optional embedder), and the
cross-author ``mark_answered`` carve-out.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import anyio

from ._common import apply_relevance_boost as _apply_relevance_boost
from ._common import cosine_similarity as _cosine
from ._common import drain_citations as _drain_citations
from ._common import (
    extract_lede,
    render_workspace_index,
    slugify_title,
    summary_from_note,
)
from ._common import log_citation as _log_citation
from ._common import rrf_fuse as _rrf_fuse
from ._common import score_bm25 as _score_bm25
from ._common import score_semantic as _score_semantic
from .disk import _should_prune
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
    from ..core.protocols import Embedder


class InMemoryWorkspace:
    """Dict-backed shared notebook.

    State lives in-process; lost on restart. The author-keyed
    ``_per_author`` index keeps the per-author counter so slugs
    stay deterministic across the run.
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        # ``(user_id, slug)`` -> Note. ``user_id=None`` is the
        # anonymous bucket and is partitioned from named users.
        self._notes: dict[tuple[str | None, str], Note] = {}
        # ``(user_id, author)`` -> next counter for slugging.
        self._counters: dict[tuple[str | None, str], int] = {}
        # ``(user_id, author, slug)`` -> list of historical Notes,
        # oldest first. Versioning parity with the disk backend.
        self._history: dict[
            tuple[str | None, str, str], list[Note]
        ] = {}
        # ``(user_id, slug)`` -> embedding vector. Computed on
        # write when an embedder is wired; lazily used by
        # search_notes(mode="semantic"|"hybrid").
        self._embeddings: dict[tuple[str | None, str], list[float]] = {}
        self._lock = anyio.Lock()
        self._embedder = embedder

    # ---- mutation --------------------------------------------------------

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
        async with self._lock:
            slug_frag = slugify_title(title)
            counter_key = (user_id, author)
            counter = self._counters.get(counter_key, 0) + 1
            self._counters[counter_key] = counter
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
            self._notes[(user_id, slug)] = note
        # Compute embedding outside the lock (it can be slow / I/O).
        await self._maybe_embed(user_id, slug, note)
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
        async with self._lock:
            existing = self._notes.get((user_id, slug))
            if existing is None:
                raise FileNotFoundError(
                    f"note {slug!r} not found in workspace"
                )
            # Cross-author mark_answered carve-out — symmetric with
            # the disk backend.
            if mark_answered is not None and existing.author != author:
                hist_key = (user_id, existing.author, slug)
                self._history.setdefault(hist_key, []).append(existing)
                updated = existing.model_copy(
                    update={
                        "answered": True,
                        "answered_by": mark_answered,
                        "updated_at": datetime.now(UTC),
                    }
                )
                self._notes[(user_id, slug)] = updated
                note_for_embed = updated
            else:
                if existing.author != author:
                    raise PermissionError(
                        f"agent {author!r} cannot update note {slug!r} "
                        f"owned by {existing.author!r}"
                    )
                hist_key = (user_id, author, slug)
                self._history.setdefault(hist_key, []).append(existing)
                update_dict: dict[str, object] = {
                    "body": body,
                    "tags": (
                        list(tags) if tags is not None
                        else list(existing.tags)
                    ),
                    "updated_at": datetime.now(UTC),
                }
                if mark_answered is not None:
                    update_dict["answered"] = True
                    update_dict["answered_by"] = mark_answered
                updated = existing.model_copy(update=update_dict)
                self._notes[(user_id, slug)] = updated
                note_for_embed = updated
        await self._maybe_embed(user_id, slug, note_for_embed)
        return updated

    async def archive_note(
        self,
        *,
        author: str,
        slug: str,
        user_id: str | None = None,
    ) -> Note:
        async with self._lock:
            existing = self._notes.get((user_id, slug))
            if existing is None:
                raise FileNotFoundError(
                    f"note {slug!r} not found in workspace"
                )
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
            self._notes[(user_id, slug)] = archived
            return archived

    # ---- reads -----------------------------------------------------------

    async def read_note(
        self,
        slug_or_title: str,
        *,
        user_id: str | None = None,
    ) -> Note | None:
        # Try slug match first (exact). Archived notes are
        # readable by slug — only listing / search exclude them.
        direct = self._notes.get((user_id, slug_or_title))
        if direct is not None:
            _log_citation(direct.slug)
            return direct
        needle = slug_or_title.lower()
        candidates = [
            n
            for (uid, _), n in self._notes.items()
            if uid == user_id and needle in n.title.lower()
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda n: n.updated_at, reverse=True)
        winner = candidates[0]
        _log_citation(winner.slug)
        return winner

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
        results: list[Note] = []
        for (uid, _), note in self._notes.items():
            if uid != user_id:
                continue
            if author is not None and note.author != author:
                continue
            if kind is not None and note.kind != kind:
                continue
            if namespace is not None and note.namespace != namespace:
                continue
            if not include_archived and note.archived_at is not None:
                continue
            results.append(note)
        results.sort(key=lambda n: n.updated_at, reverse=True)
        return [summary_from_note(n) for n in results[:limit]]

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
        """Substring + tag search (BM25-ish). When an embedder is
        wired AND mode allows, falls back to / combines with
        cosine via reciprocal rank fusion. Title hits still score
        higher than body hits in the BM25 tier."""
        q = query.lower().strip()
        if not q:
            return []
        candidates: list[Note] = []
        for (uid, _), note in self._notes.items():
            if uid != user_id:
                continue
            if namespace is not None and note.namespace != namespace:
                continue
            if not include_archived and note.archived_at is not None:
                continue
            candidates.append(note)
        has_embedder = self._embedder is not None
        effective_mode = mode
        if effective_mode == "auto":
            effective_mode = "hybrid" if has_embedder else "bm25"
        if effective_mode in ("semantic", "hybrid") and not has_embedder:
            effective_mode = "bm25"
        def _boost(results: list[NoteMatch]) -> list[NoteMatch]:
            return (
                _apply_relevance_boost(results, candidates, limit)
                if boost_relevance else results
            )

        if effective_mode == "bm25":
            return _boost(_score_bm25(q, candidates, limit))
        assert self._embedder is not None  # narrowed for type-checker
        try:
            qvec = await self._embedder.embed(query)
        except anyio.get_cancelled_exc_class():
            raise
        except Exception:  # noqa: BLE001
            return _boost(_score_bm25(q, candidates, limit))
        if not qvec:
            return _boost(_score_bm25(q, candidates, limit))
        sem_scores: dict[str, float] = {}
        for n in candidates:
            stored = self._embeddings.get((user_id, n.slug))
            if stored is None or len(stored) != len(qvec):
                continue
            sem_scores[n.slug] = _cosine(qvec, stored)
        if effective_mode == "semantic":
            return _boost(_score_semantic(sem_scores, candidates, limit))
        bm25 = _score_bm25(q, candidates, limit=len(candidates))
        return _boost(_rrf_fuse(bm25, sem_scores, candidates, limit))

    # ---- versioning ------------------------------------------------------

    async def list_versions(
        self,
        slug: str,
        *,
        author: str,
        user_id: str | None = None,
    ) -> list[NoteVersion]:
        hist = self._history.get((user_id, author, slug), [])
        out: list[NoteVersion] = []
        for i, note in enumerate(hist, start=1):
            out.append(
                NoteVersion(
                    slug=slug,
                    author=author,
                    version=i,
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
        hist = self._history.get((user_id, author, slug), [])
        if version < 1 or version > len(hist):
            return None
        _log_citation(slug)
        return hist[version - 1]

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
        async with self._lock:
            # Snapshot keys first — we mutate the dict during the loop.
            for (uid, slug) in list(self._notes.keys()):
                if uid != user_id:
                    continue
                note = self._notes[(uid, slug)]
                if _should_prune(
                    note,
                    now=now,
                    older_than=older_than,
                    min_cited_count=min_cited_count,
                    keep_kind_set=keep_kind_set,
                ):
                    del self._notes[(uid, slug)]
                    self._embeddings.pop((uid, slug), None)
                    self._history.pop((uid, note.author, slug), None)
                    notes_deleted += 1
                else:
                    notes_kept += 1
                    if keep_last_versions is not None:
                        hist_key = (uid, note.author, slug)
                        hist = self._history.get(hist_key)
                        if hist and len(hist) > keep_last_versions:
                            excess = len(hist) - keep_last_versions
                            self._history[hist_key] = hist[excess:]
                            versions_deleted += excess
        return PruneResult(
            notes_deleted=notes_deleted,
            versions_deleted=versions_deleted,
            notes_kept=notes_kept,
        )

    async def attribute_outcome(
        self,
        *,
        success: bool,
        slugs: list[str] | None = None,
        user_id: str | None = None,
    ) -> int:
        # Explicit slugs win (post-run path); fall back to draining
        # the contextvar only when none were passed (in-run path).
        cited = set(slugs) if slugs is not None else _drain_citations()
        if not cited:
            return 0
        now = datetime.now(UTC)
        updated = 0
        async with self._lock:
            for slug in cited:
                key = (user_id, slug)
                existing = self._notes.get(key)
                if existing is None:
                    continue
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
                self._notes[key] = patched
                updated += 1
        return updated

    # ---- introspection / lifecycle --------------------------------------

    async def render_index(
        self,
        *,
        user_id: str | None = None,
    ) -> str:
        summaries = await self.list_notes(user_id=user_id, limit=10_000)
        return render_workspace_index(summaries)

    async def aclose(self) -> None:
        return None

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

    # ---- helpers ---------------------------------------------------------

    async def _maybe_embed(
        self, user_id: str | None, slug: str, note: Note
    ) -> None:
        if self._embedder is None:
            return
        try:
            text = f"{note.title}\n\n{note.body}"
            vector = await self._embedder.embed(text)
            if vector:
                self._embeddings[(user_id, slug)] = list(vector)
        except anyio.get_cancelled_exc_class():
            raise
        except Exception:  # noqa: BLE001 — embedding is best-effort
            pass


# Scoring helpers (BM25 / semantic cosine / RRF fusion / relevance
# boost) and the citation-logging helpers live in ``_common`` and are
# shared with the disk backend so the two can't drift — see the
# aliased imports at the top of this module.
