"""In-memory :class:`Workspace` backend.

Zero-dependency, no persistence, no filesystem. Useful for tests,
ephemeral coordination inside a single process, and as a reference
implementation for what the Protocol promises behaviourally.

Same multi-tenant partition rules as the disk backend — notes from
different ``user_id`` runs are never visible to each other.
"""

from __future__ import annotations

from datetime import UTC, datetime

import anyio

from ._common import (
    extract_lede,
    render_workspace_index,
    slugify_title,
    summary_from_note,
)
from .types import Note, NoteKind, NoteMatch, NoteSummary, WorkspaceMembership


class InMemoryWorkspace:
    """Dict-backed shared notebook.

    State lives in-process; lost on restart. The author-keyed
    ``_per_author`` index keeps the per-author counter so slugs
    stay deterministic across the run.
    """

    def __init__(self) -> None:
        # ``(user_id, slug)`` -> Note. ``user_id=None`` is the
        # anonymous bucket and is partitioned from named users.
        self._notes: dict[tuple[str | None, str], Note] = {}
        # ``(user_id, author)`` -> next counter for slugging.
        self._counters: dict[tuple[str | None, str], int] = {}
        self._lock = anyio.Lock()

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
            )
            self._notes[(user_id, slug)] = note
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
        async with self._lock:
            existing = self._notes.get((user_id, slug))
            if existing is None:
                raise FileNotFoundError(
                    f"note {slug!r} not found in workspace"
                )
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
            self._notes[(user_id, slug)] = updated
            return updated

    # ---- reads -----------------------------------------------------------

    async def read_note(
        self,
        slug_or_title: str,
        *,
        user_id: str | None = None,
    ) -> Note | None:
        # Try slug match first (exact).
        direct = self._notes.get((user_id, slug_or_title))
        if direct is not None:
            return direct
        # Fall back to case-insensitive title substring. Ambiguous
        # matches: most recently updated wins.
        needle = slug_or_title.lower()
        candidates = [
            n
            for (uid, _), n in self._notes.items()
            if uid == user_id and needle in n.title.lower()
        ]
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
        results: list[Note] = []
        for (uid, _), note in self._notes.items():
            if uid != user_id:
                continue
            if author is not None and note.author != author:
                continue
            if kind is not None and note.kind != kind:
                continue
            results.append(note)
        results.sort(key=lambda n: n.updated_at, reverse=True)
        return [summary_from_note(n) for n in results[:limit]]

    async def search_notes(
        self,
        query: str,
        *,
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[NoteMatch]:
        """Substring + tag search. Title hits score higher than body
        hits; tag hits score in between."""
        q = query.lower().strip()
        if not q:
            return []
        scored: list[tuple[float, str, Note]] = []
        for (uid, _), note in self._notes.items():
            if uid != user_id:
                continue
            title_l = note.title.lower()
            body_l = note.body.lower()
            tag_match = any(q in t.lower() for t in note.tags)
            if q in title_l:
                snippet = note.title
                scored.append((1.0, snippet, note))
            elif tag_match:
                snippet = "tags: " + ", ".join(note.tags)
                scored.append((0.7, snippet, note))
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
