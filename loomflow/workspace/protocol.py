"""The :class:`Workspace` protocol.

Implementations live in :mod:`loomflow.workspace.disk` (the production
on-disk backend) and :mod:`loomflow.workspace.inmemory` (zero-dep
backend for tests + ephemeral coordination).

The protocol is async — all I/O methods return awaitables — and
multi-tenant — every method that filters notes accepts an optional
``user_id`` partition.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Protocol, runtime_checkable

from .types import (
    Note,
    NoteKind,
    NoteMatch,
    NoteSummary,
    NoteVersion,
    PruneResult,
    WorkspaceMembership,
)


@runtime_checkable
class Workspace(Protocol):
    """The shared-notebook backend the framework wires into agents.

    The five public methods correspond 1:1 with the five tools an
    agent sees (``note`` / ``read_note`` / ``list_notes`` /
    ``search_notes`` / ``update_note``). Additional methods
    (``render_index``, ``seed``, ``aclose``) cover lifecycle and
    introspection that agents don't need to know about.
    """

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
        """Append a new note to the workspace.

        Returns the constructed :class:`Note` with its assigned
        ``slug``. Implementations choose the slug shape (the disk
        backend uses ``<NNN>-<slugified-title>`` numbered per
        author); callers should treat the slug as opaque.

        Optional fields:

        * ``namespace`` — sub-bucket within the author's notes.
          Defaults to ``None`` (no namespace). Notes in different
          namespaces are still visible to ``list_notes`` /
          ``search_notes`` (namespace is metadata, not a partition).
        * ``answered`` — tri-state flag for ``kind="question"``
          notes. Set ``False`` when asking, flipped via
          :meth:`mark_answered`.
        * ``parent_slug`` — link to a parent note (e.g. an answer
          pointing at its question).
        """
        ...

    async def read_note(
        self,
        slug_or_title: str,
        *,
        user_id: str | None = None,
    ) -> Note | None:
        """Look up a note by slug or partial title match.

        Slug matches win over title matches. Title matching is
        case-insensitive and substring-based; ambiguous title
        matches return the most recently updated one.

        Archived notes ARE returned by this method — only listing
        and search exclude them by default.
        """
        ...

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
        """Return notes filtered by author / kind, newest first.

        Optional filters:

        * ``namespace`` — restrict to notes in this namespace.
          ``None`` (default) returns notes across all namespaces.
        * ``include_archived`` — when ``False`` (default), archived
          notes are excluded. Pass ``True`` to see them.
        """
        ...

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
        """Text-search notes. Implementations may use BM25, token
        overlap, semantic similarity (when an embedder is wired),
        or hybrid scoring — return :class:`NoteMatch` objects with
        backend-specific scores.

        ``mode`` selects scoring:

        * ``"auto"`` (default) — hybrid (BM25 + cosine via RRF)
          when an embedder is wired, BM25 otherwise.
        * ``"bm25"`` — text-only, even when an embedder is wired.
        * ``"semantic"`` — cosine-only; falls back to BM25 if no
          embedder is wired.
        * ``"hybrid"`` — explicit hybrid; falls back to BM25 if no
          embedder is wired.

        ``boost_relevance`` (default ``False``): when ``True``,
        notes' base scores get a multiplicative boost based on
        their citation metadata (``cited_count`` +
        ``success_count``). Frequently-read AND
        frequently-validated notes rank higher than mere
        text-matches. Opt-in to preserve back-compat for callers
        that don't want this signal.
        """
        ...

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
        """Replace the body of a note this author previously wrote.

        Raises :class:`PermissionError` if ``author`` doesn't own
        the note. Backends preserve the original ``created_at`` and
        bump ``updated_at``.

        Before overwriting, the prior body is snapshotted into the
        note's revision history (accessible via
        :meth:`list_versions` / :meth:`read_version`).

        ``mark_answered`` is the documented cross-author carve-out
        for the question / answer pattern: passing a non-None slug
        flips the target note's ``answered=True`` + ``answered_by=
        <slug>`` even when the calling author doesn't own the
        note. Used internally by ``answer_question``; rarely
        needed by user code.
        """
        ...

    async def archive_note(
        self,
        *,
        author: str,
        slug: str,
        user_id: str | None = None,
    ) -> Note:
        """Mark a note as archived in-place. Returns the updated
        note with ``archived_at`` set.

        Archived notes are excluded from ``list_notes`` /
        ``search_notes`` by default (opt back in via
        ``include_archived=True``). They remain directly readable
        by slug via :meth:`read_note`.

        Raises :class:`PermissionError` if ``author`` doesn't own
        the note.
        """
        ...

    async def list_versions(
        self,
        slug: str,
        *,
        author: str,
        user_id: str | None = None,
    ) -> list[NoteVersion]:
        """Return the revision history of a note, oldest first.

        Each :meth:`update_note` call snapshots the prior body as
        version N (monotonic starting at 1). Versions are
        immutable; deleting them is not part of the public
        protocol.
        """
        ...

    async def read_version(
        self,
        slug: str,
        version: int,
        *,
        author: str,
        user_id: str | None = None,
    ) -> Note | None:
        """Return the full :class:`Note` as it was at revision
        ``version``. Returns ``None`` if the version doesn't exist.

        The returned note's ``updated_at`` reflects the revision
        timestamp, not the current note's timestamp.
        """
        ...

    async def prune(
        self,
        *,
        older_than: timedelta | None = None,
        min_cited_count: int = 1,
        keep_kinds: list[NoteKind] | None = None,
        keep_last_versions: int | None = None,
        user_id: str | None = None,
    ) -> PruneResult:
        """Garbage-collect stale, low-value notes. **Hard-deletes**
        — the removed notes are gone, not archived.

        A note is **pruned** only when ALL of these hold:

        * ``older_than`` is set AND the note's last activity
          (``max(updated_at, last_cited_at)``) is older than that
          window. When ``older_than`` is ``None``, age is NOT a
          filter — every note becomes age-eligible. **Strongly
          recommended to pass ``older_than``** so a freshly-
          written note can't be pruned before it's had a chance
          to be cited.
        * the note's ``cited_count`` is BELOW ``min_cited_count``
          (default 1 — i.e. a note cited at least once survives).
        * the note's ``kind`` is NOT in ``keep_kinds`` (e.g. pass
          ``["decision"]`` to never prune decisions).

        ``keep_last_versions``: when set, each surviving note's
        revision history is trimmed to the most recent N
        revisions. ``None`` (default) leaves history untouched.

        Returns a :class:`PruneResult` with counts. Idempotent in
        spirit — running it twice with the same args just deletes
        nothing the second time.

        ``prune`` is **observation-class** like
        :meth:`attribute_outcome` — it does not check author
        ownership, because it's an operator / maintenance
        operation, not an agent action. Don't wire it as an agent
        tool; call it from a cron job, an end-of-benchmark hook,
        or manually.
        """
        ...

    async def attribute_outcome(
        self,
        *,
        success: bool,
        user_id: str | None = None,
    ) -> int:
        """Close the self-improvement loop: take every note the
        agent cited during the current run (tracked via the
        ambient ``_ambient_citations_var`` contextvar) and update
        the per-note relevance metadata.

        Specifically:

        * ``cited_count`` += 1 for every cited note
        * ``success_count`` += 1 if ``success=True``
        * ``last_cited_at`` = now for every cited note

        Returns the number of notes whose metadata was updated
        (zero outside a run, or if the run cited nothing).

        ``attribute_outcome`` is **observation-class** — it does
        not check author ownership. Anyone with workspace access
        can report what they cited and the outcome; this is how
        you know which past notes were USEFUL versus just
        present.

        Idempotency: calling twice with ``success=True`` for the
        same run double-counts. Callers should call it once per
        run. To opt out of citation tracking, simply don't call
        this method — the per-run citation log evaporates with
        the contextvar.
        """
        ...

    async def render_index(
        self,
        *,
        user_id: str | None = None,
    ) -> str:
        """Return the rendered ``WORKSPACE.md`` index — the
        human-readable table of contents the framework keeps in
        sync with note writes. Workspaces also write this to disk
        on-disk-backed implementations."""
        ...

    async def aclose(self) -> None:
        """Release any resources (file handles, locks). Idempotent."""
        ...

    def member(
        self,
        name: str | None = None,
        *,
        teammates: list[str] | None = None,
    ) -> WorkspaceMembership:
        """Return a :class:`WorkspaceMembership` joining this workspace
        as ``name`` with the given teammates.

        Use at the call site to collapse three Agent kwargs
        (``workspace`` + ``workspace_name`` + ``workspace_teammates``)
        into one::

            Agent(
                "...",
                workspace=ws.member("researcher", teammates=["analyst", "writer"]),
            )

        Default implementations on both shipped backends just return
        a ``WorkspaceMembership(self, name, teammates)``; custom
        backends rarely need to override.
        """
        ...
