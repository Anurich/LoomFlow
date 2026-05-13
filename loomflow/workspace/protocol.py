"""The :class:`Workspace` protocol.

Implementations live in :mod:`loomflow.workspace.disk` (the production
on-disk backend) and :mod:`loomflow.workspace.inmemory` (zero-dep
backend for tests + ephemeral coordination).

The protocol is async — all I/O methods return awaitables — and
multi-tenant — every method that filters notes accepts an optional
``user_id`` partition.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import Note, NoteKind, NoteMatch, NoteSummary, WorkspaceMembership


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
    ) -> Note:
        """Append a new note to the workspace.

        Returns the constructed :class:`Note` with its assigned
        ``slug``. Implementations choose the slug shape (the disk
        backend uses ``<NNN>-<slugified-title>`` numbered per
        author); callers should treat the slug as opaque.
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
        """
        ...

    async def list_notes(
        self,
        *,
        author: str | None = None,
        kind: NoteKind | None = None,
        user_id: str | None = None,
        limit: int = 50,
    ) -> list[NoteSummary]:
        """Return notes filtered by author / kind, newest first."""
        ...

    async def search_notes(
        self,
        query: str,
        *,
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[NoteMatch]:
        """Text-search notes. Implementations may use BM25, token
        overlap, or simple substring — return :class:`NoteMatch`
        objects with backend-specific scores."""
        ...

    async def update_note(
        self,
        *,
        author: str,
        slug: str,
        body: str,
        tags: list[str] | None = None,
        user_id: str | None = None,
    ) -> Note:
        """Replace the body of a note this author previously wrote.

        Raises :class:`PermissionError` if ``author`` doesn't own
        the note. Backends preserve the original ``created_at`` and
        bump ``updated_at``.
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
