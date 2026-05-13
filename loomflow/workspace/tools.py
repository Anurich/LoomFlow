"""Tool factory wiring the five workspace tools onto an agent.

``make_workspace_tools(workspace, author)`` returns a list of five
:class:`Tool` instances:

* ``note(title, content, kind="finding", tags=None)`` — write to
  the shared notebook. Author + slug + timestamps auto-injected.
* ``read_note(slug_or_title)`` — read a teammate's note.
* ``list_notes(author=None, kind=None)`` — overview of what's in
  the notebook.
* ``search_notes(query)`` — text search across all notes.
* ``update_note(slug, content, tags=None)`` — revise your own
  note (you can only update notes you authored).

The agent's ``author`` identity is **baked in via closure** so
the model never has to type it. Multi-tenant ``user_id`` flows
through :func:`loomflow.core.get_run_context` at call time, so the
same tool instance partitions correctly across runs.

:func:`workspace_prompt_section` produces the markdown chunk
:meth:`Agent.__init__` appends to the system prompt when a
workspace is wired — tells the model the tools exist and how to
use them.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..core.context import get_run_context
from ..tools.registry import Tool
from .protocol import Workspace
from .types import NoteKind

__all__ = [
    "make_workspace_tools",
    "workspace_prompt_section",
]


_VALID_KINDS: tuple[NoteKind, ...] = (
    "finding",
    "question",
    "decision",
    "summary",
    "artifact",
    "plan",
    "note",
)


def _coerce_kind(value: str | None) -> NoteKind:
    if value is None:
        return "finding"
    low = value.lower()
    if low in _VALID_KINDS:
        return low  # type: ignore[return-value]
    return "note"


def _coerce_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        # Single string → split on commas for ergonomics.
        return [t.strip() for t in value.split(",") if t.strip()]
    if isinstance(value, Iterable):
        return [str(t).strip() for t in value if str(t).strip()]
    return []


def _current_user_id() -> str | None:
    """Read the multi-tenant partition key from the ambient
    :class:`~loomflow.RunContext`. Returns ``None`` (anonymous
    bucket) when called outside any active run."""
    return get_run_context().user_id


def _current_run_id() -> str | None:
    ctx = get_run_context()
    # ``RunContext.run_id`` is an empty string when no run is active.
    return ctx.run_id or None


def make_workspace_tools(
    workspace: Workspace,
    *,
    author: str = "agent",
    tool_prefix: str = "",
) -> list[Tool]:
    """Build the five agent-facing tools that consume ``workspace``.

    ``author`` is baked into every write so the agent never has
    to attribute itself. For a single :class:`Agent`, the default
    ``"agent"`` is fine; :class:`Team` builders override it with
    the worker's role name so the notebook shows
    ``researcher``, ``analyst``, etc. instead of generic
    ``agent`` entries.

    ``tool_prefix`` namespaces every tool name with ``"<prefix>__"``
    — useful if you need two workspaces wired to one agent (rare).
    Default empty string keeps the natural names.
    """

    def _name(base: str) -> str:
        return f"{tool_prefix}__{base}" if tool_prefix else base

    async def _note(
        title: str,
        content: str,
        kind: str = "finding",
        tags: Any = None,
    ) -> str:
        """Write a note to the team's shared notebook.

        Use this for findings, decisions, open questions, plans, or
        synthesis you want teammates to see. The note is auto-tagged
        with your name + timestamp; you don't manage filenames.

        Returns the note's slug — pass this to other agents (or to
        ``update_note``) if you want to point at a specific entry.
        """
        note = await workspace.write_note(
            author=author,
            title=title,
            body=content,
            kind=_coerce_kind(kind),
            tags=_coerce_tags(tags),
            user_id=_current_user_id(),
            run_id=_current_run_id(),
        )
        return (
            f"Saved as `{note.slug}` (author={note.author}, "
            f"kind={note.kind}). "
            f"Other agents will see this in `list_notes()`."
        )

    async def _read_note(slug_or_title: str) -> str:
        """Read a note from the team notebook by slug (``003-foo``)
        or by partial title match (case-insensitive)."""
        note = await workspace.read_note(
            slug_or_title,
            user_id=_current_user_id(),
        )
        if note is None:
            return (
                f"No note matched `{slug_or_title}`. "
                f"Try `list_notes()` to see what's available."
            )
        return (
            f"# {note.title}\n"
            f"_by {note.author} · kind={note.kind} · "
            f"updated {note.updated_at.isoformat()}_\n"
            f"\n{note.body}"
        )

    async def _list_notes(
        author_filter: str | None = None,
        kind: str | None = None,
    ) -> str:
        """List the team notebook's notes with one-line summaries.

        Returns the current index. Filter by ``author_filter`` /
        ``kind`` (finding / question / decision / summary / artifact /
        plan / note) to narrow.
        """
        kind_norm = _coerce_kind(kind) if kind else None
        # Pass kind_norm=None back through as None — the protocol
        # accepts that to mean "all kinds".
        summaries = await workspace.list_notes(
            author=author_filter,
            kind=kind_norm,
            user_id=_current_user_id(),
        )
        if not summaries:
            return (
                "The notebook is empty. You're the first contributor — "
                "call `note(title, content)` to share your findings."
            )
        lines = [f"# Notebook ({len(summaries)} notes)\n"]
        for s in summaries:
            tag_suffix = f" `[{', '.join(s.tags)}]`" if s.tags else ""
            lines.append(
                f"- **`{s.slug}`** [{s.author} · {s.kind}]{tag_suffix} "
                f"— {s.title}"
            )
            if s.lede:
                lines.append(f"  > {s.lede}")
        return "\n".join(lines)

    async def _search_notes(query: str) -> str:
        """Free-text search across every note in the team notebook.

        Returns ranked matches with snippets. Title hits rank higher
        than tag matches; tag matches rank higher than body hits.
        """
        matches = await workspace.search_notes(
            query, user_id=_current_user_id()
        )
        if not matches:
            return f"No notes matched `{query}`."
        lines = [f"# Search: {query} ({len(matches)} hit(s))\n"]
        for m in matches:
            s = m.summary
            lines.append(
                f"- **`{s.slug}`** [{s.author} · {s.kind}] — {s.title}"
            )
            lines.append(f"  > {m.snippet}")
        return "\n".join(lines)

    async def _update_note(
        slug: str,
        content: str,
        tags: Any = None,
    ) -> str:
        """Replace the body of a note you previously wrote.

        You may only update your own notes — attempting to overwrite
        a teammate's note returns an error.
        """
        try:
            note = await workspace.update_note(
                author=author,
                slug=slug,
                body=content,
                tags=_coerce_tags(tags) if tags is not None else None,
                user_id=_current_user_id(),
            )
        except FileNotFoundError:
            return f"ERROR: note `{slug}` not found in your namespace."
        except PermissionError as exc:
            return f"ERROR: {exc}"
        return (
            f"Updated `{note.slug}` (now {len(note.body)} chars)."
        )

    note_tool = Tool(
        name=_name("note"),
        description=(
            "Write a note to the team's shared notebook. Use for "
            "findings, decisions, open questions, plans. Returns the "
            "note's slug for later reference."
        ),
        fn=_note,
        input_schema={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short, descriptive title — slugified into the filename.",
                },
                "content": {
                    "type": "string",
                    "description": "Markdown body of the note.",
                },
                "kind": {
                    "type": "string",
                    "enum": list(_VALID_KINDS),
                    "default": "finding",
                    "description": (
                        "Note category — finding / question / decision / "
                        "summary / artifact / plan / note."
                    ),
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
            },
            "required": ["title", "content"],
        },
    )

    read_tool = Tool(
        name=_name("read_note"),
        description=(
            "Read a note from the team notebook by slug or partial "
            "title match (case-insensitive)."
        ),
        fn=_read_note,
        input_schema={
            "type": "object",
            "properties": {
                "slug_or_title": {
                    "type": "string",
                    "description": "Slug like '003-foo' or a title substring.",
                },
            },
            "required": ["slug_or_title"],
        },
    )

    list_tool = Tool(
        name=_name("list_notes"),
        description=(
            "List the team notebook's notes (newest first), with "
            "optional filters by author or kind."
        ),
        fn=_list_notes,
        input_schema={
            "type": "object",
            "properties": {
                "author_filter": {
                    "type": "string",
                    "description": "Only return notes by this author.",
                },
                "kind": {
                    "type": "string",
                    "enum": list(_VALID_KINDS),
                    "description": "Only return notes of this kind.",
                },
            },
            "required": [],
        },
    )

    search_tool = Tool(
        name=_name("search_notes"),
        description=(
            "Free-text search the team notebook. Returns ranked hits "
            "with snippets."
        ),
        fn=_search_notes,
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    )

    update_tool = Tool(
        name=_name("update_note"),
        description=(
            "Revise the body of a note you previously wrote. You "
            "may only update notes you authored."
        ),
        fn=_update_note,
        input_schema={
            "type": "object",
            "properties": {
                "slug": {"type": "string"},
                "content": {"type": "string"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["slug", "content"],
        },
    )

    return [note_tool, read_tool, list_tool, search_tool, update_tool]


def workspace_prompt_section(
    *,
    author: str = "agent",
    teammates: list[str] | None = None,
) -> str:
    """Return the markdown chunk :class:`Agent.__init__` appends to
    the system prompt when a workspace is wired.

    Includes the agent's identity and (when provided) the names of
    teammates so the model knows who else is contributing. Heavy
    nudging toward "list first, then act" to keep teams from
    duplicating work.
    """
    team_line = ""
    if teammates:
        others = [t for t in teammates if t != author]
        if others:
            team_line = (
                f"Your teammates are: {', '.join(others)}.\n"
            )
    return (
        "## Shared notebook\n\n"
        f"You are `{author}` on this team. {team_line}"
        "You share a notebook with teammates. Five tools are "
        "available:\n\n"
        "- `list_notes()` — what teammates have already written.\n"
        "- `read_note(slug_or_title)` — read a specific note.\n"
        "- `search_notes(query)` — find notes by free-text query.\n"
        "- `note(title, content, kind=)` — share findings, "
        "decisions, questions, plans.\n"
        "- `update_note(slug, content)` — revise your own notes.\n\n"
        "**Before doing significant work**, call `list_notes()` to "
        "check whether a teammate has already done it (or flagged it "
        "as in-progress with a `plan` note). If yes, build on their "
        "work; don't duplicate.\n\n"
        "**When you find something worth sharing**, call `note(...)`. "
        "One note per finding. Be concise — teammates will read this. "
        "Use `kind=\"question\"` to flag open problems for others to "
        "pick up; `kind=\"plan\"` to announce you're working on "
        "something; `kind=\"decision\"` for sticky choices.\n"
    )
