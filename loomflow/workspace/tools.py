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
    namespace: str | None = None,
    questions: bool = False,
    include_archive: bool = True,
) -> list[Tool]:
    """Build the agent-facing tools that consume ``workspace``.

    Default tool set (always-on):

    * ``note`` / ``read_note`` / ``list_notes`` / ``search_notes``
      / ``update_note`` — the five-tool core surface.
    * ``archive_note`` (when ``include_archive=True``, default) —
      marks a note as archived. Excluded from listings by default;
      still readable by slug.

    Optional tool set (``questions=True``):

    * ``ask_question`` / ``answer_question`` / ``list_open_questions``
      — async-message pattern over notes. ``ask_question`` writes
      a ``kind="question"`` note with ``answered=False``;
      ``answer_question`` writes a child finding note and flips
      the question's ``answered=True`` (cross-author safe via the
      ``mark_answered`` carve-out in :meth:`Workspace.update_note`).

    Kwargs:

    * ``author`` is baked into every write so the agent never has
      to attribute itself. :class:`Team` builders override it with
      the worker's role name.
    * ``tool_prefix`` namespaces every tool name with
      ``"<prefix>__"`` — for when two workspaces are wired to one
      agent (rare).
    * ``namespace`` scopes WRITES to a sub-bucket within this
      author's notes. ``list_notes`` / ``search_notes`` still see
      every namespace by default — namespace is metadata, not a
      filter, so teammates' work in adjacent namespaces stays
      visible (you can filter explicitly on a per-call basis).
    * ``questions`` enables the three question tools.
    * ``include_archive`` (default True) wires the ``archive_note``
      tool.
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
            namespace=namespace,
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

    tools_out: list[Tool] = [
        note_tool, read_tool, list_tool, search_tool, update_tool,
    ]

    # ---- Optional: archive_note (default-on) ----------------------------

    if include_archive:
        async def _archive_note(slug: str) -> str:
            """Mark a note you wrote as archived.

            Archived notes are excluded from ``list_notes`` /
            ``search_notes`` by default but remain readable by
            slug. Use to clean up stale notes without losing the
            record. You may only archive notes you authored.
            """
            try:
                note = await workspace.archive_note(
                    author=author,
                    slug=slug,
                    user_id=_current_user_id(),
                )
            except FileNotFoundError:
                return f"ERROR: note `{slug}` not found."
            except PermissionError as exc:
                return f"ERROR: {exc}"
            return f"Archived `{note.slug}`."

        archive_tool = Tool(
            name=_name("archive_note"),
            description=(
                "Mark a note as archived. Archived notes are hidden "
                "from list/search by default but still readable by "
                "slug. You may only archive your own notes."
            ),
            fn=_archive_note,
            input_schema={
                "type": "object",
                "properties": {"slug": {"type": "string"}},
                "required": ["slug"],
            },
        )
        tools_out.append(archive_tool)

    # ---- Optional: question / answer / list_open_questions --------------

    if questions:
        async def _ask_question(title: str, content: str) -> str:
            """Post an open question to the notebook for someone
            else (or future-you) to answer.

            Writes a ``kind="question"`` note with ``answered=False``.
            Use ``list_open_questions()`` to find unanswered ones,
            and ``answer_question(slug, content)`` to respond.
            """
            note = await workspace.write_note(
                author=author,
                title=title,
                body=content,
                kind="question",
                user_id=_current_user_id(),
                run_id=_current_run_id(),
                namespace=namespace,
                answered=False,
            )
            return (
                f"Asked as `{note.slug}`. Teammates can find this "
                "via `list_open_questions()` and respond with "
                f"`answer_question('{note.slug}', ...)`."
            )

        async def _answer_question(slug: str, content: str) -> str:
            """Answer an open question and mark it resolved.

            Writes a child ``kind="finding"`` note linked to the
            question, then flips the question's ``answered=True``
            (cross-author safe — you can answer questions other
            agents asked).
            """
            question = await workspace.read_note(
                slug, user_id=_current_user_id()
            )
            if question is None:
                return f"ERROR: question `{slug}` not found."
            answer = await workspace.write_note(
                author=author,
                title=f"Answer: {question.title}",
                body=content,
                kind="finding",
                user_id=_current_user_id(),
                run_id=_current_run_id(),
                namespace=namespace,
                parent_slug=slug,
            )
            try:
                # Cross-author carve-out: pass ``mark_answered`` so
                # the workspace flips ``answered=True`` even when
                # `author` doesn't own the question.
                await workspace.update_note(
                    author=author,
                    slug=slug,
                    body=question.body,
                    tags=list(question.tags),
                    user_id=_current_user_id(),
                    mark_answered=answer.slug,
                )
            except Exception as exc:  # noqa: BLE001 — surface as text
                return (
                    f"Answer saved as `{answer.slug}` but could not "
                    f"mark question answered: {exc}"
                )
            return (
                f"Answered. Answer saved as `{answer.slug}` and "
                f"question `{slug}` is now marked answered."
            )

        async def _list_open_questions() -> str:
            """List unanswered questions (``kind="question"`` notes
            where ``answered`` is False or absent)."""
            summaries = await workspace.list_notes(
                kind="question",
                user_id=_current_user_id(),
                limit=100,
            )
            # The summary doesn't carry the ``answered`` flag
            # directly; re-fetch the full Notes to filter.
            open_q: list[str] = []
            for s in summaries:
                note = await workspace.read_note(
                    s.slug, user_id=_current_user_id()
                )
                if note is None:
                    continue
                if note.answered is True:
                    continue
                tag_suffix = (
                    f" `[{', '.join(note.tags)}]`" if note.tags else ""
                )
                open_q.append(
                    f"- **`{note.slug}`** [{note.author}]{tag_suffix} "
                    f"— {note.title}"
                )
                if note.body:
                    preview = note.body[:120].replace("\n", " ")
                    open_q.append(f"  > {preview}")
            if not open_q:
                return "No open questions."
            return (
                f"# Open questions ({len(open_q) // 2})\n\n"
                + "\n".join(open_q)
            )

        tools_out.append(
            Tool(
                name=_name("ask_question"),
                description=(
                    "Post an open question to the notebook for "
                    "another agent (or future-you) to answer."
                ),
                fn=_ask_question,
                input_schema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["title", "content"],
                },
            )
        )
        tools_out.append(
            Tool(
                name=_name("answer_question"),
                description=(
                    "Answer an open question and mark it resolved. "
                    "Cross-author safe — answer anyone's question."
                ),
                fn=_answer_question,
                input_schema={
                    "type": "object",
                    "properties": {
                        "slug": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["slug", "content"],
                },
            )
        )
        tools_out.append(
            Tool(
                name=_name("list_open_questions"),
                description=(
                    "List unanswered questions in the notebook — "
                    "things teammates flagged for someone else to "
                    "pick up."
                ),
                fn=_list_open_questions,
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            )
        )

    return tools_out


def workspace_prompt_section(
    *,
    author: str = "agent",
    teammates: list[str] | None = None,
    include_archive: bool = True,
    questions: bool = False,
) -> str:
    """Return the markdown chunk :class:`Agent.__init__` appends to
    the system prompt when a workspace is wired.

    The prompt is **shape-aware**:

    * When ``teammates`` is non-empty (multi-agent team mode), the
      copy emphasises COORDINATION: "share findings with
      teammates, list first to avoid duplicating their work."
    * When ``teammates`` is None / empty (single-agent cross-run
      mode), the copy emphasises CROSS-RUN PERSISTENCE: "this is
      YOUR persistent knowledge across runs; check what past-you
      wrote, leave findings for future-you."

    Why two variants? Empirical: when the prompt mentioned
    "teammates" but no teammates existed, models concluded the
    notebook was empty / not relevant and skipped it. The
    single-agent variant tells the model the notebook IS its
    own continuity. (Observed in Terminal-Bench runs, May 2026.)
    """
    others = (
        [t for t in (teammates or []) if t != author] if teammates else []
    )
    is_team_mode = bool(others)
    sections: list[str] = []

    # Tool list (shared across both modes).
    tool_lines = [
        "- `list_notes()` — see what's already written.",
        "- `read_note(slug_or_title)` — read a specific note.",
        "- `search_notes(query)` — find notes by free-text query.",
        "- `note(title, content, kind=)` — share findings, "
        "decisions, plans.",
        "- `update_note(slug, content)` — revise your own notes.",
    ]
    if include_archive:
        tool_lines.append(
            "- `archive_note(slug)` — mark a stale note archived "
            "(hidden from listings but still readable by slug)."
        )
    if questions:
        tool_lines.append(
            "- `ask_question(title, content)` — flag an open "
            "problem for someone else (or future-you) to answer."
        )
        tool_lines.append(
            "- `answer_question(slug, content)` — answer an open "
            "question and mark it resolved."
        )
        tool_lines.append(
            "- `list_open_questions()` — see what's unanswered."
        )

    if is_team_mode:
        # ---- Multi-agent team variant ----
        sections.append("## Shared notebook")
        sections.append("")
        sections.append(
            f"You are `{author}` on this team. Your teammates are: "
            f"{', '.join(others)}."
        )
        sections.append("")
        sections.append(
            "You share a notebook with teammates. Tools available:"
        )
        sections.append("")
        sections.extend(tool_lines)
        sections.append("")
        sections.append(
            "**Before doing significant work**, call `list_notes()` "
            "to check whether a teammate has already done it (or "
            "flagged it as in-progress with a `plan` note). If yes, "
            "build on their work; don't duplicate."
        )
        sections.append("")
        sections.append(
            "**When you find something worth sharing**, call "
            "`note(...)`. One note per finding. Be concise — "
            "teammates will read this. Use `kind=\"question\"` to "
            "flag open problems; `kind=\"plan\"` to announce work "
            "in progress; `kind=\"decision\"` for sticky choices."
        )
    else:
        # ---- Single-agent cross-run variant ----
        sections.append("## Your persistent notebook")
        sections.append("")
        sections.append(
            f"You are `{author}`. This notebook is YOUR persistent "
            "knowledge across runs — every note you write now is "
            "readable by future-you in subsequent runs of this "
            "agent. Tools available:"
        )
        sections.append("")
        sections.extend(tool_lines)
        sections.append("")
        sections.append(
            "**At the start of every task**, call `list_notes()` "
            "(or `search_notes(query)` with terms from the task) "
            "to see what past-you already learned. If a past note "
            "is relevant, `read_note(slug)` it before diving in. "
            "The notebook may be empty on your first run — that's "
            "expected; you're building the knowledge base."
        )
        sections.append("")
        sections.append(
            "**As you work**, call `note(...)` for findings worth "
            "preserving. Be concise. Use `kind=\"decision\"` for "
            "sticky choices you'll want to remember; "
            "`kind=\"finding\"` for analysis results; "
            "`kind=\"plan\"` for in-progress strategies. "
            "Future-you (next task) will thank present-you."
        )

    return "\n".join(sections) + "\n"
