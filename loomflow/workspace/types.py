"""Value types for the shared-notebook workspace.

Three Pydantic models flow through the workspace surface:

* :class:`Note` — one entry in the notebook with author, slug,
  title, body, kind, timestamps, and free-form tags.
* :class:`NoteSummary` — the cheap projection of a :class:`Note`
  used in index views; carries only the metadata + one-line lede.
* :class:`NoteMatch` — a search result wrapping a
  :class:`NoteSummary` with a relevance score and a snippet.

Notes carry YAML frontmatter on disk; the workspace serialises /
parses it transparently. Agents never see frontmatter directly —
they get :class:`Note` / :class:`NoteSummary` objects (or rendered
markdown) through the tool surface.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

NoteKind = Literal[
    "finding",     # a fact / result the team should know
    "question",    # an open problem for someone else to pick up
    "decision",    # a choice made (sticky; reference for later turns)
    "summary",     # a synthesis across other notes
    "artifact",    # large output (code, draft, table); body is the artifact
    "plan",        # what an agent is about to do (avoid duplicate work)
    "note",        # generic fallback
]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Note(BaseModel):
    """One notebook entry.

    Immutable from the agent's point of view — to revise, call
    ``update_note(slug, ...)`` and a fresh :class:`Note` is written
    in its place. The old version stays in the audit log.

    ``slug`` is the stable identifier — generated from the title +
    a per-author counter. Other agents can refer to a note by slug
    (``read_note("003-population-trends")``) or by partial title
    match (``read_note("population")``).
    """

    # ``extra="ignore"`` makes future schema additions soft: if a
    # newer-version workspace writes a note with a field this
    # version doesn't know about, we drop the field rather than
    # raise. Combined with the optional new fields below, this
    # gives forward + backward compatibility for free.
    model_config = ConfigDict(frozen=True, extra="ignore")

    slug: str
    """``<NNN>-<slugified-title>`` — stable identifier within the
    author's namespace."""

    author: str
    """The agent name that wrote this note. The framework injects
    this from the workspace tool factory; agents never type it."""

    title: str
    """One-line human-readable title."""

    body: str
    """Markdown body. May span many lines; no length cap is
    enforced by the workspace, but agents are nudged toward
    concise notes via the prompt injection."""

    kind: NoteKind = "finding"

    tags: list[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    user_id: str | None = None
    """Multi-tenant partition. Notes from different ``user_id`` runs
    never appear in each other's listings even on a shared workspace
    root."""

    run_id: str | None = None
    """The :class:`RunContext.run_id` that produced this note.
    Useful for replay / per-run filtering."""

    namespace: str | None = None
    """Optional sub-bucket within the author's notes. Lets one
    workspace hold multiple logically-distinct sub-projects without
    spinning up separate workspaces. Defaults to ``None`` (no
    namespace subdir; behaves exactly like the pre-namespace
    workspace). See :meth:`Workspace.write_note`."""

    archived_at: datetime | None = None
    """Set by :meth:`Workspace.archive_note` to mark a stale note
    as archived. Archived notes are excluded from ``list_notes`` /
    ``search_notes`` by default (opt-in via ``include_archived=
    True``) but remain readable by slug via ``read_note``."""

    answered: bool | None = None
    """For ``kind="question"`` notes: ``False`` = open, ``True`` =
    answered. ``None`` means "not a question / not tracked" — the
    tri-state lets non-question notes leave the field absent rather
    than lie with ``False``. Flipped by
    :meth:`Workspace.mark_answered`."""

    answered_by: str | None = None
    """Slug of the answer note that resolved a question. Set by
    :meth:`Workspace.mark_answered` alongside ``answered=True``."""

    parent_slug: str | None = None
    """Slug of a parent note this note is a child of (typically an
    answer pointing back at its question). Lets the workspace build
    threads / link graphs without a separate edge table."""


class NoteSummary(BaseModel):
    """Cheap projection of :class:`Note` for index views.

    ``lede`` is the first non-empty line of the body, truncated to
    ~120 chars. Lets ``list_notes()`` render a useful overview
    without loading every body into the prompt.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    slug: str
    author: str
    title: str
    kind: NoteKind
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    lede: str
    """First ~120 chars of the body for quick scanning."""

    namespace: str | None = None
    """Optional sub-bucket; mirrors :attr:`Note.namespace`. Surfaced
    in the summary so callers can render namespace tags without
    fetching the full body."""

    archived_at: datetime | None = None
    """Mirrors :attr:`Note.archived_at`. Lets index renderers show
    an "(archived)" badge without re-reading the body."""


class NoteMatch(BaseModel):
    """A search hit — :class:`NoteSummary` plus relevance + snippet."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    summary: NoteSummary
    score: float
    """Higher is better; not necessarily normalised — backends pick
    a comparable-within-result-set scoring scheme. ``1.0`` is a
    title hit; lower values are body matches."""

    snippet: str
    """A short excerpt around the matched phrase (or the lede when
    the match is title-only). ~140 chars."""


class NoteVersion(BaseModel):
    """One historical revision of a note.

    Returned by :meth:`Workspace.list_versions` (one per revision)
    and :meth:`Workspace.read_version` (the full body of one
    revision). Versions are immutable; ``update_note`` appends a
    new version, never modifies an old one.

    Counter is monotonic per-slug starting at 1; ``0001.md``,
    ``0002.md``, ... on the disk backend. Re-using version numbers
    is a bug.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    slug: str
    """Slug of the live note this is a revision of."""

    author: str
    """Author of the note at the time of this revision."""

    version: int
    """Monotonic revision number starting at 1. ``list_versions``
    returns these in ascending order."""

    created_at: datetime
    """When this revision was written (i.e. when ``update_note``
    superseded the prior content)."""

    body_preview: str
    """First ~120 chars of the historical body. Lets
    ``list_versions`` render scannable history without loading
    every revision body. Fetch the full body with
    :meth:`Workspace.read_version`."""


class WorkspaceMembership(BaseModel):
    """A :class:`~loomflow.Workspace` plus the agent's identity in
    it — the single argument :class:`~loomflow.Agent` accepts when
    you want to join a shared notebook as a named role.

    Construct via :meth:`Workspace.member` (chained, IDE-friendly)::

        ws = LocalDiskWorkspace.temp()
        Agent(
            "...",
            workspace=ws.member("researcher", teammates=["analyst", "writer"]),
        )

    Or via the dict form (declarative / TOML-friendly)::

        Agent(
            "...",
            workspace={
                "backend": ws,
                "author": "researcher",
                "teammates": ["analyst", "writer"],
            },
        )

    Both end up as a frozen :class:`WorkspaceMembership`. The
    framework unpacks ``workspace`` / ``name`` / ``teammates`` and
    wires the five notebook tools attributed to ``name``, with
    a prompt section listing the teammates by role.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    workspace: Any
    """The shared :class:`Workspace` instance. Multiple agents in
    one team should share the SAME workspace instance; this field
    is intentionally not typed as the Protocol so we don't drag the
    runtime-checkable Protocol into Pydantic's validation."""

    name: str | None = None
    """The agent's author identity in the notebook. Notes get
    attributed to this name (``[researcher]`` rather than the
    generic ``[agent]``). ``None`` falls back to the framework
    default."""

    teammates: list[str] | None = None
    """Names of other agents in the same team. Surfaced in the
    workspace prompt section so the model knows who else is
    contributing. ``None`` omits the teammates line."""
