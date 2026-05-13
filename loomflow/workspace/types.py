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

    model_config = ConfigDict(frozen=True)

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


class NoteSummary(BaseModel):
    """Cheap projection of :class:`Note` for index views.

    ``lede`` is the first non-empty line of the body, truncated to
    ~120 chars. Lets ``list_notes()`` render a useful overview
    without loading every body into the prompt.
    """

    model_config = ConfigDict(frozen=True)

    slug: str
    author: str
    title: str
    kind: NoteKind
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    lede: str
    """First ~120 chars of the body for quick scanning."""


class NoteMatch(BaseModel):
    """A search hit — :class:`NoteSummary` plus relevance + snippet."""

    model_config = ConfigDict(frozen=True)

    summary: NoteSummary
    score: float
    """Higher is better; not necessarily normalised — backends pick
    a comparable-within-result-set scoring scheme. ``1.0`` is a
    title hit; lower values are body matches."""

    snippet: str
    """A short excerpt around the matched phrase (or the lede when
    the match is title-only). ~140 chars."""


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
