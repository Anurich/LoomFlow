"""Shared-notebook workspace — coordination layer for multi-agent
teams via auto-indexed markdown files.

Public surface:

* :class:`Workspace` — the protocol every backend implements.
* :class:`LocalDiskWorkspace` — production backend; file-per-note
  with auto-regenerated ``WORKSPACE.md`` index.
* :class:`InMemoryWorkspace` — zero-dep backend for tests +
  ephemeral coordination inside one process.
* :class:`Note` / :class:`NoteSummary` / :class:`NoteMatch` —
  the value types the protocol returns.
* :func:`resolve_workspace` — string/dict/instance dispatcher
  used by :class:`Agent`, :class:`Workflow`, and :class:`Team`.
* :func:`make_workspace_tools` — builds the five
  agent-facing :class:`Tool` instances (``note``, ``read_note``,
  ``list_notes``, ``search_notes``, ``update_note``) attributed
  to a given author.

The metaphor is **shared notebook**, not filesystem. Agents call
``note(title, content)`` and the workspace handles slugs, frontmatter,
indexing, and write atomicity. Filenames are invisible to agents.
"""

from .disk import LocalDiskWorkspace
from .inmemory import InMemoryWorkspace
from .protocol import Workspace
from .resolver import resolve_workspace
from .tools import make_workspace_tools, workspace_prompt_section
from .types import Note, NoteKind, NoteMatch, NoteSummary, WorkspaceMembership

__all__ = [
    "InMemoryWorkspace",
    "LocalDiskWorkspace",
    "Note",
    "NoteKind",
    "NoteMatch",
    "NoteSummary",
    "Workspace",
    "WorkspaceMembership",
    "make_workspace_tools",
    "resolve_workspace",
    "workspace_prompt_section",
]
