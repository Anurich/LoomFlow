"""Tool registry and decorators + built-in filesystem / shell tools.

Users typically construct tools via :func:`tool` (decorator) and pass
the resulting :class:`Tool` objects to :class:`Agent`. The agent wraps
them in an :class:`InProcessToolHost`.

For the canonical "Claude-Code-shaped" tool set (read / write / edit
/ bash), import the four factory functions from
:mod:`jeevesagent.tools.builtin` (also re-exported at the top level).
"""

from .builtin import (
    PathEscapeError,
    bash_tool,
    default_workdir,
    edit_tool,
    filesystem_tools,
    read_tool,
    write_tool,
)
from .registry import InProcessToolHost, Tool, tool

__all__ = [
    "InProcessToolHost",
    "PathEscapeError",
    "Tool",
    "bash_tool",
    "default_workdir",
    "edit_tool",
    "filesystem_tools",
    "read_tool",
    "tool",
    "write_tool",
]
