"""Tool registry and decorators.

Users typically construct tools via :func:`tool` (decorator) and pass
the resulting :class:`Tool` objects to :class:`Agent`. The agent wraps
them in an :class:`InProcessToolHost`.
"""

from .registry import InProcessToolHost, Tool, tool

__all__ = ["InProcessToolHost", "Tool", "tool"]
