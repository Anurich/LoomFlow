"""Exception hierarchy.

All harness-raised exceptions inherit from :class:`JeevesAgentError` so
callers can catch the family without binding to specific subtypes.
"""

from __future__ import annotations


class JeevesAgentError(Exception):
    """Base class for all harness errors."""


class ConfigError(JeevesAgentError):
    """Invalid or unresolvable configuration passed to ``Agent``."""


class BudgetExceeded(JeevesAgentError):
    """A run was halted because a budget limit was hit."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class PermissionDenied(JeevesAgentError):
    """A tool call was denied by the permission layer or a user hook."""

    def __init__(self, tool: str, reason: str) -> None:
        super().__init__(f"{tool}: {reason}")
        self.tool = tool
        self.reason = reason


class ToolError(JeevesAgentError):
    """A tool invocation failed at the tool's own boundary."""


class SandboxError(JeevesAgentError):
    """The sandbox refused or failed to execute a tool."""


class RuntimeJournalError(JeevesAgentError):
    """The durable runtime journal is unreadable or inconsistent."""


class MemoryStoreError(JeevesAgentError):
    """The memory backend failed an operation."""


class MCPError(JeevesAgentError):
    """An MCP transport, handshake, or protocol error."""


class FreshnessError(JeevesAgentError):
    """A certified value failed its freshness policy."""


class LineageError(JeevesAgentError):
    """A certified value failed its lineage policy."""


class CancelledByUser(JeevesAgentError):
    """A user-driven interruption (signal, timeout) ended the run."""
