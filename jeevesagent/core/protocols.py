"""Protocol definitions for every module boundary.

These structural types are the contract surface of the harness. Every
implementation — first-party or third-party — satisfies one of these. The
loop and the agent only depend on the protocols, never on concrete
implementations.

The protocols are intentionally async-only: every method that performs
I/O is a coroutine, every stream is an :class:`AsyncIterator`, every
resource is an :class:`AsyncContextManager`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from .types import (
    BudgetStatus,
    Episode,
    Event,
    MemoryBlock,
    Message,
    ModelChunk,
    PermissionDecision,
    Span,
    ToolCall,
    ToolDef,
    ToolEvent,
    ToolResult,
)


@runtime_checkable
class Model(Protocol):
    """LLM provider interface. One adapter per lab (Anthropic, OpenAI, ...)."""

    name: str

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ModelChunk]:
        """Stream completion chunks. Each chunk is text, tool_call, or finish."""
        ...


@runtime_checkable
class Memory(Protocol):
    """Tiered memory: working blocks, episodic store, semantic graph."""

    async def working(self) -> list[MemoryBlock]:
        """All in-context blocks. Pinned to every prompt."""
        ...

    async def update_block(self, name: str, content: str) -> None:
        """Replace the contents of a named block."""
        ...

    async def append_block(self, name: str, content: str) -> None:
        """Append to a named block, creating it if absent."""
        ...

    async def remember(self, episode: Episode) -> str:
        """Persist an episode. Returns the episode ID."""
        ...

    async def recall(
        self,
        query: str,
        *,
        kind: str = "episodic",
        limit: int = 5,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list[Episode]:
        """Retrieve episodes (or facts, when ``kind='semantic'``)."""
        ...

    async def consolidate(self) -> None:
        """Background: extract semantic facts from recent episodes."""
        ...


class RuntimeSession(Protocol):
    """Handle to an open durable session held by a :class:`Runtime`."""

    id: str

    async def deliver(self, name: str, payload: Any) -> None:
        ...


@runtime_checkable
class Runtime(Protocol):
    """Durable execution. Wraps every side effect in a journal entry."""

    name: str

    async def step(
        self,
        name: str,
        fn: Callable[..., Awaitable[Any]],
        *args: Any,
        idempotency_key: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute ``fn`` as a journaled step. Replays cached on resume."""
        ...

    def stream_step(
        self,
        name: str,
        fn: Callable[..., AsyncIterator[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Execute a streaming step. Replays the aggregate on resume."""
        ...

    def session(
        self,
        session_id: str,
    ) -> AbstractAsyncContextManager[RuntimeSession]:
        """Open or resume a durable session."""
        ...

    async def signal(self, session_id: str, name: str, payload: Any) -> None:
        """Send an external signal (e.g., human approval) to a session."""
        ...


@runtime_checkable
class ToolHost(Protocol):
    """MCP-aware tool registry. Lazy-loads schemas on demand."""

    async def list_tools(self, *, query: str | None = None) -> list[ToolDef]:
        ...

    async def call(
        self,
        tool: str,
        args: Mapping[str, Any],
        *,
        call_id: str = "",
    ) -> ToolResult:
        """Invoke ``tool`` with ``args``. The ``call_id`` is propagated into
        the returned :class:`ToolResult` so the loop can correlate
        results with the originating model-emitted call.
        """
        ...

    def watch(self) -> AsyncIterator[ToolEvent]:
        """Notifications when the tool list changes (MCP listChanged)."""
        ...


class Sandbox(Protocol):
    """Isolation layer for tool execution."""

    async def execute(self, tool: ToolDef, args: Mapping[str, Any]) -> ToolResult:
        ...

    def with_filesystem(
        self, root: str
    ) -> AbstractAsyncContextManager[None]:
        """Temporary filesystem sandbox for the duration of the context."""
        ...


class Permissions(Protocol):
    """Decides whether a tool call is allowed."""

    async def check(
        self, call: ToolCall, *, context: Mapping[str, Any]
    ) -> PermissionDecision:
        ...


class HookHost(Protocol):
    """Aggregator over user-registered lifecycle callbacks."""

    async def pre_tool(self, call: ToolCall) -> PermissionDecision:
        ...

    async def post_tool(self, call: ToolCall, result: ToolResult) -> None:
        ...

    async def on_event(self, event: Event) -> None:
        ...


class Budget(Protocol):
    """Resource governance — tokens, calls, cost, wall clock."""

    async def allows_step(self) -> BudgetStatus:
        ...

    async def consume(
        self,
        *,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
    ) -> None:
        ...


class Telemetry(Protocol):
    """OpenTelemetry-compatible tracing/metrics surface."""

    def trace(
        self, name: str, **attrs: Any
    ) -> AbstractAsyncContextManager[Span]:
        ...

    async def emit_metric(self, name: str, value: float, **attrs: Any) -> None:
        ...


class Embedder(Protocol):
    """Text-to-vector embedding model used by the memory subsystem."""

    name: str
    dimensions: int

    async def embed(self, text: str) -> list[float]:
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...


class Secrets(Protocol):
    """Resolution and redaction of named secrets."""

    async def resolve(self, ref: str) -> str:
        ...

    async def store(self, ref: str, value: str) -> None:
        ...

    def redact(self, text: str) -> str:
        ...
