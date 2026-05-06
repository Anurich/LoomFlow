"""Pydantic types for messages, events, tools, memory, and runtime state.

These are the value objects that flow across module boundaries. They are
immutable where possible, validated on construction, and free of behavior
that requires I/O.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .ids import new_id


def _utcnow() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Messages and model I/O
# ---------------------------------------------------------------------------


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolDef(BaseModel):
    """Schema description of a tool the model can call.

    Mirrors the JSON-Schema-flavored shape used across MCP and provider APIs.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    server: str | None = None  # MCP server name, if applicable


class ToolCall(BaseModel):
    """A model-emitted request to invoke a tool."""

    id: str = Field(default_factory=lambda: new_id("tcall"))
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    tool_def: ToolDef | None = None
    destructive: bool = False

    def is_destructive(self) -> bool:
        return self.destructive

    def idempotency_key(self) -> str:
        from .ids import deterministic_hash

        return deterministic_hash(self.tool, self.args)


class Message(BaseModel):
    """A single chat message in the model's conversation.

    ``tool_calls`` is populated on assistant messages that emitted tool
    calls in the previous turn — real provider adapters (Anthropic
    ``tool_use`` blocks, OpenAI ``tool_calls`` array) need to reconstruct
    the right wire format from this.
    """

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


class ToolResult(BaseModel):
    """Outcome of a tool invocation."""

    call_id: str
    ok: bool
    output: Any = None
    error: str | None = None
    denied: bool = False
    reason: str | None = None
    duration_ms: float | None = None
    started_at: datetime = Field(default_factory=_utcnow)

    @classmethod
    def success(cls, call_id: str, output: Any, **kwargs: Any) -> "ToolResult":
        return cls(call_id=call_id, ok=True, output=output, **kwargs)

    @classmethod
    def error_(cls, call_id: str, message: str, **kwargs: Any) -> "ToolResult":
        return cls(call_id=call_id, ok=False, error=message, **kwargs)

    @classmethod
    def denied_(cls, call_id: str, reason: str, **kwargs: Any) -> "ToolResult":
        return cls(call_id=call_id, ok=False, denied=True, reason=reason, **kwargs)


class Usage(BaseModel):
    """Token and cost accounting for a model call."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class ModelChunk(BaseModel):
    """A single chunk from a streaming model call.

    Discriminated by ``kind``. Exactly one of the optional fields is set
    depending on the kind.
    """

    kind: Literal["text", "tool_call", "finish"]
    text: str | None = None
    tool_call: ToolCall | None = None
    finish_reason: str | None = None
    usage: Usage | None = None


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class MemoryBlock(BaseModel):
    """An in-context memory block, pinned to every prompt."""

    name: str
    content: str
    updated_at: datetime = Field(default_factory=_utcnow)
    pinned_order: int = 0

    def format(self) -> str:
        return f"<{self.name}>\n{self.content}\n</{self.name}>"


class Episode(BaseModel):
    """A single (input, decisions, tool calls, output) tuple from history."""

    id: str = Field(default_factory=lambda: new_id("ep"))
    session_id: str
    occurred_at: datetime = Field(default_factory=_utcnow)
    input: str
    output: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    embedding: list[float] | None = None

    def format(self) -> str:
        return f"[{self.occurred_at.isoformat()}] {self.input!r} -> {self.output!r}"


class Fact(BaseModel):
    """A semantic claim extracted from one or more episodes.

    Bi-temporal: ``valid_from``/``valid_until`` tracks when the fact was
    true in the world; ``recorded_at`` tracks when we learned it.
    """

    id: str = Field(default_factory=lambda: new_id("fact"))
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    valid_from: datetime = Field(default_factory=_utcnow)
    valid_until: datetime | None = None
    recorded_at: datetime = Field(default_factory=_utcnow)
    sources: list[str] = Field(default_factory=list)

    def format(self) -> str:
        suffix = ""
        if self.valid_until is not None:
            suffix = f" (until {self.valid_until.isoformat()})"
        return (
            f"{self.subject} {self.predicate} {self.object}"
            f" [confidence {self.confidence:.2f}]{suffix}"
        )


# ---------------------------------------------------------------------------
# Decisions and control signals
# ---------------------------------------------------------------------------


class PermissionDecision(BaseModel):
    """Outcome of a permission check or pre-tool hook."""

    model_config = ConfigDict(frozen=True)

    decision: Literal["allow", "deny", "ask"]
    reason: str | None = None

    @property
    def allow(self) -> bool:
        return self.decision == "allow"

    @property
    def deny(self) -> bool:
        return self.decision == "deny"

    @property
    def ask(self) -> bool:
        return self.decision == "ask"

    @classmethod
    def allow_(cls, reason: str | None = None) -> "PermissionDecision":
        return cls(decision="allow", reason=reason)

    @classmethod
    def deny_(cls, reason: str) -> "PermissionDecision":
        return cls(decision="deny", reason=reason)

    @classmethod
    def ask_(cls, reason: str | None = None) -> "PermissionDecision":
        return cls(decision="ask", reason=reason)


class BudgetStatus(BaseModel):
    """Result of a budget check before each step."""

    model_config = ConfigDict(frozen=True)

    state: Literal["ok", "warn", "blocked"]
    reason: str | None = None

    @property
    def blocked(self) -> bool:
        return self.state == "blocked"

    @property
    def warn(self) -> bool:
        return self.state == "warn"

    @classmethod
    def ok_(cls) -> "BudgetStatus":
        return cls(state="ok")

    @classmethod
    def warn_(cls, reason: str) -> "BudgetStatus":
        return cls(state="warn", reason=reason)

    @classmethod
    def blocked_(cls, reason: str) -> "BudgetStatus":
        return cls(state="blocked", reason=reason)


# ---------------------------------------------------------------------------
# Events (the streamed observation channel)
# ---------------------------------------------------------------------------


class EventKind(StrEnum):
    STARTED = "started"
    MODEL_CHUNK = "model_chunk"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    MEMORY_RECALL = "memory_recall"
    MEMORY_WRITE = "memory_write"
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXCEEDED = "budget_exceeded"
    PERMISSION_ASK = "permission_ask"
    PERMISSION_DECISION = "permission_decision"
    ERROR = "error"
    COMPLETED = "completed"


class Event(BaseModel):
    """A single observable record from a running session.

    Carries a discriminator (``kind``) plus a free-form payload. Construct
    via the class methods to ensure consistent shapes.
    """

    kind: EventKind
    session_id: str
    at: datetime = Field(default_factory=_utcnow)
    payload: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def started(cls, session_id: str, prompt: str) -> "Event":
        return cls(kind=EventKind.STARTED, session_id=session_id, payload={"prompt": prompt})

    @classmethod
    def model_chunk(cls, session_id: str, chunk: ModelChunk) -> "Event":
        return cls(
            kind=EventKind.MODEL_CHUNK,
            session_id=session_id,
            payload={"chunk": chunk.model_dump()},
        )

    @classmethod
    def tool_call(cls, session_id: str, call: ToolCall) -> "Event":
        return cls(
            kind=EventKind.TOOL_CALL,
            session_id=session_id,
            payload={"call": call.model_dump()},
        )

    @classmethod
    def tool_result(cls, session_id: str, result: ToolResult) -> "Event":
        return cls(
            kind=EventKind.TOOL_RESULT,
            session_id=session_id,
            payload={"result": result.model_dump()},
        )

    @classmethod
    def budget_warning(cls, session_id: str, status: BudgetStatus) -> "Event":
        return cls(
            kind=EventKind.BUDGET_WARNING,
            session_id=session_id,
            payload={"status": status.model_dump()},
        )

    @classmethod
    def budget_exceeded(cls, session_id: str, status: BudgetStatus) -> "Event":
        return cls(
            kind=EventKind.BUDGET_EXCEEDED,
            session_id=session_id,
            payload={"status": status.model_dump()},
        )

    @classmethod
    def error(cls, session_id: str, exc: BaseException) -> "Event":
        return cls(
            kind=EventKind.ERROR,
            session_id=session_id,
            payload={"type": type(exc).__name__, "message": str(exc)},
        )

    @classmethod
    def completed(cls, session_id: str, result: Any) -> "Event":
        return cls(
            kind=EventKind.COMPLETED,
            session_id=session_id,
            payload={"result": result},
        )


# ---------------------------------------------------------------------------
# Run results, audit, certified values
# ---------------------------------------------------------------------------


class RunResult(BaseModel):
    """Final outcome of an ``Agent.run`` call."""

    session_id: str
    output: str
    turns: int
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    started_at: datetime
    finished_at: datetime
    interrupted: bool = False
    interruption_reason: str | None = None


class CertifiedValue(BaseModel):
    """A value carrying provenance metadata for freshness/lineage checks."""

    model_config = ConfigDict(frozen=True)

    value: Any
    source: str
    fetched_at: datetime
    valid_until: datetime | None = None
    schema_version: str = "1"
    lineage: tuple[str, ...] = ()


class AuditEntry(BaseModel):
    """An immutable, signed entry in the audit log."""

    model_config = ConfigDict(frozen=True)

    seq: int
    timestamp: datetime
    session_id: str
    actor: str
    action: str
    payload: dict[str, Any]
    signature: str


class Span(BaseModel):
    """A trace span handle. Concrete telemetry adapters return their own
    representation; this is the value-object contract for in-process use."""

    name: str
    trace_id: str
    span_id: str
    started_at: datetime = Field(default_factory=_utcnow)
    attributes: dict[str, Any] = Field(default_factory=dict)


class ToolEvent(BaseModel):
    """Tool registry change notification (MCP listChanged etc.)."""

    kind: Literal["added", "removed", "updated"]
    tool: str
    server: str | None = None
    at: datetime = Field(default_factory=_utcnow)
