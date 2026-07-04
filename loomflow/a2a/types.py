"""Wire types for the A2A (Agent-to-Agent) protocol, v1.0-shaped (G10).

Self-contained pydantic models for the JSON-RPC 2.0 surface A2A uses —
implemented directly from the published spec rather than depending on
the reference ``a2a-sdk``, so ``loomflow[a2a]`` needs only ``httpx``
(and only for the *client*; the server side is dependency-free).

Field names are deliberately camelCase (``messageId``, ``contextId``,
``artifactId``, ...) so the models serialize to the exact wire shape
with a plain ``model_dump()`` — no alias machinery to forget at a
call site. All wire models use ``extra="allow"`` so richer payloads
from other SDKs (protocolVersion extensions, ``metadata`` bags,
transport hints) round-trip instead of failing validation.

v1 scope notes:

* **Parts** — only ``{kind: "text", text}`` parts are produced and
  consumed. ``DataPart`` (``kind: "data"``) and ``FilePart``
  (``kind: "file"``) parse without error (extra-tolerant model) but
  are ignored by text extraction; full support is future work.
* **TaskState** — the five states loomflow emits/accepts are
  ``submitted | working | completed | failed | input-required``.
  (The spec's ``canceled`` / ``rejected`` / ``auth-required`` states
  are not produced by this implementation.)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..core.ids import new_id

__all__ = [
    "INTERNAL_ERROR",
    "INVALID_PARAMS",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "PARSE_ERROR",
    "TASK_NOT_FOUND",
    "A2AError",
    "AgentCapabilities",
    "AgentCard",
    "AgentSkill",
    "Artifact",
    "Message",
    "Part",
    "Task",
    "TaskArtifactUpdateEvent",
    "TaskState",
    "TaskStatus",
    "TaskStatusUpdateEvent",
    "message_text",
    "text_artifact",
    "text_message",
]

# ---------------------------------------------------------------------------
# JSON-RPC 2.0 error codes (standard) + A2A-specific codes
# ---------------------------------------------------------------------------

PARSE_ERROR = -32700
"""Request body is not valid JSON."""
INVALID_REQUEST = -32600
"""Body is JSON but not a valid JSON-RPC request object."""
METHOD_NOT_FOUND = -32601
"""Unknown A2A method."""
INVALID_PARAMS = -32602
"""Params are malformed for the method (bad Message, no text parts, ...)."""
INTERNAL_ERROR = -32603
"""Server-side fault outside the agent run itself."""
TASK_NOT_FOUND = -32001
"""A2A-specific: ``tasks/get`` for an id the server does not know."""


class A2AError(Exception):
    """A JSON-RPC / protocol-level error from a remote A2A agent.

    Raised by :class:`~loomflow.a2a.A2AClient` when the remote returns
    a JSON-RPC error envelope, a non-200 HTTP status, or a task in the
    ``failed`` state. ``code`` carries the JSON-RPC error code when one
    was present (``None`` for transport-level failures).
    """

    def __init__(self, message: str, *, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Message / Part / Task shapes
# ---------------------------------------------------------------------------


class Part(BaseModel):
    """One content part of a message or artifact.

    v1: text parts only (``kind="text"``). Parts of other kinds parse
    (extra-tolerant) but contribute nothing to text extraction.
    """

    model_config = ConfigDict(extra="allow")

    kind: str = "text"
    text: str | None = None
    metadata: dict[str, Any] | None = None


class Message(BaseModel):
    """A single conversational turn between a user and an agent."""

    model_config = ConfigDict(extra="allow")

    role: Literal["user", "agent"]
    parts: list[Part] = Field(default_factory=list)
    messageId: str = Field(default_factory=lambda: new_id("msg"))
    contextId: str | None = None
    taskId: str | None = None
    kind: Literal["message"] = "message"
    metadata: dict[str, Any] | None = None


TaskState = Literal["submitted", "working", "completed", "failed", "input-required"]


class TaskStatus(BaseModel):
    """Current state of a task, optionally with an explanatory message."""

    model_config = ConfigDict(extra="allow")

    state: TaskState
    message: Message | None = None
    timestamp: str | None = None


class Artifact(BaseModel):
    """An output produced by the agent for a task."""

    model_config = ConfigDict(extra="allow")

    artifactId: str = Field(default_factory=lambda: new_id("artifact"))
    name: str | None = None
    parts: list[Part] = Field(default_factory=list)


class Task(BaseModel):
    """The unit of work A2A clients submit and poll."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(default_factory=lambda: new_id("task"))
    contextId: str
    status: TaskStatus
    artifacts: list[Artifact] = Field(default_factory=list)
    history: list[Message] | None = None
    kind: Literal["task"] = "task"
    metadata: dict[str, Any] | None = None


class TaskStatusUpdateEvent(BaseModel):
    """Streaming event: the task moved to a new state."""

    model_config = ConfigDict(extra="allow")

    taskId: str
    contextId: str
    status: TaskStatus
    final: bool = False
    kind: Literal["status-update"] = "status-update"


class TaskArtifactUpdateEvent(BaseModel):
    """Streaming event: the task produced (part of) an artifact."""

    model_config = ConfigDict(extra="allow")

    taskId: str
    contextId: str
    artifact: Artifact
    kind: Literal["artifact-update"] = "artifact-update"


# ---------------------------------------------------------------------------
# Agent card
# ---------------------------------------------------------------------------


class AgentCapabilities(BaseModel):
    """What optional protocol features this agent supports."""

    model_config = ConfigDict(extra="allow")

    streaming: bool = True
    pushNotifications: bool = False


class AgentSkill(BaseModel):
    """A discrete capability advertised on the agent card."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    description: str


class AgentCard(BaseModel):
    """The discovery document served at ``/.well-known/agent-card.json``."""

    model_config = ConfigDict(extra="allow")

    name: str
    description: str
    url: str = ""
    version: str = ""
    protocolVersion: str = "1.0"
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    skills: list[AgentSkill] = Field(default_factory=list)
    defaultInputModes: list[str] = Field(default_factory=lambda: ["text"])
    defaultOutputModes: list[str] = Field(default_factory=lambda: ["text"])


# ---------------------------------------------------------------------------
# Small factories / extractors used by both server and client
# ---------------------------------------------------------------------------


def message_text(message: Message) -> str:
    """Concatenate the text of a message's text parts (newline-joined)."""
    return "\n".join(p.text for p in message.parts if p.kind == "text" and isinstance(p.text, str))


def text_message(role: Literal["user", "agent"], text: str, **kwargs: Any) -> Message:
    """Build a single-text-part :class:`Message`."""
    return Message(role=role, parts=[Part(kind="text", text=text)], **kwargs)


def text_artifact(text: str, *, name: str | None = None) -> Artifact:
    """Build a single-text-part :class:`Artifact`."""
    return Artifact(name=name, parts=[Part(kind="text", text=text)])
