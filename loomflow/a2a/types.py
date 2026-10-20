"""Wire types for the A2A (Agent-to-Agent) protocol (G10).

Self-contained pydantic models for the JSON-RPC 2.0 surface A2A uses —
implemented directly from the published spec rather than depending on
the reference ``a2a-sdk``, so ``loomflow[a2a]`` needs only ``httpx``
(and only for the *client*; the server side is dependency-free).

Two wire versions exist and loomflow speaks both:

* **v0.3** — ``kind``-tagged objects (``{"kind": "text", "text"}``),
  lowercase roles and states (``"user"``, ``"completed"``), slash
  method names (``message/send``).
* **v1.0** — the ProtoJSON rendering of ``a2a.proto``: no ``kind``
  tags (a part is ``{"text": ...}`` / ``{"raw" | "url": ...}`` /
  ``{"data": ...}``), enum names (``"ROLE_USER"``,
  ``"TASK_STATE_COMPLETED"``), PascalCase methods (``SendMessage``),
  and results wrapped as ``{"task": ...}`` / ``{"message": ...}``.

The pydantic models below are the v0.3 shape, loomflow's internal
representation; :func:`to_v1` / :func:`from_v1` convert any object to
and from the v1.0 wire form. Field names are camelCase in both, so
``model_dump()`` gives the exact wire shape. All wire models use
``extra="allow"`` so richer payloads from other SDKs round-trip.

Scope notes:

* **Parts** — loomflow produces and consumes text; file and data
  parts are carried through the codecs but ignored by text
  extraction.
* **TaskState** — every spec state parses; loomflow produces
  ``working``, ``completed``, ``failed`` and ``canceled``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..core.ids import new_id

__all__ = [
    "CONTENT_TYPE_NOT_SUPPORTED",
    "EXTENDED_CARD_NOT_CONFIGURED",
    "INTERNAL_ERROR",
    "INVALID_PARAMS",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "PARSE_ERROR",
    "PUSH_NOTIFICATION_NOT_SUPPORTED",
    "TASK_NOT_CANCELABLE",
    "TASK_NOT_FOUND",
    "UNSUPPORTED_OPERATION",
    "VERSION_NOT_SUPPORTED",
    "A2AError",
    "AgentCapabilities",
    "AgentCard",
    "AgentInterface",
    "AgentSkill",
    "Artifact",
    "Message",
    "Part",
    "Task",
    "TaskArtifactUpdateEvent",
    "TaskState",
    "TaskStatus",
    "TaskStatusUpdateEvent",
    "from_v1",
    "message_text",
    "text_artifact",
    "text_message",
    "to_v1",
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
TASK_NOT_CANCELABLE = -32002
"""A2A-specific: the task is already in a terminal state."""
PUSH_NOTIFICATION_NOT_SUPPORTED = -32003
"""A2A-specific: this server doesn't do push notifications."""
UNSUPPORTED_OPERATION = -32004
"""A2A-specific: a known operation this server doesn't support."""
CONTENT_TYPE_NOT_SUPPORTED = -32005
"""A2A-specific: no part of the message is a media type the agent takes."""
EXTENDED_CARD_NOT_CONFIGURED = -32007
"""A2A-specific: there is no authenticated extended agent card."""
VERSION_NOT_SUPPORTED = -32009
"""A2A-specific: the requested ``A2A-Version`` isn't served here."""


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


TaskState = Literal[
    "submitted",
    "working",
    "completed",
    "failed",
    "canceled",
    "input-required",
    "rejected",
    "auth-required",
]
TERMINAL_STATES: frozenset[str] = frozenset({"completed", "failed", "canceled", "rejected"})


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
    """A discrete capability advertised on the agent card.

    ``tags`` is required by the spec; left empty, the skill's ``id``
    is used.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        if not self.tags:
            self.tags = [self.id]


class AgentInterface(BaseModel):
    """One way to reach the agent: a URL, a binding and a protocol version."""

    model_config = ConfigDict(extra="allow")

    url: str
    protocolBinding: str = "JSONRPC"
    protocolVersion: str = "1.0"


class AgentCard(BaseModel):
    """The discovery document served at ``/.well-known/agent-card.json``.

    ``supportedInterfaces`` is the v1.0 field; ``url`` /
    ``protocolVersion`` / ``preferredTransport`` are the v0.3 ones,
    kept so older clients find the endpoint too.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    description: str
    supportedInterfaces: list[AgentInterface] = Field(default_factory=list)
    url: str = ""
    version: str = ""
    protocolVersion: str = "0.3"
    preferredTransport: str | None = None
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    skills: list[AgentSkill] = Field(default_factory=list)
    defaultInputModes: list[str] = Field(default_factory=lambda: ["text/plain"])
    defaultOutputModes: list[str] = Field(default_factory=lambda: ["text/plain"])

    def endpoint(self, version: str) -> str | None:
        """The JSON-RPC URL this card advertises for ``version``."""
        for iface in self.supportedInterfaces:
            if (
                iface.protocolBinding.upper() == "JSONRPC"
                and iface.protocolVersion.split(".")[:2] == version.split(".")[:2]
            ):
                return iface.url
        if version.startswith("0.3") and self.url:
            return self.url
        return None


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


# ---------------------------------------------------------------------------
# v1.0 wire codecs (ProtoJSON of a2a.proto) <-> the v0.3-shaped models
# ---------------------------------------------------------------------------

_V1_ROLE = {"user": "ROLE_USER", "agent": "ROLE_AGENT"}
_V03_ROLE = {v: k for k, v in _V1_ROLE.items()}
_V1_STATE = {
    state: "TASK_STATE_" + state.upper().replace("-", "_")
    for state in (
        "submitted", "working", "completed", "failed", "canceled",
        "input-required", "rejected", "auth-required",
    )
}
_V03_STATE = {v: k for k, v in _V1_STATE.items()}

def _part_to_v1(part: dict[str, Any]) -> dict[str, Any]:
    kind = part.get("kind", "text")
    out = {k: v for k, v in part.items() if k not in ("kind", "file", "text", "data")}
    if kind == "file":
        file = part.get("file") or {}
        if "bytes" in file:
            out["raw"] = file["bytes"]
        if "uri" in file:
            out["url"] = file["uri"]
        if file.get("mimeType"):
            out["mediaType"] = file["mimeType"]
        if file.get("name"):
            out["filename"] = file["name"]
    elif kind == "data":
        out["data"] = part.get("data")
    else:
        out["text"] = part.get("text", "")
    return out


def _part_from_v1(part: dict[str, Any]) -> dict[str, Any]:
    if "kind" in part:
        return dict(part)
    base = {k: v for k, v in part.items() if k in ("metadata",)}
    if "text" in part:
        return {**base, "kind": "text", "text": part["text"]}
    if "raw" in part or "url" in part:
        file: dict[str, Any] = {}
        if "raw" in part:
            file["bytes"] = part["raw"]
        if "url" in part:
            file["uri"] = part["url"]
        if part.get("mediaType"):
            file["mimeType"] = part["mediaType"]
        if part.get("filename"):
            file["name"] = part["filename"]
        return {**base, "kind": "file", "file": file}
    if "data" in part:
        return {**base, "kind": "data", "data": part["data"]}
    return {**base, "kind": "text", "text": ""}


def _convert(obj: dict[str, Any], *, v1: bool) -> dict[str, Any]:
    """Convert one Message / Task / Artifact / status / event dict."""
    out: dict[str, Any] = {}
    for key, value in obj.items():
        if v1 and key in ("kind", "final"):
            continue  # v1.0 has neither the tag nor ``final``
        if key == "role" and isinstance(value, str):
            out[key] = (_V1_ROLE if v1 else _V03_ROLE).get(value, value)
        elif key == "state" and isinstance(value, str):
            out[key] = (_V1_STATE if v1 else _V03_STATE).get(value, value)
        elif key == "parts" and isinstance(value, list):
            out[key] = [(_part_to_v1 if v1 else _part_from_v1)(p) for p in value]
        elif key in ("status", "message", "artifact") and isinstance(value, dict):
            out[key] = _convert(value, v1=v1)
        elif key in ("artifacts", "history") and isinstance(value, list):
            out[key] = [_convert(v, v1=v1) if isinstance(v, dict) else v for v in value]
        else:
            out[key] = value
    return out


def to_v1(obj: dict[str, Any]) -> dict[str, Any]:
    """A v0.3-shaped object (dict) in its v1.0 wire form."""
    return _convert(obj, v1=True)


def from_v1(obj: dict[str, Any], *, kind: str | None = None) -> dict[str, Any]:
    """A v1.0 wire object (or an already-v0.3 one) in v0.3 shape.

    ``kind`` (``"message"`` / ``"task"`` / ``"status-update"`` /
    ``"artifact-update"``) restores the v0.3 tag v1.0 doesn't carry.
    """
    out = _convert(obj, v1=False)
    if kind is not None:
        out.setdefault("kind", kind)
    return out
