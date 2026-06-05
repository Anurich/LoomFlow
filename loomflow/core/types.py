"""Pydantic types for messages, events, tools, memory, and runtime state.

These are the value objects that flow across module boundaries. They are
immutable where possible, validated on construction, and free of behavior
that requires I/O.
"""

import re
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .ids import new_id


def _utcnow() -> datetime:
    return datetime.now(UTC)


_PREDICATE_CAMEL_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")
_PREDICATE_NON_WORD = re.compile(r"[\s\-]+")


def _normalize_predicate(value: str) -> str:
    """Canonicalise a predicate string for stable supersession.

    Maps free-form predicates emitted by LLM extractors ("Name_Is",
    "name-is", "nameIs", "name is") onto a single canonical form
    ("name_is") so that ``InMemoryFactStore`` / SQLite / Postgres
    supersession can match equivalent claims by string equality. We
    only collapse case + word-separator variants — semantically
    distinct predicates ("name_is" vs "is_named") still differ and
    require a custom alias map at a higher layer.
    """
    s = _PREDICATE_CAMEL_BOUNDARY.sub(r"\1_\2", value)
    s = _PREDICATE_NON_WORD.sub("_", s)
    return s.lower().strip("_")


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
    destructive: bool = False
    """Carries the originating :class:`Tool`'s destructive flag
    through ``Tool.to_def()`` so downstream consumers (model
    adapters, the ReAct permissions stamp, MCP listChanged
    notifications) can decide whether a tool needs an approval
    gate without needing to look up the Tool object separately.
    Default ``False`` for backward compat with adapters that
    constructed ToolDef pre-0.10.17 without setting it."""


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
    """Token and cost accounting for a model call.

    The cache fields follow Anthropic's **separate-buckets** semantics:
    ``input_tokens`` is the count of tokens billed at the FULL input
    rate (i.e. not served from cache); ``cached_input_tokens`` is the
    count served from the prompt cache (billed at the provider's
    discount — OpenAI 0.5x, Anthropic 0.1x). Total tokens the model
    processed is ``input_tokens + cached_input_tokens``.

    OpenAI's native API uses ``input_tokens`` = total, with
    ``cached_tokens`` as a subset; the OpenAI adapter normalises to
    the loomflow convention on the way in so downstream code sees one
    shape.
    """

    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    """Prompt tokens billed at the full input rate (i.e. cache miss
    or caching disabled). Does NOT include cached tokens — they're
    counted separately in ``cached_input_tokens``."""

    cached_input_tokens: int = 0
    """Prompt tokens served from the provider's prompt cache (cache
    hits). Billed at the cache-read rate — OpenAI 0.5x, Anthropic
    0.1x of the model's base input rate. Zero when caching is
    disabled or the prompt was a cache miss."""

    cache_write_tokens: int = 0
    """Prompt tokens written to the cache on this call (Anthropic
    only — OpenAI's cache writes are free and not surfaced).
    Billed at the cache-write premium: 1.25x for 5-minute TTL,
    2x for 1-hour TTL."""

    output_tokens: int = 0
    """Completion tokens billed at the model's output rate."""

    cost_usd: float = 0.0
    """USD cost computed by :func:`loomflow.model._pricing.estimate_cost`
    from the four token buckets above + the model's pricing entry.
    Zero for unknown models."""

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of prompt tokens served from the provider cache.

        ``cached_input_tokens / (input_tokens + cache_write_tokens +
        cached_input_tokens)`` — the standard cache-efficacy ratio.
        Returns ``0.0`` when no prompt tokens were billed (so a
        zero-token call never divides by zero). A stable-prefix
        workload should trend toward 0.7+ once the cache warms.
        """
        denom = self.input_tokens + self.cache_write_tokens + self.cached_input_tokens
        if denom <= 0:
            return 0.0
        return self.cached_input_tokens / denom


class PromptCacheConfig(BaseModel):
    """Configuration for **prompt caching** across model providers.

    The :class:`~loomflow.Agent` constructor accepts three shapes for
    its ``prompt_caching=`` kwarg, all of which resolve to one of
    these:

    * ``False`` / ``None`` → ``PromptCacheConfig(enabled=False)``.
      Default. No cache markers injected; cached usage fields stay 0
      even if the provider serves cached tokens automatically
      (OpenAI). Best for first-time-correctness reviewers.
    * ``True`` → ``PromptCacheConfig(enabled=True)`` with 5-minute
      TTL. Anthropic adapters will inject ``cache_control`` on the
      system prompt + tool definitions; OpenAI adapters will parse
      ``cached_tokens`` from the response and apply the discount.
    * ``dict`` → explicit per-field config:
      ``{"enabled": True, "ttl": "1h", "cache_key": "session_42"}``.

    Provider mapping:

    * **OpenAI** — caching is automatic regardless of this flag, but
      ``cache_key`` (when set) forwards as ``prompt_cache_key`` for
      improved cache-hit routing. Read tokens are billed at 0.5x.
    * **Anthropic** — ``cache_control: {type: "ephemeral", ttl}``
      injected on the LAST system block + the LAST tool definition
      (2 of the 4 available breakpoints). Read tokens billed at
      0.1x; write tokens at 1.25x (5m) or 2x (1h).
    * **Gemini** — not supported in this release (Gemini requires a
      separate ``CachedContent.create`` API call; planned for a
      future loomflow version).
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    """Master switch. When ``False``, no cache_control markers are
    injected and cached usage fields aren't populated."""

    ttl: Literal["5m", "1h"] = "5m"
    """Cache time-to-live. ``"5m"`` (default, cheapest write) or
    ``"1h"`` (Anthropic only; 2x write premium but worth it for
    long-running sessions that re-hit the same prefix)."""

    cache_key: str | None = None
    """Optional cache-routing hint. OpenAI's ``prompt_cache_key``
    parameter; helps requests with the same prefix hit the same
    backend cache. Map to ``user_id`` or ``session_id`` for
    per-user / per-session routing. Ignored by Anthropic."""


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
    """A single (input, decisions, tool calls, output) tuple from history.

    ``user_id`` is the framework-managed namespace partition. Episodes
    persisted with one ``user_id`` value are never visible to memory
    recall queries scoped to a different ``user_id``. ``None`` is its
    own bucket — the "anonymous / single-tenant" namespace — and does
    not see episodes belonging to a non-None ``user_id`` (and vice
    versa). Set automatically from :class:`~loomflow.RunContext`
    by the agent loop; pass explicitly when constructing episodes
    outside a run.
    """

    id: str = Field(default_factory=lambda: new_id("ep"))
    session_id: str
    user_id: str | None = None
    occurred_at: datetime = Field(default_factory=_utcnow)
    input: str
    output: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    embedding: list[float] | None = None
    tool_transcript: list[Message] | None = None
    """Optional intermediate-message log: every tool_call and
    tool_result the agent emitted between ``input`` (USER) and
    ``output`` (ASSISTANT) during this episode. ``None`` means
    "transcript not captured for this episode" — the legacy /
    default behavior; the field stays out of conversation
    rehydration. A non-None list means
    ``Agent(persist_tool_transcripts=True)`` opted in, and the
    architecture's ``session_messages()`` rehydration will splice
    these messages between USER and ASSISTANT so the worker sees
    its prior tool work (file contents read, command output, etc.)
    on subsequent delegations — the fix for the
    "persistent_subagents preserves shape but not substance"
    structural gap. Storage: backends with a relational schema
    (sqlite, postgres) persist transcripts in a sidecar table keyed
    by ``episode_id``; non-schema backends (inmemory, vector stores)
    keep them inline on the Episode object. Per-entry size is
    capped (default 50KB) at construction in
    ``loomflow.agent.api`` to keep storage bounded."""

    def format(self) -> str:
        return f"[{self.occurred_at.isoformat()}] {self.input!r} -> {self.output!r}"


class EpisodeMatch(BaseModel):
    """A recalled :class:`Episode` paired with its retrieval scores.

    Returned by :meth:`Memory.recall_scored`. Carries enough metadata
    for downstream code (rerankers, MMR diversification, score-based
    filtering, A/B retrieval-quality experiments) to reason about
    *why* this episode was selected without re-running recall.

    The ``score`` field is the **fused final score** the backend used
    to rank this match — backends are free to define what "1.0 means
    best" looks like for their algorithm. The component fields
    (``vector_score``, ``bm25_score``, ``rerank_score``) are
    optional breakdowns; ``None`` means "this component wasn't
    computed or doesn't apply for this backend".

    Adding new score components is a backward-compatible field
    addition (Pydantic ignores unknown fields by default and adds
    ``None`` defaults for new ones), so the protocol can grow
    without breaking existing backends.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    episode: Episode
    score: float
    """Final fused score the backend used for ranking. Higher is
    better. Range and meaning are backend-defined."""

    vector_score: float | None = None
    """Cosine-similarity component, in ``[-1, 1]``. ``None`` when
    the backend didn't compute embeddings for this query (e.g.
    no embedder configured, or pure-lexical recall)."""

    bm25_score: float | None = None
    """BM25 lexical-match component. ``None`` when the backend
    didn't compute a BM25 ranking (e.g. pure-vector backends)."""

    rerank_score: float | None = None
    """Optional cross-encoder / LLM reranker score, computed AFTER
    the initial fused ranking. ``None`` when no reranker was
    configured."""


class Fact(BaseModel):
    """A semantic claim extracted from one or more episodes.

    Bi-temporal: ``valid_from``/``valid_until`` tracks when the fact was
    true in the world; ``recorded_at`` tracks when we learned it.

    ``user_id`` is the framework-managed namespace partition. Facts
    persisted with one ``user_id`` value are never visible to recall
    queries scoped to a different ``user_id``. Set automatically from
    :class:`~loomflow.RunContext` by the agent loop / consolidator;
    pass explicitly when constructing facts outside a run.
    """

    id: str = Field(default_factory=lambda: new_id("fact"))
    user_id: str | None = None
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    valid_from: datetime = Field(default_factory=_utcnow)
    valid_until: datetime | None = None
    recorded_at: datetime = Field(default_factory=_utcnow)
    sources: list[str] = Field(default_factory=list)

    @field_validator("predicate")
    @classmethod
    def _canonicalise_predicate(cls, v: str) -> str:
        """Normalise predicates so supersession sees equivalent claims.

        "Name_Is", "name-is", and "nameIs" all canonicalise to
        "name_is". See :func:`_normalize_predicate`.
        """
        return _normalize_predicate(v)

    def format(self) -> str:
        suffix = ""
        if self.valid_until is not None:
            suffix = f" (until {self.valid_until.isoformat()})"
        return (
            f"{self.subject} {self.predicate} {self.object}"
            f" [confidence {self.confidence:.2f}]{suffix}"
        )


# ---------------------------------------------------------------------------
# Memory inspection / GDPR
# ---------------------------------------------------------------------------


class MemoryProfile(BaseModel):
    """Summary of what a :class:`Memory` knows about a single user.

    Returned by :meth:`Memory.profile`. Cheap aggregate counts +
    last-seen timestamp + the most-recent facts; suitable for
    rendering a "what does the bot know about me?" view to the
    end user, or a tenant dashboard for ops.

    Backends that don't track full episode counts (e.g. Redis without
    `FT.SEARCH` aggregations available) report what they can; the
    counts are best-effort, never wildly wrong.
    """

    user_id: str | None
    episode_count: int = 0
    fact_count: int = 0
    last_seen: datetime | None = None
    recent_sessions: list[str] = Field(default_factory=list)
    """Up to the 10 most-recent ``session_id``s touched, newest first."""
    sample_facts: list[Fact] = Field(default_factory=list)
    """Up to 10 of the most-recently-recorded facts about the user."""


class MemoryExport(BaseModel):
    """Full data dump for a single user — GDPR / data-portability use.

    Returned by :meth:`Memory.export`. Carries the complete record of
    everything the memory holds for ``user_id``: all episodes, all
    facts, working blocks the user touched. Serialise with
    ``.model_dump_json()`` for download or downstream processing.
    """

    user_id: str | None
    episodes: list[Episode] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    exported_at: datetime = Field(default_factory=_utcnow)


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
    ARCHITECTURE_EVENT = "architecture_event"
    # Workflow events — emitted by :class:`loomflow.Workflow.stream`.
    # Distinct from agent events so consumers can filter by pattern.
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_STEP_STARTED = "workflow_step_started"
    WORKFLOW_STEP_COMPLETED = "workflow_step_completed"
    WORKFLOW_STEP_FAILED = "workflow_step_failed"
    WORKFLOW_COMPLETED = "workflow_completed"
    """Generic architecture-progress event. Carries a namespaced
    ``name`` in the payload (e.g. ``"self_refine.critique"``,
    ``"reflexion.lesson_persisted"``, ``"router.classified"``) so
    each architecture can stream its own progress signal without
    expanding :class:`EventKind`."""


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

    @classmethod
    def architecture_event(
        cls,
        session_id: str,
        name: str,
        **data: Any,
    ) -> "Event":
        """Generic architecture-progress event.

        ``name`` is a namespaced string identifying the source
        architecture and the kind of progress
        (e.g. ``"self_refine.critique"``,
        ``"reflexion.lesson_persisted"``,
        ``"router.classified"``). ``data`` is merged into the
        payload alongside ``name`` so consumers can pattern-match
        on ``name`` and read structured fields off the rest.
        """
        return cls(
            kind=EventKind.ARCHITECTURE_EVENT,
            session_id=session_id,
            payload={"name": name, **data},
        )


# ---------------------------------------------------------------------------
# Run results, audit, certified values
# ---------------------------------------------------------------------------


class RunResult(BaseModel):
    """Final outcome of an ``Agent.run`` call.

    Three accessors for the model's output, picked by what fits
    the call site:

    * ``result.value`` — **recommended for most code**. Smart
      accessor: returns the parsed Pydantic instance when an
      ``output_schema`` was supplied AND validation succeeded;
      falls through to the raw text otherwise. One name, right
      type, no surprise on the schema vs no-schema split.
    * ``result.parsed`` — explicit "give me the typed object or
      ``None``". Only populated when ``output_schema=`` was set
      and the model produced something that validated.
    * ``result.output`` — always a string. The raw text the model
      emitted (the JSON itself when a schema was supplied). Use
      for logging, audit, debugging — when you want to see what
      the model actually said, not the parsed view.

    Examples::

        # free-form text run — value === output (string)
        result = await agent.run("summarise this PDF")
        print(result.value)         # the summary text

        # structured-output run — value IS the typed instance
        result = await agent.run(prompt, output_schema=Invoice)
        invoice: Invoice = result.value     # typed, validated
        raw_json: str = result.output       # the JSON the model emitted
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    output: str
    parsed: Any | None = None
    """The validated Pydantic instance when ``output_schema=`` was
    supplied to :meth:`Agent.run`; ``None`` otherwise. Typed as
    ``Any`` to keep the runtime type free; the call site has the
    schema and can cast or annotate as needed."""
    turns: int
    tokens_in: int = 0
    """Prompt tokens billed at the full input rate (cache misses)."""
    cached_tokens_in: int = 0
    """Prompt tokens served from the provider's prompt cache. Zero
    when caching is disabled or the model doesn't support it. See
    :class:`Usage.cached_input_tokens` for the per-call equivalent."""
    cache_write_tokens: int = 0
    """Prompt tokens written to cache on this run (Anthropic only)."""
    tokens_out: int = 0
    cost_usd: float = 0.0
    started_at: datetime
    finished_at: datetime
    interrupted: bool = False
    interruption_reason: str | None = None

    cited_slugs: list[str] = Field(default_factory=list)
    """Workspace note slugs the agent READ during this run (via
    ``read_note`` / ``read_version``). Captured from the per-run
    citation contextvar just before it's reset at run-end, so the
    caller can drive the self-improvement loop AFTER the run::

        result = await agent.run(prompt, user_id="u")
        await workspace.attribute_outcome(
            success=True, slugs=result.cited_slugs, user_id="u",
        )

    Empty when no workspace is wired, when nothing was read, or
    when ``living_plan`` / workspace tooling is disabled."""

    @property
    def value(self) -> Any:
        """Smart accessor: ``parsed`` when set, else ``output``.

        For schema-typed runs this is the typed Pydantic instance.
        For free-form text runs it's the same string as
        ``result.output``. The recommended way to read the result
        in 90% of code — you don't have to branch on whether a
        schema was passed.
        """
        return self.parsed if self.parsed is not None else self.output

    @property
    def total_tokens(self) -> int:
        """Convenience: ``tokens_in + tokens_out``."""
        return self.tokens_in + self.tokens_out

    @property
    def cache_hit_rate(self) -> float:
        """Run-level prompt-cache hit rate over all turns.

        ``cached_tokens_in / (tokens_in + cache_write_tokens +
        cached_tokens_in)``. Returns ``0.0`` when no prompt tokens
        were billed. The run-level analogue of
        :attr:`Usage.cache_hit_rate`; a stable-prefix workload with
        caching on should trend toward 0.7+ once warm."""
        denom = self.tokens_in + self.cache_write_tokens + self.cached_tokens_in
        if denom <= 0:
            return 0.0
        return self.cached_tokens_in / denom

    @property
    def duration(self) -> timedelta:
        """Wall-clock latency between ``started_at`` and ``finished_at``."""
        return self.finished_at - self.started_at


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
    """An immutable, signed entry in the audit log.

    ``user_id`` (M9) is a top-level field for multi-tenant audit
    queries — `query(user_id="alice")` returns Alice's entries
    cleanly, no JSON-payload digging. Optional for back-compat
    with single-tenant deployments; populated automatically by the
    agent loop from the live :class:`~loomflow.RunContext`.
    """

    model_config = ConfigDict(frozen=True)

    seq: int
    timestamp: datetime
    session_id: str
    user_id: str | None = None
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


# Reasoning-effort dial for thinking-capable models. Maps to
# different per-provider shapes inside each Model adapter:
#
# * OpenAI o1/o3/o4/GPT-5  → ``reasoning_effort`` string
# * Anthropic Claude 4.6+  → ``output_config.effort`` (adaptive)
# * Anthropic Claude 3.7-4.5 → ``thinking.budget_tokens`` int
# * LiteLLM passthrough    → ``reasoning_effort`` (normalised)
#
# Models that don't support reasoning effort silently drop the
# kwarg with a one-time warning (see ``Model`` protocol docstring).
# ``strict_effort=True`` on the Agent makes the drop a hard error
# instead, for callers who want typo / capability checks loud.
#
# ``minimal`` was added with OpenAI GPT-5 and represents the
# "skip nearly all reasoning" path. ``xhigh`` is currently
# Anthropic Opus 4.7 only. ``max`` is Anthropic Opus 4.7 and
# DeepSeek V4-pro.
Effort = Literal["minimal", "low", "medium", "high", "xhigh", "max"]
