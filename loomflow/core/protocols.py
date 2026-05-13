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
    EpisodeMatch,
    Event,
    Fact,
    MemoryBlock,
    MemoryExport,
    MemoryProfile,
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
    """LLM provider interface. One adapter per lab (Anthropic, OpenAI, ...).

    The required surface is ``stream(...)`` — every adapter must
    implement it. Adapters MAY additionally override ``complete(...)``
    with a non-streaming (single-shot) call; if not, ``complete``
    falls back to consuming the stream internally and assembling the
    full response, which is correct but slower (per-chunk wire +
    parsing overhead). Architectures use ``complete`` on the
    non-streaming hot path (``agent.run()``) and ``stream`` when a
    consumer is reading from ``agent.stream()``.
    """

    name: str

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        output_schema: Any | None = None,
        effort: str | None = None,
        strict_effort: bool = False,
        prompt_caching: Any = None,
    ) -> AsyncIterator[ModelChunk]:
        """Stream completion chunks. Each chunk is text, tool_call, or finish.

        ``output_schema`` is an optional Pydantic ``BaseModel`` subclass
        (typed loosely as ``Any`` to keep the protocol pydantic-free).
        Adapters that support a provider-native structured-output API
        (OpenAI's ``response_format=json_schema``, Anthropic's forced
        tool-call pattern, LiteLLM's passthrough) translate the schema
        into the provider's local idiom so the model is constrained to
        produce valid JSON matching it. Adapters without native support
        ignore the kwarg silently — the agent loop's prompt-augmentation
        path still produces JSON, just with retry-on-validation-fail
        instead of hard guarantees.

        ``effort`` is an optional reasoning-effort dial that maps to
        whatever each provider supports (see :class:`~loomflow.Effort`).
        Adapters whose model doesn't support reasoning effort drop the
        kwarg with a one-time per-(model, effort) warning. Pass
        ``strict_effort=True`` on the agent to make the drop a hard
        error instead — useful for catching typos / capability mismatches
        loudly during development.
        """
        ...


@runtime_checkable
class Memory(Protocol):
    """Tiered memory: working blocks, episodic store, semantic graph."""

    async def working(
        self, *, user_id: str | None = None
    ) -> list[MemoryBlock]:
        """All in-context blocks for ``user_id``. Pinned to every prompt.

        Like every other memory primitive, working blocks are
        user-partitioned: blocks set under one ``user_id`` are
        invisible to a query scoped to a different one. Backends
        MUST honour this — passing alice's user_id never returns
        bob's pinned blocks.
        """
        ...

    async def update_block(
        self, name: str, content: str, *, user_id: str | None = None
    ) -> None:
        """Replace the contents of a named block in ``user_id``'s
        namespace. ``None`` is the anonymous bucket."""
        ...

    async def append_block(
        self, name: str, content: str, *, user_id: str | None = None
    ) -> None:
        """Append to a named block in ``user_id``'s namespace,
        creating it if absent."""
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
        user_id: str | None = None,
    ) -> list[Episode]:
        """Retrieve episodes (or facts, when ``kind='semantic'``).

        When ``user_id`` is supplied, results are restricted to
        episodes stored with that exact ``user_id`` value. ``None``
        is its own bucket (the "anonymous / single-tenant"
        namespace) — episodes stored with ``user_id=None`` are never
        visible to a query with ``user_id="alice"`` and vice versa.
        Backends MUST honour this filter to preserve the framework's
        multi-tenant safety contract.
        """
        ...

    async def recall_scored(
        self,
        query: str,
        *,
        kind: str = "episodic",
        limit: int = 5,
        time_range: tuple[datetime, datetime] | None = None,
        user_id: str | None = None,
        alpha: float = 0.5,
    ) -> list[EpisodeMatch]:
        """Hybrid recall returning episodes paired with retrieval
        scores.

        Same filtering semantics as :meth:`recall` (``user_id``
        partition, ``time_range``, ``kind``), but instead of bare
        :class:`Episode` rows the result is a list of
        :class:`EpisodeMatch` carrying the score breakdown — vector,
        BM25, optional reranker — used to rank each match.

        ``alpha`` weights the lexical-vs-vector mix in backends
        that compute both: ``0.0`` is pure BM25, ``1.0`` is pure
        vector cosine, ``0.5`` (default) is balanced via Reciprocal
        Rank Fusion. Backends without one of the rankings ignore
        ``alpha`` for that direction.

        **Backends without a native hybrid implementation MAY
        delegate to** :meth:`recall` and wrap each :class:`Episode`
        with a neutral ``score=1.0`` — see
        :func:`loomflow.memory.default_recall_scored` for the
        helper. This keeps the protocol coherent while making
        native hybrid implementations strictly opt-in for richer
        ranking.

        Use this method when you care about *which* episodes won
        and *why* — for downstream rerankers, MMR diversification,
        score-threshold filters, or A/B retrieval-quality tests.
        Use :meth:`recall` when you only need the rows.
        """
        ...

    async def recall_facts(
        self,
        query: str,
        *,
        limit: int = 5,
        valid_at: datetime | None = None,
        user_id: str | None = None,
    ) -> list[Fact]:
        """Retrieve bi-temporal facts matching ``query``.

        Backends that don't expose a fact store return ``[]``. The agent
        loop calls this directly rather than duck-typing on
        ``memory.facts`` so backends without fact support don't need
        any opt-out mechanism.

        ``user_id`` filters by namespace partition with the same
        semantics as :meth:`recall`: ``None`` is its own bucket and
        does not cross-contaminate with non-None values.
        """
        ...

    async def consolidate(self) -> None:
        """Background: extract semantic facts from recent episodes."""
        ...

    async def session_messages(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
        limit: int = 20,
    ) -> list[Message]:
        """Return the most-recent ``limit`` user/assistant turns from
        the conversation identified by ``session_id``, in order
        (oldest first).

        This is the conversation-continuity primitive — the agent
        loop calls it at the top of every run so that reusing a
        ``session_id`` actually continues the chat (the model sees
        previous turns as real :class:`Message` history) rather than
        starting fresh and relying solely on semantic recall.

        ``user_id`` MUST be respected by backends as a hard
        namespace partition: messages persisted under one
        ``user_id`` are never visible to a query scoped to a
        different one. Backends without persisted message logs
        return ``[]`` — the agent loop falls back to the
        semantic-recall path in that case.
        """
        ...

    async def profile(
        self, *, user_id: str | None = None
    ) -> MemoryProfile:
        """Summary of what this memory knows about ``user_id``.

        Returns counts (episodes, facts), the most-recent sessions
        the user touched, the last-seen timestamp, and a sample of
        the most-recently-recorded facts. Suitable for rendering a
        "what does the bot know about me?" view to the end user, or
        for an ops dashboard.

        Backends MUST honour ``user_id`` as a hard partition —
        passing one user's id never returns counts derived from
        another user's data.
        """
        ...

    async def forget(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        before: datetime | None = None,
    ) -> int:
        """Erase memory for a user — GDPR / "right to be forgotten".

        With ``user_id`` only: erase EVERYTHING (episodes + facts +
        session messages + working blocks) belonging to that user.
        With ``session_id``: erase only that conversation thread for
        that user. With ``before``: erase only data older than the
        timestamp (other args still scope it). All filters AND
        together.

        Returns the total number of records deleted (episodes +
        facts; backends without precise counts may return their
        best estimate).

        ``user_id=None`` erases the anonymous bucket only — same
        partition rule as :meth:`recall`. To erase everything
        across all users, callers must enumerate users and call
        ``forget`` per-user; the framework deliberately makes the
        "delete every user" path explicit so it's not done by
        accident.
        """
        ...

    async def export(
        self, *, user_id: str | None = None
    ) -> MemoryExport:
        """Full data dump for ``user_id`` — GDPR / data portability.

        Returns every episode, every fact, and the export
        timestamp. Serialise with ``MemoryExport.model_dump_json()``
        for download or downstream processing. Honours the
        ``user_id`` partition: never includes data belonging to a
        different user.
        """
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
    """Decides whether a tool call is allowed.

    ``user_id`` (M9): the agent loop forwards the live
    :class:`~loomflow.RunContext`'s user_id so multi-tenant
    permission impls (e.g. :class:`~loomflow.PerUserPermissions`)
    can route to per-user policies. Implementations that don't
    care about the user can ignore the kwarg; the framework's
    fallback ``except TypeError`` covers legacy impls.
    """

    async def check(
        self,
        call: ToolCall,
        *,
        context: Mapping[str, Any],
        user_id: str | None = None,
    ) -> PermissionDecision:
        ...


class HookHost(Protocol):
    """Aggregator over user-registered lifecycle callbacks.

    ``user_id`` (M9): same contract as :class:`Permissions` —
    forwarded from the live RunContext so per-user hooks can
    route. Legacy hook impls without the kwarg fall back via
    ``except TypeError`` in the agent loop.
    """

    async def pre_tool(
        self, call: ToolCall, *, user_id: str | None = None
    ) -> PermissionDecision:
        ...

    async def post_tool(self, call: ToolCall, result: ToolResult) -> None:
        ...

    async def on_event(self, event: Event) -> None:
        ...


class Budget(Protocol):
    """Resource governance — tokens, calls, cost, wall clock.

    ``user_id`` (M9): the agent loop forwards the live
    :class:`~loomflow.RunContext`'s user_id into every
    ``allows_step`` and ``consume`` call so multi-tenant budget
    impls can enforce per-user caps. Implementations that don't
    track per-user usage may ignore the kwarg; the framework
    falls back gracefully when the kwarg isn't accepted.
    """

    async def allows_step(
        self, *, user_id: str | None = None
    ) -> BudgetStatus:
        ...

    async def consume(
        self,
        *,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        user_id: str | None = None,
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
    """Resolution and redaction of named secrets.

    ``resolve`` / ``store`` are async because most production secrets
    backends (Vault, AWS Secrets Manager, GCP Secret Manager) talk
    over the network. ``lookup_sync`` exists for the
    *constructor-time* path: when the framework needs to wire an
    API key into a model adapter before any event loop is running
    (e.g. ``OpenAIModel(...)`` from inside ``Agent.__init__``).
    Concrete impls returning ``None`` from ``lookup_sync`` for refs
    that can't be resolved synchronously are fine — callers should
    fall back to ``os.environ`` or to the explicit ``api_key=``
    argument.
    """

    async def resolve(self, ref: str) -> str:
        ...

    async def store(self, ref: str, value: str) -> None:
        ...

    def redact(self, text: str) -> str:
        ...

    def lookup_sync(self, ref: str) -> str | None:
        """Synchronous best-effort lookup, for constructor-time
        callers that can't await. Returns ``None`` when the ref
        isn't available synchronously (e.g. the impl needs a
        network round-trip). Default impls in :mod:`loomflow.
        security.secrets` cover env-var and in-memory lookups."""
        ...
