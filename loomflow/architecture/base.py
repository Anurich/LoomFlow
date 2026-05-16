"""Architecture protocol + supporting types.

Three pieces:

* :class:`AgentSession` — mutable per-run state shared between
  :class:`Agent` and the :class:`Architecture`. The architecture
  reads ``messages`` and writes ``turns``, ``output``,
  ``cumulative_usage``, ``interrupted``, ``interruption_reason``,
  and ``metadata`` as iteration progresses. The :class:`Agent` reads
  the final state to build a :class:`RunResult`.

* :class:`Dependencies` — every protocol implementation an
  architecture might need (model, memory, runtime, tools, budget,
  permissions, hooks, telemetry, audit log, ``max_turns``), bundled
  into one struct so an architecture's ``run()`` signature stays
  short. Stable for the lifetime of a run.

* :class:`Architecture` — the protocol architectures implement. One
  method (``run``) plus a ``name`` and ``declared_workers`` for
  introspection.

Setup events (``Event.started``) and teardown events
(``Event.completed``) are emitted by :class:`Agent`, NOT the
architecture. Architectures yield the events that happen *during*
iteration: per-turn, per-tool, per-step, budget warnings, errors.

This keeps every architecture's ``run()`` focused on its own
strategy without re-implementing setup/teardown plumbing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..core.context import RunContext
from ..core.protocols import (
    Budget,
    Memory,
    Model,
    Permissions,
    Runtime,
    Telemetry,
    ToolHost,
)
from ..core.types import Event, Message, ToolCall, Usage
from ..security.audit import AuditLog
from ..security.hooks import HookRegistry

if TYPE_CHECKING:
    from ..agent.api import Agent

# An approval handler is the bridge between a permissions policy
# returning ``Decision.ask_(...)`` and an actual decision. The
# framework calls the handler with the pending tool call and the
# resolved ``user_id`` for the run; the handler returns ``True``
# to allow the call or ``False`` to deny. Handlers are async so
# they can await UI prompts, Slack approvals, ticketing systems,
# etc. without blocking the agent loop.
ApprovalHandler = Callable[[ToolCall, str | None], Awaitable[bool]]


@dataclass
class AgentSession:
    """Mutable per-run state shared between :class:`Agent` and an
    :class:`Architecture`.

    The :class:`Agent` constructs this once per run, the architecture
    mutates it as iteration progresses, and the :class:`Agent` reads
    the final state to build a :class:`RunResult`.

    ``metadata`` is a free-form dict architectures use for things
    that don't deserve their own field — multi-agent architectures
    stash worker handoff state, planners stash plans, etc.
    """

    id: str
    instructions: str
    messages: list[Message] = field(default_factory=list)
    turns: int = 0
    output: str = ""
    cumulative_usage: Usage = field(default_factory=Usage)
    interrupted: bool = False
    interruption_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Dependencies:
    """Bundled protocol implementations passed to every architecture.

    Constructed once per run from the :class:`Agent`'s configured
    backends. Architectures treat this as read-only — they call
    methods on the contained protocols but don't mutate the struct
    itself.

    Multi-agent architectures (Supervisor, Router, etc.) will grow
    helper methods on this class — ``fresh_session``,
    ``scope_for_worker``, ``with_extra_tools``, ``spawn_child`` — as
    they land in v0.5+. v0.3 keeps it as a passive struct.
    """

    model: Model
    memory: Memory
    runtime: Runtime
    tools: ToolHost
    budget: Budget
    permissions: Permissions
    hooks: HookRegistry
    telemetry: Telemetry
    audit_log: AuditLog | None
    max_turns: int
    approval_handler: ApprovalHandler | None = None
    """Resolves :class:`Decision.ask_` outcomes from the permissions
    layer. When unset, an ``ask`` decision is treated as a deny —
    historical behaviour preserved so single-tenant code without an
    approval flow still works. When set, the architecture calls
    this handler and uses the returned bool as the decision."""
    output_schema: Any | None = None
    """Pydantic ``BaseModel`` subclass requested via
    ``Agent.run(output_schema=...)``. Forwarded by the architecture
    to ``model.complete()`` / ``model.stream()`` so adapters with
    native structured-output support (OpenAI ``response_format``,
    Anthropic forced-tool-call, LiteLLM passthrough) can constrain
    the model to produce valid JSON. Adapters without native support
    ignore it silently and the prompt-augmentation path (system
    prompt carries the schema) still applies."""
    effort: str | None = None
    """Reasoning-effort dial — one of ``"minimal" | "low" | "medium"
    | "high" | "xhigh" | "max"``. Forwarded to ``model.complete()``
    / ``model.stream()`` where each adapter translates it into the
    provider's native shape (OpenAI ``reasoning_effort``, Anthropic
    ``thinking`` / ``output_config.effort``, LiteLLM passthrough).
    Models that don't support reasoning effort emit a one-time
    warning per ``(model, effort)`` pair and drop the kwarg — opt
    into hard-fail via ``strict_effort=True`` on the Agent."""
    strict_effort: bool = False
    """When True, ``effort`` against an unsupported model raises
    ``EffortNotSupportedError`` instead of warn-and-drop. Useful for
    pipelines where silently downgrading a reasoning request would
    be worse than failing fast."""
    prompt_caching: Any = None
    """Resolved :class:`~loomflow.core.types.PromptCacheConfig`
    (or ``None`` when caching is disabled). Forwarded to
    ``model.complete()`` / ``model.stream()`` so the Anthropic
    adapter can inject ``cache_control`` markers and the OpenAI
    adapter can pass an optional ``prompt_cache_key``. Typed as
    ``Any`` here to avoid pulling a value-type dependency into
    the architecture base module — the consuming adapters know
    the shape."""
    streaming: bool = False
    """Whether a downstream consumer is reading from
    ``agent.stream()``. When True, architectures should preserve
    real-time event-arrival semantics so a consumer that breaks
    out of the iterator triggers prompt cancellation. When False
    (the default for ``agent.run()``), architectures may batch
    events for fewer task-group / channel allocations on the
    hot path."""

    # ---------------------------------------------------------------
    # Fast-mode flags — auto-set by Agent._loop based on which
    # protocol implementations are no-op defaults vs production-
    # configured. Hot paths skip integration points when their
    # layer is no-op so users with a default agent get LangChain-
    # class latency. The moment a user wires up a real audit log /
    # telemetry exporter / permissions policy / etc., the
    # corresponding flag flips False and the integration point
    # becomes active.
    # ---------------------------------------------------------------
    fast_audit: bool = True
    """Skip ``_audit(...)`` calls when ``audit_log`` is ``None``."""
    fast_telemetry: bool = True
    """Skip ``telemetry.trace(...)`` contextmanagers + ``emit_metric``
    calls when ``telemetry`` is ``NoTelemetry``."""
    fast_permissions: bool = True
    """Skip per-tool ``permissions.check(...)`` when permissions is
    the no-op ``AllowAll``."""
    fast_hooks: bool = True
    """Skip ``hooks.pre_tool`` / ``hooks.post_tool`` dispatch when
    no hooks have been registered."""
    fast_runtime: bool = True
    """Inline ``await fn(*args)`` (skipping ``runtime.step(...)``
    wrapping + idempotency-key derivation) when runtime is
    ``InProcRuntime``."""
    fast_budget: bool = True
    """Skip ``budget.allows_step()`` and ``budget.consume(...)``
    when budget is ``NoBudget``."""
    fast_stop_hooks: bool = True
    """Skip the stop-hook re-invocation loop when no stop hooks
    are registered. Auto-set ``False`` when
    ``Agent(stop_hooks=[...])`` is non-empty OR a framework auto-
    registered hook fires (e.g. ``living_plan=True``). When True,
    ``Agent._loop`` runs ``architecture.run(...)`` exactly once
    and proceeds to teardown; when False, ``_loop`` wraps the
    architecture in the Ralph-loop bounded by
    ``Agent.max_stop_hook_iterations``."""

    # ---------------------------------------------------------------
    # Per-run context — populated from :class:`~loomflow.RunContext`
    # at the top of :meth:`Agent.run`. Architectures forward
    # ``context.user_id`` to :meth:`Memory.recall` so episodic /
    # factual recall is namespace-partitioned, and pass the whole
    # ``context`` to spawned sub-agents (with possibly-derived
    # ``session_id``) so multi-agent orchestration preserves
    # isolation. The same ``RunContext`` is also installed in a
    # :class:`contextvars.ContextVar` for the duration of the run
    # so tools and hooks can read it via ``get_run_context()``.
    # ---------------------------------------------------------------
    context: RunContext = field(default_factory=RunContext)
    """Typed scope for the run — ``user_id`` (memory namespace),
    ``session_id`` (conversation thread), ``run_id`` (this specific
    invocation), and ``metadata`` (free-form app context). See
    :class:`~loomflow.RunContext` for the per-field semantics."""


@runtime_checkable
class Architecture(Protocol):
    """Strategy interface for driving the agent loop.

    Implementations are async generators: they ``yield`` :class:`Event`
    values for every milestone they want surfaced (model chunks, tool
    calls, tool results, budget warnings, errors, architecture-specific
    progress events).

    See ``Subagent.md`` for the catalogue of architectures and the
    design rationale behind the protocol shape.
    """

    name: str

    def run(
        self,
        session: AgentSession,
        deps: Dependencies,
        prompt: str,
    ) -> AsyncIterator[Event]:
        """Drive iteration; yield events as they happen.

        The architecture mutates ``session`` (turns, output,
        cumulative_usage, messages, interrupted, interruption_reason,
        metadata) as it iterates and yields :class:`Event`\\ s for the
        caller to forward (or ignore, in non-streaming runs).

        Implementations are *async generators* — declared
        ``async def run(...) -> AsyncIterator[Event]:`` with ``yield``
        statements in the body.

        **Re-invocation contract.** ``Agent._loop`` MAY call
        ``run(session, deps, new_prompt)`` a second (or Nth) time
        on the same ``session`` when a registered
        :class:`~loomflow.StopHook` votes to continue. The new
        ``prompt`` should be treated as a fresh user turn
        appended to the running conversation; implementations
        MAY assume ``len(session.messages) > 0`` on re-entry and
        SHOULD append ``prompt`` as a ``Role.USER`` message to
        preserve the conversation. Built-in architectures (ReAct,
        Reflexion, Supervisor, …) all honour this contract; third-
        party architectures that ignore ``prompt`` on re-entry
        will silently drop the StopHook's directive — document
        the deviation prominently if your custom architecture
        differs.
        """
        ...

    def declared_workers(self) -> dict[str, Agent]:
        """Sub-Agents this architecture composes, keyed by role name.

        Used by multi-agent architectures (Supervisor, Actor-Critic,
        Debate, Router, Blackboard, Swarm) to expose their workers for
        introspection (logging, telemetry, eval). Single-agent
        architectures return ``{}``.
        """
        ...
