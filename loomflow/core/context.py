"""Per-run context propagation.

A single :class:`RunContext` is built at the top of every
:meth:`Agent.run` (or :meth:`Agent.stream`) call and stored in a
:class:`contextvars.ContextVar` for the duration of the run. Tools,
hooks, sub-agents, and architectures all read it through
:func:`get_run_context` rather than threading it through every
signature.

The framework treats ``user_id`` and ``session_id`` as first-class
typed primitives — not strings buried in a free-form ``configurable``
dict. ``user_id`` partitions memory recall; ``session_id`` identifies
the conversation thread for replay and continuity. Application-
specific keys go in the ``metadata`` mapping, where the framework
makes no claim to understand them.

The contextvar is automatically propagated by ``anyio``'s structured
concurrency primitives (``create_task_group``, ``start_soon``), so
parallel tool dispatch, sub-agent spawning, and streaming consumers
all see the same context without any explicit plumbing.

Tests that call ``@tool`` functions directly (no active agent run)
get a default empty :class:`RunContext` rather than an exception —
preserving direct-invocation ergonomics.
"""

from __future__ import annotations

import enum
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .protocols import Memory

__all__ = [
    "IsolationWarning",
    "RunContext",
    "get_run_context",
    "set_run_context",
]


class IsolationWarning(UserWarning):
    """Emitted when a memory query is likely to silently miss data
    because the caller forgot to pass ``user_id``.

    Concrete trigger: a backend's ``recall`` / ``recall_facts`` runs
    with ``user_id=None`` against a store whose persisted records
    include at least one non-None ``user_id`` — the partition is
    safe (the anonymous bucket and named-user buckets are isolated),
    but the developer probably wired up multi-tenancy somewhere and
    forgot to pass ``user_id`` here, so they will see suspiciously
    empty recall results.

    Subclass of :class:`UserWarning` so it goes through Python's
    standard ``warnings`` filter machinery — apps can silence,
    promote-to-error, or log it however they want, e.g.::

        import warnings
        from loomflow import IsolationWarning
        warnings.simplefilter("error", IsolationWarning)  # raise on hit
    """


class _Sentinel(enum.Enum):
    """Sentinel for "field not provided" — distinct from ``None``,
    which is a valid value for both ``user_id`` and ``session_id``."""

    UNSET = "UNSET"


@dataclass(frozen=True, slots=True)
class RunContext:
    """Typed, immutable context for one agent run.

    Set once at the start of :meth:`Agent.run` and propagated to
    every architecture, tool, hook, sub-agent, and memory operation
    via a :class:`contextvars.ContextVar`. The framework treats
    ``user_id`` and ``session_id`` as first-class fields (typed,
    namespaced); ``metadata`` is an opaque bag for app-specific keys
    the framework does not interpret.

    Construct one directly when you need to spawn work outside an
    active run with explicit scope:

    .. code-block:: python

        ctx = RunContext(user_id="alice", session_id="conv_42")
        async with set_run_context(ctx):
            await my_tool(...)

    Inside an agent run, prefer :func:`get_run_context` over
    constructing a new one — that gives you the live context the
    framework set up.
    """

    user_id: str | None = None
    """Namespace for memory recall + persistence. ``None`` is the
    "anonymous / single-tenant" bucket; episodes / facts stored
    with ``user_id=None`` never see episodes / facts stored with
    a non-None ``user_id`` and vice versa. The framework treats
    this as a hard partition key, not a soft filter."""

    session_id: str | None = None
    """Conversation thread identifier. Reusing the same ``session_id``
    across calls signals "continue this conversation" — the
    framework will rehydrate prior session messages so the model
    sees real chat history, not just memory recall. ``None`` means
    "fresh conversation"; the framework auto-generates one inside
    :meth:`Agent.run` if not supplied."""

    run_id: str = ""
    """Unique identifier for this single :meth:`Agent.run` invocation.
    Distinct from ``session_id`` (which identifies a conversation
    that may span many runs). Auto-set by :meth:`Agent.run`; an
    explicit value passed in by the caller is overridden."""

    metadata: Mapping[str, Any] = field(default_factory=dict)
    """Free-form application context. Use this for keys the framework
    does not need to understand — locale, request id, feature flags,
    tenant id beyond ``user_id``, etc. Read inside tools / hooks via
    ``get_run_context().metadata``."""

    # --- Convenience -------------------------------------------------

    def with_overrides(
        self,
        *,
        user_id: str | None | _Sentinel = _Sentinel.UNSET,
        session_id: str | None | _Sentinel = _Sentinel.UNSET,
        run_id: str | _Sentinel = _Sentinel.UNSET,
        metadata: Mapping[str, Any] | _Sentinel = _Sentinel.UNSET,
    ) -> RunContext:
        """Return a new context with selected fields replaced.

        Used by multi-agent architectures when spawning sub-agents
        that need to inherit most of the parent's context but with
        a derived ``session_id`` or augmented ``metadata``. The
        sentinel makes "leave this field unchanged" distinguishable
        from "explicitly set this field to ``None``".
        """
        return RunContext(
            user_id=self.user_id if user_id is _Sentinel.UNSET else user_id,
            session_id=(
                self.session_id if session_id is _Sentinel.UNSET else session_id
            ),
            run_id=self.run_id if run_id is _Sentinel.UNSET else run_id,
            metadata=(
                self.metadata if metadata is _Sentinel.UNSET else metadata
            ),
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Shorthand for ``self.metadata.get(key, default)``."""
        return self.metadata.get(key, default)


# ---------------------------------------------------------------------------
# ContextVar plumbing
# ---------------------------------------------------------------------------


_DEFAULT_CONTEXT = RunContext()
"""The context returned by :func:`get_run_context` when no run is
active. All-None / empty so test code that calls ``@tool`` functions
directly (with no agent loop running) gets a sane object back rather
than an exception."""


_ctx_var: ContextVar[RunContext] = ContextVar(
    "loomflow_run_context", default=_DEFAULT_CONTEXT
)


# ``Memory`` ambient — used when a :class:`Workflow` is configured
# with ``memory=`` and a nested :class:`Agent` did not specify its
# own. The Workflow installs this for the duration of a run; nested
# agents that left ``memory=`` unset read it from here as a fallback
# in :meth:`Agent._loop`. Carried separately from ``RunContext`` to
# keep that frozen-dataclass small and avoid pulling the ``Memory``
# protocol into the data class definition.
_ambient_memory_var: ContextVar[Any] = ContextVar(
    "loomflow_ambient_memory", default=None
)

# ``response_tone`` ambient — same propagation pattern as memory.
# A :class:`Workflow` configured with ``response_tone=`` installs
# the spec here for the duration of a run; nested :class:`Agent`
# steps that left ``response_tone=`` unset on construction read it
# as a fallback. Carries the raw string spec (preset name or
# free-form); :func:`loomflow.core.tone.resolve_response_tone` is
# responsible for translating it into a directive.
_ambient_response_tone_var: ContextVar[str | None] = ContextVar(
    "loomflow_ambient_response_tone", default=None
)

# ``Workspace`` ambient — same propagation pattern as memory + tone.
# A :class:`Workflow` configured with ``workspace=`` installs the
# resolved :class:`~loomflow.workspace.Workspace` here for the
# duration of a run; nested :class:`Agent` steps that left
# ``workspace=`` unset on construction read it as a fallback and
# auto-wire the five notebook tools onto themselves at run start.
# Lets a team share one notebook by wiring it once at the workflow
# level instead of per-agent.
_ambient_workspace_var: ContextVar[Any] = ContextVar(
    "loomflow_ambient_workspace", default=None
)

# ``LivingPlan`` ambient — per-run plan storage for the TodoWrite-
# style ``plan_write`` / ``plan_read`` tools. :meth:`Agent.run`
# allocates a fresh :class:`LivingPlan` (or uses the seeded one
# from ``living_plan=<LivingPlan instance>``) at run start, installs
# it here for the duration, and resets the token at run end. Plan
# tools read this contextvar at call time so concurrent
# ``agent.run()`` invocations on the same :class:`Agent` instance
# operate on isolated plan state. Held separately from
# ``RunContext`` because :class:`LivingPlan` is mutable and the
# ``RunContext`` is a frozen dataclass.
_ambient_living_plan_var: ContextVar[Any] = ContextVar(
    "loomflow_ambient_living_plan", default=None
)

# ``Workspace citation tracking`` ambient — per-run set of note
# slugs the agent READ during this run. ``Workspace.read_note`` /
# ``read_version`` add to it; ``Workspace.attribute_outcome`` reads
# it after the run to update each cited note's relevance metadata
# (``cited_count``, ``success_count``, ``last_cited_at``).
#
# Default is ``None`` (no tracking — outside a run, in tests, or
# when the workspace deliberately disables it). The ``Agent._loop``
# sets a fresh empty set per run; the contextvar mechanism keeps
# concurrent runs isolated. Citations are an OBSERVATION, not a
# write — they don't need author-ownership checks.
#
# Implementation note: we use a set wrapper rather than a frozenset
# so the workspace can mutate the set in-place during the run. The
# Token returned by .set() resets to the prior value on exit.
_ambient_citations_var: ContextVar[Any] = ContextVar(
    "loomflow_ambient_citations", default=None
)


def get_run_context() -> RunContext:
    """Return the :class:`RunContext` for the currently-running agent.

    Inside an active :meth:`Agent.run` call this returns the live
    context with ``user_id``, ``session_id``, ``run_id``, and
    ``metadata`` populated. Outside any active run (test code,
    direct ``@tool`` invocation, REPL exploration) this returns the
    default empty :class:`RunContext` — never raises.

    Tools that need scope information call this rather than taking
    extra parameters:

    .. code-block:: python

        @tool
        async def fetch_user_orders() -> str:
            ctx = get_run_context()
            return await db.query("orders", user_id=ctx.user_id)
    """
    return _ctx_var.get()


class set_run_context:  # noqa: N801 — context-manager class, lowercase by convention
    """Context manager that installs a :class:`RunContext` for the
    duration of an ``async with`` block.

    The framework uses this internally inside :meth:`Agent.run` to
    expose the live context to tools and hooks. Application code
    rarely needs it, but it is the supported way to invoke a tool
    *outside* an agent loop with explicit scope — for example in
    background workers that share tool implementations with the
    agent::

        async with set_run_context(RunContext(user_id="alice")):
            await some_tool(...)

    Behaves correctly under structured concurrency: nested
    ``async with`` blocks restore the prior context on exit, and
    ``anyio`` task-group spawns inherit the active context
    automatically.
    """

    __slots__ = ("_context", "_token")

    def __init__(self, context: RunContext) -> None:
        self._context = context
        self._token: Token[RunContext] | None = None

    async def __aenter__(self) -> RunContext:
        self._token = _ctx_var.set(self._context)
        return self._context

    async def __aexit__(self, *exc_info: object) -> None:
        if self._token is not None:
            _ctx_var.reset(self._token)
            self._token = None


@contextmanager
def inherit_ambient_memory(memory: Memory) -> Iterator[None]:
    """Install ``memory`` as the ambient memory for the duration of
    the ``with`` block. A spawned worker :class:`Agent` whose
    ``memory=`` was NOT explicitly set at construction will pick this
    up via :meth:`Agent._resolve_run_memory` and use it instead of
    its private default :class:`InMemoryMemory`.

    Mirrors the propagation :class:`Workflow.stream` does at
    ``workflow/__init__.py:720``. Multi-agent architectures
    (Supervisor, Swarm, Router, Debate, ActorCritic, Blackboard) wrap
    every worker-spawn site with this so workers inherit the
    coordinator's memory backend — closing the gap where
    ``Team.supervisor(memory=...)`` was silently NOT propagating to
    workers that had been constructed without their own ``memory=``.

    Idempotent + nest-safe: contextvars are restored on exit, so
    nested ``with`` blocks and ``anyio`` task-group spawns interact
    correctly. Workers that DID set ``memory=`` at construction
    (``_memory_was_explicit=True``) ignore this — explicit always
    wins over ambient, matching the workflow precedence rule.

    Example::

        from loomflow.core.context import inherit_ambient_memory

        with inherit_ambient_memory(deps.memory):
            result = await worker.run(
                instructions,
                session_id=worker_session_id,
                context=get_run_context(),
            )
    """
    token = _ambient_memory_var.set(memory)
    try:
        yield
    finally:
        _ambient_memory_var.reset(token)
