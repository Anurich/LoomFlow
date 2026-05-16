"""The public ``Agent`` class.

Conventions:

* Pass a string of instructions for a working agent backed by sensible
  defaults: :class:`EchoModel`, :class:`InMemoryMemory`,
  :class:`InProcRuntime`, :class:`NoBudget`, :class:`AllowAll`,
  :class:`HookRegistry`, an empty :class:`InProcessToolHost`.
* Pass ``tools=[fn_or_Tool, ...]`` to register Python callables; the
  agent wraps them in an in-process :class:`ToolHost`.
* Override any subsystem by passing a concrete implementation of the
  matching protocol from :mod:`loomflow.core.protocols`.

Two execution surfaces share a single internal loop:

* :meth:`Agent.run` runs to completion and returns a :class:`RunResult`.
* :meth:`Agent.stream` returns an ``AsyncIterator[Event]`` of milestones
  as they happen — STARTED, MODEL_CHUNK, TOOL_CALL, TOOL_RESULT,
  BUDGET_WARNING/EXCEEDED, ERROR, COMPLETED.

Internally, :meth:`_loop` accepts an ``emit`` callback and threads it
through every milestone. ``run()`` passes a no-op emit; ``stream()``
pipes events through an :func:`anyio.create_memory_object_stream` so a
slow consumer applies backpressure to the loop instead of buffering
unboundedly.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import anyio
from pydantic import BaseModel, ValidationError

from ..architecture import (
    AgentSession,
    Architecture,
    Dependencies,
    resolve_architecture,
)
from ..architecture.tool_host_wrappers import ExtendedToolHost
from ..core.context import RunContext, set_run_context
from ..core.errors import OutputValidationError
from ..core.ids import new_id
from ..core.protocols import (
    Budget,
    HookHost,
    Memory,
    Model,
    Permissions,
    Runtime,
    Telemetry,
    ToolHost,
)
from ..core.types import (
    Episode,
    Event,
    Message,
    PromptCacheConfig,
    Role,
    RunResult,
    Usage,
)
from ..governance.budget import NoBudget
from ..governance.retry import RetryPolicy
from ..model.echo import EchoModel
from ..observability.tracing import NoTelemetry
from ..runtime.inproc import InProcRuntime
from ..security.audit import AuditLog
from ..security.hooks import HookRegistry, PostToolHook, PreToolHook
from ..security.permissions import AllowAll
from ..tools.registry import InProcessToolHost, Tool
from .stop_hooks import StopHook, StopHookResult

DEFAULT_MAX_TURNS = 50
DEFAULT_STREAM_BUFFER = 128

Emit = Callable[[Event], Awaitable[None]]

# Module-level singleton no-op async context manager. ``contextlib.nullcontext``
# implements both the sync and async protocols (since Python 3.10), so we can
# reuse a single instance everywhere a hot path wants to *maybe* enter a
# telemetry span: ``async with (NULL_CTX if fast else tel.trace(...)):``.
_NULL_CTX: contextlib.AbstractAsyncContextManager[None] = contextlib.nullcontext()


class Agent:
    """A fully-async, MCP-native, model-agnostic agent harness."""

    def __init__(
        self,
        instructions: str,
        *,
        model: Model | str | dict[str, Any] | None = None,
        memory: Memory | str | Mapping[str, Any] | None = None,
        runtime: Runtime | str | Mapping[str, Any] | None = None,
        budget: Budget | None = None,
        permissions: Permissions | None = None,
        hooks: HookRegistry | None = None,
        tools: list[Tool | Callable[..., object]]
        | ToolHost
        | Tool
        | Callable[..., object]
        | None = None,
        telemetry: Telemetry | str | Mapping[str, Any] | None = None,
        audit_log: AuditLog | str | Path | dict[str, Any] | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        auto_consolidate: bool = False,
        architecture: Architecture | str | None = None,
        skills: list[Any] | None = None,
        retry_policy: RetryPolicy | None = None,
        auto_extract: bool | None = None,
        approval_handler: Any | None = None,
        secrets: Any | None = None,
        output_schema: type[BaseModel] | Any | None = None,
        response_tone: str | None = None,
        effort: str | None = None,
        strict_effort: bool = False,
        prompt_caching: bool | Mapping[str, Any] | None = None,
        workspace: Any | str | Mapping[str, Any] | None = None,
        living_plan: Any = None,
        stop_hooks: list[StopHook] | None = None,
        max_stop_hook_iterations: int = 15,
        tool_result_summarizer: Model | str | None = None,
        tool_result_summary_threshold: int = 500,
        snip_window: int = 0,
        auto_compact_at_tokens: int | None = None,
        auto_compact_summariser: Model | str | None = None,
        auto_compact_keep_recent_turns: int = 4,
    ) -> None:
        # Skills — packaged on-disk playbooks loaded on demand.
        # Build the registry first so frontmatter validation fires
        # at construction time (not later, when the model first
        # tries to load).
        from ..architecture.tool_host_wrappers import ExtendedToolHost
        from ..skills import SkillRegistry, make_load_skill_tool

        skill_registry = (
            SkillRegistry(skills) if skills else None
        )
        has_skills = skill_registry is not None and len(skill_registry) > 0

        self._instructions = instructions
        if has_skills:
            # Append the skill catalog to the system prompt — the
            # cheap "metadata" tier of progressive disclosure
            # (~50 tokens per skill, regardless of body size).
            assert skill_registry is not None
            self._instructions = (
                f"{instructions.rstrip()}\n\n"
                f"{skill_registry.catalog_section()}"
            )

        # ``_model`` is the inner, concrete adapter (OpenAIModel,
        # AnthropicModel, etc.) — kept as-is so introspection +
        # tests see the real type. ``_wrapped_model`` is the
        # retry-decorated version that gets handed to
        # :class:`Dependencies` and used by every architecture's
        # model call. We wrap when (a) a caller-supplied policy
        # says to retry, or (b) the default policy is appropriate
        # for this model class — ScriptedModel and EchoModel are
        # tests / dev fakes (no retry needed); real network-backed
        # models (OpenAI, Anthropic, LiteLLM) get the sensible
        # default unless the caller passed ``RetryPolicy.disabled()``.
        # Default to ``EnvSecrets`` so today's behaviour is
        # preserved (API keys come from ``os.environ``); callers
        # who want a vault / dict-backed lookup pass an explicit
        # ``secrets=`` instance.
        if secrets is None:
            from ..security.secrets import EnvSecrets
            secrets = EnvSecrets()
        self._secrets = secrets
        # Dict-form ``model={"name": ..., "effort": ..., "strict_effort":
        # ...}`` is sugar for passing the model spec + agent-level
        # reasoning-effort config in one place. Top-level kwargs win when
        # both are specified — explicit beats config.
        model, effort, strict_effort = _normalize_model_spec(
            model, effort=effort, strict_effort=strict_effort,
        )
        self._model: Model = _resolve_model(model, secrets=self._secrets)
        self._retry_policy: RetryPolicy = (
            retry_policy
            if retry_policy is not None
            else _default_retry_policy_for(self._model)
        )
        if self._retry_policy.is_enabled():
            from ..model.retrying import RetryingModel
            self._wrapped_model: Model = RetryingModel(
                self._model, self._retry_policy
            )
        else:
            self._wrapped_model = self._model
        # Telemetry is initialised *before* the memory wrapping so
        # the AutoExtractMemory wrapper (below) can take a reference
        # and emit duration / extraction-count metrics. The resolver
        # accepts string ("console", "file:./spans.jsonl", "otel"),
        # dict ({"backend": "file", "path": "..."}), or an already-
        # built Telemetry instance.
        from ..observability.resolver import resolve_telemetry
        self._telemetry: Telemetry = resolve_telemetry(telemetry)
        # Resolve ``memory=`` into a concrete :class:`Memory`.
        # The resolver handles None (default in-memory), strings
        # ("sqlite:./bot.db", "postgres://..."), config dicts
        # ({"backend": ..., ...}), and explicit Memory instances.
        from ..memory.resolver import resolve_memory
        # Track whether the caller explicitly supplied a memory.
        # When False, ``_loop`` checks the ambient workflow memory
        # contextvar (set by ``Workflow(memory=...)``) and uses
        # that as a fallback. Explicit-on-Agent ALWAYS wins so
        # opt-out by passing your own memory= still works.
        self._memory_was_explicit: bool = memory is not None
        self._memory: Memory = resolve_memory(memory)

        # Auto fact extraction. When enabled, the agent loop's
        # ``remember(episode)`` calls go through an
        # :class:`AutoExtractMemory` wrapper that runs a small
        # Consolidator pass after each write, pulling structured
        # (subject, predicate, object) claims out of the
        # conversation into the bi-temporal fact store.
        # ``_wrapped_memory`` is the loop-facing view; ``_memory``
        # stays as the user-supplied inner so introspection +
        # tests see the real backend type. Same dual-attribute
        # pattern as ``_model`` / ``_wrapped_model`` for retries.
        # Extraction is best-effort — failures (model errors,
        # malformed JSON) never break the run.
        #
        # Default is auto-picked: ON for real network adapters
        # (OpenAI / Anthropic / LiteLLM) where extraction is the
        # whole point of "your bot just remembers"; OFF for in-
        # process fakes (ScriptedModel / EchoModel) since those
        # produce canned output that confuses the Consolidator
        # and make tests non-deterministic. Pass ``auto_extract=
        # True/False`` to override.
        # Track whether auto_extract was explicitly chosen by the
        # caller or default-picked from the model class — the
        # AutoExtractMemory wrapper uses this to decide whether to
        # emit a one-time-per-process startup notice (ops visibility
        # for the default-on path).
        auto_extract_was_default = auto_extract is None
        if auto_extract is None:
            auto_extract = _default_auto_extract_for(self._model)
        if auto_extract:
            from ..memory.auto_extract import AutoExtractMemory
            from ..memory.consolidator import Consolidator
            consolidator = Consolidator(model=self._model)
            self._wrapped_memory: Memory = AutoExtractMemory(
                self._memory,
                consolidator,
                telemetry=self._telemetry,
                auto_picked=auto_extract_was_default,
            )
        else:
            self._wrapped_memory = self._memory
        # Resolve ``runtime=`` (string / dict / instance) the same way
        # model / memory / telemetry do. Sync resolution only —
        # PostgresRuntime needs an async connect, so users wire it up
        # themselves and pass the instance through.
        from ..runtime.resolver import resolve_runtime
        self._runtime: Runtime = resolve_runtime(runtime)
        self._budget: Budget = budget if budget is not None else NoBudget()
        self._permissions: Permissions = (
            permissions if permissions is not None else AllowAll()
        )
        self._hooks = hooks if hooks is not None else HookRegistry()
        self._skills = skill_registry

        host = _coerce_tool_host(tools)
        if has_skills:
            assert skill_registry is not None
            # InProcessToolHost and ExtendedToolHost both expose
            # ``register``; for any other host implementation
            # (MCP / custom), wrap with ExtendedToolHost so we
            # have a place to push the load_skill tool plus any
            # pending tools the skills lazy-register on demand.
            if not isinstance(host, InProcessToolHost):
                host = ExtendedToolHost(host, [])
            load_tool = make_load_skill_tool(
                skill_registry, host=host
            )
            host.register(load_tool)

        # ----- Shared workspace ----------------------------------------
        # When ``workspace=`` is wired, the framework auto-installs
        # five tools onto the agent (note / read_note / list_notes /
        # search_notes / update_note) and appends a prompt section
        # nudging the model to coordinate with teammates via the
        # shared notebook. Author identity is baked into the tool
        # closures so the agent never has to attribute itself.
        from ..workspace.resolver import resolve_workspace as _resolve_ws
        from ..workspace.tools import (
            make_workspace_tools as _make_ws_tools,
        )
        from ..workspace.tools import (
            workspace_prompt_section as _ws_prompt,
        )
        from ..workspace.types import WorkspaceMembership as _WSMember

        # The ``workspace=`` kwarg accepts three shapes:
        #   * raw :class:`Workspace` instance — share the notebook,
        #     no specific identity (notes attributed to "agent")
        #   * :class:`WorkspaceMembership` (typed) — chained from
        #     ``ws.member("researcher", teammates=[...])`` — name
        #     and teammates bundled in
        #   * ``Mapping`` with ``backend`` / ``author`` / ``teammates``
        #     — declarative dict form, resolved to a Membership
        # The resolver returns either a bare Workspace or a
        # Membership; we unpack identity here so the rest of
        # ``_loop`` sees ``self._workspace`` / ``self._workspace_name``
        # / ``self._workspace_teammates`` as before.
        _resolved = _resolve_ws(workspace)
        _ws_name: str | None = None
        _ws_teammates: list[str] | None = None
        if isinstance(_resolved, _WSMember):
            self._workspace: Any | None = _resolved.workspace
            _ws_name = _resolved.name
            _ws_teammates = (
                list(_resolved.teammates) if _resolved.teammates else None
            )
        else:
            self._workspace = _resolved
        self._workspace_was_explicit: bool = workspace is not None
        self._workspace_name: str | None = _ws_name
        # Names of teammates this agent collaborates with on the
        # shared workspace. The prompt section names them so the
        # model knows who else is contributing. Populated by the
        # ``WorkspaceMembership`` form OR mutated by a Team builder
        # at the moment the agent is wrapped into a team.
        self._workspace_teammates: list[str] | None = _ws_teammates
        if self._workspace is not None:
            author = self._workspace_name or "agent"
            ws_tools = _make_ws_tools(self._workspace, author=author)
            if not isinstance(host, InProcessToolHost) and not hasattr(
                host, "register"
            ):
                # Non-register-capable host (raw MCP, etc.) — wrap so
                # we can add the workspace tools alongside its native
                # ones. Mirrors the skills branch above.
                host = ExtendedToolHost(host, [])
            for t in ws_tools:
                host.register(t)  # type: ignore[union-attr]
            # Append the workspace prompt section after instructions
            # (and after the skill catalog if present).
            self._instructions = (
                f"{self._instructions.rstrip()}\n\n"
                f"{_ws_prompt(author=author, teammates=self._workspace_teammates)}"
            )

        # ----- Living plan ----------------------------------------------
        # When ``living_plan=`` is enabled, the framework auto-installs
        # two TodoWrite-style tools (``plan_write`` / ``plan_read``)
        # onto the agent, plus optionally ``recall_past_plans`` when
        # the workspace mirror is on. The tools read per-run state
        # from the ambient :data:`_ambient_living_plan_var` contextvar
        # (set in :meth:`_loop`) so concurrent ``agent.run()`` calls
        # on the same Agent have isolated plans.
        #
        # Default for v0.10.0 is OPT-IN (``None`` → disabled). The
        # roadmap is to flip ``None`` to a smart-default in v0.11 (on
        # for tool-using agents, off otherwise) once the primitive
        # has been dogfooded.
        from ..tools.plan import (
            living_plan_prompt_section as _lp_prompt,
        )
        from ..tools.plan import (
            make_plan_tools as _make_plan_tools,
        )
        from ..tools.plan import (
            make_recall_past_plans_tool as _make_recall_past_plans,
        )
        from ..tools.plan_resolver import resolve_living_plan

        self._living_plan_spec = resolve_living_plan(
            living_plan,
            workspace_present=self._workspace is not None,
        )
        if self._living_plan_spec.enabled:
            plan_author = (
                self._living_plan_spec.author
                or self._workspace_name
                or "agent"
            )
            plan_workspace = (
                self._workspace
                if self._living_plan_spec.mirror_to_workspace
                else None
            )
            plan_tools = _make_plan_tools(
                workspace=plan_workspace,
                task_id=self._living_plan_spec.task_id,
                author=plan_author,
            )
            if self._living_plan_spec.include_recall and plan_workspace is not None:
                plan_tools = [
                    *plan_tools,
                    _make_recall_past_plans(plan_workspace, author=plan_author),
                ]
            if not isinstance(host, InProcessToolHost) and not hasattr(
                host, "register"
            ):
                host = ExtendedToolHost(host, [])
            for t in plan_tools:
                host.register(t)  # type: ignore[union-attr]
            self._instructions = (
                f"{self._instructions.rstrip()}\n\n"
                f"{_lp_prompt(has_workspace_mirror=self._living_plan_spec.mirror_to_workspace)}"
            )
        self._tool_host: ToolHost = host

        # ``self._telemetry`` already initialised above (before the
        # memory-wrap step so AutoExtractMemory can take a reference).
        from ..security.audit import resolve_audit_log
        self._audit_log: AuditLog | None = resolve_audit_log(audit_log)
        self._max_turns = max_turns
        self._auto_consolidate = auto_consolidate
        # Approval handler resolves :class:`Decision.ask_` outcomes
        # from the permissions layer. Without one, ``ask`` falls
        # back to deny — see ``_resolve_ask_decision`` in
        # ``architecture/react.py``.
        self._approval_handler = approval_handler
        # Default ``output_schema`` for runs that don't supply one.
        # ``Agent(output_schema=Receipt)`` makes "this agent always
        # returns a Receipt" the contract; per-call ``run(...,
        # output_schema=)`` still wins for callers that want to
        # override on a per-prompt basis.
        self._default_output_schema: Any | None = output_schema
        # Agent-default response tone. Per-call ``run(response_tone=)``
        # wins; if neither is set, the workflow ambient (set by
        # ``Workflow(response_tone=)``) is the next fallback;
        # otherwise no tone directive is appended at all.
        self._default_response_tone: str | None = response_tone
        # Default ``effort`` for runs that don't supply one.
        # ``Agent(effort="high")`` makes "this agent always thinks
        # hard" the contract; per-call ``run(..., effort=)`` still
        # wins for callers that want to override per-prompt.
        self._default_effort: str | None = effort
        # ``strict_effort`` is agent-level only — no per-call
        # override. If you want a run to fail when the model can't
        # honour the effort dial, that's a property of how the
        # whole agent is wired up.
        self._strict_effort: bool = strict_effort
        # Prompt caching is also agent-level: caching is a property
        # of how the agent is wired (which model, which TTL,
        # whether to route per-session). The Anthropic adapter
        # reads this to decide whether to inject ``cache_control``
        # markers on the system prompt + tool definitions; the
        # OpenAI adapter reads it for the optional
        # ``prompt_cache_key`` routing hint (OpenAI caching itself
        # is automatic regardless).
        self._prompt_caching: PromptCacheConfig = _resolve_prompt_caching(
            prompt_caching
        )
        self._architecture: Architecture = resolve_architecture(architecture)

        # Stop hooks — framework-level Ralph loop. Auto-hooks
        # (today: living_plan in-progress checker) prepend; user-
        # supplied hooks append. First hook to return a non-None
        # StopHookResult in each iteration wins. Bounded by
        # ``max_stop_hook_iterations`` so a flaky hook can't burn
        # the user's budget unbounded.
        auto_hooks: list[StopHook] = []
        if (
            self._living_plan_spec.enabled
            and self._living_plan_spec.auto_stop_hook
        ):
            # Import here to avoid a circular cycle at module
            # load: plan.py imports types that resolve back into
            # the agent package.
            from ..tools.plan import make_plan_stop_hook
            auto_hooks.append(make_plan_stop_hook())
        self._stop_hooks: list[StopHook] = [
            *auto_hooks,
            *(stop_hooks or []),
        ]
        if max_stop_hook_iterations < 0:
            raise ValueError(
                "max_stop_hook_iterations must be >= 0 "
                "(0 disables the loop entirely)"
            )
        self._max_stop_hook_iterations: int = max_stop_hook_iterations

        # Tool-result summariser. ``None`` (the default) disables
        # summarisation — tool results ship verbatim, behaviour
        # identical to pre-0.10.14. When a model is wired, the
        # ReAct loop replaces oversized tool results with a model-
        # generated summary before they enter conversation history
        # (see :mod:`loomflow.tools.result_summarizer`). Accepts the
        # same shapes as the main ``model=`` kwarg: a model name
        # string, a dict config, or a :class:`Model` instance.
        self._tool_result_summarizer: Model | None = (
            None
            if tool_result_summarizer is None
            else _resolve_model(
                tool_result_summarizer, secrets=self._secrets
            )
        )
        if tool_result_summary_threshold < 0:
            raise ValueError(
                "tool_result_summary_threshold must be >= 0"
            )
        self._tool_result_summary_threshold: int = (
            tool_result_summary_threshold
        )

        # Snip — bounded conversation window. ``0`` (default)
        # disables; positive integer keeps the last N user-
        # anchored turn groups in ``session.messages`` before
        # each architecture invocation. See
        # :mod:`loomflow.agent.snip` for the slicing semantics.
        # Snipping is pure list-slicing — no API call, no model
        # required. Pairs with tool_result_summarizer (0.10.14)
        # and the future auto-compact (0.10.19) as the three
        # tiers of context-budget defence.
        if snip_window < 0:
            raise ValueError(
                "snip_window must be >= 0 (0 disables snipping)"
            )
        self._snip_window: int = snip_window

        # Auto-compact (0.10.19) — third tier of context-budget
        # defence. None (default) → disabled. Pass an int (token
        # threshold) to enable; the summariser defaults to the
        # main model if not supplied separately. Fires inside
        # the Ralph loop between iterations when
        # ``session.messages`` accumulates past the threshold.
        # See :mod:`loomflow.agent.auto_compact` for the
        # compaction algorithm and trigger semantics.
        if (
            auto_compact_at_tokens is not None
            and auto_compact_at_tokens <= 0
        ):
            raise ValueError(
                "auto_compact_at_tokens must be > 0 or None "
                "(None disables auto-compact entirely)"
            )
        self._auto_compact_at_tokens: int | None = (
            auto_compact_at_tokens
        )
        # Summariser defaults to the main model when not given.
        # That keeps the opt-in API to a single kwarg
        # (``auto_compact_at_tokens=N``) for the common case;
        # users who want a cheaper model (Haiku for an Opus
        # main) pass ``auto_compact_summariser="haiku"``.
        if auto_compact_summariser is None:
            self._auto_compact_summariser: Model | None = (
                None if auto_compact_at_tokens is None
                else self._model
            )
        else:
            self._auto_compact_summariser = _resolve_model(
                auto_compact_summariser, secrets=self._secrets
            )
        if auto_compact_keep_recent_turns < 1:
            raise ValueError(
                "auto_compact_keep_recent_turns must be >= 1"
            )
        self._auto_compact_keep_recent_turns: int = (
            auto_compact_keep_recent_turns
        )

        # Persistent-subagent registry. Populated by
        # ``Team.supervisor(persistent_subagents=True)`` (the
        # default in v0.10.10+) with one :class:`_WorkerHandle` per
        # worker the coordinator can delegate to. Plain dict for
        # v1 — concrete-first, extract :class:`WorkerRegistry`
        # Protocol when a second backend (durable-on-disk for
        # /resume, Redis for distributed CLI) materialises.
        #
        # Empty dict for every non-coordinator Agent (no kwarg,
        # no overhead — 64 bytes per Agent instance). Mutable
        # cross-run state by design; durability across
        # ``Agent.run`` calls IS the feature.
        from .worker_registry import _WorkerHandle  # noqa: I001 — local to avoid circ
        self._worker_registry: dict[str, _WorkerHandle] = {}

    # ---- hook decorators (user-facing sugar) ----------------------------

    def before_tool(self, fn: PreToolHook) -> PreToolHook:
        """Register a pre-tool hook. First denial wins; allow otherwise."""
        return self._hooks.register_pre_tool(fn)

    def after_tool(self, fn: PostToolHook) -> PostToolHook:
        """Register a best-effort post-tool callback."""
        return self._hooks.register_post_tool(fn)

    @property
    def hooks(self) -> HookHost:
        return self._hooks

    def __repr__(self) -> str:
        host_name = type(self._tool_host).__name__
        return (
            f"Agent(model={self._model.name!r}, "
            f"memory={type(self._memory).__name__}, "
            f"runtime={type(self._runtime).__name__}, "
            f"tools={host_name}, "
            f"max_turns={self._max_turns})"
        )

    # ---- public introspection ------------------------------------------

    @property
    def model(self) -> Model:
        """The configured :class:`Model` adapter."""
        return self._model

    @property
    def memory(self) -> Memory:
        """The configured :class:`Memory` backend."""
        return self._memory

    @property
    def runtime(self) -> Runtime:
        """The configured :class:`Runtime`."""
        return self._runtime

    @property
    def tool_host(self) -> ToolHost:
        """The configured :class:`ToolHost`."""
        return self._tool_host

    @property
    def skills(self) -> Any | None:
        """The :class:`SkillRegistry` of skills registered on this
        agent (or ``None`` if no skills were configured). Useful for
        inspecting / mutating the skill set after construction."""
        return self._skills

    @property
    def budget(self) -> Budget:
        """The configured :class:`Budget`."""
        return self._budget

    @property
    def permissions(self) -> Permissions:
        """The configured :class:`Permissions` policy."""
        return self._permissions

    @property
    def instructions(self) -> str:
        """The system prompt the agent runs with.

        Surfaced as a public property so multi-agent architectures
        (e.g. :class:`~loomflow.architecture.Supervisor`) can read
        each worker's intended role when composing instructions for
        the supervising model.
        """
        return self._instructions

    @property
    def architecture(self) -> Architecture:
        """The configured :class:`Architecture` strategy.

        Default is :class:`~loomflow.architecture.ReAct`. Pass
        ``architecture=`` to ``Agent(...)`` to override.
        """
        return self._architecture

    # ---- graph visualization -------------------------------------------

    async def generate_graph(
        self,
        path: str | Path | None = None,
        *,
        title: str | None = None,
    ) -> str:
        """Render this agent's structure as a Mermaid graph.

        Walks the agent + its architecture + all sub-agents +
        every agent's tools, producing a graph that captures the
        full team, tool attachments, and architecture-specific
        relationships (delegate / handoff / classify / etc.).

        Returns the Mermaid text. If ``path`` is provided, also
        writes to disk — extension determines the format:

        * ``.mmd`` — raw Mermaid source
        * ``.md``  — Markdown with the diagram in a ``mermaid``
          fence (renders on GitHub, IDE markdown previews,
          Jupyter)
        * ``.png`` / ``.svg`` — rendered via ``mermaid.ink``;
          falls back to ``.mmd`` next to the path on network
          failure

        Example::

            mermaid_text = await agent.generate_graph("graph.md")
            print(mermaid_text)

        Pass ``title=`` to override the diagram title (defaults to
        the file's stem, or ``"Agent"`` if no path is given).
        """
        from ..graph import build_graph, write_graph

        if path is None:
            graph = await build_graph(self, title=title or "Agent")
            return graph.to_mermaid()
        return await write_graph(self, path, title=title)

    # ---- public memory shortcuts ---------------------------------------

    async def recall(
        self,
        query: str,
        *,
        kind: str = "episodic",
        limit: int = 5,
    ) -> list[Any]:
        """Convenience wrapper around ``self.memory.recall(query, ...)``.

        Returns episodes matching ``query``. For semantic / fact-store
        recall, use ``self.memory.facts.recall_text(...)`` directly.
        """
        return await self._memory.recall(query, kind=kind, limit=limit)

    # ---- plugin API ----------------------------------------------------

    def add_tool(self, item: Tool | Callable[..., object]) -> Tool:
        """Register a tool after construction.

        Convenience for plugin-style code that adds tools after the
        ``Agent`` exists. Only works when the underlying tool host is
        an :class:`InProcessToolHost` (the default — and the only host
        that has a writable registry today).

        Returns the constructed :class:`Tool` so callers can introspect
        the auto-derived schema.
        """
        if not isinstance(self._tool_host, InProcessToolHost):
            from ..core.errors import ConfigError

            raise ConfigError(
                f"add_tool requires InProcessToolHost; got "
                f"{type(self._tool_host).__name__}. Pass the tool at "
                "construction time, or wrap it in a custom ToolHost."
            )
        return self._tool_host.register(item)

    def with_tool(
        self, fn: Callable[..., object]
    ) -> Callable[..., object]:
        """Decorator-style equivalent of :meth:`add_tool`.

        Usage::

            @agent.with_tool
            async def search(query: str) -> str:
                '''Search a knowledge base.'''
                return f"results for {query}"

        Returns the original function unchanged (so it can still be
        called normally), and registers it as a tool on the agent's
        underlying :class:`InProcessToolHost`. Same constraint as
        :meth:`add_tool`: the host must be writable.
        """
        self.add_tool(fn)
        return fn

    def remove_tool(self, name: str) -> bool:
        """Unregister a tool by name. Returns ``True`` if a tool was
        removed, ``False`` if no tool with that name was registered.

        Same constraint as :meth:`add_tool`: only works with
        :class:`InProcessToolHost`.
        """
        if not isinstance(self._tool_host, InProcessToolHost):
            from ..core.errors import ConfigError

            raise ConfigError(
                f"remove_tool requires InProcessToolHost; got "
                f"{type(self._tool_host).__name__}."
            )
        return self._tool_host.unregister(name)

    async def tools_list(self) -> list[str]:
        """Return the names of all currently-registered tools.

        Convenience that works for any :class:`ToolHost`. Calls
        ``tool_host.list_tools()`` under the hood and returns just the
        names; use ``self.tool_host.list_tools()`` directly for the
        full :class:`ToolDef` records.
        """
        defs = await self._tool_host.list_tools()
        return [d.name for d in defs]

    async def consolidate(self) -> int:
        """Manually trigger memory consolidation.

        Returns the number of new facts the consolidator extracted,
        or ``0`` when the memory backend doesn't expose a fact store.

        Useful when ``auto_consolidate=False`` (the default) and you
        want to batch consolidation at a controlled cadence — e.g.
        once a day, or before shutdown.
        """
        fact_store = getattr(self._memory, "facts", None)
        before = 0
        if fact_store is not None:
            before = len(await fact_store.all_facts())
        await self._memory.consolidate()
        if fact_store is None:
            return 0
        after = len(await fact_store.all_facts())
        return max(0, after - before)

    async def _audit(
        self,
        *,
        session_id: str,
        actor: str,
        action: str,
        payload: dict[str, Any],
        user_id: str | None = None,
    ) -> None:
        if self._audit_log is None:
            return
        # Forward user_id as a top-level field. Older AuditLog
        # impls without the kwarg fall back via the
        # ``except TypeError`` so the agent never breaks on a
        # legacy custom log.
        try:
            await self._audit_log.append(
                session_id=session_id,
                actor=actor,
                action=action,
                payload=payload,
                user_id=user_id,
            )
        except TypeError:
            from ..core._deprecation import warn_legacy_protocol
            warn_legacy_protocol("AuditLog", "append")
            await self._audit_log.append(
                session_id=session_id,
                actor=actor,
                action=action,
                payload=payload,
            )

    def _resolve_run_memory(self) -> Memory:
        """Pick the :class:`Memory` to use for the current run.

        Resolution order:

        1. **Explicit-on-Agent wins.** If the caller passed
           ``memory=`` to :class:`Agent`, that instance is used —
           wrapped with :class:`AutoExtractMemory` when
           ``auto_extract=True``. Workflow-level memory is ignored.
        2. **Ambient workflow memory** (set by
           ``Workflow(memory=...)``) is used as a fallback when the
           Agent had no explicit memory. This makes
           ``wf = Workflow.chain([agent_a, agent_b], memory=mem)``
           propagate ``mem`` to both agents without per-agent wiring.
           Auto-extract wrapping is *not* re-applied here — the
           workflow's memory is taken as-is so the user's choice of
           backend is respected exactly.
        3. **Agent default** (in-memory store) when neither was
           supplied — preserves the standalone-Agent behaviour.

        Called once per ``_loop`` invocation; the resolved memory
        is then used uniformly for the architecture's reads and
        the post-run episode write.
        """
        if self._memory_was_explicit:
            return self._wrapped_memory
        from ..core.context import _ambient_memory_var
        ambient = _ambient_memory_var.get()
        if ambient is not None:
            # Workflow memory wins over the agent's default. We use
            # the raw ambient instance (no auto-extract wrap) so
            # the workflow's choice of backend is taken at face
            # value — consistent with "explicit always wins".
            return cast(Memory, ambient)
        return self._wrapped_memory

    async def _validate_output_with_retry(
        self,
        *,
        session: AgentSession,
        schema: Any,
        retries: int,
        deps: Dependencies,
    ) -> BaseModel:
        """Validate ``session.output`` against ``schema``; on failure,
        give the model up to ``retries`` follow-up turns to fix it.

        ``schema`` may be a single ``BaseModel`` subclass or a tagged
        union (``A | B`` / ``Union[A, B]``). For unions we try each
        member in order and accept the first that validates — letting
        an agent return one of several shapes per call (e.g. a
        ``Receipt`` on success or a structured ``Error`` on
        failure).

        Retry turns are single-shot model calls (no tools, no full
        ReAct loop) because the architecture has already terminated
        — the model just needs to re-emit the JSON. Each retry
        appends the validation error as a USER message so the model
        sees what went wrong, regenerates, and we update
        ``session.output`` + ``cumulative_usage`` accordingly.

        Raises :class:`OutputValidationError` when the retry budget
        is exhausted.
        """
        from ..architecture.helpers import add_usage

        types = _extract_schema_types(schema)
        if not types:
            raise OutputValidationError(
                "output_schema must be a Pydantic BaseModel subclass or a "
                "Union of BaseModel subclasses.",
                raw=session.output,
                schema=schema,
                cause=None,
            )

        last_error: ValidationError | None = None
        for attempt in range(retries + 1):
            text = _strip_json_fences(session.output)
            parsed, last_error = _try_validate_union(text, types)
            if parsed is not None:
                # Stash the cleaned text so the persisted episode
                # has parseable JSON, not the fenced version.
                session.output = text
                return parsed
            if attempt >= retries:
                break

            # Re-prompt: tell the model what was wrong and ask
            # for a clean JSON re-emission.
            assert last_error is not None
            error_summary = _summarise_validation_error(last_error)
            schema_json = json.dumps(
                _union_json_schema(types), separators=(",", ":")
            )
            retry_prompt = (
                "Your previous response failed schema validation:\n"
                f"{error_summary}\n\n"
                "Return a corrected response — ONLY a single valid "
                "JSON object that matches this schema, with no "
                "prose, no markdown fences, no explanation:\n"
                f"{schema_json}"
            )
            session.messages.append(
                Message(role=Role.USER, content=retry_prompt)
            )

            # Single-shot model call. Use ``complete`` when
            # available (no streaming overhead); fall back to
            # consuming ``stream`` otherwise.
            if hasattr(deps.model, "complete"):
                new_text, _calls, usage, _finish = (
                    await deps.model.complete(
                        session.messages,
                        tools=None,
                        effort=deps.effort,
                        strict_effort=deps.strict_effort,
                    )
                )
            else:
                parts: list[str] = []
                usage = Usage()
                async for chunk in deps.model.stream(
                    session.messages,
                    tools=None,
                    effort=deps.effort,
                    strict_effort=deps.strict_effort,
                ):
                    if chunk.kind == "text" and chunk.text:
                        parts.append(chunk.text)
                    elif (
                        chunk.kind == "finish"
                        and chunk.usage is not None
                    ):
                        usage = chunk.usage
                new_text = "".join(parts)

            session.messages.append(
                Message(role=Role.ASSISTANT, content=new_text)
            )
            session.output = new_text
            session.cumulative_usage = add_usage(
                session.cumulative_usage, usage
            )
            session.turns += 1

        assert last_error is not None  # we only break above on a failure
        type_names = " | ".join(t.__name__ for t in types)
        raise OutputValidationError(
            f"Model output did not validate against {type_names} "
            f"after {retries} retry attempt(s).",
            raw=session.output,
            schema=schema,
            cause=last_error,
        ) from last_error

    # ---- factory: from dict / TOML config ------------------------------

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        model: Model | str | dict[str, Any] | None = None,
        memory: Memory | str | Mapping[str, Any] | None = None,
        runtime: Runtime | str | Mapping[str, Any] | None = None,
        telemetry: Telemetry | str | Mapping[str, Any] | None = None,
        audit_log: AuditLog | str | Path | dict[str, Any] | None = None,
        permissions: Permissions | str | Mapping[str, Any] | None = None,
        tools: list[Tool | Callable[..., object]] | ToolHost | None = None,
        secrets: Any | None = None,
        hooks: HookRegistry | None = None,
        retry_policy: RetryPolicy | None = None,
        approval_handler: Any | None = None,
    ) -> Agent:
        """Construct an ``Agent`` from a parsed config dict.

        Same shape as :meth:`from_config` but skips the file read.
        Useful when the config comes from somewhere other than a TOML
        file — environment variables, a Pydantic settings model, a
        ``yaml.safe_load`` result, an HTTP API, etc.

        Each backend kwarg overrides the corresponding ``cfg`` entry
        when both are supplied. Callables (tools, hooks, approval
        handlers, secrets stores) and pre-built instances are passed
        through kwargs because TOML / JSON / YAML can't express them.

        Recognised keys (all optional except ``instructions`` and
        ``model``):

        * ``instructions: str`` — required
        * ``model: str | dict`` — required (or pass ``model=`` kwarg).
          Dict form: ``{"name": "...", "effort": "...",
          "strict_effort": bool}``.
        * ``memory: str | dict`` — string spec ("sqlite:./bot.db",
          "postgres://...") or a ``{"backend": ..., ...}`` dict.
        * ``runtime: str | dict`` — "inproc" / "sqlite:./j.db" or
          ``{"backend": "sqlite", "path": "..."}``.
        * ``telemetry: str | dict`` — "none" / "console" / "memory" /
          "file:./spans.jsonl" / "otel" or matching dict form.
        * ``audit_log: str | dict`` — file path or
          ``{"name": "file", "path": "...", "scope_full": bool,
          "secret": "..."}``.
        * ``permissions: str | dict`` — "allow_all" / "strict" /
          "accept_edits" / "bypass" or
          ``{"backend": "standard", "mode": ..., "allowed_tools":
          [...], "denied_tools": [...]}``.
        * ``architecture: str`` — "react" / "reflexion" / etc.
        * ``effort: str``, ``strict_effort: bool``,
          ``response_tone: str``.
        * ``max_turns: int``, ``auto_consolidate: bool``,
          ``auto_extract: bool``.
        * ``budget: dict`` with any of ``max_tokens``,
          ``max_input_tokens``, ``max_output_tokens``, ``max_cost_usd``,
          ``max_wall_clock_minutes``, ``soft_warning_at``.
        * ``skills: list`` — strings (directory paths) or dicts
          ({"path": "...", "label": "..."}).
        * ``mcp: list[dict]`` — each entry has ``name`` + ``transport``
          ("stdio" with ``command`` / ``args`` / ``env`` OR "http"
          with ``url`` / ``headers``). Wrapped in an
          :class:`~loomflow.mcp.MCPRegistry` and threaded into
          ``tools=`` (unless a tool host is also supplied via the
          kwarg, in which case ``mcp`` is rejected).
        """
        from datetime import timedelta

        from ..core.errors import ConfigError
        from ..governance.budget import BudgetConfig, StandardBudget
        from ..mcp import MCPRegistry, MCPServerSpec
        from ..skills import SkillSource

        instructions = cfg.get("instructions")
        if not isinstance(instructions, str):
            raise ConfigError(
                "Agent.from_dict: missing or non-string 'instructions' field"
            )

        # ----- model (kwarg overrides cfg) ----------------------------
        model_spec = model if model is not None else cfg.get("model")
        if model_spec is None:
            raise ConfigError(
                "Agent.from_dict: missing 'model' field. Add e.g. "
                "model = 'claude-opus-4-7' (or pass model= explicitly)."
            )

        # ----- scalar agent-level toggles -----------------------------
        max_turns = cfg.get("max_turns", DEFAULT_MAX_TURNS)
        auto_consolidate = bool(cfg.get("auto_consolidate", False))
        architecture = cfg.get("architecture")
        effort = cfg.get("effort")
        strict_effort = bool(cfg.get("strict_effort", False))
        response_tone = cfg.get("response_tone")
        auto_extract = cfg.get("auto_extract")

        # ----- backend kwargs (kwarg wins, otherwise cfg) -------------
        memory_arg = memory if memory is not None else cfg.get("memory")
        runtime_arg = runtime if runtime is not None else cfg.get("runtime")
        telemetry_arg = (
            telemetry if telemetry is not None else cfg.get("telemetry")
        )
        audit_log_arg = (
            audit_log if audit_log is not None else cfg.get("audit_log")
        )
        permissions_arg = (
            permissions if permissions is not None else cfg.get("permissions")
        )
        # Run the permissions string/dict through the resolver here so
        # the Agent constructor (which expects ``Permissions | None``)
        # gets a concrete instance.
        from ..security.permissions_resolver import resolve_permissions
        permissions_obj = (
            resolve_permissions(permissions_arg)
            if permissions_arg is not None
            else None
        )

        # ----- budget (scalar dict) -----------------------------------
        budget: Budget | None = None
        if "budget" in cfg:
            b = cfg["budget"]
            wall_clock = None
            if "max_wall_clock_minutes" in b:
                wall_clock = timedelta(minutes=float(b["max_wall_clock_minutes"]))
            budget = StandardBudget(
                BudgetConfig(
                    max_tokens=b.get("max_tokens"),
                    max_input_tokens=b.get("max_input_tokens"),
                    max_output_tokens=b.get("max_output_tokens"),
                    max_cost_usd=b.get("max_cost_usd"),
                    max_wall_clock=wall_clock,
                    soft_warning_at=float(b.get("soft_warning_at", 0.8)),
                )
            )

        # ----- skills (list of paths / dicts) -------------------------
        skill_specs: list[Any] = []
        for raw in cfg.get("skills", []) or []:
            if isinstance(raw, str):
                skill_specs.append(raw)
            elif isinstance(raw, Mapping):
                path = raw.get("path")
                if not isinstance(path, str):
                    raise ConfigError(
                        "Agent.from_dict: skills[*] dicts must include a "
                        "string 'path'."
                    )
                label = raw.get("label")
                # Path strings → SkillSource via coerce so the Path
                # gets ``expanduser``'d and validated against the
                # filesystem the same way the bare-string form is.
                if label is None:
                    skill_specs.append(SkillSource.coerce(path))
                else:
                    skill_specs.append(SkillSource.coerce((path, label)))
            else:
                raise ConfigError(
                    "Agent.from_dict: skills entries must be strings or "
                    f"dicts; got {type(raw).__name__}."
                )

        # ----- mcp (list of server dicts → MCPRegistry) ---------------
        mcp_entries = cfg.get("mcp") or []
        if mcp_entries:
            if tools is not None:
                raise ConfigError(
                    "Agent.from_dict: cannot mix 'mcp' config with a "
                    "tools= kwarg. Either drop one or build a combined "
                    "ToolHost yourself (e.g. ExtendedToolHost over both)."
                )
            specs: list[MCPServerSpec] = []
            for entry in mcp_entries:
                if not isinstance(entry, Mapping):
                    raise ConfigError(
                        "Agent.from_dict: mcp[*] entries must be dicts."
                    )
                specs.append(_mcp_spec_from_dict(entry))
            tools = MCPRegistry(list(specs))  # type: ignore[assignment]

        # ``living_plan`` accepts the same bool / str / dict / instance
        # forms as the constructor kwarg. The resolver inside
        # ``Agent.__init__`` does the validation.
        living_plan_arg: Any = cfg.get("living_plan")

        return cls(
            instructions,
            model=model_spec,
            memory=memory_arg,
            runtime=runtime_arg,
            telemetry=telemetry_arg,
            audit_log=audit_log_arg,
            permissions=permissions_obj,
            tools=tools,
            budget=budget,
            architecture=architecture,
            effort=effort,
            strict_effort=strict_effort,
            response_tone=response_tone,
            max_turns=int(max_turns),
            auto_consolidate=auto_consolidate,
            auto_extract=auto_extract,
            skills=skill_specs or None,
            secrets=secrets,
            hooks=hooks,
            retry_policy=retry_policy,
            approval_handler=approval_handler,
            living_plan=living_plan_arg,
        )

    @classmethod
    def from_config(
        cls,
        path: str | Path,
        *,
        model: Model | str | dict[str, Any] | None = None,
        memory: Memory | str | Mapping[str, Any] | None = None,
        runtime: Runtime | str | Mapping[str, Any] | None = None,
        telemetry: Telemetry | str | Mapping[str, Any] | None = None,
        audit_log: AuditLog | str | Path | dict[str, Any] | None = None,
        permissions: Permissions | str | Mapping[str, Any] | None = None,
        tools: list[Tool | Callable[..., object]] | ToolHost | None = None,
        secrets: Any | None = None,
        hooks: HookRegistry | None = None,
        retry_policy: RetryPolicy | None = None,
        approval_handler: Any | None = None,
    ) -> Agent:
        """Construct an ``Agent`` from a TOML config file.

        Designed for ops/devops users who want declarative agent
        config separate from code. The TOML covers every backend the
        framework can build sync — model, memory, runtime, telemetry,
        audit log, permissions, budget, architecture, effort, skills,
        and MCP servers. Things TOML can't naturally express (real
        callables, custom hook objects, secret stores, retry policies)
        come in through kwargs that override matching cfg entries.

        Example ``agent.toml`` covering most options::

            instructions = "You are a research assistant."
            model = "claude-opus-4-7"
            max_turns = 100
            architecture = "react"
            auto_consolidate = true
            effort = "medium"

            [memory]
            backend = "sqlite"
            path = "./memory.db"

            [runtime]
            backend = "sqlite"
            path = "./journal.db"

            [telemetry]
            backend = "file"
            path = "./spans.jsonl"

            [audit_log]
            name = "file"
            path = "./audit.jsonl"
            scope_full = true

            [permissions]
            backend = "standard"
            mode = "default"
            denied_tools = ["bash"]

            [budget]
            max_tokens = 200_000
            max_cost_usd = 5.0
            max_wall_clock_minutes = 10

            [[skills]]
            path = "./skills/research"

            [[mcp]]
            name = "git"
            transport = "stdio"
            command = "uvx"
            args = ["mcp-server-git", "--repo", "."]

        Then::

            agent = Agent.from_config("agent.toml")
        """
        from ..core.errors import ConfigError

        try:
            import tomllib  # py3.11+
        except ImportError as exc:  # pragma: no cover — should never hit on 3.11+
            raise ConfigError(
                "tomllib is required (Python 3.11+)."
            ) from exc

        with Path(path).open("rb") as fh:
            cfg = tomllib.load(fh)

        try:
            return cls.from_dict(
                cfg,
                model=model,
                memory=memory,
                runtime=runtime,
                telemetry=telemetry,
                audit_log=audit_log,
                permissions=permissions,
                tools=tools,
                secrets=secrets,
                hooks=hooks,
                retry_policy=retry_policy,
                approval_handler=approval_handler,
            )
        except ConfigError as exc:
            # Re-raise with the file path in the message so users know
            # which TOML produced the error.
            raise ConfigError(f"{path}: {exc}") from None

    # ---- public API ------------------------------------------------------

    async def run(
        self,
        prompt: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        context: RunContext | None = None,
        extra_tools: list[Tool] | None = None,
        emit: Emit | None = None,
        output_schema: type[BaseModel] | Any | None = None,
        output_validation_retries: int = 1,
        response_tone: str | None = None,
        effort: str | None = None,
    ) -> RunResult:
        """Run the agent to completion and return its :class:`RunResult`.

        ``user_id`` is the namespace partition for memory recall and
        persistence — episodes and facts stored with one ``user_id``
        are never visible to a query scoped to a different ``user_id``.
        ``None`` is the "anonymous / single-tenant" bucket. See
        :class:`~loomflow.RunContext` for the partitioning
        contract.

        Pass ``session_id`` to resume a journaled run — when paired with
        a durable runtime (e.g. :class:`SqliteRuntime`), already-completed
        steps replay from the journal instead of re-executing. Without a
        durable runtime, ``session_id`` just labels the run.

        ``metadata`` is a free-form bag for application context the
        framework does not interpret (locale, request id, feature
        flags). Tools and hooks read it via
        ``get_run_context().metadata``.

        ``context`` accepts a fully-formed :class:`RunContext` instead
        of the individual kwargs — useful when passing context through
        multi-agent boundaries that received their parent's context as
        a single object. When both ``context`` and the individual
        kwargs are provided, the kwargs override the corresponding
        fields on ``context``.

        ``extra_tools`` injects additional :class:`Tool`\\ s for this
        run only — the agent's configured ``ToolHost`` is wrapped so
        the model sees the extras alongside whatever tools were
        registered at construction. Used by multi-agent architectures
        that need to inject coordination tools (e.g. Swarm's
        ``handoff(target, message)``) into a peer agent's loop without
        permanently mutating that agent's static configuration.

        ``emit`` is an awaitable callback invoked once per
        :class:`Event` produced during the run (model chunks, tool
        calls, tool results, architecture progress, errors, ...).
        Default ``None`` drops events on the floor (regular ``run``
        semantics — return only the final ``RunResult``). Multi-agent
        architectures pass an emit that forwards a sub-Agent's events
        into the parent's stream, so calls like ``await
        worker.run(prompt, emit=parent_send)`` surface the worker's
        token-by-token streaming to the outermost ``agent.stream(...)``
        consumer.

        ``output_schema`` requests a structured, validated final
        answer. Pass any Pydantic ``BaseModel`` subclass and the
        framework will (1) append a JSON-schema directive to the
        system prompt instructing the model to emit a final answer
        that matches, (2) parse the final assistant text against the
        schema, and (3) populate :attr:`RunResult.parsed` with the
        validated instance. ``RunResult.output`` keeps the raw text
        so you can log or display it. Up to
        ``output_validation_retries`` extra turns are spent
        recovering from a parse failure (the model is given the
        validation error as feedback and asked to try again); if it
        still fails after the retry budget, the run raises
        :class:`~loomflow.OutputValidationError`. Set retries to
        0 to fail fast.
        """
        return await self._loop(
            prompt,
            emit=emit if emit is not None else _noop_emit,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            context=context,
            extra_tools=extra_tools,
            # Per-call schema wins; fall back to the agent's default
            # so ``Agent(output_schema=Receipt)`` doesn't need to be
            # repeated on every ``run()``.
            output_schema=(
                output_schema
                if output_schema is not None
                else self._default_output_schema
            ),
            output_validation_retries=output_validation_retries,
            response_tone=response_tone,
            effort=effort if effort is not None else self._default_effort,
        )

    async def resume(
        self,
        session_id: str,
        prompt: str,
        *,
        user_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        context: RunContext | None = None,
        extra_tools: list[Tool] | None = None,
        emit: Emit | None = None,
        output_schema: type[BaseModel] | Any | None = None,
        output_validation_retries: int = 1,
    ) -> RunResult:
        """Resume a previously-interrupted run from its journal.

        Equivalent to ``agent.run(prompt, session_id=session_id, ...)``
        with the same kwarg surface as :meth:`run`. Exists as a
        separate method so the intent is explicit at the call site
        — when a durable :class:`Runtime` (e.g. :class:`SqliteRuntime`)
        is configured, completed steps replay from the journal
        instead of re-executing.
        """
        return await self.run(
            prompt,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            context=context,
            extra_tools=extra_tools,
            emit=emit,
            output_schema=output_schema,
            output_validation_retries=output_validation_retries,
        )

    async def stream(
        self,
        prompt: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        context: RunContext | None = None,
        extra_tools: list[Tool] | None = None,
        output_schema: type[BaseModel] | Any | None = None,
        output_validation_retries: int = 1,
        response_tone: str | None = None,
        effort: str | None = None,
    ) -> AsyncIterator[Event]:
        """Stream :class:`Event`\\ s as the loop produces them.

        The loop runs as a background task; events are pushed through a
        bounded memory stream so a slow consumer applies backpressure.
        Breaking out of the iteration cancels the producer cleanly.
        ``session_id`` works the same as :meth:`run`'s — pass an
        existing one to resume against a durable runtime's journal.
        ``extra_tools`` works the same as :meth:`run`'s.
        """
        send, receive = anyio.create_memory_object_stream[Event](
            max_buffer_size=DEFAULT_STREAM_BUFFER
        )

        # Same fallback as ``run``: per-call schema wins; agent's
        # default is used otherwise.
        effective_schema = (
            output_schema
            if output_schema is not None
            else self._default_output_schema
        )
        effective_effort = (
            effort if effort is not None else self._default_effort
        )

        async def _produce() -> None:
            try:
                await self._loop(
                    prompt,
                    emit=send.send,
                    user_id=user_id,
                    session_id=session_id,
                    metadata=metadata,
                    context=context,
                    extra_tools=extra_tools,
                    output_schema=effective_schema,
                    output_validation_retries=output_validation_retries,
                    response_tone=response_tone,
                    effort=effective_effort,
                )
            except Exception as exc:  # noqa: BLE001 — surface as ERROR + re-raise
                with anyio.CancelScope(shield=True):
                    await send.send(Event.error("", exc))
                raise
            finally:
                send.close()

        async with anyio.create_task_group() as tg:
            tg.start_soon(_produce)
            try:
                async with receive:
                    async for event in receive:
                        yield event
            finally:
                tg.cancel_scope.cancel()

    # ---- the loop --------------------------------------------------------

    async def _loop(
        self,
        prompt: str,
        *,
        emit: Emit,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        context: RunContext | None = None,
        extra_tools: list[Tool] | None = None,
        output_schema: type[BaseModel] | Any | None = None,
        output_validation_retries: int = 1,
        response_tone: str | None = None,
        effort: str | None = None,
    ) -> RunResult:
        """Setup → delegate iteration to the architecture → teardown.

        The architecture (default :class:`ReAct`) drives the iteration
        and yields events as it goes. Setup wraps the run in a runtime
        session + telemetry trace so every ``runtime.step`` recorded
        from inside the architecture lands on the same journal entry,
        and installs a :class:`~loomflow.RunContext` in a
        contextvar so tools / hooks / sub-agents see ``user_id``,
        ``session_id``, ``run_id``, and ``metadata`` without having
        to thread them through every signature. Teardown persists
        the episode (tagged with ``user_id`` for namespace
        partitioning), builds the :class:`RunResult`, emits final
        metrics, and triggers ``auto_consolidate``.
        """
        started_at = datetime.now(UTC)
        run_id = new_id("run")
        loop_started = anyio.current_time()

        # Resolve scope from kwargs + optional ``context``: kwargs
        # win when explicitly supplied; otherwise fall back to
        # ``context``'s value; otherwise the framework default
        # (auto-generated session_id; None for everything else).
        # ``run_id`` is always framework-assigned — caller-supplied
        # values on ``context.run_id`` are overridden because each
        # ``Agent.run`` invocation is its own run.
        ctx_user_id = (
            user_id if user_id is not None
            else (context.user_id if context is not None else None)
        )
        ctx_session_id = (
            session_id if session_id is not None
            else (context.session_id if context is not None else None)
        )
        if ctx_session_id is None:
            ctx_session_id = new_id("sess")
        # ``session_id`` is the public-facing id used for
        # journal/audit/telemetry; mirror it back so the rest of
        # ``_loop`` (which references it directly) stays consistent.
        session_id = ctx_session_id
        ctx_metadata: Mapping[str, Any] = (
            metadata if metadata is not None
            else (context.metadata if context is not None else {})
        )
        run_ctx = RunContext(
            user_id=ctx_user_id,
            session_id=ctx_session_id,
            run_id=run_id,
            metadata=ctx_metadata,
        )

        # Fast-mode flags — auto-detect "no-op default" implementations
        # so hot-path call sites can skip the integration layer
        # entirely. The moment a user wires up a real audit log /
        # telemetry exporter / permission policy / etc., the
        # corresponding flag flips False and the integration becomes
        # active. See ``Dependencies`` for the field-level docstrings
        # and the README "Fast path by default" section for the user-
        # facing story.
        fast_audit = self._audit_log is None
        fast_telemetry = isinstance(self._telemetry, NoTelemetry)
        fast_permissions = isinstance(self._permissions, AllowAll)
        fast_hooks = (
            len(self._hooks.pre_tool_hooks) == 0
            and len(self._hooks.post_tool_hooks) == 0
        )
        fast_runtime = isinstance(self._runtime, InProcRuntime)
        fast_budget = isinstance(self._budget, NoBudget)
        # No registered stop hooks → skip the Ralph-loop wrapper
        # entirely so the no-op default keeps the LangChain-class
        # latency promise. Hooks are mostly used for living_plan
        # auto-registration and bespoke consumer-side checks.
        fast_stop_hooks = len(self._stop_hooks) == 0
        # ``fast_tool_summary`` mirrors the same pattern: True when
        # no summariser model is wired (the default), so the ReAct
        # tool-dispatch loop can short-circuit the summarisation
        # call site and ship tool results verbatim with no extra
        # allocation. Flips False only when the user passed
        # ``tool_result_summarizer=`` at Agent construction.
        fast_tool_summary = self._tool_result_summarizer is None
        # ``fast_snip`` mirrors the same pattern. Snip is the
        # cheap context-budget defence; when disabled (window=0,
        # the default) the architecture skips the snip pass
        # entirely.
        fast_snip = self._snip_window <= 0

        run_trace: contextlib.AbstractAsyncContextManager[Any] = (
            _NULL_CTX
            if fast_telemetry
            else self._telemetry.trace(
                "loom.run",
                session_id=session_id,
                max_turns=self._max_turns,
                model=self._model.name,
                architecture=self._architecture.name,
            )
        )

        # When this Agent owns a workspace, also install it as the
        # ambient for the duration of the run. Nested Agents (sub-
        # agents spawned by a Supervisor / Swarm / Blackboard / Team
        # builder) that didn't get their own ``workspace=`` at
        # construction will pick it up via ``_ambient_workspace_var``
        # in their own ``_loop``, so one Team workspace cascades
        # automatically through every worker. We reset the token in
        # a ``finally`` below so the contextvar doesn't leak past
        # this run.
        from ..core.context import (
            _ambient_citations_var,
            _ambient_living_plan_var,
            _ambient_workspace_var,
        )

        ws_token: Any = (
            _ambient_workspace_var.set(self._workspace)
            if self._workspace is not None
            else None
        )

        # Install a fresh per-run citation set when a workspace is
        # wired. ``Workspace.read_note`` / ``read_version`` log
        # slugs into this set; ``Workspace.attribute_outcome`` (called
        # by user code after the run) drains it to update per-note
        # relevance metadata. Without an active set (no workspace),
        # the helpers no-op.
        citation_token: Any = (
            _ambient_citations_var.set(set())
            if self._workspace is not None
            else None
        )

        # Install a fresh per-run living-plan state for the duration
        # of this run when enabled. The tools registered at __init__
        # read this state via the contextvar at call time, so
        # concurrent runs of the same Agent see isolated plans.
        # Pre-seed from the construction-time spec when supplied
        # (``Agent(living_plan=LivingPlan(...))``).
        plan_token: Any = None
        if self._living_plan_spec.enabled:
            from ..tools.plan import LivingPlan, _LivingPlanState
            seed = self._living_plan_spec.seed_plan
            plan_state = _LivingPlanState(
                plan=seed if seed is not None else LivingPlan(),
            )
            plan_token = _ambient_living_plan_var.set(plan_state)

        async with (
            self._runtime.session(session_id),
            run_trace,
            set_run_context(run_ctx),
        ):
            if not fast_audit:
                # ``FullTranscriptAuditLog`` opts into verbatim
                # prompt capture; default audit truncates to 500
                # chars to avoid logging customer PII by accident.
                from ..security.audit import wants_full_transcripts
                full_audit = wants_full_transcripts(self._audit_log)
                await self._audit(
                    session_id=session_id,
                    user_id=run_ctx.user_id,
                    actor="user",
                    action="run_started",
                    payload={
                        "prompt": prompt if full_audit else prompt[:500],
                        "model": self._model.name,
                        "max_turns": self._max_turns,
                        "architecture": self._architecture.name,
                    },
                )
            await emit(Event.started(session_id, prompt))

            # Append the JSON-schema directive when a structured
            # output is requested AND the model adapter doesn't have
            # a provider-native structured-output API. Adapters
            # flagged ``supports_native_structured_output = True``
            # (OpenAI, Anthropic, LiteLLM-passthrough) translate the
            # schema into a decode-time constraint, so duplicating
            # the schema as text in the system prompt is just dead
            # tokens. Saves ~2k input tokens per structured-output
            # call. Validation-retry still injects the schema into
            # the retry message if the model ever produces invalid
            # JSON, so reliability is preserved.
            #
            # We check ``self._model`` (the raw adapter) rather than
            # ``_wrapped_model`` (which adds retry decoration) — the
            # flag is an adapter-level capability, not a wrapper one.
            native_structured = (
                output_schema is not None
                and getattr(
                    self._model,
                    "supports_native_structured_output",
                    False,
                )
            )
            effective_instructions = (
                _augment_instructions_for_schema(
                    self._instructions, output_schema
                )
                if output_schema is not None and not native_structured
                else self._instructions
            )

            # Resolve the effective response tone for THIS run:
            # per-call wins, then agent default, then workflow
            # ambient (set by ``Workflow(response_tone=...)``), then
            # None. When None, ``append_tone_directive`` is a no-op.
            from ..core.context import _ambient_response_tone_var
            from ..core.tone import append_tone_directive

            effective_tone = (
                response_tone
                if response_tone is not None
                else (
                    self._default_response_tone
                    if self._default_response_tone is not None
                    else _ambient_response_tone_var.get()
                )
            )
            effective_instructions = append_tone_directive(
                effective_instructions, effective_tone
            )

            session = AgentSession(
                id=session_id,
                instructions=effective_instructions,
            )
            # Per-run tool injection: if extra_tools provided, wrap
            # the agent's host so the model sees them alongside the
            # statically-configured tools. The wrap is local to this
            # run; the agent's _tool_host is unchanged.
            per_run_extras: list[Tool] = list(extra_tools or [])

            # Ambient workspace inheritance: when this Agent didn't
            # get its own ``workspace=`` at construction but a parent
            # ``Workflow(workspace=...)`` is active, materialise the
            # five notebook tools for THIS run only. The agent's
            # ``_tool_host`` stays unchanged; the workspace prompt
            # nudges happen via ``_default_instructions_with_ambient``
            # below.
            ambient_ws_tools: list[Tool] = []
            ambient_ws_section: str = ""
            if not self._workspace_was_explicit:
                from ..core.context import _ambient_workspace_var
                from ..workspace.tools import make_workspace_tools as _make_ws_tools
                from ..workspace.tools import workspace_prompt_section as _ws_prompt

                amb_ws = _ambient_workspace_var.get()
                if amb_ws is not None:
                    author = self._workspace_name or "agent"
                    ambient_ws_tools = _make_ws_tools(amb_ws, author=author)
                    ambient_ws_section = _ws_prompt(
                        author=author,
                        teammates=self._workspace_teammates,
                    )
                    per_run_extras.extend(ambient_ws_tools)
            if ambient_ws_section:
                # Same pattern as the skills catalog + the agent-level
                # workspace section: append the nudges after the user
                # instructions so the agent sees them at every turn.
                effective_instructions = (
                    f"{effective_instructions.rstrip()}\n\n{ambient_ws_section}"
                )

            effective_tools = (
                ExtendedToolHost(self._tool_host, per_run_extras)
                if per_run_extras
                else self._tool_host
            )
            # Resolve the memory for THIS run: explicit-on-Agent
            # always wins; otherwise check the ambient workflow
            # memory contextvar so ``Workflow(memory=...)`` flows
            # through to nested agents that left ``memory=`` blank.
            # See :func:`_resolve_run_memory` below for full rules.
            effective_memory = self._resolve_run_memory()
            deps = Dependencies(
                # ``_wrapped_model`` is the retry-decorated view —
                # falls through to ``_model`` when the policy is
                # disabled, so the architecture loop never sees a
                # raw SDK exception when retries could have helped.
                model=self._wrapped_model,
                # See ``_resolve_run_memory``: this is either
                # ``_wrapped_memory`` (the per-Agent default with
                # auto-extract wrapping) or, when the user left
                # ``memory=`` unset on this Agent and a parent
                # ``Workflow(memory=...)`` is active, the workflow's
                # shared memory.
                memory=effective_memory,
                runtime=self._runtime,
                tools=effective_tools,
                budget=self._budget,
                permissions=self._permissions,
                hooks=self._hooks,
                telemetry=self._telemetry,
                audit_log=self._audit_log,
                max_turns=self._max_turns,
                approval_handler=self._approval_handler,
                # Forward the per-call output_schema so architectures
                # can hand it to model.complete()/.stream(); adapters
                # with native structured-output APIs use it to
                # constrain the model and skip the validation retry.
                output_schema=output_schema,
                # Reasoning-effort dial; architectures forward to
                # model.complete()/.stream(), where each adapter
                # translates into its provider's native shape.
                # strict_effort is agent-level; effort is per-call.
                effort=effort,
                strict_effort=self._strict_effort,
                # Prompt cache config — Anthropic adapters use it to
                # inject ``cache_control`` markers; OpenAI uses the
                # optional ``cache_key`` as a routing hint. Carries
                # ``enabled=False`` when caching is off, so adapters
                # check ``deps.prompt_caching.enabled`` and skip.
                prompt_caching=self._prompt_caching,
                # Architectures consult this to pick fast (buffered)
                # vs streaming (channel) paths for parallel work.
                streaming=emit is not _noop_emit,
                fast_audit=fast_audit,
                fast_telemetry=fast_telemetry,
                fast_permissions=fast_permissions,
                fast_hooks=fast_hooks,
                fast_runtime=fast_runtime,
                fast_budget=fast_budget,
                fast_stop_hooks=fast_stop_hooks,
                fast_tool_summary=fast_tool_summary,
                tool_result_summarizer=self._tool_result_summarizer,
                tool_result_summary_threshold=(
                    self._tool_result_summary_threshold
                ),
                fast_snip=fast_snip,
                snip_window=self._snip_window,
                context=run_ctx,
            )

            # First architecture pass — same as before.
            async for event in self._architecture.run(session, deps, prompt):
                await emit(event)

            # Framework Ralph loop — when stop hooks are registered,
            # run them after the architecture exits. Any hook
            # returning a StopHookResult triggers a re-invocation
            # of architecture.run() with the inject_message as a
            # fresh user prompt. Bounded by
            # ``self._max_stop_hook_iterations`` so a flaky hook
            # can't burn budget unbounded.
            #
            # First-non-None wins per iteration; remaining hooks
            # for that iteration are skipped. Re-uses the same
            # session (messages carry forward as conversation
            # history) and the same deps (already includes
            # ``context``, ``fast_*`` flags, the wrapped model +
            # memory).
            if not fast_stop_hooks:
                iter_count = 0
                while iter_count < self._max_stop_hook_iterations:
                    hook_result: StopHookResult | None = None
                    fired_hook_name: str = ""
                    for hook in self._stop_hooks:
                        hook_result = await hook(
                            session, deps, iteration=iter_count
                        )
                        if hook_result is not None:
                            fired_hook_name = getattr(
                                hook, "name", type(hook).__name__
                            )
                            break
                    if hook_result is None:
                        # All hooks voted "stop" — natural exit.
                        break
                    iter_count += 1
                    await emit(
                        Event.architecture_event(
                            session_id,
                            "stop_hook.fired",
                            payload={
                                "hook": fired_hook_name,
                                "iteration": iter_count,
                                "reason": hook_result.reason,
                            },
                        )
                    )

                    # Auto-compact between Ralph iterations.
                    # Counts tokens in session.messages; if past
                    # threshold, summarises older turns into a
                    # single system message + keeps the last N
                    # turn groups verbatim. Pure no-op when not
                    # opted in (``auto_compact_at_tokens=None``).
                    # Failures are graceful — the compactor
                    # NEVER kills a turn.
                    if (
                        self._auto_compact_at_tokens is not None
                        and self._auto_compact_summariser is not None
                    ):
                        from ..model.count_tokens import count_tokens
                        from .auto_compact import maybe_auto_compact
                        try:
                            current = await count_tokens(
                                self._model, session.messages
                            )
                        except Exception:  # noqa: BLE001
                            current = 0
                        if current > self._auto_compact_at_tokens:
                            new_msgs, summary = await maybe_auto_compact(
                                session.messages,
                                summariser=self._auto_compact_summariser,
                                at_tokens=self._auto_compact_at_tokens,
                                current_token_count=current,
                                keep_recent_turns=(
                                    self._auto_compact_keep_recent_turns
                                ),
                            )
                            if new_msgs is not None:
                                dropped = (
                                    len(session.messages)
                                    - len(new_msgs)
                                )
                                session.messages = new_msgs
                                await emit(
                                    Event.architecture_event(
                                        session_id,
                                        "auto_compacted",
                                        payload={
                                            "tokens_before": current,
                                            "messages_before": (
                                                len(session.messages)
                                                + dropped
                                            ),
                                            "messages_after": (
                                                len(session.messages)
                                            ),
                                            "messages_dropped": dropped,
                                            "summary_chars": (
                                                len(summary)
                                            ),
                                        },
                                    )
                                )

                    async for event in self._architecture.run(
                        session, deps, hook_result.inject_message
                    ):
                        await emit(event)
                else:
                    # Loop exhausted via the cap — record the
                    # interruption on the session so RunResult
                    # surfaces it back to the caller.
                    session.interrupted = True
                    session.interruption_reason = (
                        "stop_hook_iterations_exhausted"
                    )
                    await emit(
                        Event.architecture_event(
                            session_id,
                            "stop_hook.exhausted",
                            payload={
                                "iterations": iter_count,
                                "limit": self._max_stop_hook_iterations,
                            },
                        )
                    )

            # Structured output validation + retry. Only kicks in
            # when the caller requested a schema; happens AFTER the
            # architecture loop has finalised ``session.output`` and
            # BEFORE we persist the episode (so the persisted text
            # is the validated one). Up to
            # ``output_validation_retries`` extra single-turn model
            # calls are spent fixing the output; on the last
            # failure :class:`OutputValidationError` is raised.
            parsed: Any | None = None
            if output_schema is not None:
                parsed = await self._validate_output_with_retry(
                    session=session,
                    schema=output_schema,
                    retries=output_validation_retries,
                    deps=deps,
                )

            episode = Episode(
                session_id=session_id,
                user_id=run_ctx.user_id,
                input=prompt,
                output=session.output,
            )
            if fast_runtime:
                await effective_memory.remember(episode)
            else:
                await self._runtime.step(
                    f"persist_episode_{session.turns}",
                    effective_memory.remember,
                    episode,
                )

            # Snapshot the per-run citation set BEFORE the
            # contextvar is reset below. Without this, calling
            # ``Workspace.attribute_outcome`` after ``run()``
            # returns would find an empty set (the contextvar is
            # already reset) — the self-improvement loop would be
            # a silent no-op. Carrying the slugs on the RunResult
            # lets the caller drive ``attribute_outcome`` with an
            # explicit ``slugs=`` after the run.
            _cited = _ambient_citations_var.get()
            cited_slugs = (
                sorted(_cited) if isinstance(_cited, set) else []
            )

            result = RunResult(
                session_id=session_id,
                output=session.output,
                parsed=parsed,
                turns=session.turns,
                tokens_in=session.cumulative_usage.input_tokens,
                cached_tokens_in=session.cumulative_usage.cached_input_tokens,
                cache_write_tokens=session.cumulative_usage.cache_write_tokens,
                tokens_out=session.cumulative_usage.output_tokens,
                cost_usd=session.cumulative_usage.cost_usd,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                interrupted=session.interrupted,
                interruption_reason=session.interruption_reason,
                cited_slugs=cited_slugs,
            )

            elapsed_ms = (anyio.current_time() - loop_started) * 1000
            if not fast_telemetry:
                await self._telemetry.emit_metric(
                    "loom.session.duration_ms",
                    elapsed_ms,
                    session_id=session_id,
                    interrupted=session.interrupted,
                    turns=session.turns,
                )

            # Auto-consolidate runs after the response is finalized but
            # before the COMPLETED event so observers see it as part of
            # the same run. Failures surface as ERROR events but never
            # break the run — consolidation is best-effort.
            if self._auto_consolidate:
                try:
                    await self._memory.consolidate()
                except Exception as exc:  # noqa: BLE001
                    await emit(Event.error(session_id, exc))

            if not fast_audit:
                from ..security.audit import wants_full_transcripts
                completion_payload: dict[str, Any] = {
                    "turns": session.turns,
                    "interrupted": session.interrupted,
                    "interruption_reason": session.interruption_reason,
                    "tokens_in": session.cumulative_usage.input_tokens,
                    "tokens_out": session.cumulative_usage.output_tokens,
                    "cost_usd": session.cumulative_usage.cost_usd,
                    "elapsed_ms": elapsed_ms,
                }
                # When ``FullTranscriptAuditLog`` wraps the log,
                # include the final model output so the entry is
                # self-contained for replay / investigation. The
                # ``parsed`` field is serialised via ``model_dump``
                # when it's a Pydantic instance; raw types pass
                # through as-is.
                if wants_full_transcripts(self._audit_log):
                    completion_payload["output"] = session.output
                    if parsed is not None:
                        completion_payload["parsed"] = (
                            parsed.model_dump(mode="json")
                            if isinstance(parsed, BaseModel)
                            else parsed
                        )
                await self._audit(
                    session_id=session_id,
                    user_id=run_ctx.user_id,
                    actor="system",
                    action="run_completed",
                    payload=completion_payload,
                )
            await emit(Event.completed(session_id, result.model_dump(mode="json")))
            # Reset the workspace contextvar set above so the
            # ambient doesn't leak past this run. We do it inside
            # the ``async with`` so it fires before the run-context
            # teardown — keeps the cleanup order symmetric with the
            # set-up order above.
            if ws_token is not None:
                _ambient_workspace_var.reset(ws_token)
            if plan_token is not None:
                _ambient_living_plan_var.reset(plan_token)
            if citation_token is not None:
                _ambient_citations_var.reset(citation_token)
            return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _noop_emit(_event: Event) -> None:
    return None


_NETWORK_MODEL_CLASS_NAMES = frozenset(
    {"OpenAIModel", "AnthropicModel", "LiteLLMModel"}
)


def _default_auto_extract_for(model: Model) -> bool:
    """Auto-pick the default for ``auto_extract`` based on the model.

    On for in-tree network adapters (OpenAI / Anthropic / LiteLLM):
    extraction is what makes "your bot just remembers" work. Off
    for in-process fakes (ScriptedModel / EchoModel) and unrecognised
    custom Models: their canned responses confuse the Consolidator
    and would make tests non-deterministic. Custom-model users opt
    in by passing ``auto_extract=True`` explicitly.
    """
    return type(model).__name__ in _NETWORK_MODEL_CLASS_NAMES


def _default_retry_policy_for(model: Model) -> RetryPolicy:
    """Pick a sensible default retry policy based on the model type.

    The framework wraps **only** the in-tree network adapters
    (``OpenAIModel`` / ``AnthropicModel`` / ``LiteLLMModel``) by
    default — those are the ones whose call sites can hit transient
    failures (5xx, rate limit, network blip). Everything else
    (in-process fakes, custom user-supplied :class:`Model`
    implementations, test mocks) is left unwrapped: the framework
    cannot assume a custom Model's exception classes match our
    classifier, and silently retrying its calls could mask real
    bugs. Callers who DO want retries on a custom model pass
    ``retry_policy=RetryPolicy()`` (or any enabled policy)
    explicitly to opt in.
    """
    if type(model).__name__ in _NETWORK_MODEL_CLASS_NAMES:
        return RetryPolicy()
    return RetryPolicy.disabled()


# ---------------------------------------------------------------------------
# Structured-output helpers
# ---------------------------------------------------------------------------


_SCHEMA_DIRECTIVE_TEMPLATE = """

---
**STRUCTURED OUTPUT REQUIRED.**
Your final answer must be a single valid JSON object that conforms
exactly to this JSON Schema:

```json
{schema_json}
```

Return ONLY the JSON object — no surrounding prose, no markdown
fences, no explanation. The receiver will validate your response
with Pydantic and reject any extra text.
""".rstrip()


def _augment_instructions_for_schema(
    base_instructions: str, schema: Any
) -> str:
    """Build the run-scoped system prompt when ``output_schema`` is
    requested. Appends a clear, schema-specific directive to the
    agent's static instructions so the model knows it must emit JSON
    matching the schema. Tagged unions (``A | B``) become a single
    ``anyOf`` schema so the model sees all valid shapes at once."""
    types = _extract_schema_types(schema)
    schema_json = json.dumps(
        _union_json_schema(types), indent=2, sort_keys=True
    )
    return base_instructions.rstrip() + _SCHEMA_DIRECTIVE_TEMPLATE.format(
        schema_json=schema_json
    )


def _union_json_schema(types: list[type[BaseModel]]) -> dict[str, Any]:
    """Build a single JSON Schema spanning one or more Pydantic models.

    * Single type: returns that model's ``model_json_schema()``.
    * Multiple types: returns an ``anyOf`` whose branches are the
      member schemas, with their ``$defs`` merged into a single
      top-level ``$defs`` so ``$ref`` resolution stays valid.
    """
    if len(types) == 1:
        return types[0].model_json_schema()
    branches: list[dict[str, Any]] = []
    merged_defs: dict[str, Any] = {}
    for t in types:
        s = dict(t.model_json_schema())
        # Hoist nested $defs to the union root so cross-branch refs
        # remain resolvable. Last writer wins on name collisions —
        # that's acceptable since identical model names should
        # have identical schemas.
        nested = s.pop("$defs", None)
        if isinstance(nested, dict):
            merged_defs.update(nested)
        branches.append(s)
    out: dict[str, Any] = {"anyOf": branches}
    if merged_defs:
        out["$defs"] = merged_defs
    return out


def _try_validate_union(
    text: str, types: list[type[BaseModel]]
) -> tuple[BaseModel | None, ValidationError | None]:
    """Try each Pydantic type in order; return the first that
    validates. When all fail, return ``(None, last_error)`` for the
    retry path. The order of ``types`` is the order the user wrote
    them (``A | B`` → ``[A, B]``), so callers can put the more
    specific / preferred type first."""
    last_error: ValidationError | None = None
    for t in types:
        try:
            return t.model_validate_json(text), None
        except ValidationError as exc:
            last_error = exc
    return None, last_error


_FENCE_RE_PREFIX = "```"


def _extract_schema_types(schema: Any) -> list[type[BaseModel]]:
    """Normalise an ``output_schema`` argument into a list of
    Pydantic model types.

    Accepts:
      * A single ``BaseModel`` subclass — returns ``[schema]``.
      * A ``Union[A, B]`` or ``A | B`` (PEP 604) of ``BaseModel``
        subclasses — returns ``[A, B]`` (with non-Pydantic members
        like ``None`` filtered out).
      * Anything else — returns ``[]``.

    Tagged-union outputs let an agent return one of multiple
    types: ``output_schema=Invoice | Error`` lets the model pick
    "valid invoice" vs "structured error" per call. The framework
    builds an ``anyOf`` JSON schema, asks the model for one,
    and tries each member type during validation.
    """
    import types as _typing_types
    from typing import Union, get_args, get_origin

    if schema is None:
        return []
    # Single Pydantic model.
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return [schema]
    # Union or PEP 604 union (X | Y).
    origin = get_origin(schema)
    if origin is Union or isinstance(schema, _typing_types.UnionType):
        members: list[type[BaseModel]] = []
        for arg in get_args(schema):
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                members.append(arg)
        return members
    return []


def _strip_json_fences(text: str) -> str:
    """Strip markdown code fences a model might wrap its JSON in.

    Tolerates ```json``` and bare ``` fences, with or without
    trailing newlines. Idempotent for already-clean JSON.
    """
    stripped = text.strip()
    if not stripped.startswith(_FENCE_RE_PREFIX):
        return stripped
    # Drop opening fence (and optional language tag).
    after_open = stripped[len(_FENCE_RE_PREFIX):]
    newline_idx = after_open.find("\n")
    if newline_idx == -1:
        return stripped
    body = after_open[newline_idx + 1:]
    if body.endswith(_FENCE_RE_PREFIX):
        body = body[: -len(_FENCE_RE_PREFIX)]
    return body.strip()


def _summarise_validation_error(exc: ValidationError) -> str:
    """Compact, retry-friendly description of a Pydantic
    :class:`ValidationError` — one bullet per error, capped to keep
    the retry prompt small enough not to blow the context window
    on pathological cases."""
    lines: list[str] = []
    for err in exc.errors()[:10]:
        loc = ".".join(str(p) for p in err.get("loc", ()))
        msg = err.get("msg", "invalid")
        kind = err.get("type", "")
        lines.append(f"- {loc}: {msg} (type={kind})")
    if len(exc.errors()) > 10:
        lines.append(f"- ... ({len(exc.errors()) - 10} more error(s))")
    return "\n".join(lines)


_LITELLM_PREFIXES: tuple[str, ...] = (
    "mistral-",
    "command-",       # Cohere
    "bedrock/",       # AWS Bedrock
    "vertex_ai/",     # Google Vertex
    "together_ai/",   # Together AI
    "ollama/",        # Local Ollama
    "gemini/",        # Google Gemini
    "groq/",          # Groq
    "replicate/",     # Replicate
    "azure/",         # Azure OpenAI
    "litellm/",       # explicit opt-in: ``litellm/<any>`` strips the prefix
)


_MODEL_DICT_KEYS = frozenset({"name", "model", "effort", "strict_effort"})


def _normalize_model_spec(
    spec: Model | str | dict[str, Any] | None,
    *,
    effort: str | None,
    strict_effort: bool,
) -> tuple[Model | str | None, str | None, bool]:
    """Unpack ``model={"name": ..., "effort": ..., "strict_effort": ...}``
    into a plain spec + the corresponding agent kwargs.

    Mirrors the ``audit_log={...}`` dict pattern: one parameter
    carries both the resource identifier and any related defaults.
    Explicit top-level kwargs win over dict-embedded values so
    callers who pass *both* can override per-Agent without
    rewriting the dict.

    Returns ``(spec, effort, strict_effort)`` ready for the existing
    resolution path. Non-dict inputs pass through unchanged.
    """
    if not isinstance(spec, dict):
        return spec, effort, strict_effort

    from ..core.errors import ConfigError

    d = dict(spec)  # shallow copy so we don't mutate the caller's dict
    extras = set(d) - _MODEL_DICT_KEYS
    if extras:
        raise ConfigError(
            f"model= dict has unknown key(s): {sorted(extras)}. "
            f"Recognised keys: {sorted(_MODEL_DICT_KEYS)}."
        )
    # ``name`` is the preferred key; ``model`` is the same thing
    # under a different name so users who think of "model name"
    # don't get tripped up.
    name = d.pop("name", None)
    alt = d.pop("model", None)
    if name is None and alt is None:
        raise ConfigError(
            "model= dict requires a 'name' key (or 'model' alias)."
        )
    if name is not None and alt is not None:
        raise ConfigError(
            "model= dict has both 'name' and 'model' — pick one."
        )
    resolved_spec: Model | str = name if name is not None else alt

    dict_effort = d.pop("effort", None)
    dict_strict = d.pop("strict_effort", None)

    # Explicit top-level kwarg wins; otherwise inherit from the dict.
    if effort is None and dict_effort is not None:
        effort = dict_effort
    if not strict_effort and dict_strict is not None:
        strict_effort = bool(dict_strict)

    return resolved_spec, effort, strict_effort


def _resolve_prompt_caching(
    spec: bool | Mapping[str, Any] | None,
) -> PromptCacheConfig:
    """Normalise the ``prompt_caching=`` kwarg into a typed config.

    Accepted shapes:

    * ``None`` / ``False`` → disabled config
    * ``True`` → enabled with default TTL (``"5m"``)
    * ``Mapping`` → ``{"enabled": bool, "ttl": "5m"|"1h",
      "cache_key": "..."}``. Missing keys take their defaults.

    Anything else raises :class:`ConfigError` with the recognised
    forms enumerated.
    """
    from ..core.errors import ConfigError

    if spec is None or spec is False:
        return PromptCacheConfig(enabled=False)
    if spec is True:
        return PromptCacheConfig(enabled=True)
    if isinstance(spec, Mapping):
        enabled = bool(spec.get("enabled", True))
        ttl = spec.get("ttl", "5m")
        if ttl not in ("5m", "1h"):
            raise ConfigError(
                f"prompt_caching= 'ttl' must be '5m' or '1h'; got {ttl!r}."
            )
        cache_key = spec.get("cache_key")
        if cache_key is not None and not isinstance(cache_key, str):
            raise ConfigError(
                "prompt_caching= 'cache_key' must be a string or None."
            )
        return PromptCacheConfig(
            enabled=enabled, ttl=ttl, cache_key=cache_key,
        )
    raise ConfigError(
        f"prompt_caching= unrecognised value {spec!r}. Use True / False / "
        "None, or a dict like {'enabled': True, 'ttl': '5m'|'1h', "
        "'cache_key': '<session-id>'}."
    )


def _resolve_model(
    spec: Model | str | None, *, secrets: Any | None = None
) -> Model:
    """Resolve a string spec or instance to a concrete :class:`Model`.

    Strings dispatch by prefix:

    * ``claude-*`` -> :class:`AnthropicModel` (direct, no LiteLLM hop)
    * ``gpt-*`` / ``o1-*`` / ``o3-*`` -> :class:`OpenAIModel` (direct)
    * ``echo`` -> :class:`EchoModel` (zero-key dev / tests)
    * ``mistral-``, ``command-``, ``bedrock/``, ``vertex_ai/``,
      ``together_ai/``, ``ollama/``, ``gemini/``, ``groq/``,
      ``replicate/``, ``azure/``, ``litellm/`` -> :class:`LiteLLMModel`
      which fans out to ~100 providers via the LiteLLM SDK
    * ``litellm/<spec>`` strips the prefix before forwarding (handy
      when you want LiteLLM to handle a spec the direct paths would
      otherwise grab)

    ``None`` raises :class:`~loomflow.core.errors.ConfigError` with a
    helpful suggestion list. Unknown specs raise
    :class:`~loomflow.core.errors.ConfigError` too (was ``ValueError``
    in 0.1.x — harmonised in 0.2.0).
    """
    from ..core.errors import ConfigError

    if spec is None:
        raise ConfigError(
            "Agent() requires a `model` argument. Pass one of:\n"
            "  model='claude-opus-4-7'   "
            "(Anthropic, needs ANTHROPIC_API_KEY)\n"
            "  model='gpt-4o'            "
            "(OpenAI, needs OPENAI_API_KEY)\n"
            "  model='echo'              "
            "(zero-key fake — text echoes the prompt; for dev/tests)\n"
            "  model='mistral-large'     "
            "(LiteLLM, also: command-, bedrock/, vertex_ai/, ollama/, ...)\n"
            "  model=AnthropicModel(...) "
            "or any Model-protocol instance for full control."
        )
    if not isinstance(spec, str):
        return spec
    if spec.startswith("claude-"):
        from ..model.anthropic import AnthropicModel
        return AnthropicModel(spec, secrets=secrets)
    if spec.startswith(("gpt-", "o1-", "o3-")):
        from ..model.openai import OpenAIModel
        return OpenAIModel(spec, secrets=secrets)
    if spec == "echo":
        return EchoModel()
    if spec.startswith(_LITELLM_PREFIXES):
        from ..model.litellm import LiteLLMModel

        # ``litellm/<inner>`` strips the explicit-opt-in prefix.
        inner = spec[len("litellm/"):] if spec.startswith("litellm/") else spec
        return LiteLLMModel(inner, secrets=secrets)
    raise ConfigError(
        f"unknown model spec: {spec!r}. Recognised prefixes:\n"
        "  claude-*, gpt-*, o1-*, o3-* (direct adapters)\n"
        "  mistral-, command-, bedrock/, vertex_ai/, ollama/, "
        "gemini/, groq/, together_ai/, replicate/, azure/ "
        "(via LiteLLM)\n"
        "  echo (zero-key fake)\n"
        "Or pass a Model-protocol instance directly. To force the "
        "LiteLLM path for any spec, prefix with 'litellm/'."
    )


def _coerce_tool_host(
    tools: list[Tool | Callable[..., object]]
    | ToolHost
    | Tool
    | Callable[..., object]
    | None,
) -> ToolHost:
    """Normalize ``tools=`` to a ``ToolHost``.

    Accepts:

    * ``None`` -> empty in-process host
    * a ``ToolHost`` instance (anything with ``list_tools`` + ``call``)
    * a list of ``Tool`` / callable
    * a single ``Tool`` (auto-wrapped in a one-tool list)
    * a single callable (auto-wrapped via ``@tool``)

    The single-callable / single-``Tool`` shorthand is friendlier
    when you only have one tool; ``tools=my_fn`` is shorter and
    less error-prone than ``tools=[my_fn]``.
    """
    if tools is None:
        return InProcessToolHost([])
    # Duck-type: anything with ``list_tools`` and ``call`` is a host.
    if hasattr(tools, "list_tools") and hasattr(tools, "call"):
        return tools  # type: ignore[return-value]
    if isinstance(tools, list):
        return InProcessToolHost(tools)
    if isinstance(tools, Tool):
        return InProcessToolHost([tools])
    if callable(tools):
        return InProcessToolHost([tools])
    raise TypeError(
        f"tools= must be None, a list, a Tool, a callable, or a "
        f"ToolHost; got {type(tools).__name__}: {tools!r}.\n"
        f"Valid forms:\n"
        f"  • tools=None — no tools\n"
        f"  • tools=[my_fn, other_fn] — list of @tool functions / callables\n"
        f"  • tools=my_fn — single tool (auto-wrapped in a list)\n"
        f"  • tools=my_tool_host — a ToolHost instance (has list_tools + call)"
    )


def _mcp_spec_from_dict(entry: Mapping[str, Any]) -> Any:
    """Translate a parsed-from-config dict into an :class:`MCPServerSpec`.

    Schema::

        { name = "git",
          transport = "stdio",         # or "http"
          # stdio:
          command = "uvx",
          args = ["mcp-server-git", "--repo", "."],
          env = { GIT_PAGER = "cat" }, # optional
          # http:
          url = "https://example.com/mcp",
          headers = { Authorization = "Bearer ..." },
          description = "...",         # optional, free-form
        }
    """
    from ..core.errors import ConfigError
    from ..mcp import MCPServerSpec

    name = entry.get("name")
    transport = entry.get("transport")
    if not isinstance(name, str) or not name:
        raise ConfigError(
            "Agent.from_dict: mcp[*] entry needs a non-empty 'name' string."
        )
    if transport not in ("stdio", "http"):
        raise ConfigError(
            f"Agent.from_dict: mcp[{name!r}].transport must be 'stdio' or "
            f"'http' (got {transport!r})."
        )
    description = entry.get("description", "")
    if not isinstance(description, str):
        raise ConfigError(
            f"Agent.from_dict: mcp[{name!r}].description must be a string."
        )
    if transport == "stdio":
        command = entry.get("command")
        if not isinstance(command, str):
            raise ConfigError(
                f"Agent.from_dict: mcp[{name!r}] stdio transport needs "
                "a string 'command'."
            )
        args = entry.get("args") or []
        if not isinstance(args, list) or not all(
            isinstance(a, str) for a in args
        ):
            raise ConfigError(
                f"Agent.from_dict: mcp[{name!r}].args must be a list of strings."
            )
        env = entry.get("env")
        if env is not None and not (
            isinstance(env, Mapping)
            and all(
                isinstance(k, str) and isinstance(v, str)
                for k, v in env.items()
            )
        ):
            raise ConfigError(
                f"Agent.from_dict: mcp[{name!r}].env must be a string→string mapping."
            )
        return MCPServerSpec.stdio(
            name,
            command,
            list(args),
            env=dict(env) if env is not None else None,
            description=description,
        )
    # http
    url = entry.get("url")
    if not isinstance(url, str):
        raise ConfigError(
            f"Agent.from_dict: mcp[{name!r}] http transport needs a string 'url'."
        )
    headers = entry.get("headers")
    if headers is not None and not (
        isinstance(headers, Mapping)
        and all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in headers.items()
        )
    ):
        raise ConfigError(
            f"Agent.from_dict: mcp[{name!r}].headers must be a string→string mapping."
        )
    return MCPServerSpec.http(
        name,
        url,
        headers=dict(headers) if headers is not None else None,
        description=description,
    )


