"""The public ``Agent`` class.

Conventions:

* Pass a string of instructions for a working agent backed by sensible
  defaults: :class:`EchoModel`, :class:`InMemoryMemory`,
  :class:`InProcRuntime`, :class:`NoBudget`, :class:`AllowAll`,
  :class:`HookRegistry`, an empty :class:`InProcessToolHost`.
* Pass ``tools=[fn_or_Tool, ...]`` to register Python callables; the
  agent wraps them in an in-process :class:`ToolHost`.
* Override any subsystem by passing a concrete implementation of the
  matching protocol from :mod:`jeevesagent.core.protocols`.

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
from typing import Any

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
    Role,
    RunResult,
    Usage,
)
from ..governance.budget import NoBudget
from ..governance.retry import RetryPolicy
from ..memory.inmemory import InMemoryMemory
from ..model.echo import EchoModel
from ..observability.tracing import NoTelemetry
from ..runtime.inproc import InProcRuntime
from ..security.audit import AuditLog
from ..security.hooks import HookRegistry, PostToolHook, PreToolHook
from ..security.permissions import AllowAll
from ..tools.registry import InProcessToolHost, Tool

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
        model: Model | str | None = None,
        memory: Memory | None = None,
        runtime: Runtime | None = None,
        budget: Budget | None = None,
        permissions: Permissions | None = None,
        hooks: HookRegistry | None = None,
        tools: list[Tool | Callable[..., object]]
        | ToolHost
        | Tool
        | Callable[..., object]
        | None = None,
        telemetry: Telemetry | None = None,
        audit_log: AuditLog | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        auto_consolidate: bool = False,
        architecture: Architecture | str | None = None,
        skills: list[Any] | None = None,
        retry_policy: RetryPolicy | None = None,
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
        self._model: Model = _resolve_model(model)
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
        self._memory: Memory = memory if memory is not None else InMemoryMemory()
        self._runtime: Runtime = runtime if runtime is not None else InProcRuntime()
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
        self._tool_host: ToolHost = host

        self._telemetry: Telemetry = (
            telemetry if telemetry is not None else NoTelemetry()
        )
        self._audit_log: AuditLog | None = audit_log
        self._max_turns = max_turns
        self._auto_consolidate = auto_consolidate
        self._architecture: Architecture = resolve_architecture(architecture)

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
        (e.g. :class:`~jeevesagent.architecture.Supervisor`) can read
        each worker's intended role when composing instructions for
        the supervising model.
        """
        return self._instructions

    @property
    def architecture(self) -> Architecture:
        """The configured :class:`Architecture` strategy.

        Default is :class:`~jeevesagent.architecture.ReAct`. Pass
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
    ) -> None:
        if self._audit_log is None:
            return
        await self._audit_log.append(
            session_id=session_id,
            actor=actor,
            action=action,
            payload=payload,
        )

    async def _validate_output_with_retry(
        self,
        *,
        session: AgentSession,
        schema: type[BaseModel],
        retries: int,
        deps: Dependencies,
    ) -> BaseModel:
        """Validate ``session.output`` against ``schema``; on failure,
        give the model up to ``retries`` follow-up turns to fix it.

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

        last_error: ValidationError | None = None
        for attempt in range(retries + 1):
            text = _strip_json_fences(session.output)
            try:
                parsed = schema.model_validate_json(text)
                # Stash the cleaned text so the persisted episode
                # has parseable JSON, not the fenced version.
                session.output = text
                return parsed
            except ValidationError as exc:
                last_error = exc
                if attempt >= retries:
                    break

                # Re-prompt: tell the model what was wrong and ask
                # for a clean JSON re-emission.
                error_summary = _summarise_validation_error(exc)
                schema_json = json.dumps(
                    schema.model_json_schema(), separators=(",", ":")
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
                            session.messages, tools=None
                        )
                    )
                else:
                    parts: list[str] = []
                    usage = Usage()
                    async for chunk in deps.model.stream(
                        session.messages, tools=None
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
        raise OutputValidationError(
            f"Model output did not validate against {schema.__name__} "
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
        model: Model | None = None,
        memory: Memory | None = None,
        runtime: Runtime | None = None,
        tools: list[Tool | Callable[..., object]] | ToolHost | None = None,
    ) -> Agent:
        """Construct an ``Agent`` from a parsed config dict.

        Same shape as :meth:`from_config` but skips the file read.
        Useful when the config comes from somewhere other than a TOML
        file — environment variables, a Pydantic settings model, a
        ``yaml.safe_load`` result, an HTTP API, etc.

        Recognised keys (all optional except ``instructions`` and
        ``model``):

        * ``instructions: str`` — required
        * ``model: str`` — required (or pass ``model=`` kwarg)
        * ``max_turns: int``
        * ``auto_consolidate: bool``
        * ``budget: dict`` with any of ``max_tokens``,
          ``max_input_tokens``, ``max_output_tokens``, ``max_cost_usd``,
          ``max_wall_clock_minutes``, ``soft_warning_at``
        """
        from datetime import timedelta

        from ..core.errors import ConfigError
        from ..governance.budget import BudgetConfig, StandardBudget

        instructions = cfg.get("instructions")
        if not isinstance(instructions, str):
            raise ConfigError(
                "Agent.from_dict: missing or non-string 'instructions' field"
            )

        model_spec = model if model is not None else cfg.get("model")
        if model_spec is None:
            raise ConfigError(
                "Agent.from_dict: missing 'model' field. Add e.g. "
                "model = 'claude-opus-4-7' (or pass model= explicitly)."
            )

        max_turns = cfg.get("max_turns", DEFAULT_MAX_TURNS)
        auto_consolidate = bool(cfg.get("auto_consolidate", False))

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

        return cls(
            instructions,
            model=model_spec,
            memory=memory,
            runtime=runtime,
            tools=tools,
            budget=budget,
            max_turns=int(max_turns),
            auto_consolidate=auto_consolidate,
        )

    @classmethod
    def from_config(
        cls,
        path: str | Path,
        *,
        model: Model | None = None,
        memory: Memory | None = None,
        runtime: Runtime | None = None,
        tools: list[Tool | Callable[..., object]] | ToolHost | None = None,
    ) -> Agent:
        """Construct an ``Agent`` from a TOML config file.

        Designed for ops/devops users who want declarative agent
        config separate from code. Supports the textual / numeric
        bits — instructions, model spec (string), max_turns,
        auto_consolidate, budget — and lets callers pass concrete
        instances for the things TOML can't reasonably express
        (real ``Memory``, ``Runtime``, custom ``Model``, tools).

        Example ``agent.toml``::

            instructions = "You are a research assistant."
            model = "claude-opus-4-7"
            max_turns = 100
            auto_consolidate = true

            [budget]
            max_tokens = 200_000
            max_cost_usd = 5.0
            max_wall_clock_minutes = 10
            soft_warning_at = 0.8

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
                tools=tools,
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
        output_schema: type[BaseModel] | None = None,
        output_validation_retries: int = 1,
    ) -> RunResult:
        """Run the agent to completion and return its :class:`RunResult`.

        ``user_id`` is the namespace partition for memory recall and
        persistence — episodes and facts stored with one ``user_id``
        are never visible to a query scoped to a different ``user_id``.
        ``None`` is the "anonymous / single-tenant" bucket. See
        :class:`~jeevesagent.RunContext` for the partitioning
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
        :class:`~jeevesagent.OutputValidationError`. Set retries to
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
            output_schema=output_schema,
            output_validation_retries=output_validation_retries,
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
        output_schema: type[BaseModel] | None = None,
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
        output_schema: type[BaseModel] | None = None,
        output_validation_retries: int = 1,
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
                    output_schema=output_schema,
                    output_validation_retries=output_validation_retries,
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
        output_schema: type[BaseModel] | None = None,
        output_validation_retries: int = 1,
    ) -> RunResult:
        """Setup → delegate iteration to the architecture → teardown.

        The architecture (default :class:`ReAct`) drives the iteration
        and yields events as it goes. Setup wraps the run in a runtime
        session + telemetry trace so every ``runtime.step`` recorded
        from inside the architecture lands on the same journal entry,
        and installs a :class:`~jeevesagent.RunContext` in a
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

        run_trace: contextlib.AbstractAsyncContextManager[Any] = (
            _NULL_CTX
            if fast_telemetry
            else self._telemetry.trace(
                "jeeves.run",
                session_id=session_id,
                max_turns=self._max_turns,
                model=self._model.name,
                architecture=self._architecture.name,
            )
        )

        async with (
            self._runtime.session(session_id),
            run_trace,
            set_run_context(run_ctx),
        ):
            if not fast_audit:
                await self._audit(
                    session_id=session_id,
                    actor="user",
                    action="run_started",
                    payload={
                        "prompt": prompt[:500],
                        "model": self._model.name,
                        "max_turns": self._max_turns,
                        "architecture": self._architecture.name,
                        "user_id": run_ctx.user_id,
                    },
                )
            await emit(Event.started(session_id, prompt))

            # Append the JSON-schema directive when a structured
            # output is requested — augments the agent's base
            # instructions for this run only, leaving the static
            # ``self._instructions`` unchanged.
            effective_instructions = (
                _augment_instructions_for_schema(
                    self._instructions, output_schema
                )
                if output_schema is not None
                else self._instructions
            )

            session = AgentSession(
                id=session_id,
                instructions=effective_instructions,
            )
            # Per-run tool injection: if extra_tools provided, wrap
            # the agent's host so the model sees them alongside the
            # statically-configured tools. The wrap is local to this
            # run; the agent's _tool_host is unchanged.
            effective_tools = (
                ExtendedToolHost(self._tool_host, extra_tools)
                if extra_tools
                else self._tool_host
            )
            deps = Dependencies(
                # ``_wrapped_model`` is the retry-decorated view —
                # falls through to ``_model`` when the policy is
                # disabled, so the architecture loop never sees a
                # raw SDK exception when retries could have helped.
                model=self._wrapped_model,
                memory=self._memory,
                runtime=self._runtime,
                tools=effective_tools,
                budget=self._budget,
                permissions=self._permissions,
                hooks=self._hooks,
                telemetry=self._telemetry,
                audit_log=self._audit_log,
                max_turns=self._max_turns,
                # Architectures consult this to pick fast (buffered)
                # vs streaming (channel) paths for parallel work.
                streaming=emit is not _noop_emit,
                fast_audit=fast_audit,
                fast_telemetry=fast_telemetry,
                fast_permissions=fast_permissions,
                fast_hooks=fast_hooks,
                fast_runtime=fast_runtime,
                fast_budget=fast_budget,
                context=run_ctx,
            )

            async for event in self._architecture.run(session, deps, prompt):
                await emit(event)

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
                await self._memory.remember(episode)
            else:
                await self._runtime.step(
                    f"persist_episode_{session.turns}",
                    self._memory.remember,
                    episode,
                )

            result = RunResult(
                session_id=session_id,
                output=session.output,
                parsed=parsed,
                turns=session.turns,
                tokens_in=session.cumulative_usage.input_tokens,
                tokens_out=session.cumulative_usage.output_tokens,
                cost_usd=session.cumulative_usage.cost_usd,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                interrupted=session.interrupted,
                interruption_reason=session.interruption_reason,
            )

            elapsed_ms = (anyio.current_time() - loop_started) * 1000
            if not fast_telemetry:
                await self._telemetry.emit_metric(
                    "jeeves.session.duration_ms",
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
                await self._audit(
                    session_id=session_id,
                    actor="system",
                    action="run_completed",
                    payload={
                        "turns": session.turns,
                        "interrupted": session.interrupted,
                        "interruption_reason": session.interruption_reason,
                        "tokens_in": session.cumulative_usage.input_tokens,
                        "tokens_out": session.cumulative_usage.output_tokens,
                        "cost_usd": session.cumulative_usage.cost_usd,
                        "elapsed_ms": elapsed_ms,
                    },
                )
            await emit(Event.completed(session_id, result.model_dump(mode="json")))
            return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _noop_emit(_event: Event) -> None:
    return None


_NETWORK_MODEL_CLASS_NAMES = frozenset(
    {"OpenAIModel", "AnthropicModel", "LiteLLMModel"}
)


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
    base_instructions: str, schema: type[BaseModel]
) -> str:
    """Build the run-scoped system prompt when ``output_schema`` is
    requested. Appends a clear, schema-specific directive to the
    agent's static instructions so the model knows it must emit JSON
    matching the schema."""
    schema_json = json.dumps(
        schema.model_json_schema(), indent=2, sort_keys=True
    )
    return base_instructions.rstrip() + _SCHEMA_DIRECTIVE_TEMPLATE.format(
        schema_json=schema_json
    )


_FENCE_RE_PREFIX = "```"


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


def _resolve_model(spec: Model | str | None) -> Model:
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

    ``None`` raises :class:`~jeevesagent.core.errors.ConfigError` with a
    helpful suggestion list. Unknown specs raise
    :class:`~jeevesagent.core.errors.ConfigError` too (was ``ValueError``
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
        return AnthropicModel(spec)
    if spec.startswith(("gpt-", "o1-", "o3-")):
        from ..model.openai import OpenAIModel
        return OpenAIModel(spec)
    if spec == "echo":
        return EchoModel()
    if spec.startswith(_LITELLM_PREFIXES):
        from ..model.litellm import LiteLLMModel

        # ``litellm/<inner>`` strips the explicit-opt-in prefix.
        inner = spec[len("litellm/"):] if spec.startswith("litellm/") else spec
        return LiteLLMModel(inner)
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
    raise TypeError(f"unsupported tools= argument: {type(tools).__name__}")


