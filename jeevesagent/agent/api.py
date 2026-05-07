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

from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio

from ..architecture import (
    AgentSession,
    Architecture,
    Dependencies,
    resolve_architecture,
)
from ..architecture.tool_host_wrappers import ExtendedToolHost
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
    RunResult,
)
from ..governance.budget import NoBudget
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
    ) -> None:
        self._instructions = instructions
        self._model: Model = _resolve_model(model)
        self._memory: Memory = memory if memory is not None else InMemoryMemory()
        self._runtime: Runtime = runtime if runtime is not None else InProcRuntime()
        self._budget: Budget = budget if budget is not None else NoBudget()
        self._permissions: Permissions = (
            permissions if permissions is not None else AllowAll()
        )
        self._hooks = hooks if hooks is not None else HookRegistry()
        self._tool_host: ToolHost = _coerce_tool_host(tools)
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
        session_id: str | None = None,
        extra_tools: list[Tool] | None = None,
        emit: Emit | None = None,
    ) -> RunResult:
        """Run the agent to completion and return its :class:`RunResult`.

        Pass ``session_id`` to resume a journaled run — when paired with
        a durable runtime (e.g. :class:`SqliteRuntime`), already-completed
        steps replay from the journal instead of re-executing. Without a
        durable runtime, ``session_id`` just labels the run.

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
        """
        return await self._loop(
            prompt,
            emit=emit if emit is not None else _noop_emit,
            session_id=session_id,
            extra_tools=extra_tools,
        )

    async def resume(
        self,
        session_id: str,
        prompt: str,
        *,
        extra_tools: list[Tool] | None = None,
        emit: Emit | None = None,
    ) -> RunResult:
        """Resume a previously-interrupted run from its journal.

        Equivalent to ``agent.run(prompt, session_id=session_id)``.
        Exists as a separate method so the intent is explicit at the
        call site and to match the surface advertised by the engineering
        plan.
        """
        return await self.run(
            prompt,
            session_id=session_id,
            extra_tools=extra_tools,
            emit=emit,
        )

    async def stream(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        extra_tools: list[Tool] | None = None,
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
                    session_id=session_id,
                    extra_tools=extra_tools,
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
        session_id: str | None = None,
        extra_tools: list[Tool] | None = None,
    ) -> RunResult:
        """Setup → delegate iteration to the architecture → teardown.

        The architecture (default :class:`ReAct`) drives the iteration
        and yields events as it goes. Setup wraps the run in a runtime
        session + telemetry trace so every ``runtime.step`` recorded
        from inside the architecture lands on the same journal entry.
        Teardown persists the episode, builds the :class:`RunResult`,
        emits final metrics, and triggers ``auto_consolidate``.
        """
        started_at = datetime.now(UTC)
        if session_id is None:
            session_id = new_id("sess")
        loop_started = anyio.current_time()

        async with (
            self._runtime.session(session_id),
            self._telemetry.trace(
                "jeeves.run",
                session_id=session_id,
                max_turns=self._max_turns,
                model=self._model.name,
                architecture=self._architecture.name,
            ),
        ):
            await self._audit(
                session_id=session_id,
                actor="user",
                action="run_started",
                payload={
                    "prompt": prompt[:500],
                    "model": self._model.name,
                    "max_turns": self._max_turns,
                    "architecture": self._architecture.name,
                },
            )
            await emit(Event.started(session_id, prompt))

            session = AgentSession(
                id=session_id,
                instructions=self._instructions,
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
                model=self._model,
                memory=self._memory,
                runtime=self._runtime,
                tools=effective_tools,
                budget=self._budget,
                permissions=self._permissions,
                hooks=self._hooks,
                telemetry=self._telemetry,
                audit_log=self._audit_log,
                max_turns=self._max_turns,
            )

            async for event in self._architecture.run(session, deps, prompt):
                await emit(event)

            await self._runtime.step(
                f"persist_episode_{session.turns}",
                self._memory.remember,
                Episode(
                    session_id=session_id,
                    input=prompt,
                    output=session.output,
                ),
            )

            result = RunResult(
                session_id=session_id,
                output=session.output,
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


