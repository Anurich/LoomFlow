# JeevesAgent — Engineering Plan

> **JeevesAgent** is a model-agnostic, MCP-native, fully-async agent harness. Users define an `Agent`. The harness handles memory, durability, security, observability, and tool plumbing. Configurable at every layer; zero-config out of the box.

This document is the engineering blueprint. It covers architecture, async design, module topology, public/private contracts, plugin system, testing strategy, and the phased build. Every architectural choice is justified with primary-source research, and every snippet is real Python that compiles.

---

## Table of Contents

1. [What we're actually building](#1-what-were-actually-building)
2. [Engineering principles (non-negotiable)](#2-engineering-principles-non-negotiable)
3. [Async foundations](#3-async-foundations)
4. [Module topology](#4-module-topology)
5. [Core protocols (the interface layer)](#5-core-protocols-the-interface-layer)
6. [The Agent class — public API](#6-the-agent-class--public-api)
7. [The Loop — internal architecture](#7-the-loop--internal-architecture)
8. [Layer 0: Durable runtime](#8-layer-0-durable-runtime)
9. [Layer 1: Memory subsystem](#9-layer-1-memory-subsystem)
10. [Layer 2: Security harness](#10-layer-2-security-harness)
11. [Layer 3: MCP spine](#11-layer-3-mcp-spine)
12. [Layer 4: Data certification](#12-layer-4-data-certification)
13. [Layer 5: Resource governance](#13-layer-5-resource-governance)
14. [Layer 6: Observability](#14-layer-6-observability)
15. [Plugin architecture](#15-plugin-architecture)
16. [Concurrency patterns](#16-concurrency-patterns)
17. [Testing strategy](#17-testing-strategy)
18. [Performance & scalability](#18-performance--scalability)
19. [Phased build with verification gates](#19-phased-build-with-verification-gates)
20. [The first PR: package skeleton](#20-the-first-pr-package-skeleton)

---

## 1. What we're actually building

### 1.1 The product

A Python library — `jeevesagent` — where this works:

```python
from jeevesagent import Agent

agent = Agent("You are a research assistant")
result = await agent.run("Summarize this week's AI news")
```

…and *also* this works:

```python
agent = Agent(
    "You are a research assistant",
    model="claude-opus-4-7",
    tools=[gmail, drive, jeeves_gateway],
    memory=Memory(
        backend=PostgresBackend(dsn=DSN),
        strategy=TemporalGraph(),
        working_tokens=20_000,
    ),
    runtime=DurableRuntime("dbos"),
    permissions=Permissions.strict(),
    budget=Budget(max_tokens=200_000, max_cost_usd=5.0),
)

@agent.before_tool
async def review(call):
    if call.is_destructive():
        return await ask_user_for_approval(call)

async for event in agent.stream("complex task here"):
    handle(event)
```

Same `Agent` class. A string becomes a config object becomes an injected implementation. Convention-over-configuration, all the way down.

### 1.2 The wedge

> **JeevesAgent is the model-agnostic, MCP-native harness with memory done right.**

That sentence has three pieces and they appear in priority order:

1. **Model-agnostic** — Claude, GPT, Gemini, Llama, all behind one interface
2. **MCP-native** — MCP isn't an integration; it's the spine of the entire system
3. **Memory done right** — tiered like MemGPT, fast like Mem0, temporal like Zep

If a user is happy on Anthropic and never plans to switch, they should use Claude Agent SDK. JeevesAgent is for everyone else — the people who want a real harness without binding their stack to one model lab.

### 1.3 The user

**Primary**: Pragmatic engineers building production agents — solo or small team — who want to ship without assembling a Frankenstein. The kind of person who'd otherwise reach for FastAPI rather than Django or Flask.

**Secondary**: Teams whose existing infra includes Jeeves MCP gateway, who want their agent to plug into that ecosystem natively.

**Not the target (yet)**: Enterprise procurement, compliance-first orgs, no-code builders. We earn that audience in v2.

---

## 2. Engineering principles (non-negotiable)

These are the rules that govern every PR. If a change violates one of these, it doesn't merge.

### 2.1 Async-only, structured concurrency only

Every public function returning anything other than a value is `async`. Every fan-out uses `anyio` task groups. **Zero** raw `asyncio.create_task` or `asyncio.gather` calls in the codebase. Source: the OpenAI, Anthropic, and Google Gemini Python SDKs all use anyio internally — this is the proven pattern.

Why this matters: `asyncio.gather()` does not cancel sibling tasks when one fails, leaks orphaned tasks under cancellation, and makes correct shutdown nearly impossible. We are not paying that tax.

### 2.2 Protocol-typed interfaces everywhere

Every module boundary is a `typing.Protocol`. We use `runtime_checkable` only on stable surfaces. Concrete implementations get injected through constructor parameters; no magic resolution, no globals.

Why: structural typing means anyone can implement an interface without inheriting from our types. Tests pass fakes; users pass replacements. No DI framework needed for v1 — Python's type system is enough.

### 2.3 The loop is deterministic; the world isn't

The agent's control loop is pure Python that can be replayed from a journal. **Every** non-deterministic operation — LLM calls, tool calls, memory writes — runs as a journaled "step" that returns cached results on replay. This is the Temporal/DBOS pattern, and it's why OpenAI's Codex runs on Temporal in production.

Concretely: `runtime.step("name", fn, *args)` is the only way side effects happen.

### 2.4 Validate state on write, not on read

LangGraph's #6491 bug — invalid state saves successfully then can never be loaded — is unforgivable. Every state mutation runs through Pydantic validation **before** it's persisted. Bad state never lands.

### 2.5 Trust boundary stays outside the sandbox

The harness runs the agent's tools inside a sandbox (bubblewrap/Seatbelt/gVisor). The harness itself does not run inside that sandbox. If the model can write files that mutate orchestration state, the trust model is broken. This is what Anthropic and OpenAI both warn about explicitly.

### 2.6 Observable by default

Every model call, tool call, memory op, hook decision, and budget check emits an OpenTelemetry span. No proprietary telemetry format. Users get to use Honeycomb, Grafana, Datadog, LangSmith — whatever they already have.

### 2.7 No hidden state

There are no module-level globals beyond the entry point registry. Every dependency the loop needs is passed in via constructor. This makes the system trivially testable and trivially reentrant. Two `Agent` instances in the same process never interfere.

### 2.8 Backwards compatibility is sacred

Once a Protocol is published, it doesn't change without a major version bump. Plugin authors and downstream users build against these contracts. Breaking them is a tax we don't impose lightly.

---

## 3. Async foundations

This section is the most important early decision. Get it wrong and every other layer suffers.

### 3.1 Library choice: anyio over raw asyncio

We use `anyio` as the concurrency primitives layer for the entire codebase. We do **not** import from `asyncio` directly except for protocol compatibility (e.g., when a third-party library requires it).

Why:

| Feature | `asyncio` | `anyio` |
|---|---|---|
| Task groups (structured) | 3.11+ (`TaskGroup`) | Yes (`create_task_group`) |
| Cancel scopes | No (timeouts only) | Yes (full control) |
| `move_on_after` (best-effort timeout) | No | Yes |
| Cancel-on-error semantics for siblings | No (`gather`) | Yes |
| Memory object streams | `Queue` (poor cleanup) | `create_memory_object_stream` |
| Backend portability (asyncio + trio) | No | Yes |
| Used by major SDKs | — | OpenAI, Anthropic, Google Gemini |

The killer feature for an agent harness is `move_on_after`:

```python
# "Take whatever results arrived in 3 seconds, proceed with those"
async with anyio.move_on_after(3.0):
    async with anyio.create_task_group() as tg:
        for query in queries:
            tg.start_soon(search, query)
# Outside the scope: completed results are still in our list
```

This pattern matters constantly — parallel search, optimistic tool fan-out, "best of N" sampling. `asyncio.gather()` cannot express it cleanly.

### 3.2 Cancel scopes — the discipline

Cancel scopes are the single most underrated tool in async Python. Every place we spawn parallel work, we wrap it in a scope. Every long-running operation gets a timeout scope. Cleanup is guaranteed because scope exit waits for all child tasks.

Three patterns we use everywhere:

**Pattern A — fail fast on first error:**
```python
async with anyio.create_task_group() as tg:
    for call in tool_calls:
        tg.start_soon(_run_one, call)
# If any task raises, all siblings are cancelled and the
# exception propagates as ExceptionGroup
```

**Pattern B — best effort with deadline:**
```python
results: list[ToolResult] = []
async with anyio.move_on_after(timeout_s):
    async with anyio.create_task_group() as tg:
        for call in tool_calls:
            tg.start_soon(_run_into, call, results)
# After the scope, `results` has whatever finished
```

**Pattern C — shielded cleanup:**
```python
try:
    await long_op()
finally:
    # Cleanup must complete even if we're being cancelled
    with anyio.CancelScope(shield=True):
        await release_resources()
```

### 3.3 Streaming — async generators end to end

Every step that produces incremental output is an `AsyncIterator`. Model output streams chunk-by-chunk. Tool execution streams progress events. The agent's main loop streams `Event` records to whoever is listening.

```python
async def stream(self, prompt: str) -> AsyncIterator[Event]:
    async for event in self._loop.iter_events(prompt):
        yield event
```

Backpressure is automatic — `anyio.create_memory_object_stream()` with a capacity buffer between producer and consumer ensures slow consumers can't OOM the producer.

### 3.4 Resource lifecycle — async context managers

Every resource that needs setup/teardown is an `AsyncContextManager`. The MCP client connection, the database pool, the tracing exporter, all of them. Composition uses `AsyncExitStack`:

```python
async with AsyncExitStack() as stack:
    db = await stack.enter_async_context(self.db_pool.acquire())
    mcp = await stack.enter_async_context(self.mcp_client.connect())
    # ... if anything fails, both unwind cleanly
```

### 3.5 What we never do

- `asyncio.create_task()` — unstructured spawning, leaks tasks
- `asyncio.gather()` — no sibling cancellation on error
- `asyncio.wait()` — even worse than gather
- `loop.call_soon()` / `loop.run_until_complete()` — implicit loop access
- `time.sleep()` — blocks the event loop
- Threading without `anyio.to_thread.run_sync()` — bypasses cancellation
- Sync database libraries — `psycopg2` is banned, only `asyncpg` or `psycopg[binary,pool]` async API

### 3.6 Threadpool only for blocking I/O we can't replace

When we *must* call sync code (e.g., a library has no async version), we use `anyio.to_thread.run_sync(fn)`. This runs in a worker thread, respects cancellation, and doesn't block the event loop. Used sparingly.

---

## 4. Module topology

The package is laid out as a constellation of small, focused modules. Each module exports protocols and concrete implementations. There's a strict dependency direction (no cycles).

```
jeevesagent/
├── __init__.py             # Public API: Agent, Memory, Runtime, etc.
├── core/                   # Layer-free primitives
│   ├── types.py            # Pydantic models (Message, Event, ToolCall, ...)
│   ├── errors.py           # Exception hierarchy
│   ├── protocols.py        # All Protocol definitions
│   └── ids.py              # ULID generation, deterministic hashes
├── async_/                 # Async utilities (note: name avoids stdlib clash)
│   ├── streams.py          # Memory object streams helpers
│   ├── timeouts.py         # Standard timeout policies
│   └── lifecycle.py        # AsyncExitStack helpers
├── model/                  # LLM provider adapters
│   ├── base.py             # Model protocol
│   ├── anthropic.py        # Claude adapter
│   ├── openai.py           # GPT adapter
│   ├── litellm.py          # Catch-all via LiteLLM
│   └── streaming.py        # Chunk normalization across providers
├── memory/                 # Memory subsystem
│   ├── base.py             # Memory protocol + Episode/Block types
│   ├── working.py          # Working memory (in-context blocks)
│   ├── episodic.py         # Episodic store
│   ├── semantic.py         # Semantic graph
│   ├── backends/
│   │   ├── postgres.py     # Postgres + pgvector backend
│   │   └── memory.py       # In-memory dev backend
│   └── server.py           # Expose memory as MCP server
├── runtime/                # Durable execution
│   ├── base.py             # Runtime protocol
│   ├── inproc.py           # In-process (no durability) — default
│   ├── dbos.py             # DBOS adapter
│   └── temporal.py         # Temporal adapter
├── mcp/                    # MCP client + registry
│   ├── client.py           # MCP client (stdio + Streamable HTTP)
│   ├── registry.py         # Lazy tool loading, server discovery
│   ├── auth.py             # OAuth 2.1 flow
│   └── transport.py        # Transport abstractions
├── security/               # Security harness
│   ├── permissions.py      # Permission modes, ACL
│   ├── sandbox/
│   │   ├── base.py         # Sandbox protocol
│   │   ├── bubblewrap.py   # Linux
│   │   ├── seatbelt.py     # macOS
│   │   └── docker.py       # Container fallback
│   ├── secrets.py          # Secret resolver
│   ├── hooks.py            # PreToolUse / PostToolUse / etc.
│   └── audit.py            # Append-only audit log
├── governance/             # Resource governance
│   ├── budget.py           # Token / call / cost budgets
│   └── enforcer.py         # Hooks into the loop
├── data/                   # Data certification
│   ├── lineage.py          # CertifiedValue type
│   ├── policy.py           # Freshness rules
│   └── extractor.py        # Pulls metadata from MCP responses
├── observability/
│   ├── tracing.py          # OpenTelemetry adapter
│   ├── metrics.py          # OTel metrics
│   └── evals.py            # Inline eval signals
├── agent/                  # The orchestrator
│   ├── api.py              # The public Agent class
│   ├── config.py           # AgentConfig resolution from kwargs
│   ├── session.py          # Per-run session state
│   ├── loop.py             # The actual agent loop
│   └── deps.py             # Dependency container
├── jeeves/                 # First-party Jeeves integration
│   ├── client.py           # Jeeves gateway client
│   └── auth.py             # jm_sk_* token handling
└── cli/                    # Optional: dev CLI
    └── main.py
```

### Dependency direction

```
       agent/  ← public layer, depends on everything below
         │
   ┌─────┴─────┐
   ▼           ▼
governance/  observability/
   │           │
   ├───────────┤
   ▼           ▼
security/    data/
   │           │
   └─────┬─────┘
         ▼
       mcp/
         │
   ┌─────┴─────┐
   ▼           ▼
runtime/    memory/
   │           │
   └─────┬─────┘
         ▼
       model/
         │
         ▼
     core/, async_/   (no dependencies)
```

A module never imports from a module above it. Tests for each layer use fakes for the layers below. This is enforced by import-linter or similar in CI.

---

## 5. Core protocols (the interface layer)

This is where the engineering rigor lives. Every interface is a `Protocol`. Concrete implementations live in their own modules. The loop knows nothing about which implementation it has — only the interface.

```python
# jeevesagent/core/protocols.py

from __future__ import annotations
from typing import Protocol, AsyncIterator, AsyncContextManager, Mapping, Any
from typing import runtime_checkable
from datetime import datetime
from .types import (
    Message, ModelChunk, ToolDef, ToolCall, ToolResult,
    Episode, MemoryBlock, Span, Event, RunResult,
    PermissionDecision, BudgetStatus,
)


@runtime_checkable
class Model(Protocol):
    """LLM provider interface. Adapter classes implement this for each lab."""

    name: str  # e.g. "claude-opus-4-7"

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ModelChunk]:
        """Stream completion chunks. Each chunk is text, tool_call, or finish."""
        ...


@runtime_checkable
class Memory(Protocol):
    """Tiered memory. Backed by working / episodic / semantic stores."""

    async def working(self) -> list[MemoryBlock]:
        """Return all in-context blocks. Pinned to context window."""
        ...

    async def update_block(self, name: str, content: str) -> None:
        ...

    async def append_block(self, name: str, content: str) -> None:
        ...

    async def remember(self, episode: Episode) -> str:
        """Persist an episode. Returns episode ID."""
        ...

    async def recall(
        self,
        query: str,
        *,
        kind: str = "episodic",  # "episodic" | "semantic" | "all"
        limit: int = 5,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list[Episode]:
        ...

    async def consolidate(self) -> None:
        """Background: extract semantic facts from recent episodes."""
        ...


@runtime_checkable
class Runtime(Protocol):
    """Durable execution. The journal of the loop."""

    async def step(
        self,
        name: str,
        fn,
        *args,
        idempotency_key: str | None = None,
        **kwargs,
    ) -> Any:
        """Execute fn as a journaled step. Replays cached on resume."""
        ...

    async def session(self, session_id: str) -> AsyncContextManager["RuntimeSession"]:
        """Open or resume a durable session."""
        ...

    async def signal(self, session_id: str, name: str, payload: Any) -> None:
        """Send an external signal to a running session (e.g., human approval)."""
        ...


@runtime_checkable
class ToolHost(Protocol):
    """MCP-aware tool registry. Lazy-loads tool schemas on demand."""

    async def list_tools(self, *, query: str | None = None) -> list[ToolDef]:
        """If query given, search; else return cached snapshot."""
        ...

    async def call(self, tool: str, args: Mapping[str, Any]) -> ToolResult:
        ...

    async def watch(self) -> AsyncIterator["ToolEvent"]:
        """Notifications when tool list changes (MCP listChanged)."""
        ...


@runtime_checkable
class Sandbox(Protocol):
    """Isolation layer for tool execution."""

    async def execute(self, tool: ToolDef, args: Mapping[str, Any]) -> ToolResult:
        ...

    async def with_filesystem(self, root: str) -> AsyncContextManager[None]:
        """Temporary filesystem sandbox for the duration of the context."""
        ...


@runtime_checkable
class Permissions(Protocol):
    """Decides whether a tool call is allowed."""

    async def check(self, call: ToolCall, *, context: Mapping[str, Any]) -> PermissionDecision:
        ...


@runtime_checkable
class HookHost(Protocol):
    """User-registered callbacks for lifecycle events."""

    async def pre_tool(self, call: ToolCall) -> PermissionDecision:
        ...

    async def post_tool(self, call: ToolCall, result: ToolResult) -> None:
        ...

    async def on_event(self, event: Event) -> None:
        ...


@runtime_checkable
class Budget(Protocol):
    """Resource governance — tokens, calls, cost."""

    async def allows_step(self) -> BudgetStatus:
        ...

    async def consume(self, *, tokens_in: int, tokens_out: int, cost_usd: float) -> None:
        ...


@runtime_checkable
class Telemetry(Protocol):
    """OpenTelemetry-compatible tracing/metrics."""

    def trace(self, name: str, **attrs: Any) -> AsyncContextManager[Span]:
        ...

    async def emit_metric(self, name: str, value: float, **attrs: Any) -> None:
        ...
```

A few things worth highlighting:

- **Every method that does I/O is `async`.** No exceptions.
- **`AsyncContextManager` for resources, `AsyncIterator` for streams.** No callback APIs.
- **`runtime_checkable` is selective** — it adds runtime overhead and prevents future method additions, so we use it only where users genuinely need `isinstance()` checks (typically the four core protocols).
- **No inheritance hierarchies in the protocol layer.** A `PostgresMemory` and an `InMemoryMemory` both satisfy `Memory` without sharing a base class.

---

## 6. The Agent class — public API

This is the surface 95% of users will touch. Every other module exists to make this class do its job.

```python
# jeevesagent/agent/api.py

from __future__ import annotations
from typing import AsyncIterator, Awaitable, Callable
import anyio
from contextlib import asynccontextmanager

from ..core.types import RunResult, Event, ToolCall, PermissionDecision
from .config import AgentConfig
from .deps import Dependencies
from .session import Session
from .loop import AgentLoop


class Agent:
    """The harness. Pass instructions; everything else has sensible defaults."""

    def __init__(
        self,
        instructions: str,
        *,
        # All optional. Strings are profile names; objects are explicit configs.
        model: str | "Model" = "claude-opus-4-7",
        tools: list | None = None,
        memory: str | "Memory" | "MemoryConfig" | None = None,
        runtime: str | "Runtime" | None = None,
        permissions: "Permissions" | None = None,
        budget: "Budget" | "BudgetConfig" | None = None,
        telemetry: str | "Telemetry" | None = None,
        # Hook-style callbacks
        on_event: Callable[[Event], Awaitable[None]] | None = None,
    ) -> None:
        self._cfg = AgentConfig.resolve(
            instructions=instructions,
            model=model,
            tools=tools or [],
            memory=memory,
            runtime=runtime,
            permissions=permissions,
            budget=budget,
            telemetry=telemetry,
        )
        self._on_event = on_event
        self._user_hooks = _HookRegistry()

    # --- Hook decorators -------------------------------------------------

    def before_tool(self, fn: Callable[[ToolCall], Awaitable[PermissionDecision]]):
        self._user_hooks.before_tool.append(fn)
        return fn

    def after_tool(self, fn):
        self._user_hooks.after_tool.append(fn)
        return fn

    def on(self, event_name: str):
        def deco(fn):
            self._user_hooks.named[event_name].append(fn)
            return fn
        return deco

    # --- Public API ------------------------------------------------------

    async def run(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        timeout: float | None = None,
    ) -> RunResult:
        """Run to completion. Returns the final result."""
        async with self._open_session(session_id) as session:
            if timeout is not None:
                with anyio.fail_after(timeout):
                    return await session.run(prompt)
            return await session.run(prompt)

    async def stream(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
    ) -> AsyncIterator[Event]:
        """Stream Events as they happen. Yields until terminal event."""
        async with self._open_session(session_id) as session:
            async for event in session.stream(prompt):
                if self._on_event:
                    await self._on_event(event)
                yield event

    async def resume(self, session_id: str) -> RunResult:
        """Resume a previously-interrupted session from the journal."""
        async with self._open_session(session_id, resume=True) as session:
            return await session.resume()

    # --- Session lifecycle ----------------------------------------------

    @asynccontextmanager
    async def _open_session(self, session_id: str | None, *, resume: bool = False):
        deps = await Dependencies.build(self._cfg, self._user_hooks)
        async with deps.lifecycle():
            session = Session(deps, session_id=session_id, resume=resume)
            yield session
```

Three things to notice:

1. **No subclassing required.** Hooks are decorators that register callbacks. Users compose, not extend.
2. **`run`, `stream`, `resume` are the only entry points.** Three verbs cover every use case. `run` for fire-and-forget, `stream` for UI feedback, `resume` for crash recovery.
3. **`Dependencies.build()` is where the magic happens.** This resolves strings/configs to real objects, wires them together, and returns a fully-formed dependency graph. It's the single place where "the framework decides things."

---

## 7. The Loop — internal architecture

The loop is the heart. It must be:
- **Deterministic** (replayable from the journal)
- **Cancellable** (cooperates with cancel scopes)
- **Streaming** (yields events as they happen)
- **Budget-aware** (terminates cleanly when limits are hit)

```python
# jeevesagent/agent/loop.py

from __future__ import annotations
from typing import AsyncIterator
import anyio
from anyio.abc import TaskGroup

from ..core.types import (
    Event, EventKind, ToolCall, ToolResult, Message, ModelChunk
)
from .deps import Dependencies
from .session import Session


class AgentLoop:
    """The agent's main loop. Pure orchestration; all I/O delegated."""

    def __init__(self, deps: Dependencies):
        self.d = deps  # short alias; we reference this a lot

    async def run(self, session: Session, prompt: str) -> AsyncIterator[Event]:
        """Run the loop, yielding events. Caller decides whether to consume all."""
        # 1. Build initial context: working memory + recalled episodes + prompt
        await self._seed_context(session, prompt)
        yield Event.started(session.id, prompt)

        # 2. Main loop — model calls + tool dispatches alternating
        async with anyio.create_task_group() as tg:
            try:
                async for event in self._iterate(session, tg):
                    yield event
            except* Exception as eg:  # noqa
                # ExceptionGroup from task group; surface first as Event.error
                yield Event.error(session.id, eg)
                raise

        yield Event.completed(session.id, session.result)

    # ----------------------------------------------------------------

    async def _seed_context(self, session: Session, prompt: str) -> None:
        # Working memory blocks always present
        blocks = await self.d.memory.working()
        session.append_system("\n\n".join(b.format() for b in blocks))

        # Recall relevant episodes (parallel: episodic + semantic)
        async with anyio.create_task_group() as tg:
            episodes_handle: list = [None]
            facts_handle: list = [None]

            async def _recall_episodic():
                episodes_handle[0] = await self.d.memory.recall(prompt, kind="episodic", limit=3)

            async def _recall_semantic():
                facts_handle[0] = await self.d.memory.recall(prompt, kind="semantic", limit=5)

            tg.start_soon(_recall_episodic)
            tg.start_soon(_recall_semantic)

        if episodes_handle[0]:
            session.append_system(_format_episodes(episodes_handle[0]))
        if facts_handle[0]:
            session.append_system(_format_facts(facts_handle[0]))

        session.append_user(prompt)

    # ----------------------------------------------------------------

    async def _iterate(self, session: Session, tg: TaskGroup) -> AsyncIterator[Event]:
        while not session.complete:
            # Budget check first — fail closed if we're over.
            status = await self.d.budget.allows_step()
            if status.blocked:
                yield Event.budget_exceeded(session.id, status)
                session.complete = True
                return

            # Model call (durable step — replayable on resume)
            chunk_aggregate = ModelChunkAggregate()
            async for chunk in self._stream_model(session):
                chunk_aggregate.feed(chunk)
                yield Event.model_chunk(session.id, chunk)

            # Track tokens, costs
            await self.d.budget.consume(
                tokens_in=chunk_aggregate.usage.input_tokens,
                tokens_out=chunk_aggregate.usage.output_tokens,
                cost_usd=chunk_aggregate.usage.cost_usd,
            )

            # If model called tools, dispatch them (parallel) and loop again.
            # If model finished without tools, we're done.
            if chunk_aggregate.tool_calls:
                results = await self._dispatch_tools(
                    session, chunk_aggregate.tool_calls, tg
                )
                for r in results:
                    yield Event.tool_result(session.id, r)
                session.append_tool_results(results)
            else:
                session.complete = True
                session.result = chunk_aggregate.text

            # Persist this turn as an episode (fire-and-forget durable step)
            await self.d.runtime.step(
                f"persist_turn_{session.turn_count}",
                self.d.memory.remember,
                session.last_episode(),
            )
            session.turn_count += 1

    # ----------------------------------------------------------------

    async def _stream_model(self, session: Session) -> AsyncIterator[ModelChunk]:
        """Wraps the model stream as a single durable step.

        On replay, the cached final chunk-aggregate is returned and we
        re-emit it as a single chunk (no double charging tokens).
        """
        # The runtime decides: real call, or replay from journal?
        async for chunk in self.d.runtime.stream_step(
            f"model_call_{session.turn_count}",
            self.d.model.stream,
            session.messages,
            tools=session.tools_view(),
        ):
            yield chunk

    # ----------------------------------------------------------------

    async def _dispatch_tools(
        self,
        session: Session,
        calls: list[ToolCall],
        tg: TaskGroup,
    ) -> list[ToolResult]:
        """Parallel tool dispatch with structured concurrency.

        Each tool call is its own durable step. Failures of individual
        tools become ToolResult.error; the loop continues.
        """
        results: list[ToolResult | None] = [None] * len(calls)

        async def _run_one(i: int, call: ToolCall) -> None:
            # Pre-tool hook (user code)
            decision = await self.d.hooks.pre_tool(call)
            if decision.deny:
                results[i] = ToolResult.denied(call.id, decision.reason)
                return

            # Permission check (system)
            perm = await self.d.permissions.check(call, context=session.context)
            if perm.deny:
                results[i] = ToolResult.denied(call.id, perm.reason)
                return

            # Execute through the sandbox, journaled by the runtime
            try:
                result = await self.d.runtime.step(
                    f"tool_call_{session.turn_count}_{i}",
                    self.d.sandbox.execute,
                    call.tool_def,
                    call.args,
                    idempotency_key=call.idempotency_key(),
                )
            except Exception as e:
                result = ToolResult.error(call.id, str(e))

            # Post-tool hook (user code, async, can't fail the loop)
            with anyio.move_on_after(5.0):
                await self.d.hooks.post_tool(call, result)

            results[i] = result

        for i, call in enumerate(calls):
            tg.start_soon(_run_one, i, call)
        # Task group exit waits for all child tasks. Cancellation propagates.

        # Replace any None with denial (shouldn't happen, defensive)
        return [r or ToolResult.error(c.id, "no_result") for r, c in zip(results, calls)]
```

### Why this loop design

**Deterministic surface, non-deterministic core.** The loop is pure Python that walks a state machine. Every side-effecting call goes through `runtime.step()` or `runtime.stream_step()`, which decides at runtime whether to actually execute or replay from journal. This is exactly the Temporal/DBOS pattern that lets OpenAI's Codex resume across crashes.

**Parallel tool dispatch with structured concurrency.** When the model emits N tool calls, we run them concurrently in a task group. If one throws, the task group cancels siblings and the exception surfaces as an `ExceptionGroup`. We catch that with `except*` syntax (PEP 654) and convert to an error event.

**Per-turn durability.** Each turn ends with persisting an episode. If the process crashes mid-turn, the journal has the LLM call result already; on resume we skip back to the tool dispatch.

**Hook isolation.** User hooks run in a `move_on_after(5.0)` scope so a buggy `after_tool` callback can't deadlock the loop. The loop never trusts user code to be fast or correct.

---

## 8. Layer 0: Durable runtime

### 8.1 The contract

```python
class Runtime(Protocol):
    async def step(self, name: str, fn, *args, **kwargs) -> Any: ...
    async def stream_step(self, name: str, fn, *args, **kwargs) -> AsyncIterator: ...
    async def session(self, session_id: str) -> AsyncContextManager[RuntimeSession]: ...
    async def signal(self, session_id: str, name: str, payload: Any) -> None: ...
```

Two semantics:
- `step`: regular journaled call. Returns a value. On replay, returns the cached value.
- `stream_step`: streaming journaled call. Yields chunks. On replay, yields a single aggregate chunk.

### 8.2 Three implementations

| Implementation | When to use | Infra cost |
|---|---|---|
| `InProcRuntime` | Dev, simple scripts, low-stakes | Nothing |
| `DBOSRuntime` | Production, single-service | Postgres only |
| `TemporalRuntime` | Multi-service, weeks-long workflows | Temporal cluster |

All three implement the same `Runtime` protocol. Switching is a string change.

### 8.3 InProcRuntime — the default

```python
# jeevesagent/runtime/inproc.py

class InProcRuntime:
    """No durability. Each step just runs. Used in dev, tests, and demos."""

    name = "inproc"

    def __init__(self) -> None:
        self._sessions: dict[str, "InProcSession"] = {}

    async def step(self, name, fn, *args, idempotency_key=None, **kwargs):
        return await fn(*args, **kwargs)

    async def stream_step(self, name, fn, *args, **kwargs):
        async for chunk in fn(*args, **kwargs):
            yield chunk

    @asynccontextmanager
    async def session(self, session_id: str):
        s = self._sessions.setdefault(session_id, InProcSession(session_id))
        try:
            yield s
        finally:
            pass  # No persistence

    async def signal(self, session_id, name, payload):
        s = self._sessions.get(session_id)
        if s:
            await s.deliver(name, payload)
```

### 8.4 DBOSRuntime — the production default

DBOS gives us journal-based replay using just Postgres. The runtime wraps DBOS workflows with our protocol surface. Critical: every `step()` call becomes a DBOS communicator (their term for an external side-effecting call), and the journal is keyed on `(session_id, step_name)`.

```python
# jeevesagent/runtime/dbos.py

# Pseudo-code; real impl uses dbos-py decorators
class DBOSRuntime:
    name = "dbos"

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    async def step(self, name, fn, *args, idempotency_key=None, **kwargs):
        # DBOS records (workflow_id, step_name) → result in Postgres
        # On replay, returns cached result without executing fn
        return await dbos_communicator(name, idempotency_key)(fn)(*args, **kwargs)

    # ... etc
```

The user toggles this with a string:

```python
agent = Agent("...", runtime="dbos")  # or "inproc" or "temporal"
```

Or with explicit config:

```python
agent = Agent("...", runtime=DBOSRuntime(dsn="postgres://..."))
```

### 8.5 The replay contract

For replay to work, every `step()`'s `fn` must be:
- **Idempotent** when called with the same args (or use `idempotency_key`)
- **Deterministic given its inputs** (no `time.time()`, no `random` without a seed, no implicit reads of mutable state)

We do not enforce this with linters in v1, but we document it loudly. Future: a static check that scans `step()` callees for forbidden symbols.

---

## 9. Layer 1: Memory subsystem

This is your biggest differentiator. Get this right and people will switch from LangGraph for this reason alone.

### 9.1 The three tiers

Following MemGPT (UC Berkeley, peer-reviewed) but with Zep-style temporal graph for semantic.

```python
@dataclass
class MemoryBlock:
    """In-context, agent-editable. Pinned to every prompt."""
    name: str            # e.g. "user_profile", "task_state"
    content: str
    updated_at: datetime

@dataclass
class Episode:
    """A single (input, decisions, tool calls, output) tuple from history."""
    id: str
    session_id: str
    occurred_at: datetime
    input: str
    output: str
    tool_calls: list[ToolCall]
    embedding: list[float] | None = None

@dataclass
class Fact:
    """A semantic claim extracted from one or more episodes."""
    id: str
    subject: str
    predicate: str
    object: str
    confidence: float
    valid_from: datetime
    valid_until: datetime | None  # None = currently valid
    recorded_at: datetime
    sources: list[str]  # episode IDs
```

The bi-temporal tracking on `Fact` is the Zep insight — "valid_from/valid_until" tracks when the fact was *true in the world*; "recorded_at" tracks when *we learned it*. This lets the agent reason about evolving state correctly.

### 9.2 The Memory implementation

```python
# jeevesagent/memory/postgres.py

class PostgresMemory:
    """Production memory backend. Postgres + pgvector."""

    name = "postgres"

    def __init__(
        self,
        pool: "asyncpg.Pool",
        embedder: "Embedder",
        working_token_budget: int = 10_000,
    ) -> None:
        self._pool = pool
        self._emb = embedder
        self._working_budget = working_token_budget

    async def working(self) -> list[MemoryBlock]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT name, content, updated_at FROM memory_blocks "
                "WHERE namespace = $1 ORDER BY pinned_order ASC",
                self._namespace,
            )
        return [MemoryBlock(**r) for r in rows]

    async def update_block(self, name: str, content: str) -> None:
        # Validate before write — see Principle 2.4
        if len(content) > self._max_block_chars:
            raise ValueError(f"block content exceeds {self._max_block_chars} chars")
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO memory_blocks(namespace, name, content, updated_at) "
                "VALUES($1, $2, $3, NOW()) "
                "ON CONFLICT(namespace, name) DO UPDATE "
                "SET content = $3, updated_at = NOW()",
                self._namespace, name, content,
            )

    async def remember(self, episode: Episode) -> str:
        # Embed in parallel with insert prep (anyio.create_task_group)
        async with anyio.create_task_group() as tg:
            embed_handle: list = [None]

            async def _embed():
                embed_handle[0] = await self._emb.embed(
                    f"{episode.input}\n{episode.output}"
                )
            tg.start_soon(_embed)

        episode.embedding = embed_handle[0]
        async with self._pool.acquire() as conn:
            await conn.execute(_INSERT_EPISODE_SQL, *_episode_params(episode))
        return episode.id

    async def recall(self, query, *, kind="episodic", limit=5, time_range=None):
        if kind == "episodic":
            return await self._recall_episodic(query, limit, time_range)
        elif kind == "semantic":
            return await self._recall_semantic(query, limit)
        elif kind == "all":
            # Parallel fan-out
            async with anyio.create_task_group() as tg:
                ep_h, sem_h = [None], [None]
                async def _ep(): ep_h[0] = await self._recall_episodic(query, limit, time_range)
                async def _sem(): sem_h[0] = await self._recall_semantic(query, limit)
                tg.start_soon(_ep)
                tg.start_soon(_sem)
            return (ep_h[0] or []) + (sem_h[0] or [])

    async def _recall_episodic(self, query, limit, time_range):
        emb = await self._emb.embed(query)
        # Vector search via pgvector
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, session_id, occurred_at, input, output, tool_calls "
                "FROM episodes "
                "WHERE namespace = $1 "
                "AND ($2::timestamptz IS NULL OR occurred_at >= $2) "
                "AND ($3::timestamptz IS NULL OR occurred_at <= $3) "
                "ORDER BY embedding <=> $4 "
                "LIMIT $5",
                self._namespace,
                time_range[0] if time_range else None,
                time_range[1] if time_range else None,
                emb,
                limit,
            )
        return [Episode(**r) for r in rows]

    async def consolidate(self) -> None:
        """Background extraction of facts from recent episodes.

        Runs as a separate worker process in production; called manually
        in tests. Uses an LLM to extract (subject, predicate, object) tuples
        and writes them to the facts table with bi-temporal tracking.
        """
        # ... (covered in detail in §9.4)
        pass
```

### 9.3 Memory exposed as MCP server

This is the differentiator. The `Memory` is *itself* an MCP server, so the agent can be told "you can call `memory.recall(...)`" the same way it'd be told about Gmail. Internal in-process; same protocol as remote.

```python
# jeevesagent/memory/server.py

from mcp.server import Server  # python-mcp-sdk

def build_memory_server(memory: Memory) -> Server:
    server = Server("jeeves-memory")

    @server.tool()
    async def recall(query: str, kind: str = "episodic", limit: int = 5):
        episodes = await memory.recall(query, kind=kind, limit=limit)
        return [e.format() for e in episodes]

    @server.tool()
    async def update_block(name: str, content: str):
        await memory.update_block(name, content)
        return {"ok": True}

    @server.tool()
    async def append_block(name: str, content: str):
        await memory.append_block(name, content)
        return {"ok": True}

    return server
```

Result: when the user adds custom memory tools, or wants to swap the memory backend for Mem0, they just point the agent's MCP registry at a different server. **Same interface. Same agent code.**

### 9.4 Background semantic consolidation

Episodes pile up; we don't want to reason over raw episodes forever. A background worker reads episodes, asks an LLM to extract facts, and writes them to the temporal graph.

```python
# jeevesagent/memory/semantic.py

class Consolidator:
    """Runs as a background worker — separate process or asyncio task.

    Pulls unconsolidated episodes in batches, extracts facts via LLM,
    writes them with bi-temporal tracking.
    """

    def __init__(
        self,
        memory: Memory,
        model: Model,
        batch_size: int = 50,
        cadence: float = 60.0,  # seconds between batches
    ):
        self._memory = memory
        self._model = model
        self._batch_size = batch_size
        self._cadence = cadence

    async def run(self) -> None:
        """Run forever. Cancel scope to stop."""
        while True:
            batch = await self._memory.unconsolidated(limit=self._batch_size)
            if not batch:
                await anyio.sleep(self._cadence)
                continue

            # Process batch in parallel (with concurrency limit)
            sem = anyio.Semaphore(10)  # max 10 concurrent extractions
            async with anyio.create_task_group() as tg:
                for ep in batch:
                    tg.start_soon(self._consolidate_one, ep, sem)

    async def _consolidate_one(self, episode: Episode, sem: anyio.Semaphore):
        async with sem:
            facts = await self._extract_facts(episode)
            await self._memory.write_facts(facts, source=episode.id)
            await self._memory.mark_consolidated(episode.id)
```

This runs as a separate `anyio` task spawned by the harness on `Agent` startup, or as its own process for production scale.

### 9.5 Memory performance target

LangMem's 59.82s p95 on LOCOMO is unusable. Mem0's 200ms is the bar.

We target **p95 ≤ 300ms** for `recall()` on a 1M-episode corpus, with:
- pgvector HNSW index for embedding search
- Connection pooling via asyncpg (target 20 max conns)
- Embedding cache (LRU on query text)
- Batched writes for `remember()` (flush every 100ms or 50 episodes)

---

## 10. Layer 2: Security harness

### 10.1 Permissions

Three modes, matching the Claude Agent SDK so users don't relearn:

```python
class Mode(StrEnum):
    DEFAULT = "default"           # prompt for approval on sensitive tools
    ACCEPT_EDITS = "acceptEdits"  # auto-approve fs writes; prompt for bash
    BYPASS = "bypassPermissions"  # CI / sandbox only

@dataclass
class PermissionsConfig:
    mode: Mode = Mode.DEFAULT
    allowed_tools: list[str] | None = None     # allow-list
    denied_tools: list[str] | None = None      # deny-list (wins over allow)
    allowed_filesystem_roots: list[str] | None = None
    sandbox: str | Sandbox | None = "auto"     # auto-detect bubblewrap/Seatbelt
```

The check itself is a small protocol method:

```python
class StandardPermissions:
    async def check(self, call: ToolCall, *, context) -> PermissionDecision:
        # 1. Check deny-list first
        if self._cfg.denied_tools and call.tool in self._cfg.denied_tools:
            return PermissionDecision.deny("tool denied by policy")

        # 2. Check allow-list
        if self._cfg.allowed_tools and call.tool not in self._cfg.allowed_tools:
            return PermissionDecision.deny("tool not in allow-list")

        # 3. Mode-specific logic
        if self._cfg.mode == Mode.BYPASS:
            return PermissionDecision.allow()

        if self._cfg.mode == Mode.ACCEPT_EDITS and _is_safe_edit(call):
            return PermissionDecision.allow()

        # 4. Default: ask user
        return PermissionDecision.ask()  # loop will surface to hook
```

### 10.2 Sandbox

Three implementations, picked at runtime based on platform:

```python
class Sandbox(Protocol):
    async def execute(self, tool: ToolDef, args: Mapping[str, Any]) -> ToolResult: ...

class BubblewrapSandbox: ...   # Linux
class SeatbeltSandbox: ...     # macOS
class DockerSandbox: ...       # Cross-platform fallback
class NoSandbox: ...           # Dev only, never default in prod
```

The sandbox boundary is defined per tool category:
- Filesystem tools → restricted to declared roots
- Network tools → restricted to declared domains
- Code execution → fully isolated (gVisor/Firecracker for Docker variant)

**Anthropic's reported impact**: 84% reduction in approval prompts when sandbox-based scoping is on. We aim for similar.

### 10.3 Hooks

Hooks are user-registered callbacks at lifecycle points. The harness calls them; user code decides what happens.

```python
class HookRegistry:
    pre_tool: list[Callable[[ToolCall], Awaitable[PermissionDecision]]]
    post_tool: list[Callable[[ToolCall, ToolResult], Awaitable[None]]]
    pre_model: list[Callable[[list[Message]], Awaitable[None]]]
    post_model: list[Callable[[ModelChunk], Awaitable[None]]]
    on_event: dict[str, list[Callable[[Event], Awaitable[None]]]]

    async def fire_pre_tool(self, call: ToolCall) -> PermissionDecision:
        """Run all pre_tool hooks; first deny wins."""
        for hook in self.pre_tool:
            with anyio.move_on_after(5.0):
                decision = await hook(call)
                if decision and decision.deny:
                    return decision
        return PermissionDecision.allow()
```

Hooks always run in a timeout scope so a buggy hook can't hang the loop. Decorator API is on `Agent`:

```python
@agent.before_tool
async def review(call):
    if call.tool == "fs.write" and call.args["path"].startswith("/etc/"):
        return PermissionDecision.deny("no system path writes")
```

### 10.4 Audit log

Every tool call, permission decision, hook outcome, and memory write is appended to an immutable log. Postgres table with monotonic sequence. Queryable for compliance and post-mortems.

```python
@dataclass(frozen=True)
class AuditEntry:
    seq: int
    timestamp: datetime
    session_id: str
    actor: str  # "user" | "model" | "system" | "hook:<name>"
    action: str  # "tool_call" | "permission_decision" | ...
    payload: dict
    signature: str  # HMAC for tamper detection
```

### 10.5 Secret handling

Secrets never reach the LLM. Tools that need them reference by name; the harness resolves at call time and strips from logged transcripts.

```python
class Secrets(Protocol):
    async def resolve(self, ref: str) -> str: ...
    async def store(self, ref: str, value: str) -> None: ...
    def redact(self, text: str) -> str: ...  # find known secrets, replace with [REDACTED]
```

---

## 11. Layer 3: MCP spine

This is where Jeeves directly plugs in.

### 11.1 The registry

```python
# jeevesagent/mcp/registry.py

class MCPRegistry:
    """Aggregates many MCP servers into a single ToolHost.

    Implements lazy tool loading: schemas are not pulled until queried,
    matching Claude Code's 95% context reduction pattern.
    """

    def __init__(self, servers: list["MCPServerSpec"]) -> None:
        self._servers = servers
        self._clients: dict[str, "MCPClient"] = {}
        self._tool_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)
        self._search_index: BM25Index | None = None  # for tool_search

    async def __aenter__(self):
        # Connect all servers in parallel
        async with anyio.create_task_group() as tg:
            for spec in self._servers:
                tg.start_soon(self._connect, spec)
        return self

    async def _connect(self, spec: MCPServerSpec):
        client = await MCPClient.connect(spec)
        self._clients[spec.name] = client

    async def list_tools(self, *, query: str | None = None) -> list[ToolDef]:
        if query:
            # Tool search: small set of tools relevant to query
            return await self._search_tools(query, limit=10)
        # Otherwise: full tool list (schemas) — heavy
        return await self._all_tools()

    async def _search_tools(self, query: str, limit: int) -> list[ToolDef]:
        if self._search_index is None:
            await self._build_search_index()
        names = self._search_index.search(query, limit=limit)
        return [self._tool_cache[n] for n in names if n in self._tool_cache]

    async def call(self, tool: str, args) -> ToolResult:
        # Tool name format: "{server}.{tool}" or just "{tool}" if unique
        server_name, tool_name = self._resolve(tool)
        client = self._clients[server_name]
        return await client.call_tool(tool_name, args)

    async def watch(self) -> AsyncIterator[ToolEvent]:
        """Aggregate tool-list-changed notifications from all servers."""
        send, receive = anyio.create_memory_object_stream(max_buffer_size=100)
        async with anyio.create_task_group() as tg:
            for client in self._clients.values():
                tg.start_soon(self._forward_events, client, send)
            async with receive:
                async for event in receive:
                    yield event
```

### 11.2 Lazy tool loading (the killer feature)

Loading every MCP tool's schema into context every turn is the single biggest source of context bloat. Claude Code achieves 95% context reduction by *not doing this*. Instead:

1. At startup: build a search index over tool names + descriptions
2. The loop exposes a meta-tool: `tool_search(query)` → returns 5-10 tool schemas
3. The agent calls `tool_search` first when it needs a tool, gets a small set, then calls one
4. Most turns never load most tools

We adopt this pattern by default. It's a single line for the user to change:

```python
agent = Agent("...", mcp=MCPConfig(lazy_load=True))   # default
agent = Agent("...", mcp=MCPConfig(lazy_load=False))  # all tools always loaded
```

### 11.3 Jeeves first-class integration

The `jeeves` module wraps the user's Jeeves gateway as a pre-built MCP server config:

```python
# jeevesagent/jeeves/client.py

@dataclass
class JeevesConfig:
    api_key: str  # jm_sk_*
    base_url: str = "https://jeeves.works/mcp"
    transport: str = "streamable_http"

class JeevesGateway:
    """Convenience wrapper around the Jeeves MCP gateway."""

    @classmethod
    def from_env(cls) -> "JeevesGateway":
        key = os.environ["JEEVES_API_KEY"]
        return cls(JeevesConfig(api_key=key))

    def as_mcp_server(self) -> MCPServerSpec:
        return MCPServerSpec(
            name="jeeves",
            transport=self._cfg.transport,
            url=f"{self._cfg.base_url}/{self._cfg.api_key}",
        )
```

Usage:

```python
from jeevesagent import Agent
from jeevesagent.jeeves import JeevesGateway

agent = Agent(
    "You are a productivity assistant",
    tools=[JeevesGateway.from_env()],
)
```

The harness recognizes `JeevesGateway` as a tool spec and adds its MCP server to the registry.

### 11.4 Transports

Two transports, matching MCP spec 2025-11-25:
- **stdio** — local subprocess, JSON-RPC over pipes
- **Streamable HTTP** — remote, JSON-RPC over HTTP with optional SSE

OAuth 2.1 flow for hosted MCP servers (incremental scope consent per the new spec). Tokens stored via the `Secrets` interface.

---

## 12. Layer 4: Data certification

Every value the agent reasons over carries provenance.

```python
@dataclass(frozen=True)
class CertifiedValue:
    value: Any
    source: str                    # MCP URI or "user_input"
    fetched_at: datetime
    valid_until: datetime | None
    schema_version: str
    lineage: tuple[str, ...]       # IDs of upstream values

    def is_fresh(self, policy: FreshnessPolicy) -> bool:
        max_age = policy.max_age_for(self.source)
        return (datetime.utcnow() - self.fetched_at) <= max_age

    def lineage_ok(self, policy: LineagePolicy) -> bool:
        return all(s in policy.allowed_sources for s in self.lineage)
```

Three checks happen at the MCP boundary:
1. Schema validation (Pydantic)
2. Freshness check (age vs. policy)
3. Lineage check (source allowed for this task)

When a check fails:
- Schema fail → exception, agent sees `ToolResult.error`
- Freshness fail → either re-fetch silently (default) or surface to agent ("data is 14 days stale, refetch?")
- Lineage fail → block the call, log as policy violation

Zero current frameworks do this. It's how we prevent the schema-drift / stale-data failure mode that's the #1 enterprise issue.

---

## 13. Layer 5: Resource governance

Five hard limits, all enforced before each step:

```python
@dataclass
class BudgetConfig:
    max_tokens: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    max_spawn_depth: int = 3
    max_wall_clock: timedelta | None = None
    max_cost_usd: float | None = None
    soft_warning_at: float = 0.8  # 80% triggers system message to agent
```

```python
class StandardBudget:
    async def allows_step(self) -> BudgetStatus:
        async with self._lock:
            if self._cfg.max_tokens and self._tokens_used >= self._cfg.max_tokens:
                return BudgetStatus.blocked("max_tokens")
            if self._cfg.max_cost_usd and self._cost_used >= self._cfg.max_cost_usd:
                return BudgetStatus.blocked("max_cost_usd")
            if self._cfg.max_wall_clock and self._elapsed() >= self._cfg.max_wall_clock:
                return BudgetStatus.blocked("max_wall_clock")
            if self._tokens_used >= self._cfg.max_tokens * self._cfg.soft_warning_at:
                return BudgetStatus.warn(f"tokens at {self._pct():.0%}")
            return BudgetStatus.ok()
```

The loop checks budget before every step. Soft warnings inject a system message ("80% of budget used, please wrap up"). Hard limits terminate cleanly with a `BudgetExceeded` event.

This is how we prevent the runaway loops Gartner reviewers complain about with LangGraph.

---

## 14. Layer 6: Observability

One word: **OpenTelemetry**.

```python
class OTelTelemetry:
    @asynccontextmanager
    async def trace(self, name: str, **attrs):
        with self._tracer.start_as_current_span(name, attributes=attrs) as span:
            yield span

    async def emit_metric(self, name: str, value: float, **attrs):
        self._meter.create_counter(name).add(value, attributes=attrs)
```

Spans we emit:
- `jeeves.run` (root span per agent.run() call)
- `jeeves.turn` (per loop iteration)
- `jeeves.model.stream` (per LLM call)
- `jeeves.tool.{name}` (per tool call)
- `jeeves.memory.recall`
- `jeeves.memory.remember`
- `jeeves.hook.{phase}`
- `jeeves.runtime.step`

Metrics:
- `jeeves.tokens.input` / `jeeves.tokens.output` (counter)
- `jeeves.cost.usd` (counter)
- `jeeves.tool.duration_ms` (histogram)
- `jeeves.session.duration_ms` (histogram)
- `jeeves.budget.exceeded` (counter)

Inline evals are a separate stream, emitted as events:

```python
class EvalSignal(BaseModel):
    name: str  # "schema_pass", "citation_present", "tool_success"
    value: float  # 0-1 score
    session_id: str
    turn: int
```

These accumulate per agent definition and trigger drift alerts.

---

## 15. Plugin architecture

Two layers of extension. Both are how you add new memory backends, model adapters, tools, sandboxes, runtimes — anything.

### 15.1 Direct construction (for in-tree code)

The most direct path: instantiate and pass.

```python
from my_company.memory import RedisMemory

agent = Agent("...", memory=RedisMemory(url=REDIS_URL))
```

`RedisMemory` just needs to satisfy the `Memory` protocol. No registration step.

### 15.2 Entry points (for distributed plugins)

Python's native plugin mechanism. A third-party package declares an entry point:

```toml
# pyproject.toml of jeevesagent-redis-memory
[project.entry-points."jeevesagent.memory"]
redis = "jeevesagent_redis_memory:RedisMemory"
```

Then users get string-based selection:

```python
agent = Agent("...", memory="redis")  # discovered via entry point
```

The harness resolves `"redis"` by scanning the `jeevesagent.memory` entry-point group at startup.

### 15.3 Entry-point groups

We publish these groups for the ecosystem:

| Group | What plugs in |
|---|---|
| `jeevesagent.model` | Model adapters |
| `jeevesagent.memory` | Memory backends |
| `jeevesagent.runtime` | Durable runtimes |
| `jeevesagent.sandbox` | Sandbox implementations |
| `jeevesagent.telemetry` | Telemetry exporters |
| `jeevesagent.tool` | Pre-built tool definitions |

A first-party plugin set ships in-tree (anthropic model, postgres memory, dbos runtime, etc.). Third-party plugins are pip-installable.

### 15.4 Resolver

```python
# jeevesagent/agent/config.py

def resolve_memory(spec) -> Memory:
    if isinstance(spec, Memory):
        return spec
    if isinstance(spec, MemoryConfig):
        return spec.build()
    if isinstance(spec, str):
        cls = _load_entry_point("jeevesagent.memory", spec)
        return cls()  # zero-arg default
    raise TypeError(f"unsupported memory spec: {spec!r}")
```

---

## 16. Concurrency patterns

Beyond the basic protocols, here are the patterns you'll see throughout the codebase.

### 16.1 Bounded fan-out

When dispatching N parallel operations with concurrency limit:

```python
sem = anyio.Semaphore(max_concurrent)

async def _bounded(item):
    async with sem:
        return await do_work(item)

async with anyio.create_task_group() as tg:
    for item in items:
        tg.start_soon(_bounded, item)
```

### 16.2 Producer/consumer

For pipelines that decouple producer rate from consumer rate:

```python
send, receive = anyio.create_memory_object_stream(max_buffer_size=50)

async def _producer():
    async with send:
        async for item in source:
            await send.send(item)

async def _consumer():
    async with receive:
        async for item in receive:
            await process(item)

async with anyio.create_task_group() as tg:
    tg.start_soon(_producer)
    tg.start_soon(_consumer)
```

Used in: streaming model output to multiple consumers (UI + telemetry + audit), background memory consolidation.

### 16.3 Race-with-fallback

"Try the fast path, fall back if too slow":

```python
async def call_with_fallback(primary, fallback, *, timeout=2.0):
    result = []

    async def _try_primary():
        with anyio.fail_after(timeout):
            result.append(await primary())

    async def _try_fallback():
        await anyio.sleep(timeout)  # only fires if primary times out
        result.append(await fallback())

    async with anyio.create_task_group() as tg:
        tg.start_soon(_try_primary)
        # Falls back only if primary task fails / times out
```

Used for: model fallback (Claude → GPT if Claude is rate-limited), memory recall (semantic → episodic if semantic is slow).

### 16.4 Cancellation-safe cleanup

```python
try:
    await main_work()
finally:
    with anyio.CancelScope(shield=True):
        # This runs even if we're being cancelled
        await release_locks()
        await flush_audit()
```

Used in: session teardown, MCP client disconnect, runtime checkpoint flush.

### 16.5 Forbidden patterns

These are removed from the codebase by linter rules:

```python
# ❌ Don't do this — leaks tasks
asyncio.create_task(work())

# ❌ Don't do this — no sibling cancellation
await asyncio.gather(work_a(), work_b())

# ❌ Don't do this — blocks the event loop
time.sleep(1)

# ❌ Don't do this — bypasses cancellation
threading.Thread(target=work).start()
```

The replacement for each is in §3.

---

## 17. Testing strategy

### 17.1 Test pyramid

```
  E2E tests (slow, few)
        ▲
        │     Integration tests (per layer, with real Postgres etc)
        │
        │       Unit tests (fast, many, with fakes for protocols)
        ▼
  Property tests (loop invariants, replay determinism)
```

### 17.2 Fakes-per-protocol

Each protocol ships a `Fake*` implementation in tests:

```python
# tests/fakes/memory.py

class FakeMemory:
    def __init__(self):
        self.episodes: dict[str, Episode] = {}
        self.blocks: dict[str, MemoryBlock] = {}

    async def working(self) -> list[MemoryBlock]:
        return list(self.blocks.values())

    async def remember(self, episode: Episode) -> str:
        self.episodes[episode.id] = episode
        return episode.id

    async def recall(self, query, **kwargs) -> list[Episode]:
        # Naive: return all, sorted by recency. Real impl uses embeddings.
        return sorted(self.episodes.values(), key=lambda e: e.occurred_at, reverse=True)[:5]
```

Tests use these instead of mocks — fakes are simpler to reason about and don't break on refactors.

### 17.3 The async test runner

We use `pytest` + `pytest-anyio` (anyio's own pytest plugin). All async tests run on both asyncio and trio backends to catch backend-specific bugs.

```python
# pyproject.toml
[tool.pytest.ini_options]
addopts = "--anyio-backend=asyncio,trio"
```

```python
import pytest

@pytest.mark.anyio
async def test_loop_dispatches_tools_in_parallel():
    deps = build_test_deps(memory=FakeMemory(), model=FakeModel(...), ...)
    loop = AgentLoop(deps)
    session = Session(deps)
    events = [e async for e in loop.run(session, "test")]
    # assertions
```

### 17.4 Property tests for replay

The most important invariant: a session that runs to completion must produce *byte-identical* output when replayed from its journal. We use `hypothesis` to generate random tool sequences and verify replay.

```python
from hypothesis import given, strategies as st

@given(st.lists(st.builds(ToolCall, ...)))
@pytest.mark.anyio
async def test_replay_identical(tool_calls):
    runtime = JournaledTestRuntime()
    deps = build_test_deps(runtime=runtime, ...)

    # First run
    result_1 = await Agent("test").run("prompt")
    journal = runtime.dump_journal()

    # Second run, same journal: must replay
    runtime_2 = JournaledTestRuntime(journal=journal)
    deps_2 = build_test_deps(runtime=runtime_2, ...)
    result_2 = await Agent("test").run("prompt")

    assert result_1 == result_2
```

### 17.5 Soak tests

A 24-hour test that runs 10k sessions with random failures injected — network drops, db timeouts, sandbox aborts. Verifies:
- No leaked tasks (`anyio` exits cleanly)
- No orphan db connections
- No memory growth beyond baseline + worked-set
- All sessions either complete or are resumable

---

## 18. Performance & scalability

### 18.1 Performance targets (v1)

| Operation | p50 | p95 | p99 |
|---|---|---|---|
| `memory.recall` (1M episodes) | 50ms | 300ms | 1s |
| `mcp.tool_search` | 10ms | 50ms | 100ms |
| `runtime.step` (in-proc) | <1ms | <5ms | 10ms |
| `runtime.step` (DBOS) | 5ms | 30ms | 100ms |
| Agent loop turn (no tools) | 50ms (model excluded) | 200ms | 500ms |

### 18.2 Connection pooling

- **Postgres**: `asyncpg.Pool`, default min=5 max=20, configurable
- **HTTP** (model APIs, MCP HTTP): `httpx.AsyncClient` shared, connection limits per host
- **MCP stdio**: one subprocess per server, kept alive across calls

### 18.3 Caching

- **Embedding cache**: LRU on (model, text), TTL 1 hour, max 10k entries
- **Tool schema cache**: TTL 5 minutes, invalidated by MCP `notifications/tools/list_changed`
- **Permission cache**: per-session, never crosses sessions

### 18.4 Batching

- **Episode persistence**: batch flush every 100ms or 50 episodes
- **Embedding generation**: batch by 10 (most providers support batch APIs)
- **Audit log**: batch flush every 1s or 100 entries

### 18.5 Horizontal scale

Each `Agent` instance is fully self-contained. To scale out:
- Run N replicas behind a load balancer (sessions are sticky via session_id)
- DBOS or Temporal handles cross-replica state
- Postgres scales via read replicas for memory recall, primary for writes
- MCP servers scale independently

There's no shared in-memory state in the harness itself. Two `Agent` instances on different machines never need to communicate.

---

## 19. Phased build with verification gates

12 weeks to v1. Each phase ends with a hard verification gate.

### Phase 1 (Weeks 1-2): Skeleton + protocols

- Package layout, pyproject.toml, CI
- Define every Protocol (no implementations yet)
- `core/`, `async_/` modules
- Pydantic types for all message shapes

**Gate**: `mypy --strict` passes. `ruff` clean. `pytest` runs (no tests yet).

### Phase 2 (Weeks 3-4): Bare loop + Anthropic adapter + InProcRuntime

- `Model` protocol + Anthropic adapter
- `InProcRuntime` (no durability)
- Agent loop with single-tool dispatch (no parallel yet)
- Trivial `Memory` (in-memory dict)
- Trivial `MCPRegistry` (no real MCP — mocks)

**Gate**: A scripted demo runs end-to-end on Claude. 10-test e2e suite passes.

### Phase 3 (Weeks 5-6): Real MCP + parallel tools

- Full MCP client (stdio + Streamable HTTP)
- `MCPRegistry` with lazy tool loading
- Parallel tool dispatch via task group
- OpenAI + LiteLLM model adapters

**Gate**: Connect to a real Jeeves MCP gateway. 20-test suite covering MCP edge cases.

### Phase 4 (Weeks 7-8): Memory subsystem

- `PostgresMemory` with pgvector
- Working/episodic/semantic tiers
- Background consolidator
- Memory exposed as MCP server

**Gate**: LOCOMO benchmark — p95 ≤ 500ms (vs. LangMem's 59.82s). Cross-session recall test.

### Phase 5 (Weeks 9-10): Durability

- DBOS adapter for Runtime
- Replay tests with hypothesis
- Resume API

**Gate**: Soak test — kill/resume 100 sessions. Property test passes for 1k random sequences.

### Phase 6 (Weeks 11-12): Security + Governance + Observability

- Permissions, sandbox (bubblewrap on Linux first)
- Hooks system
- Audit log
- Budget enforcement
- OpenTelemetry instrumentation

**Gate**: CVE replay — three known LangChain CVEs cannot be reproduced. Honeycomb dashboard shows traces. Budget tests pass.

### Phase 7 (Weeks 13+): Hardening, docs, ecosystem

- Data certification
- Temporal adapter
- Docker sandbox
- Documentation, examples
- First-party plugins for major MCP gateways (Jeeves, Composio, etc.)

**Gate**: 1.0.0 release.

---

## 20. The first PR: package skeleton

Here's what to commit first. This is the package as it should look on day one — empty implementations, but every module, every Protocol, every type in place. Everything compiles, mypy passes, no functionality yet.

### 20.1 `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "jeevesagent"
version = "0.0.1"
description = "Model-agnostic, MCP-native agent harness"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "Apache-2.0" }
dependencies = [
    "anyio>=4.4.0",
    "pydantic>=2.6.0",
    "httpx>=0.27.0",
    "asyncpg>=0.29.0",
    "pgvector>=0.2.5",
    "mcp>=1.0.0",
    "opentelemetry-api>=1.24.0",
    "opentelemetry-sdk>=1.24.0",
    "ulid-py>=1.1.0",
]

[project.optional-dependencies]
anthropic = ["anthropic>=0.40.0"]
openai = ["openai>=1.30.0"]
litellm = ["litellm>=1.40.0"]
dbos = ["dbos>=0.7.0"]
temporal = ["temporalio>=1.7.0"]
dev = [
    "pytest>=8.0",
    "pytest-anyio>=0.4",
    "hypothesis>=6.100",
    "mypy>=1.10",
    "ruff>=0.4",
    "import-linter>=2.0",
]

[project.entry-points."jeevesagent.model"]
anthropic = "jeevesagent.model.anthropic:AnthropicModel"
openai = "jeevesagent.model.openai:OpenAIModel"
litellm = "jeevesagent.model.litellm:LiteLLMModel"

[project.entry-points."jeevesagent.memory"]
postgres = "jeevesagent.memory.backends.postgres:PostgresMemory"
inmemory = "jeevesagent.memory.backends.memory:InMemoryMemory"

[project.entry-points."jeevesagent.runtime"]
inproc = "jeevesagent.runtime.inproc:InProcRuntime"
dbos = "jeevesagent.runtime.dbos:DBOSRuntime"
temporal = "jeevesagent.runtime.temporal:TemporalRuntime"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
strict = true
python_version = "3.11"

[tool.pytest.ini_options]
addopts = "--anyio-backend=asyncio,trio --strict-markers"
asyncio_mode = "auto"

[tool.importlinter]
root_package = "jeevesagent"

[[tool.importlinter.contracts]]
name = "layered architecture"
type = "layers"
layers = [
    "jeevesagent.agent",
    "jeevesagent.governance | jeevesagent.observability",
    "jeevesagent.security | jeevesagent.data",
    "jeevesagent.mcp",
    "jeevesagent.runtime | jeevesagent.memory",
    "jeevesagent.model",
    "jeevesagent.async_ | jeevesagent.core",
]
```

### 20.2 `jeevesagent/__init__.py`

```python
"""JeevesAgent — model-agnostic, MCP-native agent harness."""

from jeevesagent.agent.api import Agent
from jeevesagent.core.types import (
    Event, RunResult, ToolCall, ToolResult,
    Message, ModelChunk, Episode, MemoryBlock,
    PermissionDecision, BudgetStatus,
)
from jeevesagent.memory.base import Memory, MemoryConfig
from jeevesagent.runtime.base import Runtime
from jeevesagent.security.permissions import Permissions, Mode
from jeevesagent.governance.budget import Budget, BudgetConfig

__version__ = "0.0.1"

__all__ = [
    "Agent",
    "Memory", "MemoryConfig",
    "Runtime",
    "Permissions", "Mode",
    "Budget", "BudgetConfig",
    "Event", "RunResult",
    "ToolCall", "ToolResult",
    "Message", "ModelChunk",
    "Episode", "MemoryBlock",
    "PermissionDecision", "BudgetStatus",
]
```

### 20.3 The "hello world" the harness must support

The day Phase 2 ships, this code must work end-to-end:

```python
import asyncio
from jeevesagent import Agent

async def main():
    agent = Agent(
        "You are a helpful assistant. Be concise.",
        model="claude-opus-4-7",
    )
    result = await agent.run("What is 2+2?")
    print(result.output)

asyncio.run(main())
```

Three lines of user code. That's the bar.

---

## Closing notes

This document is the contract for what we're building and how. Three things to internalize:

1. **The harness is the product.** Not a kit. Not a framework. A complete thing the user instantiates and uses.
2. **Async-first is not a flavor — it's the design.** Every choice in this document — anyio, structured concurrency, async generators, memory streams — is downstream of that single commitment.
3. **Convention over configuration is a UX claim and an engineering claim.** It means the defaults must be excellent, and the escape hatches must be real. We owe both.

When in doubt, the order of priorities is:

1. **Correctness** — does it do the right thing under failure?
2. **Observability** — can we see what it did?
3. **Performance** — is it fast?
4. **Ergonomics** — is the API pleasant?

In that order. Always.

---

*End of engineering plan. Open the first PR with the skeleton in §20 when you're ready.*