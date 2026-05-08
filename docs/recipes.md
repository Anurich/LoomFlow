# Recipes

Concrete patterns lifted from production-shaped configurations. Copy
the recipe, swap the bits that vary, ship.

## Table of contents

1. [Customer-support bot with persistent facts](#1-customer-support-bot-with-persistent-facts)
2. [Coding assistant with sandboxed filesystem access](#2-coding-assistant-with-sandboxed-filesystem-access)
3. [Long-running research agent with durable replay](#3-long-running-research-agent-with-durable-replay)
4. [Multi-server MCP setup (Jeeves + git + filesystem)](#4-multi-server-mcp-setup)
5. [Custom embedder](#5-custom-embedder)
6. [Custom permissions policy](#6-custom-permissions-policy)
7. [Streaming UI integration](#7-streaming-ui-integration)
8. [Production checklist](#8-production-checklist)

---

## 1. Customer-support bot with persistent facts

The bot remembers what each customer told it, even across process
restarts. Facts (``user X says they live in Tokyo``,
``account Y is on the enterprise plan``) get extracted automatically
after every conversation.

```python
import asyncio
from jeevesagent import Agent, SqliteRuntime

async def main():
    # Postgres URL → resolver builds PostgresMemory + pgvector
    # facts table on first agent.run. Schema migrations are
    # idempotent; nothing to remember.
    agent = Agent(
        instructions=(
            "You are a customer-support agent for Acme. "
            "Use any known facts about the user to personalize replies. "
            "Cite the fact's source when you rely on it."
        ),
        model="claude-opus-4-7",
        memory="postgres://localhost/support_bot",
        runtime=SqliteRuntime("./support_journal.db"),
        # auto_extract=True is the default — every run auto-pulls
        # structured facts from the conversation into the bi-temporal
        # store, partitioned by user_id.
    )

    while True:
        prompt = input("User> ")
        if not prompt:
            break
        # In a real bot, ``user_id`` comes from your auth layer.
        result = await agent.run(prompt, user_id="user_42")
        print(f"Bot> {result.output}")

asyncio.run(main())
```

The first time a user mentions their plan tier, the consolidator
extracts a fact like
``("user", "subscription_plan", "enterprise")``. Later runs see it in
the ``Known facts:`` section of the system message and tailor
responses without asking again. Plan changes? Supersession closes off
the old fact's validity window automatically — historical queries
still work.

---

## 2. Coding assistant with sandboxed filesystem access

The agent can read and write files only inside a workspace directory.
Symlink-based escapes are blocked; an HMAC-signed audit log records
every file access.

```python
import asyncio
from pathlib import Path
from jeevesagent import (
    Agent, FileAuditLog, FilesystemSandbox, InProcessToolHost,
    Mode, StandardPermissions, tool,
)

WORKSPACE = Path("./workspace").resolve()

@tool
def read_file(path: str) -> str:
    """Read a file from the workspace."""
    return Path(path).read_text()

@tool(destructive=True)
def write_file(path: str, content: str) -> str:
    """Write content to a file (destructive — requires approval)."""
    Path(path).write_text(content)
    return f"wrote {len(content)} bytes to {path}"

async def main():
    inner = InProcessToolHost([read_file, write_file])
    sandbox = FilesystemSandbox(inner, roots=[WORKSPACE])

    agent = Agent(
        "You are a coding assistant. Only touch files inside the workspace.",
        model="claude-opus-4-7",
        tools=sandbox,
        permissions=StandardPermissions(mode=Mode.ACCEPT_EDITS),
        audit_log=FileAuditLog("./audit.jsonl", secret="prod-secret"),
    )

    @agent.before_tool
    async def confirm_destructive(call):
        if call.tool == "write_file":
            answer = input(f"Allow write to {call.args.get('path')}? [y/N] ")
            if answer.strip().lower() != "y":
                from jeevesagent.core.types import PermissionDecision
                return PermissionDecision.deny_("user declined")
        return None

    await agent.run("Refactor utils.py to use type hints.")

asyncio.run(main())
```

The sandbox auto-detects path-typed arguments by name (``path``,
``file``, ``directory``, etc.) and by value (containing ``/`` or
``\\``). Any path that resolves outside the workspace — including via
symlink — is rejected before the tool runs.

---

## 3. Long-running research agent with durable replay

The agent runs a multi-step research task. If the process crashes or
the host reboots, restart with the same session ID and pick up where
you left off.

```python
import asyncio
from jeevesagent import Agent, AnthropicModel, JeevesGateway, SqliteRuntime
from datetime import timedelta
from jeevesagent.governance.budget import BudgetConfig, StandardBudget

async def main():
    runtime = SqliteRuntime("./research_journal.db")
    agent = Agent(
        "You are a research assistant. Plan a multi-step research task, "
        "execute each step with the available tools, then summarize.",
        model=AnthropicModel("claude-opus-4-7"),
        runtime=runtime,
        tools=JeevesGateway.from_env(),
        budget=StandardBudget(BudgetConfig(
            max_tokens=500_000,
            max_cost_usd=20.0,
            max_wall_clock=timedelta(hours=2),
        )),
    )
    result = await agent.run("Research the state of agent harnesses in 2026.")
    print(result.output)

asyncio.run(main())
```

Every model call and every tool dispatch is journaled by
``(session_id, step_name)``. On a process restart, instantiating a
new ``SqliteRuntime`` against the same DB file with the same
session ID returns cached values for completed steps and only
re-executes the un-completed work.

(Today: session IDs are auto-generated per ``run()``. The explicit
``Agent.resume(session_id)`` API lands in a follow-up slice — for
now, the journaling itself is in place and tested at the runtime
layer.)

---

## 4. Multi-server MCP setup

Compose Jeeves Gateway with a local git server and a filesystem
server. Tool name conflicts get auto-disambiguated.

```python
from jeevesagent import (
    Agent, JeevesGateway, MCPClient, MCPRegistry, MCPServerSpec,
)

registry = MCPRegistry([
    JeevesGateway.from_env().as_mcp_server(),
    MCPServerSpec.stdio(
        name="git",
        command="uvx",
        args=["mcp-server-git", "--repo", "/Users/me/code/myrepo"],
    ),
    MCPServerSpec.stdio(
        name="fs",
        command="uvx",
        args=["mcp-server-filesystem", "--root", "/Users/me/workspace"],
    ),
])

agent = Agent(
    "You are a developer assistant.",
    model="claude-opus-4-7",
    tools=registry,
)
```

If both ``git`` and ``fs`` exposed a tool named ``status``, the agent
would see ``git.status`` and ``fs.status``. Either qualified or bare
form is accepted at call time; the registry strips the prefix before
forwarding to the underlying session.

---

## 5. Custom embedder

Any class with ``name``, ``dimensions``, ``embed(text)``, and
``embed_batch(texts)`` satisfies the ``Embedder`` protocol — no
inheritance required.

```python
from typing import Any
from jeevesagent import VectorMemory

class CohereEmbedder:
    name: str = "embed-english-v3.0"
    dimensions: int = 1024

    def __init__(self, api_key: str) -> None:
        import cohere
        self._client = cohere.AsyncClient(api_key)

    async def embed(self, text: str) -> list[float]:
        result = await self._client.embed(
            texts=[text],
            model=self.name,
            input_type="search_document",
        )
        return list(result.embeddings[0])

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        result = await self._client.embed(
            texts=texts,
            model=self.name,
            input_type="search_document",
        )
        return [list(e) for e in result.embeddings]


memory = VectorMemory(embedder=CohereEmbedder(api_key="..."))
```

---

## 6. Custom permissions policy

```python
from typing import Any, Mapping
from jeevesagent import Agent
from jeevesagent.core.types import PermissionDecision, ToolCall

class BusinessHoursPermissions:
    """Block destructive tools outside 9am-5pm local time."""

    async def check(
        self,
        call: ToolCall,
        *,
        context: Mapping[str, Any],
    ) -> PermissionDecision:
        if not call.is_destructive():
            return PermissionDecision.allow_()
        from datetime import datetime
        now = datetime.now()
        if 9 <= now.hour < 17:
            return PermissionDecision.allow_()
        return PermissionDecision.deny_(
            f"destructive calls disabled outside business hours (now {now:%H:%M})"
        )

agent = Agent("...", permissions=BusinessHoursPermissions())
```

Same pattern for any custom policy — geofencing, role-based access,
cost-tier gating, etc. Just satisfy the ``Permissions`` protocol.

---

## 7. Streaming UI integration

The ``stream()`` API yields events with backpressure. Wire it into a
WebSocket / SSE / Server-Sent Events handler:

```python
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
from jeevesagent import Agent

app = FastAPI()
agent = Agent("...", model="claude-opus-4-7")

@app.get("/chat")
async def chat(prompt: str):
    async def event_source():
        async for event in agent.stream(prompt):
            yield {
                "event": event.kind.value,
                "data": event.model_dump_json(),
            }
    return EventSourceResponse(event_source())
```

Breaking out of the iteration cancels the producer cleanly — even if
a tool call is mid-flight, it'll be cancelled within the cancel scope.

---

## 8. Production checklist

Before shipping an agent to production, verify each of these:

### Reliability

- [ ] **Durable runtime**: ``runtime=SqliteRuntime(...)`` (or DBOS /
  Temporal when those land) so crashes don't lose work.
- [ ] **Persistent memory**: pass a URL — ``memory="sqlite:./bot.db"``
  for single-instance, ``memory="postgres://..."`` /
  ``memory="redis://..."`` for multi-instance. Not the default
  ``"inmemory"`` which loses everything on exit.
- [ ] **Multi-tenancy**: pass ``user_id=`` and ``session_id=`` to
  every ``agent.run``. Memory partitions automatically; no app-side
  namespace plumbing.
- [ ] **Auto fact extraction**: on by default for real models;
  facts the user tells the bot persist as structured triples for
  future runs to recall. Pass ``auto_extract=False`` to opt out.
- [ ] **Budget**: ``StandardBudget`` with ``max_tokens``,
  ``max_cost_usd``, ``max_wall_clock``. Soft warnings at 80%.
- [ ] **Max turns cap**: default 50; lower if your tools are expensive.

### Observability

- [ ] **Telemetry**: ``OTelTelemetry`` wired to your existing
  TracerProvider. At minimum, surface ``jeeves.session.duration_ms``,
  ``jeeves.tokens.input/output``, ``jeeves.cost.usd``,
  ``jeeves.budget.exceeded``.
- [ ] **Audit log**: ``FileAuditLog`` (or Postgres-backed when
  available) with a real HMAC secret. Every tool call and run-lifecycle
  transition lands here.
- [ ] **Streaming**: expose ``stream()`` so a UI / log pipeline can
  follow the loop in real time.

### Security

- [ ] **Permission policy**: ``StandardPermissions(mode=Mode.DEFAULT)``
  for interactive use; ``BYPASS`` only in CI / sandbox.
- [ ] **Filesystem sandbox**: wrap any tool that touches the FS.
  Declare the allowed roots explicitly.
- [ ] **Pre-tool hooks**: ``@agent.before_tool`` for any tool that
  sends external messages (email, Slack, etc.).
- [ ] **Secrets**: no API keys in tool args. Use the ``Secrets``
  protocol when wiring real secret resolution (follow-up slice).

### Memory

- [ ] **Embedder**: real (``OpenAIEmbedder``, ``CohereEmbedder``) for
  production. ``HashEmbedder`` is for tests / zero-key dev only.
- [ ] **Auto-consolidate**: ``Agent(..., auto_consolidate=True)`` if
  you want facts extracted automatically. Otherwise call
  ``await agent.consolidate()`` on a cadence.
- [ ] **Fact store**: explicit (``with_facts=True`` on the memory
  factory, or pass ``fact_store=...``). Don't rely on the
  in-memory default in production.

### Testing

- [ ] **Test with ScriptedModel** for deterministic multi-turn
  scenarios. ``EchoModel`` for the simplest smoke tests.
- [ ] **Mock embedders** with a ``FakeEmbedder`` that maps specific
  texts to specific vectors when you need to assert on ranking.
- [ ] **Use the in-memory backends in tests** (``InMemoryMemory``,
  ``InMemoryFactStore``, ``InMemoryAuditLog``,
  ``InMemoryJournalStore``) so tests are fast and hermetic.
- [ ] **Skip live-integration tests with env-var gates**:
  ``@pytest.mark.skipif(not os.environ.get("JEEVES_TEST_PG_DSN"))``.
