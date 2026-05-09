# Quickstart

Everything below runs as-is — copy-paste each block into a Python file
or a notebook. Start at the top; every example builds on the previous
one in concept.

## Setup

```bash
pip install loomflow
```

For real provider API access:
```bash
pip install 'loomflow[anthropic,openai]'
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

For a local-only zero-key experience, you don't need anything beyond
`pip install loomflow`.

---

## 1. Hello, agent (no API keys, no infrastructure)

```python
import asyncio
from loomflow import Agent

async def main():
    agent = Agent("You are a helpful assistant.", model="echo")
    result = await agent.run("Tell me a joke.")
    print(result.output)

asyncio.run(main())
```

`model="echo"` selects the `EchoModel` — it echoes the prompt back,
so you can verify the loop works without burning tokens.

> **`model` is required.** Forgetting it raises a `ConfigError` with
> a list of suggested values; the harness no longer silently picks
> a fake model.

`result` is a `RunResult` with `output`, `turns`, `tokens_in`,
`tokens_out`, `cost_usd`, `started_at`, `finished_at`, `interrupted`,
`interruption_reason`.

## 2. Real models

```python
from loomflow import Agent

# Strings dispatch by prefix:
agent = Agent("You are helpful.", model="claude-opus-4-7")  # → AnthropicModel
agent = Agent("You are helpful.", model="gpt-4o")           # → OpenAIModel
agent = Agent("You are helpful.", model="echo")             # → EchoModel
```

Or pass an instance for full control:

```python
from loomflow.model import AnthropicModel

agent = Agent(
    "You are helpful.",
    model=AnthropicModel(
        "claude-opus-4-7",
        api_key="...",
        max_tokens=8192,
    ),
)
```

## 3. Tools

The `@tool` decorator takes a regular Python callable (sync or async)
and derives its JSON schema from type hints.

```python
from loomflow import Agent, tool

@tool
async def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    # In real life: hit an API. For demos, return a fixed string.
    return f"Sunny, 72°F in {city}."

@tool(destructive=True)
def delete_file(path: str) -> str:
    """Delete a file. Marked destructive so default permissions ask first."""
    import os
    os.remove(path)
    return f"deleted {path}"

agent = Agent(
    "You are a productivity assistant.",
    model="claude-opus-4-7",
    tools=[get_weather, delete_file],
)
```

Sync functions are dispatched to a worker thread via
`anyio.to_thread.run_sync`, so they never block the event loop. Tool
calls in the same model turn run **in parallel** through an
`anyio.create_task_group`.

## 4. Streaming events

`agent.stream()` yields events as they happen, with backpressure.

```python
async for event in agent.stream("plan a Tokyo trip"):
    if event.kind == "model_chunk":
        chunk = event.payload["chunk"]
        if chunk["kind"] == "text":
            print(chunk["text"], end="", flush=True)
    elif event.kind == "tool_call":
        print(f"\n[calling {event.payload['call']['tool']}]")
    elif event.kind == "tool_result":
        print(f"[got result]")
```

Events: `STARTED`, `MODEL_CHUNK`, `TOOL_CALL`, `TOOL_RESULT`,
`BUDGET_WARNING`, `BUDGET_EXCEEDED`, `ERROR`, `COMPLETED`.

## 5. MCP servers

Plug an MCP server in directly:

```python
from loomflow import Agent
from loomflow.mcp import MCPRegistry, MCPServerSpec

registry = MCPRegistry([
    MCPServerSpec.stdio(
        name="git",
        command="uvx",
        args=["mcp-server-git", "--repo", "/Users/me/code/myrepo"],
    ),
    MCPServerSpec.http(
        name="hosted",
        url="https://example.com/mcp/",
        headers={"Authorization": "Bearer ..."},
    ),
])

agent = Agent(
    "You are a coding assistant.",
    model="claude-opus-4-7",
    tools=registry,
)
```

Tool name conflicts across servers are auto-disambiguated:
`git.commit` and `github.commit` if both servers expose `commit`;
just `commit` if only one does. Either form is accepted at call time.

## 6. Jeeves Gateway (one line)

```python
from loomflow import Agent
from loomflow.jeeves import JeevesGateway

agent = Agent(
    "You are a productivity assistant.",
    model="claude-opus-4-7",
    tools=JeevesGateway.from_env(),  # reads JEEVES_API_KEY
)
```

Compose with other MCP servers:

```python
gateway = JeevesGateway.from_env()
registry = MCPRegistry([
    gateway.as_mcp_server(),
    MCPServerSpec.stdio("git", "uvx", ["mcp-server-git"]),
])
agent = Agent("...", tools=registry)
```

## 7. Memory: pick a backend

The simplest way is the **`memory=` resolver** — pass a URL and the
framework picks the backend:

```python
from loomflow import Agent

# In-memory (default; lost on restart)
agent = Agent("...", memory="inmemory")

# Single-file SQLite (persistent, no server)
agent = Agent("...", memory="sqlite:./bot.db")

# Chroma (ephemeral / persistent)
agent = Agent("...", memory="chroma")
agent = Agent("...", memory="chroma:./chroma-db")

# Postgres + pgvector
agent = Agent("...", memory="postgres://user:pw@localhost/jeeves")

# Redis (with optional RediSearch HNSW vector index)
agent = Agent("...", memory="redis://localhost:6379/0")
```

What you get out of the box:

* **Auto fact extraction** — every `agent.run()` runs a small
  Consolidator pass that pulls structured `(subject, predicate,
  object)` claims from the conversation into the fact store. Default
  ON for OpenAI / Anthropic / LiteLLM models. See **Auto fact
  extraction** below.
* **Auto-attached fact store** — the resolver wires the
  bi-temporal fact store automatically (pass `with_facts=False` in
  the dict form to skip).
* **Auto-picked embedder** — `OpenAIEmbedder("text-embedding-3-small")`
  if `OPENAI_API_KEY` is set, `HashEmbedder()` otherwise.
* **`user_id` partition** — every backend honours the multi-tenant
  contract. One shared memory file or pool serves N users.
* **Lazy connect for async backends** — Postgres / Redis URLs return
  a `LazyMemory` proxy; the connection opens on the first
  `agent.run`, so `Agent(...)` stays synchronous.

For non-default tweaks, use the dict form:

```python
agent = Agent("...", memory={
    "backend": "chroma",
    "path": "./chroma-db",
    "namespace": "tenant_a",
    "embedder": "openai-large",
    "with_facts": True,
})
```

For full control, pass an explicit instance (today's API,
unchanged):

```python
from loomflow.memory import ChromaMemory, OpenAIEmbedder

memory = ChromaMemory.local(
    "./chroma-db", with_facts=True, embedder=OpenAIEmbedder()
)
agent = Agent("...", memory=memory)
```

## 8. Auto fact extraction (default ON)

Every `agent.run()` against a real model auto-extracts structured
`(subject, predicate, object)` facts from the conversation into the
bi-temporal fact store, partitioned by `user_id`. No `Consolidator`
construction; no manual `consolidate()` call.

```python
from loomflow import Agent

agent = Agent(
    "You are a personal assistant.",
    model="claude-opus-4-7",
    memory="sqlite:./bot.db",
)

await agent.run(
    "Hi, I'm Alice and I live in Tokyo.",
    user_id="alice",
)
# A Fact(user_id="alice", subject="alice", predicate="lives_in",
#        object="Tokyo") is now in memory.facts — the framework
# extracted it automatically.

# Inspect:
profile = await agent.memory.profile(user_id="alice")
print(profile.fact_count)        # > 0
print(profile.sample_facts)      # includes the lives_in fact

# Days later, fresh process, same db:
result = await agent.run(
    "Where do I live?",
    user_id="alice",
)
# → "Tokyo" — the fact gets recalled into the seed messages.
```

Defaults: ON for `OpenAIModel` / `AnthropicModel` / `LiteLLMModel`;
OFF for `ScriptedModel` / `EchoModel` / unrecognised custom Models.
Override with `Agent(..., auto_extract=True/False)`.

Facts use **bi-temporal validity** — when a new claim contradicts
an existing one (same subject + predicate, different object), the
old fact's `valid_until` is set to the new fact's `valid_from`.
Historical facts aren't deleted, just *closed off*. Query a moment
in the past:

```python
from datetime import datetime, UTC

facts_at_jan_2026 = await agent.memory.facts.query(
    user_id="alice",
    subject="alice",
    valid_at=datetime(2026, 1, 1, tzinfo=UTC),
)
```

You can also write facts manually (skip auto-extraction for
specific cases) via `agent.memory.facts.append(Fact(...))`.

When a new fact contradicts an existing one (same subject + predicate,
different object), the old fact's `valid_until` is set to the new
fact's `valid_from` — historical facts aren't deleted, just *closed
off*. Query a moment in the past:

```python
from datetime import datetime, UTC, timedelta

facts_at_jan_2026 = await memory.facts.query(
    subject="user",
    valid_at=datetime(2026, 1, 1, tzinfo=UTC),
)
```

## 9. Durable replay

```python
from loomflow import Agent
from loomflow.runtime import SqliteRuntime

agent = Agent(
    "...",
    model="claude-opus-4-7",
    runtime=SqliteRuntime("./journal.db"),
)

result = await agent.run("complex multi-step task")
# Process crashes mid-run? Restart with same session ID:
# (Resume API is a follow-up; for now session IDs are auto-generated)
```

The runtime journals every model call and tool dispatch by
`(session_id, step_name)`. On a fresh `SqliteRuntime` against the
same DB file, replaying the same session returns cached results
without re-executing anything.

To resume an interrupted run explicitly:

```python
# First run — interrupted by Ctrl-C / OOM / power outage:
result = await agent.run("complex task", session_id="my-task-2026-05-01")

# Later, after the process restarted — same session_id picks up
# where the journal left off. Already-completed model calls and
# tool dispatches replay from the SQLite journal; only the
# un-completed work runs fresh.
result = await agent.resume("my-task-2026-05-01", "complex task")
```

`resume(session_id, prompt)` is just sugar for
`run(prompt, session_id=session_id)`.

## 10. Telemetry (OpenTelemetry)

```python
from loomflow import Agent
from loomflow.observability import OTelTelemetry
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)

tracer_provider = TracerProvider()
tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

agent = Agent(
    "...",
    telemetry=OTelTelemetry(tracer_provider=tracer_provider),
)
```

Spans emitted: `jeeves.run`, `jeeves.turn`, `jeeves.model.stream`,
`jeeves.tool`. Metrics: `jeeves.tokens.input/output`,
`jeeves.cost.usd`, `jeeves.tool.duration_ms`,
`jeeves.session.duration_ms`, `jeeves.budget.exceeded`.

Wire any OTel exporter (Honeycomb, Datadog, LangSmith, OTLP, ...).

## 11. Audit log

```python
from loomflow import Agent
from loomflow.security import FileAuditLog

audit = FileAuditLog("./audit.jsonl", secret="prod-secret")
agent = Agent("...", audit_log=audit)

await agent.run("anything")
# audit.jsonl now has run_started + tool_call + tool_result +
# run_completed entries, each HMAC-signed.

# Compliance query:
entries = await audit.query(session_id="sess_...")
```

## 12. Permissions + hooks

```python
from loomflow import Agent, Mode, StandardPermissions

agent = Agent(
    "...",
    permissions=StandardPermissions(
        mode=Mode.DEFAULT,
        denied_tools=["delete_file", "format_disk"],
    ),
)

@agent.before_tool
async def review(call):
    if call.tool == "send_email" and "@enemy.com" in str(call.args):
        from loomflow.core.types import PermissionDecision
        return PermissionDecision.deny_("blocked by reviewer")
    return None  # allow

@agent.after_tool
async def log(call, result):
    print(f"{call.tool} → ok={result.ok}")
```

For destructive tools (`@tool(destructive=True)`) the default
permissions policy returns `Decision.ask_(...)`. Wire an approval
handler to route the decision through a human / Slack / ticket
queue — without one, `ask` falls back to **deny** so the agent
never silently bypasses the gate:

```python
async def approve(call, user_id: str | None) -> bool:
    """Return True to allow, False to deny."""
    return await my_slack_app.request_approval(call.tool, user_id)

agent = Agent(
    "...",
    permissions=StandardPermissions(mode=Mode.DEFAULT),
    approval_handler=approve,
)
```

A handler that raises is treated as deny + logged. See
[Production hardening](production_hardening.md#approval-handler-for-decisionask_)
for the full failure-mode contract.

## 13. Sandbox (filesystem)

```python
from loomflow import Agent, tool
from loomflow.security import FilesystemSandbox
from loomflow.tools import InProcessToolHost

@tool
def read_file(path: str) -> str:
    """Read file contents."""
    return open(path).read()

# Wrap the tool host in a filesystem sandbox:
host = InProcessToolHost([read_file])
sandbox = FilesystemSandbox(host, roots=["/Users/me/safe-workspace"])

agent = Agent("...", tools=sandbox)
# Now any path arg outside ~/safe-workspace is denied (symlinks resolved).
```

## 14. Budget

```python
from datetime import timedelta
from loomflow import Agent
from loomflow.governance.budget import BudgetConfig, StandardBudget

agent = Agent(
    "...",
    budget=StandardBudget(BudgetConfig(
        max_tokens=200_000,
        max_cost_usd=5.0,
        max_wall_clock=timedelta(minutes=10),
        soft_warning_at=0.8,
    )),
)
```

When the budget is exceeded, the run terminates cleanly with
`result.interrupted = True` and `interruption_reason = "budget:max_tokens"`.

---

## Putting it all together

```python
import asyncio
from datetime import timedelta

from loomflow import (
    Agent,
    FileAuditLog,
    JeevesGateway,
    Mode,
    OTelTelemetry,
    SqliteRuntime,
    StandardPermissions,
)
from loomflow.governance.budget import BudgetConfig, StandardBudget

async def main():
    agent = Agent(
        "You are a research assistant. Cite your sources.",
        model="claude-opus-4-7",
        # One string picks the backend; the resolver wires up the
        # bi-temporal fact store + auto-picks an embedder.
        memory="postgres://user:pw@db.internal/jeeves",
        runtime=SqliteRuntime("./journal.db"),
        tools=JeevesGateway.from_env(),
        permissions=StandardPermissions(mode=Mode.DEFAULT),
        budget=StandardBudget(BudgetConfig(
            max_tokens=200_000,
            max_cost_usd=5.0,
            max_wall_clock=timedelta(minutes=10),
        )),
        audit_log=FileAuditLog("./audit.jsonl", secret="prod-secret"),
        telemetry=OTelTelemetry(),
        # auto_extract=True is the default for real network adapters —
        # pinning explicitly so the production-shape is unambiguous.
        auto_extract=True,
    )

    async for event in agent.stream(
        "research recent advances in agent harnesses",
        user_id="user_42",
        session_id="research_2026_05_08",
    ):
        print(f"[{event.kind}]", event.payload.get("chunk", {}).get("text", ""), end="")

asyncio.run(main())
```

That's a production-shaped agent in ~25 lines. Memory persists
facts across runs (auto-extracted from each conversation), the
runtime can recover from crashes, every step lands in the audit
log, every span shows up in your OTel exporter, and the budget
enforces hard limits. Multi-tenancy is built in via the `user_id`
kwarg — the same agent serves N users with hard partition.
