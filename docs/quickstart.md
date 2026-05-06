# Quickstart

Everything below runs as-is — copy-paste each block into a Python file
or a notebook. Start at the top; every example builds on the previous
one in concept.

## Setup

```bash
pip install jeevesagent
```

For real provider API access:
```bash
pip install 'jeevesagent[anthropic,openai]'
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

For a local-only zero-key experience, you don't need anything beyond
`pip install jeevesagent`.

---

## 1. Hello, agent (no API keys, no infrastructure)

```python
import asyncio
from jeevesagent import Agent

async def main():
    agent = Agent("You are a helpful assistant.")
    result = await agent.run("Tell me a joke.")
    print(result.output)

asyncio.run(main())
```

The default model is `EchoModel` — it echoes the prompt back, so you
can verify the loop works without burning tokens. `result` is a
`RunResult` with `output`, `turns`, `tokens_in`, `tokens_out`,
`cost_usd`, `started_at`, `finished_at`, `interrupted`,
`interruption_reason`.

## 2. Real models

```python
from jeevesagent import Agent

# Strings dispatch by prefix:
agent = Agent("You are helpful.", model="claude-opus-4-7")  # → AnthropicModel
agent = Agent("You are helpful.", model="gpt-4o")           # → OpenAIModel
agent = Agent("You are helpful.", model="echo")             # → EchoModel
```

Or pass an instance for full control:

```python
from jeevesagent import AnthropicModel

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
from jeevesagent import Agent, tool

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
from jeevesagent import Agent, MCPRegistry, MCPServerSpec

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
from jeevesagent import Agent, JeevesGateway

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

### In-memory (default)

```python
from jeevesagent import Agent, InMemoryMemory

agent = Agent("...", memory=InMemoryMemory())
```

### Vector (in-memory, embedding-based recall)

```python
from jeevesagent import Agent, VectorMemory, OpenAIEmbedder

agent = Agent(
    "...",
    memory=VectorMemory(embedder=OpenAIEmbedder("text-embedding-3-small")),
)
```

### Chroma (local persistent)

```python
from jeevesagent import Agent, ChromaMemory

# Persistent on-disk:
memory = ChromaMemory.local("./chroma-db", with_facts=True)
# Or in-memory for tests:
memory = ChromaMemory.ephemeral()

agent = Agent("...", memory=memory)
```

### Postgres + pgvector

```python
from jeevesagent import Agent, PostgresMemory, OpenAIEmbedder

memory = await PostgresMemory.connect(
    dsn="postgres://user:pass@localhost/jeeves",
    embedder=OpenAIEmbedder("text-embedding-3-small"),
    with_facts=True,  # enable bi-temporal fact store on the same pool
)
await memory.init_schema()  # creates episodes + facts tables, HNSW indexes

agent = Agent("...", memory=memory)
```

### Redis

```python
from jeevesagent import Agent, RedisMemory

memory = await RedisMemory.connect(
    "redis://localhost:6379/0",
    with_facts=True,
)
agent = Agent("...", memory=memory)
```

## 8. Bi-temporal facts

Facts are semantic claims `(subject, predicate, object)` with
bi-temporal validity:

```python
from datetime import datetime, UTC
from jeevesagent import Agent, VectorMemory, Consolidator, AnthropicModel
from jeevesagent.core.types import Fact

memory = VectorMemory(
    consolidator=Consolidator(model=AnthropicModel("claude-opus-4-7")),
)

# Manually:
await memory.facts.append(
    Fact(
        subject="user",
        predicate="lives_in",
        object="Tokyo",
        valid_from=datetime.now(UTC),
        recorded_at=datetime.now(UTC),
    )
)

# Or let the agent do it automatically:
agent = Agent(
    "You are a personal assistant.",
    model=AnthropicModel("claude-opus-4-7"),
    memory=memory,
    auto_consolidate=True,  # extracts facts after every run
)

await agent.run("Hi, I'm Alice and I live in Tokyo.")
# Facts are now in memory.facts; the next run sees them.
await agent.run("Where do I live?")  # model gets "user lives_in Tokyo" in context
```

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
from jeevesagent import Agent, SqliteRuntime

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
from jeevesagent import Agent, OTelTelemetry
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
from jeevesagent import Agent, FileAuditLog

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
from jeevesagent import Agent, Mode, StandardPermissions

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
        from jeevesagent.core.types import PermissionDecision
        return PermissionDecision.deny_("blocked by reviewer")
    return None  # allow

@agent.after_tool
async def log(call, result):
    print(f"{call.tool} → ok={result.ok}")
```

## 13. Sandbox (filesystem)

```python
from jeevesagent import Agent, FilesystemSandbox, InProcessToolHost, tool

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
from jeevesagent import Agent
from jeevesagent.governance.budget import BudgetConfig, StandardBudget

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

from jeevesagent import (
    Agent,
    AnthropicModel,
    Consolidator,
    FileAuditLog,
    JeevesGateway,
    OTelTelemetry,
    SqliteRuntime,
    StandardPermissions,
    VectorMemory,
    OpenAIEmbedder,
    Mode,
)
from jeevesagent.governance.budget import BudgetConfig, StandardBudget

async def main():
    embedder = OpenAIEmbedder("text-embedding-3-small")
    consolidator = Consolidator(model=AnthropicModel("claude-opus-4-7"))

    agent = Agent(
        "You are a research assistant. Cite your sources.",
        model="claude-opus-4-7",
        memory=VectorMemory(embedder=embedder, consolidator=consolidator),
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
        auto_consolidate=True,
    )

    async for event in agent.stream("research recent advances in agent harnesses"):
        print(f"[{event.kind}]", event.payload.get("chunk", {}).get("text", ""), end="")

asyncio.run(main())
```

That's a production-shaped agent in ~30 lines. Memory persists facts
across runs, the runtime can recover from crashes, every step lands
in the audit log, every span shows up in your OTel exporter, and the
budget enforces hard limits.
