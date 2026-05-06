# JeevesAgent

**A model-agnostic, MCP-native, fully-async agent harness with memory done right.**

```python
from jeevesagent import Agent

agent = Agent("You are a helpful assistant.")
result = await agent.run("What's 2 + 2?")
print(result.output)
```

That's the whole quickstart. No keys, no infrastructure, no scaffolding —
the default backends echo, store everything in memory, and run in-process.
Swap in real provider adapters / persistent memory / durable runtime / OTel
telemetry when you need them, all behind the same `Agent` constructor.

---

## Why this exists

Every agent framework today forces a choice you shouldn't have to make:

* **LangChain / LangGraph** lock you into a graph editor and a specific
  state model. Production teams report runaway loops, opaque debugging,
  and brittle abstractions.
* **Claude Agent SDK** is excellent if you're committed to Anthropic
  forever. It's not model-agnostic.
* **OpenAI Assistants** is a black box you don't run yourself.
* **CrewAI / AutoGen** are abstractions over LangChain — same problems.

JeevesAgent is the harness for engineers who want to **ship production
agents without binding their stack to one model lab**. It's:

* **Model-agnostic** — Anthropic, OpenAI, and (soon) LiteLLM behind one
  `Model` protocol. String-based resolver: `model="claude-opus-4-7"` or
  `model="gpt-4o"` — no decision lock-in.
* **MCP-native** — MCP isn't an integration, it's the spine. Plug
  Jeeves Gateway, Composio, or any MCP server into a single
  `MCPRegistry` and your tools just work.
* **Memory done right** — five backends (in-memory, vector, Chroma,
  Postgres+pgvector, Redis), pluggable embedders (HashEmbedder for
  zero-key, OpenAIEmbedder for production), and **bi-temporal facts**
  that track when claims were true *in the world* vs when *you learned
  them* — the Zep-style memory wedge, with native fact stores in every
  backend.
* **Durable** — `SqliteRuntime` gives you crash-recovery replay with
  zero infrastructure. DBOS / Temporal adapters land next.
* **Observable** — every step emits OpenTelemetry spans and metrics.
  Drop in your existing exporter; Honeycomb / Datadog / LangSmith just
  work.
* **Safe** — permission policies, sandbox layers, append-only HMAC-signed
  audit log, freshness/lineage policies for certified values.
* **Async-only, structured concurrency only** — anyio everywhere; zero
  raw `asyncio.create_task` / `gather`. Parallel tool dispatch via
  task groups. Backpressure-aware streaming via memory-object streams.

Three principles govern every line of code:

1. **The loop is deterministic; the world isn't.** Every side effect
   goes through `runtime.step(...)` so it can be cached and replayed.
2. **Trust boundary stays outside the sandbox.** The harness runs the
   tools inside a sandbox; the harness doesn't run inside one.
3. **Validate state on write, not on read.** Pydantic everywhere.

---

## Install

```bash
pip install jeevesagent

# Pick the extras you need:
pip install 'jeevesagent[anthropic]'    # Claude
pip install 'jeevesagent[openai]'       # GPT
pip install 'jeevesagent[postgres]'     # PostgresMemory + facts
pip install 'jeevesagent[mcp]'          # real MCP client
pip install 'jeevesagent[otel]'         # OpenTelemetry exporters

# Or install everything for development:
pip install -e '.[dev,anthropic,openai,mcp,postgres,otel]'
```

Requires Python 3.11+.

---

## 30-second quickstart

```python
import asyncio
from jeevesagent import Agent, tool

@tool
async def get_weather(city: str) -> str:
    """Look up the current weather."""
    return f"It's sunny and 72°F in {city}."

async def main():
    agent = Agent(
        "You are a travel assistant.",
        model="claude-opus-4-7",       # or "gpt-4o", or any Model instance
        tools=[get_weather],
    )
    result = await agent.run("What's the weather like in Tokyo?")
    print(result.output)
    print(f"Used {result.tokens_in + result.tokens_out} tokens, ${result.cost_usd:.4f}")

asyncio.run(main())
```

Set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) before running. That's
it — no LangChain, no LangGraph, no `chat_engine = AgentExecutor.from_llm_and_tools(...)`.

Want to see *what's happening* as the agent runs?

```python
async for event in agent.stream("plan a 3-day Tokyo trip"):
    print(f"[{event.kind}] {event.payload}")
```

You'll see `STARTED → MODEL_CHUNK × N → TOOL_CALL → TOOL_RESULT →
MODEL_CHUNK × N → COMPLETED` flow through.

---

## Capability matrix

| Capability | What you get | Where |
|---|---|---|
| **Model adapters** | Anthropic, OpenAI, Echo (zero-key), Scripted (tests) | `jeevesagent.AnthropicModel`, `OpenAIModel`, `EchoModel`, `ScriptedModel` |
| **String model resolver** | `model="claude-opus-4-7"`, `"gpt-4o"`, `"echo"` | `Agent.__init__` |
| **Tools** | `@tool` decorator with auto-schema, sync + async | `jeevesagent.tool`, `Tool` |
| **MCP servers** | stdio + Streamable HTTP, multi-server registry, name disambiguation | `MCPRegistry`, `MCPServerSpec` |
| **Jeeves Gateway** | One-line: `tools=JeevesGateway.from_env()` | `jeevesagent.jeeves` |
| **Memory backends** | In-memory dict, vector cosine, Chroma, Postgres+pgvector, Redis | `InMemoryMemory`, `VectorMemory`, `ChromaMemory`, `PostgresMemory`, `RedisMemory` |
| **Embedders** | HashEmbedder (deterministic, zero deps), OpenAIEmbedder | `HashEmbedder`, `OpenAIEmbedder` |
| **Bi-temporal facts** | All five memory backends. LLM-driven `Consolidator`. Auto-consolidate. | `Fact`, `Consolidator`, `*FactStore` |
| **Durable runtime** | sqlite-backed replay across process restarts | `SqliteRuntime`, `JournaledRuntime` |
| **Streaming** | `agent.stream()` → `AsyncIterator[Event]` with backpressure | `Agent.stream` |
| **Permissions** | mode-based + allow/deny lists, mirrors Claude Agent SDK | `StandardPermissions`, `Mode` |
| **Hooks** | `@agent.before_tool` / `@agent.after_tool` decorators | `HookRegistry` |
| **Sandbox** | `FilesystemSandbox` blocks path-arg escapes (incl. symlinks) | `FilesystemSandbox` |
| **Budget** | Per-token / per-cost / per-wall-clock limits with soft warnings | `StandardBudget`, `BudgetConfig` |
| **Telemetry** | OpenTelemetry spans + metrics for every milestone | `OTelTelemetry` |
| **Audit log** | HMAC-signed JSONL or in-memory; tracks every tool call | `FileAuditLog`, `InMemoryAuditLog` |
| **Certified values** | Freshness + lineage policies | `FreshnessPolicy`, `LineagePolicy` |

---

## Documentation

| Doc | What's there |
|---|---|
| [`docs/quickstart.md`](docs/quickstart.md) | Step-by-step examples for each backend combo |
| [`docs/recipes.md`](docs/recipes.md) | Production patterns: persistent memory, MCP, durable replay, audit |
| [`docs/architecture.md`](docs/architecture.md) | Module tour, lifecycle, extension points |
| [`project.md`](project.md) | The full engineering plan (the design doc) |
| [`BUILD_LOG.md`](BUILD_LOG.md) | Slice-by-slice changelog |

---

## Status

* **236 tests pass** in ~2.5 seconds
* **mypy `--strict`** clean across 53 production source files
* **ruff** clean including `flake8-async` lints
* Phases 1, 2, 3, 4, 5 (essentials), 6 (essentials) of the engineering
  plan all shipped. DBOS / Temporal / OS-level sandboxes / LiteLLM
  remain as follow-ups.

---

## Verify your install

```bash
git clone <repo>
cd jeevesagent
pip install -e '.[dev]'
ruff check jeevesagent
mypy --strict jeevesagent
pytest tests/ -v
```

You should see 236 passed. Two integration tests skip without
`JEEVES_TEST_PG_DSN` / `JEEVES_TEST_REDIS_URL` env vars set.

---

## Contributing

The harness has a strict CI gate: ruff + mypy `--strict` + pytest. All
three must pass. Async-only — every public function returning anything
other than a value is `async`. Every fan-out uses `anyio` task groups.
Zero raw `asyncio.create_task` or `asyncio.gather` calls.

See [`project.md`](project.md) §2 for the non-negotiable engineering
principles.

---

## License

Apache 2.0.
