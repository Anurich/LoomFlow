# JeevesAgent

**A model-agnostic, MCP-native, fully-async agent harness with memory done right.**

```python
from jeevesagent import Agent

agent = Agent("You are a helpful assistant.", model="claude-opus-4-7")
result = await agent.run("What's 2 + 2?")
print(result.output)  # "4"
```

That's the whole quickstart. Set `ANTHROPIC_API_KEY` and you're talking
to Claude. Swap `"claude-opus-4-7"` for `"gpt-4o"` to talk to GPT, or
`"echo"` to use the zero-key fake (echoes the prompt — useful for
tests and local dev). Memory, runtime, telemetry, sandbox, audit are
all opt-in behind the same `Agent` constructor.

> ⚠️ **`model` is required** as of v0.2.0. Earlier `0.1.x` releases
> silently defaulted to `EchoModel` which produced confusing output;
> now the harness fails fast with a helpful error if you forget.

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

* **Model-agnostic** — Anthropic, OpenAI, and ~100 more via LiteLLM
  (Mistral, Cohere, Bedrock, Vertex, Together, Ollama, Gemini, Groq,
  Replicate, Azure …) behind one `Model` protocol. String-based
  resolver: `model="claude-opus-4-7"`, `"gpt-4o"`, `"mistral-large"`,
  `"command-r-plus"`, … — no decision lock-in.
* **Pluggable architectures** — the agent loop is a strategy.
  Eleven shipped: ReAct (default), SelfRefine, Reflexion,
  TreeOfThoughts, PlanAndExecute (single-agent); Router,
  Supervisor, ActorCritic, MultiAgentDebate, Swarm,
  BlackboardArchitecture (multi-agent). Same `Agent` surface;
  one kwarg flips the iteration pattern.
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

## Architectures: the agent loop is a strategy

The default loop is ReAct (observe / think / act). When that doesn't
fit your problem, swap it for a different iteration pattern with one
kwarg — everything else (model, memory, tools, budget, telemetry,
runtime) stays exactly the same.

```python
from jeevesagent import (
    Agent, ReAct, SelfRefine, Reflexion, Router, RouterRoute,
    Supervisor, ActorCritic, TreeOfThoughts, MultiAgentDebate,
    Swarm, BlackboardArchitecture, PlanAndExecute,
)

# Default — observe / think / act
agent = Agent("...", model="claude-opus-4-7")

# Iterate until critic says no more issues
agent = Agent("...", model="claude-opus-4-7", architecture="self-refine")

# Verbal RL: lessons from failed attempts persist via memory.working()
# and shape future runs.
agent = Agent("...", model="claude-opus-4-7", architecture="reflexion")

# Classify input → dispatch to one specialist Agent
agent = Agent(
    "Customer support entry point",
    model="claude-haiku-4-5",        # cheap classifier
    architecture=Router(routes=[
        RouterRoute(name="billing", agent=billing_agent, description="..."),
        RouterRoute(name="tech",    agent=tech_agent,    description="..."),
        RouterRoute(name="general", agent=general_agent, description="..."),
    ], fallback_route="general"),
)

# Workers + a delegate(...) tool. Multiple delegations in one
# supervisor turn run in parallel for free.
agent = Agent(
    "You manage a small team",
    model="claude-opus-4-7",
    architecture=Supervisor(workers={
        "researcher": researcher_agent,
        "coder":      coder_agent,
        "writer":     writer_agent,
    }),
)

# Actor + adversarial critic with different models — the canonical
# pattern for code review and quality-critical work.
agent = Agent(
    "Code-quality coordinator",
    architecture=ActorCritic(
        actor=Agent("...", model="claude-opus-4-7"),
        critic=Agent("...", model="gpt-4o"),  # different model = different blind spots
        max_rounds=3, approval_threshold=0.9,
    ),
)

# BFS beam search over candidate reasoning steps. Each level proposes
# branch_factor candidates, evaluator scores them, top beam_width
# survive. Useful for combinatorial / planning / math tasks.
agent = Agent(
    "...",
    model="claude-opus-4-7",
    architecture=TreeOfThoughts(
        branch_factor=3, beam_width=2, max_depth=4, solved_threshold=0.95,
    ),
)

# N debaters argue across rounds; judge synthesizes. Use different
# MODELS for genuine prior diversity. Reserve for contested questions
# where wrong answers are expensive (3-5× cost).
agent = Agent(
    "Investment committee moderator",
    architecture=MultiAgentDebate(
        debaters=[optimist_agent, skeptic_agent, analyst_agent],
        judge=cio_agent,
        rounds=2,
    ),
)

# Plan once, execute step-by-step. Cheaper than ReAct on tasks with
# predictable structure: one planner call + N step calls + one
# synthesizer.
agent = Agent("...", model="claude-opus-4-7", architecture="plan-and-execute")

# Peer agents pass control via a handoff tool. Exploratory only —
# the spec warns of goal drift; prefer Supervisor for production.
agent = Agent(
    "...",
    model="claude-opus-4-7",
    architecture=Swarm(
        agents={"triage": triage_agent, "billing": billing_agent, "tech": tech_agent},
        entry_agent="triage",
        max_handoffs=5,
    ),
)

# Coordinator + agents share a state board. Each round the
# coordinator picks who contributes next; decider synthesizes.
agent = Agent(
    "...",
    model="claude-opus-4-7",
    architecture=BlackboardArchitecture(
        agents={"hypothesis": h_agent, "evidence": e_agent, "critic": c_agent},
        coordinator=coord_agent,
        decider=decider_agent,
    ),
)

# N debaters argue across parallel rounds; judge synthesizes the
# verdict. Use different MODELS for genuine prior diversity.
agent = Agent(
    "Investment committee moderator",
    model="claude-opus-4-7",
    architecture=MultiAgentDebate(
        debaters=[optimist_agent, skeptic_agent, analyst_agent],
        judge=cio_agent,
        rounds=2, convergence_check=True,
    ),
)

# Composition: Reflexion *of* a Supervisor — cross-session learning
# of which worker handles which intent best.
agent = Agent(
    "...",
    model="claude-opus-4-7",
    architecture=Reflexion(base=Supervisor(workers={...}), threshold=0.85),
)
```

Architectures are pluggable via the `Architecture` protocol — three
methods (`name`, `run`, `declared_workers`) and you have a custom
strategy. See [`Subagent.md`](Subagent.md) for the full design
rationale and 14-architecture catalogue covering the cases the
shipped five don't yet handle.

---

## Capability matrix

| Capability | What you get | Where |
|---|---|---|
| **Architecture protocol** | Pluggable agent-loop strategy: 11 architectures shipped | `Architecture`, `ReAct`, `SelfRefine`, `Reflexion`, `TreeOfThoughts`, `PlanAndExecute`, `Router`, `Supervisor`, `ActorCritic`, `MultiAgentDebate`, `Swarm`, `BlackboardArchitecture` |
| **Model adapters** | Anthropic, OpenAI, LiteLLM (~100 providers), Echo (zero-key), Scripted (tests) | `jeevesagent.AnthropicModel`, `OpenAIModel`, `LiteLLMModel`, `EchoModel`, `ScriptedModel` |
| **String model resolver** | `model="claude-opus-4-7"`, `"gpt-4o"`, `"mistral-large"`, `"command-r"`, `"echo"`, `"litellm/<any>"` | `Agent.__init__` |
| **Tools** | `@tool` decorator with auto-schema, sync + async; `agent.with_tool` decorator; `add_tool` / `remove_tool` / `tools_list` | `jeevesagent.tool`, `Tool` |
| **MCP servers** | stdio + Streamable HTTP, multi-server registry, name disambiguation | `MCPRegistry`, `MCPServerSpec` |
| **Jeeves Gateway** | One-line: `tools=JeevesGateway.from_env()` | `jeevesagent.jeeves` |
| **Memory backends** | In-memory dict, vector cosine, Chroma, Postgres+pgvector, Redis | `InMemoryMemory`, `VectorMemory`, `ChromaMemory`, `PostgresMemory`, `RedisMemory` |
| **Embedders** | HashEmbedder (deterministic, zero deps), OpenAIEmbedder, VoyageEmbedder, CohereEmbedder | `HashEmbedder`, `OpenAIEmbedder`, `VoyageEmbedder`, `CohereEmbedder` |
| **Bi-temporal facts** | All five memory backends. LLM-driven `Consolidator`. Auto-consolidate, plus `ConsolidationWorker` for long-lived agents. | `Fact`, `Consolidator`, `*FactStore` |
| **Durable runtime** | sqlite or postgres-backed replay across process restarts | `SqliteRuntime`, `PostgresRuntime`, `JournaledRuntime` |
| **Streaming** | `agent.stream()` → `AsyncIterator[Event]` with backpressure | `Agent.stream` |
| **Permissions** | mode-based + allow/deny lists, mirrors Claude Agent SDK | `StandardPermissions`, `Mode` |
| **Hooks** | `@agent.before_tool` / `@agent.after_tool` decorators | `HookRegistry` |
| **Sandbox** | `FilesystemSandbox` blocks path-arg escapes; `SubprocessSandbox` for full isolation | `FilesystemSandbox`, `SubprocessSandbox` |
| **Budget** | Per-token / per-cost / per-wall-clock limits with soft warnings | `StandardBudget`, `BudgetConfig` |
| **Telemetry** | OpenTelemetry spans + metrics for every milestone | `OTelTelemetry` |
| **Audit log** | HMAC-signed JSONL or in-memory; tracks every tool call | `FileAuditLog`, `InMemoryAuditLog` |
| **Certified values** | Freshness + lineage policies | `FreshnessPolicy`, `LineagePolicy` |
| **Declarative config** | Build agents from TOML or dicts | `Agent.from_config(path)`, `Agent.from_dict(cfg)` |

---

## Documentation

| Doc | What's there |
|---|---|
| [`docs/quickstart.md`](docs/quickstart.md) | Step-by-step examples for each backend combo |
| [`docs/recipes.md`](docs/recipes.md) | Production patterns: persistent memory, MCP, durable replay, audit |
| [`docs/architecture.md`](docs/architecture.md) | Module tour, lifecycle, extension points |
| [`docs/migration_0.1_to_0.2.md`](docs/migration_0.1_to_0.2.md) | What changed in 0.2.0; how to migrate |
| [`Subagent.md`](Subagent.md) | Architecture-protocol design rationale; full 14-architecture catalogue (the 5 shipped, the 9 candidates) |
| [`project.md`](project.md) | The full engineering plan (the design doc) |
| [`BUILD_LOG.md`](BUILD_LOG.md) | Slice-by-slice changelog |
| [`examples/`](examples/) | 19 runnable scripts — `00_hello.py` through `18_plan_and_execute.py`, every shipped architecture covered |

---

## Status

* **528 tests pass** in ~5 seconds (5 env-gated integrations skip
  without `JEEVES_TEST_PG_DSN` / `JEEVES_TEST_REDIS_URL`)
* **mypy `--strict`** clean across 73 production source files
* **ruff** clean including `flake8-async` lints
* Phases 1-6 of the engineering plan shipped. v0.3 added the
  `Architecture` protocol layer with **eleven** shipped
  architectures (ReAct, SelfRefine, Reflexion, TreeOfThoughts,
  PlanAndExecute, Router, Supervisor, ActorCritic,
  MultiAgentDebate, Swarm, BlackboardArchitecture). LiteLLM,
  SubprocessSandbox, PostgresRuntime, ConsolidationWorker, Voyage /
  Cohere embedders all shipped in v0.2. Temporal / OS-level
  sandboxes / the remaining 3 architectures from `Subagent.md`
  (ReWOO, Deep Agent, Graph of Thoughts) are the next chunks.

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

You should see 528 passed. Five integration tests skip without
`JEEVES_TEST_PG_DSN` / `JEEVES_TEST_REDIS_URL` / API-key env vars set.

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
