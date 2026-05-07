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
  Twelve shipped: ReAct (default), SelfRefine, Reflexion,
  TreeOfThoughts, PlanAndExecute, ReWOO (single-agent); Router,
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
fit your problem, swap it with one kwarg — everything else (model,
memory, tools, budget, telemetry, runtime) stays exactly the same.

### Single-agent loops: pass `architecture=`

```python
from jeevesagent import Agent

agent = Agent("...", model="claude-opus-4-7")                            # ReAct default
agent = Agent("...", model="...", architecture="self-refine")            # iterate until critic happy
agent = Agent("...", model="...", architecture="reflexion")              # verbal RL with lessons
agent = Agent("...", model="...", architecture="plan-and-execute")       # plan once, execute steps
agent = Agent("...", model="...", architecture="rewoo")                  # plan + parallel tools, 30-50% cheaper
agent = Agent("...", model="...", architecture="tree-of-thoughts")       # BFS beam over candidate thoughts
```

### Multi-agent teams: use `Team` builders (the ergonomic facade)

`Team` mirrors the builder shape every other framework uses
(`create_supervisor` / `Crew` / `GroupChatManager`) so migrating from
LangGraph / CrewAI / AutoGen / OpenAI Agents SDK is muscle-memory.
Each builder returns a regular `Agent` — same `.run()` / `.stream()`
interface, no special calling convention.

```python
from jeevesagent import Agent, Team, RouterRoute

# Coordinator + workers; the manager calls delegate(...) or forward_message(...)
team = Team.supervisor(
    workers={"researcher": researcher, "writer": writer, "reviewer": reviewer},
    instructions="manage the pipeline",
    model="claude-opus-4-7",
)

# Classify-and-dispatch — cheaper than Supervisor when one specialist
# is enough (1 classifier call + 1 specialist run, no synthesis pass)
team = Team.router(
    routes=[
        RouterRoute(name="billing", agent=billing, description="..."),
        RouterRoute(name="tech",    agent=tech,    description="..."),
    ],
    instructions="customer support entry point",
    model="claude-haiku-4-5",
)

# Peer agents passing control via typed handoffs (input_type= for
# structured payloads, input_filter= for selective history pruning)
team = Team.swarm(
    agents={"triage": triage, "billing": billing, "tech": tech},
    entry_agent="triage",
    model="claude-opus-4-7",
)

# Actor + critic with different models for blind-spot diversity
team = Team.actor_critic(
    actor=Agent("...", model="claude-opus-4-7"),
    critic=Agent("...", model="gpt-4o"),       # different model
    max_rounds=3,
    approval_threshold=0.9,
    model="claude-opus-4-7",                    # coordinator
)

# N debaters + optional judge with similarity-based early termination
team = Team.debate(
    debaters=[optimist, skeptic, analyst],
    judge=cio,
    rounds=2,
    convergence_similarity=0.85,
    model="claude-opus-4-7",
)

# Coordinator + agents share a workspace; decider synthesizes
team = Team.blackboard(
    agents={"hypothesis": h_agent, "evidence": e_agent, "critic": c_agent},
    coordinator=coord_agent,
    decider=decider_agent,
    model="claude-opus-4-7",
)
```

### Recursive composition (the differentiator)

Architectures wrap each other naturally — the property no
sibling-only framework gives you. Wrap a Supervisor in Reflexion for
cross-session learning of delegation patterns; nest Supervisors for
hierarchical teams; wrap an entire pipeline in `Reflexion` to retry
on low scores:

```python
from jeevesagent import Agent, Reflexion, Supervisor

agent = Agent(
    "...",
    model="claude-opus-4-7",
    architecture=Reflexion(
        base=Supervisor(workers={"researcher": ..., "writer": ...}),
        max_attempts=3,
        threshold=0.85,
        lesson_store=InMemoryVectorStore(embedder=HashEmbedder()),  # selective recall
    ),
)
```

The explicit nested form (`Agent(architecture=...)`) and `Team`
builders are interchangeable — `Team.supervisor(workers={...})` is
exactly `Agent(architecture=Supervisor(workers={...}))` under the
hood. Use `Team` for single-level teams (matches what you've seen
in other frameworks); use the nested form for recursive composition.

### Standalone testing of orchestrators

```python
from jeevesagent import Supervisor, run_architecture

sup = Supervisor(workers={"a": agent_a})
result = await run_architecture(sup, "do the thing", model="claude-opus-4-7")
```

Architectures are pluggable via the `Architecture` protocol — three
methods (`name`, `run`, `declared_workers`) and you have a custom
strategy. See [`Subagent.md`](Subagent.md) for the full design
rationale.

---

## Architecture cheat sheet

Visual reference for picking the right pattern. Each diagram shows
the actual data flow + LLM-call structure for that architecture.

### Single-agent loops

**`ReAct`** — observe / think / act loop. The default. One model call per turn; tools dispatch in parallel.

```
                 ┌────────── loop until no tool calls ──────────┐
                 │                                              │
   prompt ───► Model ───► tool calls? ──yes──► run tools ──► results
                 │                              (parallel)
                 └─────────► no calls ───► final output
```

**`SelfRefine`** — single-agent generate → critique → refine. Same model wears both hats.

```
   prompt ───► generate ───► critique ──┬── score ≥ threshold ──► output
                              ▲         │
                              │         └── below ──► refine ──┐
                              │                                │
                              └────────────────────────────────┘
```

**`Reflexion`** — wraps any base architecture with verbal-RL retry. Failed attempts produce a "lesson" stored in memory or a vector store; next attempt sees the relevant lessons.

```
   ┌─────────── attempt loop (max_attempts) ───────────┐
   │                                                    │
   │   prompt ──► [recall lessons] ──► base.run() ──► evaluator
   │                                                    │
   │                                              score < threshold?
   │                                                    │
   │                                              yes ──┴── no ──► output
   │                                                    │
   │                                              reflector ──► lesson
   │                                                    │
   └────────────────────────────────── persist ─────────┘
                                          │
                          memory block  OR  vector store (selective recall)
```

**`TreeOfThoughts`** — BFS beam search over candidate thoughts. Proposer + evaluator at every depth; beam keeps top-k; min_score floor drops weak branches early.

```
              proposer (×branch_factor)         evaluator
   prompt ──► [t1, t2, t3]  ──score──►  [0.9, 0.4, 0.7]
                                              │
                                         keep top beam_width
                                         drop below min_score
                                              │
                                              ▼
                                         [t1, t3]   ←── frontier for depth 2
                                              │
                                         (repeat to max_depth)
                                              │
                                              ▼
                                       best leaf wins
```

**`PlanAndExecute`** — planner emits a step list once; executor walks each step; synthesizer composes the final answer.

```
   prompt ───► planner ───► [step1, step2, step3]
                                     │
                                     ▼
                              executor (per step) ───► [r1, r2, r3]
                                                            │
                                                            ▼
                                                      synthesizer ───► output
```

**`ReWOO`** — like PlanAndExecute but the planner emits structured tool calls with `{{En}}` placeholders, and **independent steps run in parallel**. Two LLM calls + N tool calls — 30-50% cheaper than ReAct on tool-heavy workloads.

```
   prompt ───► planner ───► [search({{E1}}), fetch({{E2}}=search.url)]
                                          │
                                          ▼
                            parallel tool dispatch
                            (independent steps run concurrently;
                             dependent steps wait for {{En}})
                                          │
                                          ▼
                                    synthesizer ───► output
```

### Multi-agent teams

**`Router`** — classify-and-dispatch. ONE classifier call decides which specialist runs; that one specialist owns the answer.

```
                       ┌── refund_agent
   prompt ──► classifier ──► technical_agent      ◄── only ONE
                       └── faq_agent ◄── chosen      runs

   1 classifier call + 1 specialist run. The cheapest multi-agent pattern.
```

**`Supervisor`** — coordinator + workers, glued by a `delegate(worker, instructions)` tool. Multiple delegations in one supervisor turn run in parallel. `forward_message(worker)` returns a worker's output verbatim with no synthesis.

```
   prompt ───► manager ───► delegate(...) ─┬─► worker A ─┐
                              │            ├─► worker B ─┤  parallel
                              │            └─► worker C ─┤
                              ▼                          │
                          [worker outputs] ◄─────────────┘
                              │
                              ├─► synthesize ──► output
                              │
                              └─► forward_message(worker) ──► verbatim output
```

**`ActorCritic`** — actor + critic pair (use *different models* for blind-spot diversity). Critic returns structured JSON `{score, issues, summary}`; actor refines below threshold.

```
   prompt ───► actor ───► critic ──┬── score ≥ threshold ──► output
                  ▲                │
                  │                └── below ──► refine (apply rubric)
                  │                                  │
                  └──────────── max_rounds cap ──────┘
```

**`MultiAgentDebate`** — N debaters argue across rounds (in parallel each round). Jaccard convergence detects early agreement; optional judge synthesizes the final answer.

```
   prompt ──► [debater1, debater2, debater3]   ◄── round 1 (parallel)
                              │
                       converged? (Jaccard ≥ 0.85)
                       yes ───► output
                       no  ───► [responses fed back]
                              │
              [debater1, debater2, debater3]    ◄── round 2 (sees prior)
                              │
                              ▼
                          judge ──► output     (or majority vote if no judge)
```

**`Swarm`** — peer agents handing off control via a `handoff` tool (or per-target `transfer_to_<name>` tools when peers are wrapped in `Handoff` with an `input_type`). No central coordinator.

```
   prompt ──► agent A
                 │
                 │ handoff(B, payload)
                 ▼
              agent B
                 │
                 │ transfer_to_C(typed_args)
                 ▼
              agent C ──► final output
                 ▲
                 │ cycle detection: A→B→A→B kills the loop
                 │ max_handoffs caps total depth
```

**`BlackboardArchitecture`** — agents collaborate via a shared mutable workspace. Coordinator picks who acts next; decider says when work is done.

```
                ┌───────────── shared blackboard ─────────────┐
                │   facts · hypotheses · partial results       │
                └────▲──────▲──────▲──────▲────────────▲───────┘
                     │ r/w  │ r/w  │ r/w  │ r/w        │
                     │      │      │      │            │
   prompt ──► coordinator ──► picks who acts next      │
                     │      │      │      │            │
                  agent A  agent B  agent C            │
                     │      │      │      │            │
                     ▼      ▼      ▼      ▼            │
                              decider ◄────────────────┘
                                 │
                                 ├─ done? ──► output
                                 │
                                 └─ not done ──► next round
```

### Recursive composition

Any architecture can wrap any other. The killer combination: `Reflexion` *of* `Supervisor` — the team learns across attempts which worker handles which intent best.

```
   ┌────── Reflexion attempt loop ──────┐
   │                                     │
   │   prompt ──► Supervisor ──► output ─┤── score ≥ threshold ──► done
   │              (manager + 3 workers)  │
   │                                     │
   │                                     └── below ──► lesson ──► retry
   │                                                                │
   └────────────────────────────────────────────────────────────────┘
```

```python
agent = Agent(
    "...",
    model="claude-opus-4-7",
    architecture=Reflexion(
        base=Supervisor(workers={"researcher": ..., "writer": ..., "reviewer": ...}),
        lesson_store=InMemoryVectorStore(embedder=HashEmbedder()),  # selective recall
    ),
)
```

---

## Capability matrix

| Capability | What you get | Where |
|---|---|---|
| **Architecture protocol** | Pluggable agent-loop strategy: 12 architectures shipped | `Architecture`, `ReAct`, `SelfRefine`, `Reflexion`, `TreeOfThoughts`, `PlanAndExecute`, `ReWOO`, `Router`, `Supervisor`, `ActorCritic`, `MultiAgentDebate`, `Swarm`, `BlackboardArchitecture` |
| **Team facade** | Sibling-style builders (`Team.supervisor`, `Team.swarm`, `Team.router`, `Team.debate`, `Team.actor_critic`, `Team.blackboard`) for the common multi-agent shapes | `Team`, `Handoff`, `run_architecture` |
| **Vector store** | `add` / `search` / `delete` with Mongo-style filters, MMR diversity, BM25 hybrid search, save/load | `InMemoryVectorStore`, `ChromaVectorStore`, `PostgresVectorStore`, `FAISSVectorStore`, `SearchResult` |
| **Document loader** | One-line load for PDF / DOCX / Excel / CSV / HTML / Markdown into chunks | `jeevesagent.loader.load`, `MarkdownChunker`, `RecursiveChunker`, `SentenceChunker`, `TokenChunker` |
| **Built-in tools** | `read` / `write` / `edit` / `bash` factories with sandbox-aware workdirs | `read_tool`, `write_tool`, `edit_tool`, `bash_tool`, `default_workdir` |
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
| [`examples/`](examples/) | 26 runnable scripts: `00_hello`–`19_rewoo` cover every architecture; `20_rag_supervisor`–`22_rag_with_loader` are RAG patterns; `23_coding_agent`, `24_support_triage`, `25_document_pipeline`, `26_devops_diagnostic` are real-world use cases with permissions / audit / budget wired up |

---

## Status

* **743 tests pass** in ~6 seconds (5 env-gated integrations skip
  without `JEEVES_TEST_PG_DSN` / `JEEVES_TEST_REDIS_URL`)
* **mypy `--strict`** clean across 95 production source files
* **ruff** clean including `flake8-async` lints
* v0.5 ships the full vector-store stack (`InMemoryVectorStore` /
  Chroma / Postgres / FAISS, all with Mongo-style filters, MMR
  diversity, BM25 hybrid search, persistence), the document loader
  with chunking strategies, the `Team` facade for ergonomic
  multi-agent construction, and 12 architectures with selective
  lesson recall (Reflexion), typed handoffs (Swarm),
  forward_message (Supervisor), Jaccard convergence (Debate), and
  parallel proposer/evaluator with min_score floor (TreeOfThoughts).

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

You should see 743 passed. Five integration tests skip without
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
