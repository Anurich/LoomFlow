# Loom

[![CI](https://github.com/Anurich/LoomFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/Anurich/LoomFlow/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/loomflow.svg)](https://pypi.org/project/loomflow/)
[![Python versions](https://img.shields.io/pypi/pyversions/loomflow.svg)](https://pypi.org/project/loomflow/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![mypy: strict](https://img.shields.io/badge/mypy-strict-2a6db2.svg)](https://mypy.readthedocs.io/en/stable/command_line.html#cmdoption-mypy-strict)
[![Discord](https://img.shields.io/badge/discord-join-5865F2?logo=discord&logoColor=white)](https://discord.gg/X6njWztQ)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**Production workflow + agent framework. Two peer primitives —
``Workflow`` (developer-controlled DAG) and ``Agent`` (LLM-driven
loop) — that compose freely. Multi-tenant by default, typed
outputs, retries on transient errors, model-agnostic, MCP-native.**

📖 **Docs** — build locally with `pip install -e ".[docs]" && sphinx-build -b html docs docs/_build/html` (RTD hosting pending — set up the project at https://readthedocs.org once you have an account, then this link goes live)
&nbsp;&nbsp;·&nbsp;&nbsp;
**Migrating?** — [from LangGraph](docs/migrations/from-langgraph.md)
&nbsp;·&nbsp;
[from raw OpenAI SDK](docs/migrations/from-openai-sdk.md)
&nbsp;&nbsp;·&nbsp;&nbsp;
**Changelog** — [CHANGELOG.md](CHANGELOG.md)

```python
import asyncio
from pydantic import BaseModel
from loomflow import Agent

class WeatherReport(BaseModel):
    city: str
    temp_c: float
    conditions: str

async def main():
    agent = Agent("Be precise.", model="gpt-4.1-mini")

    # Free-form run, scoped to a user (memory partitions automatically).
    r = await agent.run("Hi, my name is Alice.", user_id="alice")
    print(r.output)

    # Same agent, structured output, conversation continues.
    r = await agent.run(
        "Weather in Tokyo right now: sunny, 22°C, light wind. Extract.",
        user_id="alice",
        session_id="conv_42",
        output_schema=WeatherReport,
    )
    report: WeatherReport = r.parsed   # ← typed, validated
    print(f"{report.city}: {report.temp_c}°C — {report.conditions}")

asyncio.run(main())
```

Set `OPENAI_API_KEY` and run. Swap `"gpt-4.1-mini"` for
`"claude-opus-4-7"`, `"mistral-large"`, `"command-r-plus"`,
`"echo"` (zero-key fake), or any of ~100 providers via LiteLLM.

**What's actually different about this framework:**

* `user_id` is a first-class typed primitive. One shared `Agent` +
  one shared `Memory` partitions automatically across N tenants
  with no cross-contamination. **No more "forgot to namespace" data
  leaks.**
* `output_schema=` accepts any Pydantic model. The framework
  augments the system prompt, parses the result, validates,
  retries-with-feedback on validation failure. **Typed outputs by
  default, free-text by omission.**
* Network model adapters are auto-wrapped with a typed error
  taxonomy + retry policy. Rate limits, 5xx, network blips don't
  blow up your run. **Resilient by default.**
* `session_id` is a real conversation handle. Reuse it across
  `agent.run()` calls and prior turns rehydrate as real chat
  history. **No reducer protocol, no `add_messages` magic.**
* The agent loop is a *strategy*. Twelve architectures shipped
  (ReAct, Self-Refine, Reflexion, TreeOfThoughts, PlanAndExecute,
  ReWOO, Router, Supervisor, ActorCritic, MultiAgentDebate, Swarm,
  Blackboard) behind one `Agent` constructor. **One kwarg flips
  the iteration pattern.**
* Async-only, anyio everywhere, structured concurrency cancellation
  works correctly. Fast path when production features (audit / OTel
  / permissions / hooks / journaling) aren't wired up.

> ⚠️ **`model` is required** as of v0.2.0. Earlier `0.1.x` releases
> silently defaulted to `EchoModel` which produced confusing output;
> now the harness fails fast with a helpful error if you forget.

---

## Why pick this over LangGraph / CrewAI / AutoGen

Every agent framework forces a choice you shouldn't have to make:

* **LangChain / LangGraph** lock you into a graph editor and a
  specific state model. `user_id` is a string in
  `config["configurable"]` — typo it once and you silently leak
  data across tenants. Structured outputs and retries are
  developer-side concerns.
* **Claude Agent SDK** is excellent if you're committed to Anthropic
  forever. It's not model-agnostic.
* **OpenAI Assistants** is a black box you don't run yourself.
* **CrewAI / AutoGen** are abstractions over LangChain — same
  problems.

Loom is the harness for engineers shipping production agents
without binding their stack to one model lab — and without wiring
multi-tenancy / structured outputs / retries by hand.

**Capabilities at a glance:**

* **Model-agnostic** — Anthropic, OpenAI, and ~100 more via LiteLLM
  behind one `Model` protocol. String-based resolver:
  `model="claude-opus-4-7"`, `"gpt-4.1-mini"`, `"mistral-large"`, …
* **Pluggable architectures** — twelve shipped, same `Agent`
  surface, one kwarg switches the iteration strategy.
* **MCP-native** — MCP is the tool spine, not an integration. Any
  MCP server (Composio, custom hosted gateways, local stdio
  servers) plugs into a single `MCPRegistry`.
* **Memory done right** — five backends (in-memory / vector /
  Chroma / Postgres+pgvector / Redis), pluggable embedders, and
  **bi-temporal facts** that track when claims were true in the
  world vs when you learned them. All five backends partition by
  `user_id`.
* **Durable runtime** — `SqliteRuntime` gives crash-recovery replay
  with zero infrastructure. Postgres also supported.
* **Observable** — OpenTelemetry spans and metrics for every step.
  Drop in your exporter (Honeycomb / Datadog / LangSmith).
* **Safe** — permission policies, sandbox layers, append-only
  HMAC-signed audit log, freshness/lineage policies for certified
  values.
* **Async-only, structured concurrency** — anyio everywhere, zero
  raw `asyncio.create_task` / `gather`. Parallel tool dispatch via
  task groups. Backpressure-aware streaming.

Three principles govern every line of code:

1. **The loop is deterministic; the world isn't.** Every side effect
   goes through `runtime.step(...)` so it can be cached and replayed.
2. **Trust boundary stays outside the sandbox.** The harness runs
   tools inside a sandbox; the harness doesn't run inside one.
3. **Validate state on write, not on read.** Pydantic everywhere.

---

## Install

```bash
pip install loomflow

# Pick the extras you need:
pip install 'loomflow[anthropic]'           # Claude
pip install 'loomflow[openai]'              # GPT
pip install 'loomflow[postgres]'            # PostgresMemory + facts
pip install 'loomflow[mcp]'                 # real MCP client
pip install 'loomflow[otel]'                # OpenTelemetry exporters
pip install 'loomflow[loader-pdf]'          # PDF loader (unstructured, default)
pip install 'loomflow[loader-pdf-docling]'  # alt PDF backend (docling, IBM)

# Or install everything for development:
pip install -e '.[dev,anthropic,openai,mcp,postgres,otel]'
```

Requires Python 3.11+.

---

## 30-second quickstart

```python
import asyncio
from loomflow import Agent, tool

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
from loomflow import Agent

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
from loomflow import Agent
from loomflow.architecture import RouterRoute
from loomflow.team import Team

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
from loomflow import Agent
from loomflow.architecture import Reflexion, Supervisor

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
from loomflow.architecture import Supervisor
from loomflow.team import run_architecture

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

## Workflows: pick the right shape (don't over-engineer)

`Workflow` is the developer-controlled DAG, peer of `Agent`. It
earns its weight when you need **observability**, **audit**,
**cycles**, or **graph-as-artifact**. For straight-line code
that just chains a few `await`s with no branching, *plain Python
is shorter and clearer* — don't reach for the framework.

**Decision tree, simplest first:**

| Situation | Use | One-liner |
|---|---|---|
| 2-3 sequential `await`s, no branching, no audit needs | **Plain Python** | `await c(await b(await a(x)))` |
| Linear sequence + telemetry / audit / per-step events | `Workflow.chain` | `Workflow.chain([a, b, c])` |
| Classify input, dispatch to one of N handlers | `Workflow.route` | `Workflow.route(classify, {"a": h_a, "b": h_b}, default=...)` |
| Fan-out + merge | `Workflow.parallel` | `Workflow.parallel([s1, s2], merge=combine)` |
| Cycles, mid-graph branching, conditional entry | Explicit graph | `add_node` + `add_edge` + `add_router` (+ `START` / `END`) |

Most users only ever need the sugar constructors. Reach for the
explicit builder when the graph itself is the artifact (compliance
flows, refinement loops, multi-stage routing).

**Example — branch at the entry, then terminate per branch:**

```python
from loomflow import Workflow, START, END

wf = Workflow("triage")
wf.add_node("billing", billing_agent)
wf.add_node("tech",    tech_agent)
wf.add_router(
    START,
    fn=lambda q: "billing" if "invoice" in q else "tech",
    routes={"billing": "billing", "tech": "tech"},
)
wf.add_edge("billing", END)
wf.add_edge("tech",    END)

result = await wf.run("I can't see my invoice", user_id="alice")
```

`START` and `END` are sentinels — drop them into `add_edge` /
`add_router` as edge endpoints; no special methods to remember.
Type `wf` in a Jupyter cell to render the graph inline.

**Memory propagates to nested agents** — `Workflow(memory=mem)`
is picked up by every agent in the graph that didn't specify its
own `memory=`, so episodes / facts written by one agent are
recall-able by the next without per-agent wiring.

Full rubric (when to use Workflow at all vs. Agent, anti-patterns,
both-at-once shapes): [`docs/workflow_vs_agent.md`](docs/workflow_vs_agent.md).

---

## Skills: packaged playbooks the agent loads on demand

Tools tell the agent **what** it can do. Skills tell it **how** —
domain-specific recipes the agent reads when relevant, ignores when
not. Same shape as [Anthropic Agent Skills (Oct 2025)](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview):
a directory with `SKILL.md` (frontmatter + markdown body) and
optional bundled files. Drop your existing Anthropic-format skills
into our `skills=[...]` and they Just Work.

```python
from loomflow import Agent

agent = Agent(
    "...",
    model="claude-opus-4-7",
    skills=[
        "~/.jeeves/skills/system/",          # base layer
        "~/.jeeves/skills/user/",            # user override
        ("./.jeeves-skills/", "Project"),    # project-local with label
    ],
)
```

**Progressive disclosure**: only `name` + `description` (~50 tokens
per skill) load into the system prompt at startup. The model calls
a `load_skill(name)` tool when a skill is relevant — only THEN does
the full body enter context. A 50-skill agent costs ~2,500 tokens at
rest; nothing more until the model actually loads one.

### Three skill modes — coexist freely in any skill

```
skills/my-skill/
├── SKILL.md         ← required: frontmatter + markdown body
├── tools.py         ← OPTIONAL: @tool functions (Mode B, in-process Python)
└── scripts/         ← OPTIONAL: executable scripts (Mode A or Mode C)
    └── helper.py
```

**Mode A — pure markdown.** SKILL.md teaches the model how to use
your existing tools (`read`, `write`, `bash`). The model issues
those tool calls itself based on the body's instructions.

**Mode C — frontmatter declares a script as a typed tool.** Any
language. The framework wraps the script in a subprocess-backed
`Tool` with proper args; the model calls it like any built-in tool.

```yaml
---
name: calc
description: Arithmetic helpers.
tools:
  add:
    description: Sum two integers.
    script: scripts/add.py
    args:
      a:
        type: string
        description: First int
      b:
        type: string
        description: Second int
---
```
```python
# scripts/add.py — plain Python, no decorators
import sys
print(int(sys.argv[1]) + int(sys.argv[2]))
```
The model calls `calc__add(a="2", b="3")` → framework execs the
script → captures stdout → returns to the model.

**Mode B — `tools.py` ships `@tool` functions.** Auto-discovered by
filename presence; imported at construction; registered into the
agent's tool host when the skill is loaded.

```python
# skills/greeter/tools.py
from loomflow import tool

@tool
async def say_hi(name: str) -> str:
    """Say hi."""
    return f"Hi {name}!"
```
The model calls `greeter__say_hi(name="Anupam")` directly. In-process,
fast, can share the agent's state.

### Auto-namespacing prevents collisions

Tool names get prefixed with the skill name automatically:

| Skill ships | Registered as |
|---|---|
| `add` (Mode C, calc skill) | `calc__add` |
| `say_hi` (Mode B, greeter skill) | `greeter__say_hi` |
| `search` (in two skills A and B) | `a__search` and `b__search` — no clash |

### Inline skills — one-off in code

For tiny one-off skills that don't justify a folder:

```python
from loomflow.skills import Skill

skill = Skill.from_text("""
---
name: standup
description: Format a daily standup from rough notes.
---
# Standup
Always 3 sections: Yesterday, Today, Blockers.
""")

agent = Agent("...", skills=[skill])
```

### Layered sources with last-wins override

When two sources ship a skill with the same name, the later source
wins. Lets you stack: system → user → project.

```python
agent = Agent(
    skills=[
        "~/.jeeves/skills/system/",      # base
        "~/.jeeves/skills/user/",        # user customizes
        "./.jeeves-skills/",             # project-local override
    ],
)
```

See the [`examples/`](examples/) directory for runnable end-to-end
samples — RAG over PDFs, multi-agent debate, structured outputs,
multi-user/session continuity, every memory backend behind one
parameter, and the full Workflow composition story (chain, route,
cycles, workflow-as-tool, agent architectures inside workflows,
custom-step prompt formatting).

---

## Fast path by default

Loom ships with the full production surface — audit log, OTel
telemetry, permissions, hooks, durable runtime, budget — but **you
don't pay for what you don't wire up**. Every layer has a no-op
default, and the loop detects those defaults at construction time
and skips the integration points entirely on the hot path.

A barebones `Agent("hi", model="gpt-4.1-mini", tools=[...])` runs
without going through the audit / telemetry / permissions / hook /
journaling / budget layers at all. The moment you pass
`audit_log=`, `telemetry=`, `permissions=`, `runtime=`, etc., the
corresponding layer flips on and the integration becomes active —
same `Agent` class, same API, no flags to set.

```text
                  default Agent              production Agent
                  ─────────────────         ─────────────────────
audit_log         None        → SKIP       FileAuditLog(...)    → wired
telemetry         NoTelemetry → SKIP       OTelTelemetry(...)   → wired
permissions       AllowAll    → SKIP       StandardPermissions  → wired
hooks             empty       → SKIP       @before_tool/@after_tool → wired
runtime           InProc      → INLINE     SqliteRuntime(...)   → wired
budget            NoBudget    → SKIP       StandardBudget(...)  → wired
```

When a layer is detected as no-op, the loop:

* skips the `await audit_log.append(...)` call (so even the function
  call dispatch is removed)
* skips `telemetry.trace(...)` async-context-manager entry/exit and
  the kwargs-dict construction for `emit_metric` calls
* skips `permissions.check(call, context={})` (returns `allow_()`
  inline)
* skips `hooks.pre_tool` / `hooks.post_tool` iteration
* inlines `await fn(*args)` instead of routing through
  `runtime.step(name, fn, ...)` — saves the idempotency-key hash
  derivation per tool call
* skips `budget.allows_step()` / `budget.consume(...)`

The point: "framework is slow because it's full-featured" stops
being the trade-off. You get the harness when you want it, the
speed when you don't, with no code changes between modes.

---

## Resilient by default

Real model APIs fail. Rate limits, 5xx blips, transient connection
drops happen on every production deployment. Loom ships
**retry on transient errors enabled by default** for the in-tree
network adapters (OpenAI, Anthropic, LiteLLM) — the moment you
construct a real-world agent it's already covered:

```python
agent = Agent("...", model="gpt-4.1-mini")
# Default policy: 3 attempts, 1 s → 2 s → 4 s backoff
# (capped at 30 s, ±10% jitter), respects provider Retry-After.
```

The framework normalises every model SDK's exceptions into a small
typed taxonomy so callers + the retry layer reason about failures
uniformly:

```text
ModelError                       — base (catch-all model failure)
├── TransientModelError          — retry-able
│   └── RateLimitError           — 429 / quota; carries retry_after
└── PermanentModelError          — don't retry
    ├── AuthenticationError      — bad API key
    ├── InvalidRequestError      — malformed prompt / args
    └── ContentFilterError       — safety system rejection
```

`classify_model_error(exc)` does the SDK-specific mapping (lazy
imports, no hard dependency on any provider package). The wrapper
treats `TransientModelError` as retryable, `PermanentModelError` as
fatal, and any *unrecognised* exception is propagated unchanged —
the framework refuses to silently retry errors it doesn't understand.

Tune the policy per-Agent:

```python
from loomflow import Agent
from loomflow.governance import RetryPolicy

# Default (production-sensible)
agent = Agent("...", model="gpt-4.1-mini")

# Aggressive — tolerates long provider blips
agent = Agent("...", model="gpt-4.1-mini",
              retry_policy=RetryPolicy.aggressive())

# Disabled — handle errors yourself
agent = Agent("...", model="gpt-4.1-mini",
              retry_policy=RetryPolicy.disabled())

# Custom
agent = Agent("...", model="gpt-4.1-mini",
              retry_policy=RetryPolicy(
                  max_attempts=4,
                  initial_delay_s=0.5,
                  max_delay_s=15.0,
              ))
```

Behaviour highlights:

* **Provider-supplied `Retry-After` is honoured** — when a 429
  response carries the header, the framework waits at least that
  long before the next attempt (even if it exceeds `max_delay_s`).
  Provider authority wins over local heuristics.
* **Streaming retries fire before the first chunk** — once the
  consumer has received any tokens we cannot rewind, so mid-stream
  errors propagate. Pre-first-chunk failures are retried per
  policy.
* **Custom Models are not auto-wrapped.** The framework only
  wraps its in-tree adapters by default because it knows their
  error classes. Custom Models opt in by passing `retry_policy=`
  explicitly to `Agent(...)`.

---

## Structured outputs

Production agents need to emit *data*, not free-form prose. Pass a
Pydantic `BaseModel` as `output_schema=` to `agent.run(...)` and the
framework gives you a typed, validated instance:

```python
from pydantic import BaseModel
from loomflow import Agent

class CompanyInfo(BaseModel):
    name: str
    founded_year: int
    headquarters: str

agent = Agent("extract company info", model="gpt-4.1-mini")
result = await agent.run("Tell me about Acme.", output_schema=CompanyInfo)

info: CompanyInfo = result.parsed   # ← typed, validated
print(info.founded_year)            # 2008
print(result.output)                # raw JSON text, still available
```

What the framework does:

1. **Schema-aware system prompt** — appends a `STRUCTURED OUTPUT
   REQUIRED` directive to the run's instructions, embedding the
   schema's JSON Schema. Your static `Agent(...)` instructions are
   not mutated; the augmentation is per-run.
2. **Tolerates real-world model quirks** — strips ` ```json ` /
   ` ``` ` markdown fences before parsing.
3. **Retry-with-feedback** — on a parse failure, the framework
   gives the model up to `output_validation_retries` (default `1`)
   extra single-shot turns to fix it, feeding the validation error
   back as a USER message ("Your previous response failed schema
   validation: ...; return only a corrected JSON object"). After
   the retry budget is exhausted, raises
   `OutputValidationError` with the underlying Pydantic
   `ValidationError` attached as `.cause`, the bad text on `.raw`,
   and the schema on `.schema` — so callers can build whatever
   recovery strategy they need.
4. **`result.output`** keeps the (cleaned) raw JSON text so you can
   log or audit what the model produced; **`result.parsed`** holds
   the validated Pydantic instance.

Set `output_validation_retries=0` to fail fast (no recovery turn).

End-to-end demo: [`examples/04_structured_outputs.py`](examples/04_structured_outputs.py)
extracts a structured `MeetingSummary` (with nested `ActionItem`
lists, ISO dates, and a sentiment enum) from a raw meeting transcript.

---

## Memory — one parameter, six backends

Pick a memory backend by passing a string. The framework parses the
URL scheme, builds the right backend, and (for async-connect
backends) defers the connection until first use so the `Agent(...)`
call stays synchronous.

```python
Agent(...)                                   # in-memory default; lost on restart
Agent(..., memory="inmemory")                # explicit default
Agent(..., memory="sqlite:./bot.db")         # single-file persistent — no infra
Agent(..., memory="chroma")                  # ephemeral Chroma client
Agent(..., memory="chroma:./vectors")        # persistent Chroma at path
Agent(..., memory="postgres://user:pw@host/db")    # Postgres + pgvector
Agent(..., memory="redis://localhost:6379/0")      # Redis + RediSearch
```

For non-default tweaks, pass a config dict:

```python
Agent(..., memory={
    "backend": "chroma",
    "path": "./vectors",
    "namespace": "tenant_a",
    "embedder": "openai",       # "hash" / "openai" / "openai-large" / Embedder
    "with_facts": True,
})
```

Or pass a fully-constructed Memory instance the way you do today —
power users keep every escape hatch:

```python
from loomflow.memory import ChromaMemory, OpenAIEmbedder

memory = ChromaMemory.local("./vectors", with_facts=True, embedder=OpenAIEmbedder())
agent = Agent(..., memory=memory)
```

What you get out of the box:

* **Auto-attached fact store** — string and dict specs default to
  `with_facts=True`, so semantic-recall just works. Pass
  `with_facts=False` to skip it.
* **Auto-picked embedder** — `OpenAIEmbedder("text-embedding-3-small")`
  when `OPENAI_API_KEY` is set, `HashEmbedder()` (deterministic,
  zero-key) otherwise.
* **`user_id` partition** — every backend honours the M1 multi-tenant
  contract, so one shared memory file or pool serves N users with
  no cross-contamination.
* **Lazy connection for async backends** — `Agent("...",
  memory="postgres://...")` returns immediately; the pool opens on
  first `agent.run`, errors surface there as `MemoryStoreError`.

The new `SqliteMemory` backend (`memory="sqlite:./bot.db"`) fills the
"persistent but no infra" gap — episodes, working blocks, session
messages, and bi-temporal facts all live in one `.db` file. Survives
restarts, no server, no schema migrations to think about.

### Inspect, forget, export — GDPR by default

Every backend implements the same three high-level ops, scoped by
`user_id`:

```python
# What does the agent know about Alice?
profile = await agent.memory.profile(user_id="alice")
# MemoryProfile(user_id='alice', episode_count=12, fact_count=5,
#               last_seen=..., recent_sessions=['conv_42', ...],
#               sample_facts=[Fact(subject='alice', predicate='works_at', ...), ...])

# Right-to-be-forgotten — full erasure of one user's data.
deleted = await agent.memory.forget(user_id="alice")

# Or scoped: just one conversation, or just data older than a date.
await agent.memory.forget(user_id="alice", session_id="conv_42")
await agent.memory.forget(user_id="alice", before=datetime(2026, 1, 1))

# Data-portability dump (DSAR).
export = await agent.memory.export(user_id="alice")
blob = export.model_dump_json()  # serialise for download
```

These work identically across `InMemoryMemory`, `SqliteMemory`,
`VectorMemory`, `ChromaMemory`, `PostgresMemory`, and `RedisMemory`.
Hard `user_id` partition is enforced — `profile("alice")` never
returns counts derived from bob's data, `export("alice")` never
includes bob's episodes, `forget("alice")` never touches bob.

### Auto-extract facts (default ON for real models)

The thing that turns memory into "your bot just remembers": when
`auto_extract=True` (the default for in-tree network adapters —
OpenAI, Anthropic, LiteLLM), the framework runs the bundled
`Consolidator` on every persisted episode. Structured
`(subject, predicate, object)` facts get pulled from the
conversation and stored in the bi-temporal fact store, partitioned
by `user_id`, ready to surface in future runs via `recall_facts`.

```python
agent = Agent("...", model="gpt-4.1-mini", memory="sqlite:./bot.db")

# One run; the framework auto-extracts facts in the background.
await agent.run(
    "Hi, I'm Alice and my favourite programming language is Python.",
    user_id="alice",
)

# Facts are already there — no manual Consolidator call needed.
profile = await agent.memory.profile(user_id="alice")
# profile.fact_count > 0
# profile.sample_facts contains e.g.
#   Fact(user_id="alice", subject="alice", predicate="prefers",
#        object="Python", ...)

# Days later, a different conversation:
result = await agent.run(
    "What's my favourite programming language?",
    user_id="alice",
)
# → "Your favourite is Python." (recalled from the auto-extracted fact)
```

Defaults:

* **ON** for `OpenAIModel` / `AnthropicModel` / `LiteLLMModel` — real
  network models where extraction is the whole point.
* **OFF** for `ScriptedModel` / `EchoModel` / unrecognised custom
  Models — test fakes shouldn't make extra LLM calls.

Override with `Agent(..., auto_extract=True)` (force on for a
custom model) or `auto_extract=False` (turn off for cost control,
or when wiring a different extraction model manually).

Extraction is **best-effort**: the run returns successfully even
when extraction fails (model error, malformed JSON, rate limit) —
the agent's primary contract (return a result, persist the
episode) is never blocked by this enhancement.

### Memory inspection demo

[`examples/05_memory_showcase.py`](examples/05_memory_showcase.py)
walks through every backend, the URL/dict/instance resolver, the
`profile`/`forget`/`export` GDPR ops, and runs the bundled
`Consolidator` to extract structured facts from raw chat episodes —
all in one runnable file.

---

## Multi-tenant by default — every primitive

Every stateful primitive in the framework partitions by `user_id`.
One `Agent` + one `Memory` + one `Budget` + one `AuditLog` serves N
tenants with hard isolation across **all** layers:

| Primitive | Partitioned by `user_id`? |
|---|---|
| Memory recall + facts + sessions + episodes + working blocks | ✅ |
| Auto fact extraction (extracts inherit episode's user_id) | ✅ |
| Sub-agent context inheritance | ✅ |
| Tools (read via `get_run_context().user_id`) | ✅ |
| Telemetry span attributes | ✅ |
| **Working memory blocks** (`update_block` / `working`) | ✅ M9 |
| **Budget accounting** (`StandardBudget` per-user totals) | ✅ M9 |
| **Audit log** (`AuditEntry.user_id` top-level field) | ✅ M9 |
| **Permissions** (route to per-user policy via `PerUserPermissions`) | ✅ M9 |
| **Hooks** (`pre_tool` / `post_tool` see user_id) | ✅ M9 |

Concrete examples of what M9 enables:

```python
from loomflow import Agent, StandardPermissions, Mode, BudgetConfig, StandardBudget
from loomflow.security import PerUserPermissions

# Per-user permission policies — admins get more, free users get less.
permissions = PerUserPermissions(
    policies={
        "admin_alice": StandardPermissions(mode=Mode.BYPASS),
        "paid_user_42": StandardPermissions(mode=Mode.ACCEPT_EDITS),
    },
    default=StandardPermissions(
        mode=Mode.DEFAULT, denied_tools=["bash", "delete_account"]
    ),
)

# Per-user budget caps — alice can't exhaust bob's tokens.
budget = StandardBudget(BudgetConfig(
    max_tokens=1_000_000,        # global cap (whole tenant)
    per_user_max_tokens=10_000,  # per-user cap (each tenant)
    per_user_max_cost_usd=1.0,
))

agent = Agent(
    "...",
    model="gpt-4.1-mini",
    memory="postgres://prod-db/agent",
    permissions=permissions,
    budget=budget,
    audit_log=FileAuditLog("./audit.jsonl", secret="prod-secret"),
)

# Per-user audit queries — clean SIEM integration, no payload digging.
alice_entries = await agent._audit_log.query(user_id="alice")

# Per-user budget snapshot — for ops dashboards.
print(budget.usage_for("alice"))  # {tokens_in, tokens_out, cost_usd, ...}
```

The principle: **`user_id` is the partition key for everything
stateful**. The framework forwards it from the live `RunContext`
into every primitive automatically; protocol implementations that
don't care can ignore the kwarg, and legacy impls without it fall
back gracefully.

---

## Multi-tenancy by default

Loom treats `user_id` and `session_id` as **first-class typed
primitives**, not strings buried in a free-form config dict. The
moment you pass them to `agent.run(...)`, the framework partitions
memory automatically and rehydrates conversation history without
any extra plumbing.

```python
result = await agent.run(
    "what is my favourite food?",
    user_id="alice",            # hard namespace partition for memory
    session_id="conv_42",       # conversation thread; reused = continued
    metadata={"locale": "en"},  # free-form bag for app-specific keys
)
```

What the framework does with these:

* **`user_id`** is a hard partition on every memory primitive.
  Episodes and facts stored under one `user_id` are **never visible**
  to a recall scoped to a different one. `None` is its own bucket
  ("anonymous / single-tenant"). One shared `Memory` instance can
  back N concurrent users with zero risk of cross-contamination.
* **`session_id`** is the conversation handle. Reuse the same id
  across calls and the loop rehydrates prior user/assistant turns
  as real `Message` history — the model sees the chat thread, not
  just a recall summary.
* **`metadata`** rides along on the per-run `RunContext` and is
  reachable from any tool / hook via `get_run_context()` without
  threading it through every function signature.

Inside a tool, you read scope from the live `RunContext`:

```python
from loomflow import tool, get_run_context

@tool
async def fetch_user_orders() -> str:
    """Look up the current user's recent orders."""
    ctx = get_run_context()
    return await db.query("orders", user_id=ctx.user_id)
```

The model never sees `user_id` in the tool schema, can't pass the
wrong one, and the framework guarantees the tool gets the right
value (set by `_loop`, propagated through `anyio` task groups for
parallel tool dispatch and sub-agent spawning).

**Sub-agents inherit automatically.** Every multi-agent architecture
(Supervisor, Debate, Swarm, Router, ActorCritic, Blackboard, ReWOO)
forwards the parent's `RunContext` to its workers, so `user_id`
flows through deeply nested agent trees with no per-architecture
plumbing. Workers get a fresh `session_id` so their conversation
history stays separate from the parent's.

**Footgun protection.** When a memory store contains episodes for
named users and a recall runs with `user_id=None`, the framework
emits an `IsolationWarning` — the partition is still safe, but the
dev probably forgot to pass `user_id=` somewhere. Apps that want
strict enforcement promote it to an exception:

```python
import warnings
from loomflow import IsolationWarning
warnings.simplefilter("error", IsolationWarning)
```

End-to-end demo: [`examples/03_multi_user_sessions.py`](examples/03_multi_user_sessions.py).

---

## Capability matrix

| Capability | What you get | Where |
|---|---|---|
| **Multi-tenant memory** | First-class `user_id` partition + `session_id` continuity. One shared `Memory` instance backs N users with no cross-contamination; sub-agents inherit context automatically | `RunContext`, `get_run_context`, `set_run_context`, `IsolationWarning`, `Agent.run(user_id=, session_id=, metadata=)` |
| **Memory string resolver** | Pick backend by URL scheme: `"sqlite:./bot.db"`, `"chroma:./vec"`, `"postgres://..."`, `"redis://..."`, `"inmemory"`. Async-connect backends return a lazy proxy — `Agent(...)` stays synchronous. Config-dict form for tweaks; instance pass-through for power users. | `resolve_memory` (top-level); `from loomflow.memory import SqliteMemory, LazyMemory`; `Agent(memory=)` |
| **Memory inspection / GDPR** | `profile(user_id)` summarises what the bot knows about a user; `forget(user_id, session_id=, before=)` erases (right-to-be-forgotten); `export(user_id)` dumps everything for portability / DSAR. Implemented across all six backends; hard user_id partition. | `Memory.profile`, `Memory.forget`, `Memory.export`, `MemoryProfile`, `MemoryExport` (top-level types) |
| **Auto fact extraction** | Default ON for real network models. Each `agent.run()` finishes by extracting structured `(subject, predicate, object)` facts from the conversation and writing them to the user-partitioned bi-temporal fact store. "Your bot just remembers" with no manual `Consolidator` calls. Best-effort: failures don't break the run. Emits `loom.auto_extract.duration_ms` + invocations metrics for observability. | `Agent(auto_extract=)`; `from loomflow.memory import AutoExtractMemory, Consolidator` |
| **Bounded per-user state** | `StandardBudget` and `InMemoryMemory` cap their per-user dicts at 100k users by default with 24h idle-TTL eviction so multi-tenant deployments can't OOM under runaway tenants or one-shot user_id explosions. LRU eviction order; durable backends (Sqlite/Postgres) for callers needing spill-to-disk. | `BoundedDict`, `StandardBudget(max_users=, user_idle_ttl_seconds=)`, `InMemoryMemory(max_users=, user_idle_ttl_seconds=)` |
| **Approval handler** | Permissions returning `Decision.ask_(...)` route through `Agent(approval_handler=callable)` so destructive tool calls can be gated by a human / Slack / ticket prompt. Without a handler, `ask` falls back to deny — no silent bypass. Handlers that raise are treated as deny too. | `Agent(approval_handler=)`, `ApprovalHandler` |
| **Secrets protocol** | `Agent(secrets=...)` resolves API keys via `api_key=` → `secrets.lookup_sync` → `os.environ` precedence. `EnvSecrets` (default, env-var backed) and `DictSecrets` (in-memory) ship in-tree; production callers wire their own vault adapter. `redact()` masks common API-key shapes for safe audit log output. | `EnvSecrets`, `DictSecrets`, `Secrets` protocol, `Agent(secrets=)` |
| **Structured outputs** | Pass `output_schema=` to get a typed, validated Pydantic instance back. Framework augments the system prompt with the schema, parses + validates, retries with feedback on failure | `Agent.run(output_schema=)`, `RunResult.parsed`, `OutputValidationError` |
| **Resilient model calls** | Network adapters auto-wrapped with retry-on-transient (rate limit, 5xx, network blip). Typed error taxonomy. Provider `Retry-After` honoured. | `RetryPolicy`, `RetryPolicy.disabled/aggressive`, `ModelError`, `TransientModelError`, `RateLimitError`, `PermanentModelError`, `AuthenticationError`, `InvalidRequestError`, `ContentFilterError`, `classify_model_error` |
| **Architecture protocol** | Pluggable agent-loop strategy: 12 architectures shipped. `ReAct` is top-level (the default); the others are accessed via `from loomflow.architecture import SelfRefine, Reflexion, ...` | `Architecture`, `ReAct` (top-level); `loomflow.architecture.{SelfRefine, Reflexion, TreeOfThoughts, PlanAndExecute, ReWOO, Router, Supervisor, ActorCritic, MultiAgentDebate, Swarm, BlackboardArchitecture}` |
| **Workflow primitive** | Developer-controlled DAG — peer of `Agent`, not a subtype. Use when you can draw the path on a whiteboard before writing code (`Workflow.chain` for sequences, `Workflow.route` for classify-then-dispatch, `Workflow.parallel` for fan-out + merge, explicit graph builder for custom shapes including cycles). Both composition directions are first-class: drop an `Agent` in as a workflow node, or expose a `Workflow` as an `Agent` tool via `wf.as_tool()`. Same observability spine as `Agent` — telemetry, audit log, `user_id` partition. Cycles supported with `max_visits_per_node` / `max_steps` safety caps. See [`docs/workflow_vs_agent.md`](docs/workflow_vs_agent.md) for the rubric. | `Workflow`, `WorkflowResult`, `step`, `START`, `END` |
| **Team facade** | Sibling-style builders (`Team.supervisor`, `Team.swarm`, `Team.router`, `Team.debate`, `Team.actor_critic`, `Team.blackboard`) for the common multi-agent shapes | `from loomflow.team import Team, Handoff, run_architecture` |
| **Vector store** | `add` / `search` / `delete` with Mongo-style filters, MMR diversity, BM25 hybrid search, save/load | `from loomflow.vectorstore import InMemoryVectorStore, ChromaVectorStore, PostgresVectorStore, FAISSVectorStore, SearchResult` |
| **Document loader** | One-line load for PDF / DOCX / Excel / CSV / HTML / Markdown into markdown chunks. PDF supports two interchangeable backends: `unstructured` (default, Apache 2.0, what LangChain wraps) with `fast` / `hi_res` / `ocr_only` strategies, or `docling` (MIT, IBM Research, 2026 benchmark winner on native PDFs). Per-page extraction failures emit a `RuntimeWarning` — no more silent empty pages. | `loomflow.loader.load`, `load_pdf(path, *, backend=, strategy=)`, `MarkdownChunker`, `RecursiveChunker`, `SentenceChunker`, `TokenChunker` |
| **Built-in tools** | `read` / `write` / `edit` / `bash` factories with sandbox-aware workdirs | `from loomflow.tools import read_tool, write_tool, edit_tool, bash_tool, default_workdir` |
| **Skills (Anthropic-compatible)** | Packaged playbooks loaded on demand. Three modes coexist: pure markdown, frontmatter-declared subprocess tools (any language), and `tools.py` with `@tool` (Python, in-process). Layered sources with last-wins override. | `from loomflow.skills import Skill, SkillRegistry, SkillSource, SkillMetadata, SkillError`; `Agent(skills=...)` |
| **Model adapters** | Anthropic, OpenAI, LiteLLM (~100 providers), Echo (zero-key), Scripted (tests). Echo + Scripted are top-level (zero-dep test fakes); real network adapters require their submodule for an explicit "I'm picking this provider" signal. | `EchoModel`, `ScriptedModel` (top-level); `from loomflow.model.openai import OpenAIModel`, `from loomflow.model.anthropic import AnthropicModel`, `from loomflow.model.litellm import LiteLLMModel` |
| **String model resolver** | `model="claude-opus-4-7"`, `"gpt-4o"`, `"mistral-large"`, `"command-r"`, `"echo"`, `"litellm/<any>"` | `Agent.__init__` |
| **Tools** | `@tool` decorator with auto-schema, sync + async; `agent.with_tool` decorator; `add_tool` / `remove_tool` / `tools_list` | `tool`, `Tool` (top-level) |
| **MCP servers** | stdio + Streamable HTTP, multi-server registry, name disambiguation | `from loomflow.mcp import MCPClient, MCPRegistry, MCPServerSpec` |
| **Memory backends** | In-memory dict, vector cosine, Chroma, Postgres+pgvector, Redis, single-file SQLite | `InMemoryMemory` (top-level); `from loomflow.memory.{sqlite,chroma,postgres,redis,vector} import {SqliteMemory, ChromaMemory, PostgresMemory, RedisMemory, VectorMemory}` |
| **Embedders** | HashEmbedder (deterministic, zero deps), OpenAIEmbedder, VoyageEmbedder, CohereEmbedder | `HashEmbedder` (top-level); `from loomflow.memory import OpenAIEmbedder, VoyageEmbedder, CohereEmbedder` |
| **Bi-temporal facts** | All five memory backends. LLM-driven `Consolidator`. Auto-consolidate, plus `ConsolidationWorker` for long-lived agents. | `Fact` (top-level); `from loomflow.memory import Consolidator, ConsolidationWorker, FactStore` |
| **Durable runtime** | sqlite or postgres-backed replay across process restarts | `InProcRuntime` (top-level); `from loomflow.runtime import SqliteRuntime, PostgresRuntime, JournaledRuntime` |
| **Streaming** | `agent.stream()` → `AsyncIterator[Event]` with backpressure | `Agent.stream` |
| **Permissions** | mode-based + allow/deny lists; per-user policy routing via `PerUserPermissions` | `StandardPermissions`, `AllowAll`, `Mode` (top-level); `from loomflow.security import PerUserPermissions` |
| **Hooks** | `@agent.before_tool` / `@agent.after_tool` decorators | `HookRegistry` (top-level) |
| **Sandbox** | `FilesystemSandbox` blocks path-arg escapes; `SubprocessSandbox` for full isolation | `NoSandbox` (top-level); `from loomflow.security import FilesystemSandbox, SubprocessSandbox` |
| **Budget** | Per-token / per-cost / per-wall-clock limits with soft warnings; per-user caps via `BudgetConfig(per_user_max_*)` | `StandardBudget`, `BudgetConfig`, `NoBudget` (top-level) |
| **Telemetry** | OpenTelemetry spans + metrics for every milestone | `NoTelemetry` (top-level); `from loomflow.observability import OTelTelemetry` |
| **Audit log** | HMAC-signed JSONL or in-memory; tracks every tool call; entries attributed by `user_id` | `InMemoryAuditLog`, `AuditLog` (top-level); `from loomflow.security import FileAuditLog, verify_signature` |
| **Certified values** | Freshness + lineage policies | `from loomflow.data import FreshnessPolicy, LineagePolicy` |
| **Declarative config** | Build agents from TOML or dicts | `Agent.from_config(path)`, `Agent.from_dict(cfg)` |

---

## Documentation

The full Sphinx-built documentation site is configured but not yet
deployed. To go live:

1. Sign in to <https://readthedocs.org/> with the repo's GitHub
   account
2. Click *Import Project* → pick the Loom repo
3. RTD reads `.readthedocs.yaml`, builds, and the site comes up at
   `https://<your-project-slug>.readthedocs.io/`

Until then, build locally with:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```

In-tree starting points:

| Doc | What's there |
|---|---|
| [`docs/quickstart.md`](docs/quickstart.md) | Step-by-step examples for each backend combo |
| [`docs/recipes.md`](docs/recipes.md) | Production patterns: persistent memory, MCP, durable replay, audit |
| [`docs/architecture.md`](docs/architecture.md) | Module tour, lifecycle, extension points |
| [`docs/migrations/from-langgraph.md`](docs/migrations/from-langgraph.md) | LangGraph → Loom translation guide |
| [`docs/migrations/from-openai-sdk.md`](docs/migrations/from-openai-sdk.md) | Hand-rolled OpenAI loop → Loom translation guide |
| [`docs/migration_0.1_to_0.2.md`](docs/migration_0.1_to_0.2.md) | What changed in 0.2.0; how to migrate |
| [`CHANGELOG.md`](CHANGELOG.md) | Version-by-version release notes |
| [`Subagent.md`](Subagent.md) | Architecture-protocol design rationale; full 14-architecture catalogue (the 5 shipped, the 9 candidates) |
| [`project.md`](project.md) | The full engineering plan (the design doc) |
| [`BUILD_LOG.md`](BUILD_LOG.md) | Slice-by-slice changelog |
| [`examples/`](examples/) | Eleven runnable end-to-end samples. Agent + retrieval + memory: [`01_rag_pdf.py`](examples/01_rag_pdf.py) (PDF RAG with `unstructured` / `docling` backends), [`02_specialist_debate.py`](examples/02_specialist_debate.py) (multi-agent debate), [`03_multi_user_sessions.py`](examples/03_multi_user_sessions.py) (multi-tenant + session continuity), [`04_structured_outputs.py`](examples/04_structured_outputs.py) (typed Pydantic outputs), [`05_memory_showcase.py`](examples/05_memory_showcase.py) (every memory backend behind one parameter, including GDPR ops). Workflow composition story: [`06_workflow_chain.py`](examples/06_workflow_chain.py) (no LLM — simplest possible), [`07_workflow_route.py`](examples/07_workflow_route.py) (route to specialist Agents), [`08_workflow_loop.py`](examples/08_workflow_loop.py) (refinement loop with cycles), [`09_workflow_as_tool.py`](examples/09_workflow_as_tool.py) (workflow exposed as Agent tool), [`10_workflow_architecture.py`](examples/10_workflow_architecture.py) (Agent with non-default architecture inside a workflow), [`11_workflow_custom_step.py`](examples/11_workflow_custom_step.py) (Agent wrapped in a custom step for prompt formatting + capturing `RunResult` metadata) |

---

## API stability

The framework is pre-1.0 — major versions can introduce breaking
changes — but the surface area is split into stability tiers so
adopters know what they can pin against today.

The package is split into **two import tiers**:

* **Top-level** (`from loomflow import ...`) — the daily-use surface. 66 names. Stable; will not break in 0.x without a migration note. This is what 90% of code touches.
* **Submodule** (`from loomflow.memory.postgres import PostgresMemory`, etc.) — backend-specific or specialized classes. Importing them deliberately signals "I'm picking this concrete backend / architecture / adapter."

| Tier | What's in it | Example imports |
|---|---|---|
| **Stable top-level** | `Agent`, `Workflow`, `WorkflowResult`, `step`, `START`, `END`, `tool`, `Tool`, `RunContext`, `RunResult`, `get_run_context`, `set_run_context`, common types (`Message`, `Episode`, `Fact`, `Role`, `Event`, `Usage`, `MemoryBlock`, `MemoryProfile`, `MemoryExport`, `BudgetStatus`, `PermissionDecision`, `ToolCall`, `ToolDef`, `ToolResult`), core protocols (`Memory`, `Model`, `Permissions`, `Budget`, `HookHost`, `Sandbox`, `Telemetry`, `ToolHost`, `Runtime`, `Embedder`, `Secrets`, `AuditLog`, `Architecture`), default backends (`InMemoryMemory`, `InMemoryAuditLog`, `NoBudget`, `NoTelemetry`, `NoSandbox`, `AllowAll`, `Mode`, `StandardPermissions`, `HookRegistry`, `StandardBudget`, `BudgetConfig`, `InProcRuntime`, `ReAct`, `HashEmbedder`), `resolve_memory`, test fakes (`EchoModel`, `ScriptedModel`, `ScriptedTurn`), common errors (`LoomError`, `OutputValidationError`, `IsolationWarning`, `BudgetExceeded`, `ConfigError`, `LoomDeprecationWarning`), `new_id` | `from loomflow import Agent, Workflow, step, tool` |
| **Stable submodule** | All backend-specific concrete classes — they live in their submodule and are intentionally NOT re-exported at top level. `PostgresMemory`, `ChromaMemory`, `RedisMemory`, `SqliteMemory`, `VectorMemory`, `LazyMemory`, `AutoExtractMemory`, `OpenAIModel`, `AnthropicModel`, `LiteLLMModel`, `SqliteRuntime`, `PostgresRuntime`, `JournaledRuntime`, `PerUserPermissions`, `EnvSecrets`, `DictSecrets`, `FilesystemSandbox`, `SubprocessSandbox`, `FileAuditLog`, `OTelTelemetry`, all vector stores, all embedders, all Anthropic-cookbook architectures (`SelfRefine`, `Reflexion`, `TreeOfThoughts`, `PlanAndExecute`, `ReWOO`, `Supervisor`, `Swarm`, `Router`, `MultiAgentDebate`, `BlackboardArchitecture`, `ActorCritic`), built-in tools (`read_tool`, `write_tool`, `bash_tool`, `edit_tool`), `Team`, `Skill`, `MCPClient`, `MCPRegistry` | `from loomflow.memory.postgres import PostgresMemory`<br>`from loomflow.model.openai import OpenAIModel`<br>`from loomflow.architecture import SelfRefine`<br>`from loomflow.security import PerUserPermissions, EnvSecrets`<br>`from loomflow.observability import OTelTelemetry`<br>`from loomflow.team import Team` |
| **Experimental** | `agent.generate_graph()`, the `Team.*` builders | Useful, tested, but newer — internals may change. |
| **Internal** | `_loop`, `_wrapped_model`, `_wrapped_memory`, `Dependencies`, `AgentSession`, anything starting with `_` | No stability promise. Subject to change without notice. |

If a symbol isn't listed, it's experimental by default. Open an
issue if you depend on something not yet in the Stable tier and
need it promoted.

---

## Status

* **1010 tests pass** offline in ~22 seconds; **10 live tests pass**
  against real OpenAI in ~30 seconds (5 env-gated integrations
  skip without `JEEVES_TEST_PG_DSN` / `JEEVES_TEST_REDIS_URL`)
* **mypy `--strict`** clean across 112 production source files
* **ruff** clean including `flake8-async` lints
* v0.10 ships **multi-tenancy by default**, **structured outputs**,
  **retry-on-transient by default**, and the **fast path by
  default**. Every layer (audit, telemetry, permissions, hooks,
  runtime, budget) is detected as no-op or production-wired at
  construction time, so a barebones `Agent` runs at LangChain-class
  latency with the integration skipped. `user_id` and `session_id`
  are first-class typed primitives — memory is hard-partitioned
  per `user_id`, conversations continue when a `session_id` is
  reused, and sub-agents inherit the parent's `RunContext`
  automatically via a contextvar (`get_run_context()`). Pass
  `output_schema=` (any Pydantic `BaseModel`) and `agent.run`
  returns a typed, validated instance on `result.parsed` — with
  retry-with-feedback on validation failure. Network model
  adapters are auto-wrapped with a typed error taxonomy
  (`TransientModelError` / `RateLimitError` /
  `PermanentModelError` / `AuthenticationError` /
  `InvalidRequestError` / `ContentFilterError`) and a configurable
  `RetryPolicy` so transient 5xx / 429 / network blips don't blow
  up production runs. All zero-config; no flags.
* v0.9 ships **Skills** (Anthropic Agent Skills format, with
  `tools.py` auto-discovery for in-process Python tools and
  frontmatter `tools:` manifest for any-language scripts wrapped
  as typed tools), agent-graph visualization (`agent.generate_graph()`
  → Mermaid / PNG), the `Team` facade for ergonomic multi-agent
  construction, the full vector-store stack (`InMemoryVectorStore` /
  Chroma / Postgres / FAISS — Mongo-style filters, MMR diversity,
  BM25 hybrid search, persistence), the document loader with
  chunking strategies, and 12 architectures including selective
  lesson recall (Reflexion), typed handoffs (Swarm),
  forward_message (Supervisor), Jaccard convergence (Debate), and
  parallel proposer/evaluator with min_score floor (TreeOfThoughts).

---

## Verify your install

```bash
git clone <repo>
cd loomflow
pip install -e '.[dev]'
ruff check loomflow
mypy --strict loomflow
pytest tests/ -v
```

You should see 815 passed. Five integration tests skip without
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
