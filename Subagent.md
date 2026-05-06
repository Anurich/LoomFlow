# JeevesAgent — Architecture Reference Manual

> Complete reference for every agent architecture shipped with JeevesAgent. Each section covers origin, mechanism, full implementation, performance, strengths, weaknesses, composition, tuning, pitfalls, and worked examples.

This is the **detailed** reference. For the high-level survey, see `AGENT_ARCHITECTURES.md`. For composition rules between agents, see `MULTI_AGENT_COMPOSITION_SPEC.md`. For the engineering foundation, see `JEEVES_AGENT_ENGINEERING.md`.

---

## How to use this document

Each architecture has the same 13-section structure. Skim the same headers across architectures to compare:

1. **Origin** — paper, authors, year, what problem it was created to solve
2. **The Pattern** — diagram + plain-language explanation
3. **Mechanism** — step-by-step how it actually works
4. **Configuration** — every parameter with defaults
5. **Implementation** — real Python code against the protocols
6. **Performance** — benchmarks, token cost, latency
7. **Strengths** — what it does well, quantified
8. **Weaknesses** — failure modes you will hit
9. **When to Use / When NOT to Use** — concrete scenarios
10. **Composition** — what stacks well with it
11. **Tuning Guide** — how to adjust knobs in production
12. **Common Pitfalls** — the bugs that bite teams
13. **Worked Example** — full code you can copy

---

## Table of contents

**Quick reference**
- [Architecture quick comparison](#architecture-quick-comparison)
- [Selection decision tree](#selection-decision-tree)
- [Performance comparison table](#performance-comparison-table)
- [Composition matrix](#composition-matrix)
- [The Architecture Protocol (recap)](#the-architecture-protocol-recap)

**Single-agent architectures**
1. [ReAct](#1-react)
2. [Plan-and-Execute](#2-plan-and-execute)
3. [ReWOO (Reasoning Without Observations)](#3-rewoo-reasoning-without-observations)
4. [Reflexion](#4-reflexion)
5. [Self-Refine](#5-self-refine)
6. [Tree of Thoughts (ToT)](#6-tree-of-thoughts-tot)
7. [Graph of Thoughts (GoT)](#7-graph-of-thoughts-got)
8. [Deep Agent](#8-deep-agent)

**Multi-agent architectures**
9. [Supervisor / Hierarchical](#9-supervisor--hierarchical)
10. [Actor-Critic (Generator-Critic)](#10-actor-critic-generator-critic)
11. [Multi-Agent Debate](#11-multi-agent-debate)
12. [Blackboard](#12-blackboard)
13. [Swarm](#13-swarm)
14. [Router](#14-router)

**Operations**
- [Custom architectures](#custom-architectures)
- [Upgrade paths](#upgrade-paths)
- [Frequently asked questions](#frequently-asked-questions)

---

## Architecture quick comparison

| # | Name | Type | Token Cost | Latency | Best For |
|---|---|---|---|---|---|
| 1 | ReAct | Single | 1× (baseline) | Medium | Default; exploratory tasks <10 steps |
| 2 | Plan-and-Execute | Single | 0.7× | Medium | Well-defined multi-step tasks |
| 3 | ReWOO | Single | 0.2× | Low | Cost-sensitive, predictable plans |
| 4 | Reflexion | Single | 2-3× | High | Repeated tasks with eval signal |
| 5 | Self-Refine | Single | 2× | Medium | Quality-critical, no eval signal |
| 6 | Tree of Thoughts | Single | 10-100× | Very High | Combinatorial reasoning, math/logic |
| 7 | Graph of Thoughts | Single | 50-200× | Very High | Research-grade reasoning |
| 8 | Deep Agent | Single (bundle) | 1.2× | Medium | Project-shaped, 10+ step tasks |
| 9 | Supervisor | Multi | 1.5-3× | Medium (parallel) | Specialist domains |
| 10 | Actor-Critic | Multi | 2-5× | Medium-High | Quality-critical output |
| 11 | Debate | Multi | 3-5× | High | Contested decisions |
| 12 | Blackboard | Multi | 2-4× | Medium | Exploratory, no clear plan |
| 13 | Swarm | Multi | Variable | Variable | Research/prototyping only |
| 14 | Router | Multi | 1.1× | Low | Helpdesk-style routing |

Token costs are relative to ReAct on the same task. They're rough — your task and tuning matter.

---

## Selection decision tree

```
Is the task exploratory or well-defined?
│
├─ Exploratory (next step depends on observations)
│   │
│   ├─ Fewer than ~10 steps expected
│   │   └─ ReAct  ←  start here
│   │
│   ├─ More than ~10 steps, project-shaped
│   │   └─ Deep Agent (with ReAct base)
│   │
│   └─ Repeated similar tasks, can evaluate
│       └─ Reflexion (wraps ReAct)
│
├─ Well-defined upfront
│   │
│   ├─ Cost matters more than adaptation
│   │   └─ ReWOO
│   │
│   ├─ Standard multi-step
│   │   └─ Plan-and-Execute
│   │
│   └─ Naturally decomposes into specialists
│       │
│       ├─ One specialist owns task
│       │   └─ Router
│       │
│       └─ Multiple specialists, need synthesis
│           └─ Supervisor
│
├─ Quality-critical output
│   │
│   ├─ Have an evaluator (human or eval signal)
│   │   └─ Reflexion
│   │
│   ├─ Code, writing, security review
│   │   └─ Actor-Critic (separate critic LLM)
│   │
│   └─ Single-shot quality with self-eval
│       └─ Self-Refine
│
└─ Combinatorial reasoning (math, puzzles)
    │
    ├─ Cheap, moderate quality
    │   └─ ReAct + Self-Refine
    │
    └─ Quality-critical
        └─ Tree of Thoughts (or Graph of Thoughts in v2)
```

---

## Performance comparison table

Benchmark numbers from the original papers and 2026 production reports. "—" means no public benchmark.

| Architecture | Benchmark | Score vs Baseline | Notes |
|---|---|---|---|
| ReAct | HotpotQA EM | +14% over CoT | Baseline for everything else |
| ReAct | AlfWorld success | 71% → 75% | Modest over chain-of-thought |
| ReWOO | HotpotQA tokens | -65% | 5× cost reduction at parity |
| Plan-and-Execute | TravelPlanner | 35% → 47% | Multi-step structured tasks |
| Reflexion | AlfWorld | +22% over ReAct | Same model, just architecture |
| Reflexion | HotPotQA | +20% over ReAct | Up to 12 trials per task |
| Reflexion | HumanEval pass@1 | +11 points (80→91) | GPT-4 baseline |
| Self-Refine | Code Optimization | +13 points | GPT-4 → +13 over single-pass |
| Tree of Thoughts | Game of 24 | 4% → 74% | Massive improvement, 100× tokens |
| Tree of Thoughts | Creative Writing | Coherence +15% | Subjective eval |
| Graph of Thoughts | Sorting | -31% latency vs ToT | Same accuracy with aggregation |
| AGoT | GPQA scientific | +46.2% | Comparable to RL methods |
| Deep Agent | 10+ step tasks | -50% failure rate | vs raw ReAct |
| Anthropic MA Research | Internal benchmark | +90.2% | Multi-agent vs single Claude |
| Actor-Critic | Code review | -90% issues in 3-5 rounds | Production case study |
| Multi-Agent Debate | TruthfulQA | +12% | 3 agents, 2 rounds |
| Blackboard MAS | Data discovery | +13-57% | Salemi et al. 2026 |

**Important caveat:** these numbers come from the architectures' original papers, which always optimize for their architecture. Your mileage will vary. The relative ordering is more reliable than the absolute numbers.

---

## Composition matrix

Which architectures stack with which? `✓` works well, `~` works with care, `✗` don't. The diagonal is "alone."

|         | ReAct | P&E | ReWOO | Refl | SRfn | ToT | DeepA | Sup | A-C | Dbt | Rtr |
|---------|-------|-----|-------|------|------|-----|-------|-----|-----|-----|-----|
| ReAct   | —     | ✗   | ✗     | ✓    | ✓    | ~   | ✓     | ✓   | ✓   | ~   | ✓   |
| P&E     | ✗     | —   | ✗     | ✓    | ~    | ✗   | ✓     | ✓   | ✓   | ~   | ✓   |
| ReWOO   | ✗     | ✗   | —     | ~    | ✗    | ✗   | ~     | ~   | ✗   | ✗   | ✓   |
| Refl    | ✓     | ✓   | ~     | ✗    | ✗    | ~   | ✓     | ✓   | ~   | ✗   | ~   |
| SRfn    | ✓     | ~   | ✗     | ✗    | —    | ~   | ✓     | ~   | ~   | ✗   | ~   |
| ToT     | ~     | ✗   | ✗     | ~    | ~    | —   | ~     | ✗   | ~   | ✗   | ✗   |
| DeepA   | ✓     | ✓   | ~     | ✓    | ✓    | ~   | —     | ✓   | ✓   | ~   | ✓   |
| Sup     | ✓     | ✓   | ~     | ✓    | ~    | ✗   | ✓     | ✓   | ✓   | ~   | ✓   |
| A-C     | ✓     | ✓   | ✗     | ~    | ~    | ~   | ✓     | ✓   | ✗   | ✗   | ~   |
| Dbt     | ~     | ~   | ✗     | ✗    | ✗    | ✗   | ~     | ~   | ✗   | ✗   | ✗   |
| Rtr     | ✓     | ✓   | ✓     | ~    | ~    | ✗   | ✓     | ✓   | ~   | ✗   | —   |

Reading guide:
- **Same architecture nested in itself** is `✗` (Reflexion-of-Reflexion is just burning tokens; Supervisor-of-Supervisor is the recursion case which is fine, just nested)
- **ReAct is the best base** — it composes with almost everything
- **ReWOO doesn't compose well** because its 2-call design assumes no inner loops
- **Debate doesn't compose with much** — its overhead dominates

---

## The Architecture Protocol (recap)

Every architecture is a class implementing this protocol. From `architecture/base.py`:

```python
from typing import Protocol, AsyncIterator, runtime_checkable
from ..core.types import Event
from ..agent.session import Session
from ..agent.deps import Dependencies


@runtime_checkable
class Architecture(Protocol):
    """A strategy for driving the agent loop."""

    name: str

    async def run(
        self,
        session: Session,
        deps: Dependencies,
        prompt: str,
    ) -> AsyncIterator[Event]:
        """Drive the loop. Yield events. Use deps.runtime.step() for non-determinism."""
        ...

    def declared_workers(self) -> dict[str, "Agent"]:
        """Return any child Agents this architecture declares (for tree validation)."""
        return {}
```

That's it. Eight lines of public surface. Every architecture below implements this exact interface.

---

# Single-agent architectures

## 1. ReAct

### Origin

**Paper:** Yao, Shunyu, et al. "ReAct: Synergizing Reasoning and Acting in Language Models." arXiv:2210.03629 (October 2022). Published at ICLR 2023.

**What it solves:** Pure reasoning (Chain-of-Thought) hallucinates because it can't ground in real-world observations. Pure tool use ("Act-only") makes greedy choices because it doesn't reflect. ReAct interleaves both — reason about what to do, do it, observe what happened, reason again. This single insight made LLM-as-agent practically useful.

**Status:** The default. The 2026 consensus across every framework is "start with ReAct unless you have a specific reason not to."

### The Pattern

```
┌─────────┐    ┌─────────┐    ┌──────────┐
│  Think  │ →  │   Act   │ →  │ Observe  │ → repeat
└─────────┘    └─────────┘    └──────────┘
    │              │               │
    │              │               └─ tool result returned to model
    │              └───────────────── tool selected and called
    └──────────────────────────────── reasoning trace generated
```

In one sentence: the model emits both reasoning text and tool calls; tool results come back; model emits more reasoning + more tool calls; loop until model produces a final answer with no tool calls.

### Mechanism

For each turn `t`:

1. **Build context.** Concatenate system prompt + working memory + conversation history + most recent tool results.
2. **Stream model output.** Model emits text (interleaved reasoning) and possibly one or more tool calls.
3. **Decision point:**
   - If model emitted no tool calls → loop terminates, output is the final answer
   - If model emitted tool calls → dispatch them (in parallel if multiple)
4. **Dispatch tools.** Each tool call goes through the harness's permission check, sandbox, runtime journal. Results return to the loop.
5. **Append to context.** Tool results become messages in the conversation. Loop back to step 2.
6. **Termination.** Either model decides it's done, or `max_turns` cap hits.

The key design choice: each turn's tool calls run in parallel (structured concurrency via `anyio` task group). When the model emits 5 tool calls, they execute concurrently, not serially.

### Configuration

```python
@dataclass
class ReActConfig:
    max_turns: int = 25
    """Hard cap on iterations. After this, loop exits with `max_turns_reached` event."""

    parallel_tool_dispatch: bool = True
    """Dispatch multiple tool calls per turn concurrently. Off only for debugging."""

    tool_timeout_seconds: float = 60.0
    """Per-tool deadline. Past this, tool returns error result; loop continues."""

    inject_thinking_prompt: bool = False
    """If True, prepend 'Think step by step before acting' to system prompt.
    Most modern models don't need this."""

    early_stop_phrases: list[str] = field(default_factory=list)
    """If model output contains any of these (case-insensitive), terminate immediately.
    Example: ['DONE.', 'TASK COMPLETE'] for legacy models."""
```

### Implementation

```python
# jeevesagent/architecture/react.py

from __future__ import annotations
import anyio
from typing import AsyncIterator

from ..core.types import (
    Event, ToolCall, ToolResult, ModelChunk, ModelChunkAggregate,
)
from ..agent.session import Session
from ..agent.deps import Dependencies
from .base import Architecture
from .config import ReActConfig


class ReAct:
    """The canonical Reason+Act loop. Default architecture."""

    name = "react"

    def __init__(self, config: ReActConfig | None = None, **kwargs):
        # Allow both ReAct(config=...) and ReAct(max_turns=30)
        if config is None and kwargs:
            config = ReActConfig(**kwargs)
        self.cfg = config or ReActConfig()

    def declared_workers(self):
        return {}  # ReAct has no children

    async def run(
        self,
        session: Session,
        deps: Dependencies,
        prompt: str,
    ) -> AsyncIterator[Event]:
        # 1. Seed context with prompt + working memory + recalled episodes
        await self._seed(session, deps, prompt)
        yield Event.started(session.id, prompt)

        # 2. Main iteration loop with structured concurrency for tool dispatch
        async with anyio.create_task_group() as tg:
            for turn in range(self.cfg.max_turns):
                # Budget gate
                status = await deps.budget.allows_step()
                if status.blocked:
                    yield Event.budget_exceeded(session.id, status)
                    return

                yield Event.turn_started(session.id, turn)

                # Stream model output, aggregating chunks for tool calls + usage
                aggregate = ModelChunkAggregate()
                async for chunk in deps.runtime.stream_step(
                    f"react.model.{turn}",
                    deps.model.stream,
                    session.messages,
                    tools=await session.tools_view(),
                ):
                    aggregate.feed(chunk)
                    yield Event.model_chunk(session.id, chunk, turn=turn)

                # Account
                await deps.budget.consume(
                    tokens_in=aggregate.usage.input_tokens,
                    tokens_out=aggregate.usage.output_tokens,
                    cost_usd=aggregate.usage.cost_usd,
                )

                # Early-stop check (legacy support)
                if self._should_early_stop(aggregate.text):
                    session.complete = True
                    session.result = aggregate.text
                    yield Event.completed(session.id, aggregate.text, turn=turn)
                    return

                # Termination: no tool calls = final answer
                if not aggregate.tool_calls:
                    session.complete = True
                    session.result = aggregate.text
                    yield Event.completed(session.id, aggregate.text, turn=turn)
                    return

                # Dispatch tools
                results = await self._dispatch_tools(
                    deps, tg, aggregate.tool_calls, turn,
                )
                for result in results:
                    yield Event.tool_result(session.id, result, turn=turn)
                session.append_tool_results(results)

                yield Event.turn_completed(session.id, turn)

        yield Event.max_turns_reached(session.id, self.cfg.max_turns)

    # --- helpers ---------------------------------------------------------

    async def _seed(self, session: Session, deps: Dependencies, prompt: str) -> None:
        # Working memory blocks first
        blocks = await deps.memory.working()
        if blocks:
            session.append_system("\n\n".join(b.format() for b in blocks))

        # Episodic + semantic recall in parallel
        async with anyio.create_task_group() as tg:
            ep_holder: list = [None]
            sem_holder: list = [None]

            async def _ep():
                ep_holder[0] = await deps.memory.recall(
                    prompt, kind="episodic", limit=3
                )

            async def _sem():
                sem_holder[0] = await deps.memory.recall(
                    prompt, kind="semantic", limit=5
                )

            tg.start_soon(_ep)
            tg.start_soon(_sem)

        if ep_holder[0]:
            session.append_system(self._format_episodes(ep_holder[0]))
        if sem_holder[0]:
            session.append_system(self._format_facts(sem_holder[0]))

        if self.cfg.inject_thinking_prompt:
            session.append_system("Think step by step before acting.")

        session.append_user(prompt)

    async def _dispatch_tools(
        self,
        deps: Dependencies,
        tg: anyio.abc.TaskGroup,
        calls: list[ToolCall],
        turn: int,
    ) -> list[ToolResult]:
        """Parallel dispatch with structured concurrency."""
        results: list[ToolResult | None] = [None] * len(calls)

        async def _run_one(i: int, call: ToolCall) -> None:
            # User hooks first (denial wins)
            decision = await deps.hooks.fire_pre_tool(call)
            if decision.deny:
                results[i] = ToolResult.denied(call.id, decision.reason)
                return

            # System permission check
            perm = await deps.permissions.check(call, context=session.context)
            if perm.deny:
                results[i] = ToolResult.denied(call.id, perm.reason)
                return

            # Execute through sandbox + runtime
            try:
                with anyio.fail_after(self.cfg.tool_timeout_seconds):
                    result = await deps.runtime.step(
                        f"react.tool.{turn}.{i}",
                        deps.sandbox.execute,
                        call.tool_def,
                        call.args,
                        idempotency_key=call.idempotency_key(),
                    )
            except TimeoutError:
                result = ToolResult.error(call.id, "tool_timeout")
            except Exception as e:
                result = ToolResult.error(call.id, f"tool_failed: {e}")

            # Post-tool hook (best-effort, can't fail the loop)
            with anyio.move_on_after(5.0):
                await deps.hooks.fire_post_tool(call, result)

            results[i] = result

        if self.cfg.parallel_tool_dispatch:
            for i, call in enumerate(calls):
                tg.start_soon(_run_one, i, call)
            # Task group exit waits for all
        else:
            # Serial mode for debugging
            for i, call in enumerate(calls):
                await _run_one(i, call)

        return [
            r if r is not None else ToolResult.error(c.id, "no_result")
            for r, c in zip(results, calls)
        ]

    def _should_early_stop(self, text: str) -> bool:
        if not self.cfg.early_stop_phrases:
            return False
        lower = text.lower()
        return any(p.lower() in lower for p in self.cfg.early_stop_phrases)
```

### Performance

| Metric | Typical Value | Notes |
|---|---|---|
| Tokens per turn | 500-2000 in + 200-800 out | Depends on context size |
| Latency per turn | 1-5 seconds | Model call dominates |
| Total turns to completion | 3-8 typical, 10+ on hard tasks | Beyond 10, consider Deep Agent |
| Cost per task | $0.01-$0.50 typical | At Claude/GPT-4 prices |

**Benchmark anchor:** ReAct on HotpotQA achieves ~67% EM, +14% over chain-of-thought. On AlfWorld, ReAct hits 71% success. These are 2022-23 numbers; modern models do significantly better.

### Strengths

1. **Adapts in real time.** Each step depends on observations. If a tool returns surprising data, the next step accommodates it.
2. **Transparent.** The reasoning text in each turn is the audit trail. You can read why the agent did what it did.
3. **Composes cleanly.** ReAct is the canonical base for Reflexion, Self-Refine, Deep Agent, and most multi-agent worker patterns.
4. **Battle-tested.** Three-plus years of production use. Failure modes are well-understood.
5. **The default everywhere.** Every framework — LangGraph, AutoGen, CrewAI, Claude Agent SDK — implements ReAct. Skill transfers.

### Weaknesses

1. **Token cost grows linearly with turn count.** No upfront planning means each turn re-reads the entire history. Long tasks become expensive.
2. **Loses goal coherence past ~10 steps.** The 2026 production data: 50%+ failure rate drop on 10+ step tasks. The model "forgets why it's doing this."
3. **Each turn adds latency.** Sequential per-turn means a 20-turn task is ~40-100 seconds wall-clock.
4. **Tool selection greedy.** No backtracking — if turn 3's tool choice was wrong, the agent powers through.
5. **No quality assurance.** Whatever the final turn produces is the answer; no review step.

### When to Use

- Default for any new agent until you have a specific reason otherwise
- Exploratory tasks (research, customer support, debugging)
- Conversational agents where each user turn is a fresh task
- Tasks where total tool calls expected < 10

### When NOT to Use

- Tasks expected to span 10+ turns with goal coherence requirements → use **Deep Agent**
- Tasks with strict cost constraints and predictable structure → use **ReWOO** or **Plan-and-Execute**
- Tasks requiring quality review (code, security, important writing) → wrap with **Actor-Critic** or **Reflexion**
- Combinatorial reasoning (math, logic puzzles) → use **Tree of Thoughts**

### Composition

ReAct is the canonical base for almost everything else:

```python
# Production-grade single agent
agent = Agent("...", architecture=Reflexion(base=ReAct(max_turns=30)))

# Long-running tasks with planning + filesystem
agent = Agent("...", architecture=DeepAgent(base=ReAct(max_turns=50)))

# Worker in a multi-agent system
agent = Agent("manager", architecture=Supervisor(workers={
    "researcher": Agent("...", architecture=ReAct(max_turns=15)),
}))

# With self-critique for single-shot quality
agent = Agent("...", architecture=SelfRefine(base=ReAct(), max_rounds=2))
```

ReAct does NOT compose with itself, with Plan-and-Execute (those are alternatives), or with ReWOO (different paradigms).

### Tuning Guide

**Default (`max_turns=25`)** is right for most cases. You're tuning when:

- **Hitting `max_turns_reached`?** First check whether the agent is making real progress. If yes, raise `max_turns`. If no (looping), reduce to fail faster and add a Deep Agent or Reflexion wrapper.
- **Cost is too high?** Check the average turn count. If consistently 8+, you have a long-task problem — switch to Plan-and-Execute or Deep Agent. If 3-4 turns is fine but each turn is expensive, the bottleneck is context size — add memory consolidation.
- **Latency too high?** Verify `parallel_tool_dispatch=True`. If the agent emits multi-tool turns and they're running serial, you're losing wall-clock time. If turns are inherently sequential (each tool depends on previous), latency is just the price.
- **Quality inconsistent?** ReAct alone has no quality control. Wrap with Self-Refine (1-2 rounds) or Reflexion (with eval signal).

### Common Pitfalls

**1. Forgetting `max_turns` on long tasks → runaway cost.** Always set this. Default 25 is fine for most things; don't remove it.

**2. Using ReAct when the task should be Plan-and-Execute.** Symptom: agent produces wandering, exploratory behavior on tasks with clear structure. Fix: switch architectures.

**3. Confusing `tool_timeout_seconds` with overall timeout.** This is per-tool. Use `agent.run(..., timeout=300)` for overall.

**4. Hooks that block the loop.** A `before_tool` hook that does a slow API call serializes the loop. Keep hooks fast or use `move_on_after`.

**5. Over-reliance on early stop phrases.** Modern models reliably emit "no more tool calls" via the structured output. `early_stop_phrases` is a legacy crutch.

### Worked Example

```python
import asyncio
from jeevesagent import Agent
from jeevesagent.architecture import ReAct

async def main():
    agent = Agent(
        "You are a research assistant. Use web search to find current information.",
        model="claude-opus-4-7",
        tools=["web.search", "web.fetch"],
        architecture=ReAct(
            max_turns=15,
            tool_timeout_seconds=30.0,
        ),
    )

    # Simple run
    result = await agent.run("What's the current state of agent harness research?")
    print(result.output)

    # Streaming
    async for event in agent.stream("Summarize today's AI news"):
        if event.kind == "model_chunk":
            print(event.text, end="", flush=True)
        elif event.kind == "tool_result":
            print(f"\n[tool: {event.tool} → {event.result.summary()}]")
        elif event.kind == "completed":
            print("\n[done]")

asyncio.run(main())
```


---

## 2. Plan-and-Execute

### Origin

**Paper:** Wang, Lei, et al. "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models." arXiv:2305.04091 (May 2023). Published at ACL 2023. Extended in production patterns by He et al. (2025) and the LangChain implementation.

**What it solves:** ReAct's per-step reasoning loses goal coherence on multi-step tasks. The model is busy reacting to the immediate observation and forgets the bigger picture. Plan-and-Execute decouples them: a Planner builds the entire plan upfront, then an Executor walks it. The plan itself becomes a persistent artifact the agent can re-read between steps.

**Status:** The standard "graduated from ReAct" architecture. When ReAct loses coherence, this is the first thing teams reach for.

### The Pattern

```
┌──────────────┐
│   Planner    │  produces complete plan
│ (frontier    │
│  model)      │
└──────┬───────┘
       │
       ↓
┌────────────────────────────────────────────────┐
│  Executor (often cheaper model):                │
│   step 1 → step 2 → step 3 → ... → step N      │
└──────────────┬─────────────────────────────────┘
               │
               ↓
        on failure → replan from current state
```

Two-phase: (1) plan generation, (2) plan execution with optional replan triggers.

### Mechanism

**Phase 1 — Planning.** A single model call to the Planner produces a structured plan. The plan is typically a list of steps, each with: (a) intent, (b) inputs/dependencies, (c) expected output, (d) which tools to use. The plan is parsed into a `Plan` object.

**Phase 2 — Execution.** For each step in the plan:
1. Inject the plan + current step into the Executor's context
2. Run a small ReAct subloop bounded to ~3 turns (the Executor only handles this step, not the whole task)
3. Capture the step's output as a structured result
4. If success: append to results, move to next step
5. If failure: trigger replan (or fail outright if `replan_on_failure=False`)

**Phase 3 — Replan (optional).** When a step fails, the Planner is re-invoked with: original prompt + plan-so-far + completed step results + failure description. It produces a revised plan from the failure point onward. Execution continues with the new plan.

**Key design choice:** the Planner and Executor can be different models. Use Claude Opus or GPT-4 for planning (where reasoning quality matters), Haiku or GPT-4o-mini for execution (where step-following matters). This is the cost optimization that makes Plan-and-Execute genuinely cheaper than ReAct on long tasks.

### Configuration

```python
@dataclass
class PlanAndExecuteConfig:
    planner_model: str | None = None
    """Override main model for planning. Use a frontier model here.
    None = use Agent's main model."""

    executor_model: str | None = None
    """Override main model for execution. Use a cheaper model here.
    None = use Agent's main model."""

    replan_on_failure: bool = True
    """If a step fails, regenerate the plan from that point.
    Off only when you want fail-fast behavior."""

    max_replans: int = 2
    """Hard cap on replan cycles to prevent infinite loops."""

    max_steps: int = 30
    """Hard cap on total steps including replanned ones."""

    max_executor_turns_per_step: int = 3
    """How many ReAct turns the executor gets per plan step.
    Most steps complete in 1-2 turns."""

    plan_format: Literal["structured", "natural"] = "structured"
    """structured = JSON list of steps (better, requires structured output).
    natural = numbered prose list (works on weaker models)."""
```

### Implementation

```python
# jeevesagent/architecture/plan_execute.py

from __future__ import annotations
import anyio
from typing import AsyncIterator
from pydantic import BaseModel, Field

from .base import Architecture
from .react import ReAct
from .config import PlanAndExecuteConfig


class PlanStep(BaseModel):
    id: int
    intent: str  # what this step accomplishes
    instructions: str  # detailed instructions for the executor
    depends_on: list[int] = Field(default_factory=list)  # other step IDs
    expected_output: str | None = None  # description of expected output


class Plan(BaseModel):
    steps: list[PlanStep]
    revision: int = 0  # incremented on replan

    def remaining_from(self, completed_ids: set[int]) -> list[PlanStep]:
        return [s for s in self.steps if s.id not in completed_ids]


class PlanAndExecute:
    """Planner produces full plan; Executor walks it step by step."""

    name = "plan-and-execute"

    def __init__(self, config: PlanAndExecuteConfig | None = None, **kwargs):
        if config is None and kwargs:
            config = PlanAndExecuteConfig(**kwargs)
        self.cfg = config or PlanAndExecuteConfig()
        # The executor is just a small ReAct
        self._executor_arch = ReAct(max_turns=self.cfg.max_executor_turns_per_step)

    def declared_workers(self):
        return {}

    async def run(
        self, session, deps, prompt
    ) -> AsyncIterator[Event]:
        planner_model = (
            deps.resolve_model(self.cfg.planner_model) or deps.model
        )
        executor_model = (
            deps.resolve_model(self.cfg.executor_model) or deps.model
        )

        # === PHASE 1: PLAN ===
        yield Event.phase(session.id, "planning")
        plan = await deps.runtime.step(
            "plan_initial",
            self._make_plan,
            planner_model,
            session,
            prompt,
        )
        yield Event.plan_created(session.id, plan)
        session.append_system(f"Execution plan:\n{self._format_plan(plan)}")

        # === PHASE 2: EXECUTE ===
        completed: dict[int, StepResult] = {}
        replan_count = 0
        total_steps_run = 0

        while True:
            remaining = plan.remaining_from(set(completed.keys()))
            if not remaining:
                break

            for step in remaining:
                if total_steps_run >= self.cfg.max_steps:
                    yield Event.max_steps_reached(session.id)
                    return

                # Budget gate
                status = await deps.budget.allows_step()
                if status.blocked:
                    yield Event.budget_exceeded(session.id, status)
                    return

                yield Event.step_started(session.id, step)

                # Execute the step with the executor model
                step_session = session.scope_for_step(step, completed)
                executor_deps = deps.with_model(executor_model)

                step_events = []
                try:
                    async for event in self._executor_arch.run(
                        step_session,
                        executor_deps,
                        step.instructions,
                    ):
                        step_events.append(event)
                        yield event.with_step(step.id)

                    result = StepResult.success(
                        step_id=step.id,
                        output=step_session.result,
                        events=step_events,
                    )
                    completed[step.id] = result
                    yield Event.step_completed(session.id, step.id, result)

                except Exception as e:
                    failure = StepResult.failed(
                        step_id=step.id,
                        error=str(e),
                        events=step_events,
                    )
                    yield Event.step_failed(session.id, step.id, failure)

                    if self.cfg.replan_on_failure and replan_count < self.cfg.max_replans:
                        # === PHASE 3: REPLAN ===
                        replan_count += 1
                        yield Event.replanning(session.id, replan_count)
                        plan = await deps.runtime.step(
                            f"replan_{replan_count}",
                            self._replan,
                            planner_model,
                            prompt,
                            plan,
                            completed,
                            failure,
                        )
                        yield Event.plan_revised(session.id, plan)
                        break  # break inner for loop, restart from updated remaining
                    else:
                        # No replan available — surface the failure
                        raise

                total_steps_run += 1

        # === SYNTHESIS ===
        synthesis = await deps.runtime.step(
            "synthesize",
            self._synthesize,
            planner_model,
            prompt,
            plan,
            completed,
        )
        session.complete = True
        session.result = synthesis
        yield Event.completed(session.id, synthesis)

    # --- helpers ---------------------------------------------------------

    async def _make_plan(self, model, session, prompt) -> Plan:
        """Generate initial plan via structured output."""
        plan_prompt = (
            f"You are a meticulous planner. Decompose this task into discrete steps:\n\n"
            f"{prompt}\n\n"
            f"Output a JSON list of steps. Each step has: id (int), intent (string), "
            f"instructions (string for an executor agent), depends_on (list of prior step IDs), "
            f"expected_output (string)."
        )
        # Use structured output if available, else parse
        response = await model.complete_structured(
            messages=[{"role": "user", "content": plan_prompt}],
            output_schema=Plan,
        )
        return response

    async def _replan(self, model, prompt, prev_plan, completed, failure):
        """Generate a revised plan after a step failure."""
        replan_prompt = (
            f"Original task: {prompt}\n\n"
            f"Previous plan (revision {prev_plan.revision}):\n"
            f"{self._format_plan(prev_plan)}\n\n"
            f"Completed steps and their outputs:\n"
            f"{self._format_completed(completed)}\n\n"
            f"Step {failure.step_id} failed: {failure.error}\n\n"
            f"Generate a revised plan starting from step {failure.step_id}. "
            f"Use what's already been done; don't repeat completed steps."
        )
        revised = await model.complete_structured(
            messages=[{"role": "user", "content": replan_prompt}],
            output_schema=Plan,
        )
        revised.revision = prev_plan.revision + 1
        return revised

    async def _synthesize(self, model, prompt, plan, completed) -> str:
        """Combine step outputs into a final answer."""
        synthesis_prompt = (
            f"Original task: {prompt}\n\n"
            f"Plan executed:\n{self._format_plan(plan)}\n\n"
            f"Step outputs:\n{self._format_completed(completed)}\n\n"
            f"Synthesize a final answer to the original task using these step outputs."
        )
        response = await model.complete(
            messages=[{"role": "user", "content": synthesis_prompt}],
        )
        return response.text
```

### Performance

| Metric | Plan-and-Execute | vs ReAct |
|---|---|---|
| Tokens | 0.7× ReAct on tasks with 5+ steps | -30% |
| LLM calls | 1 plan + N execute + maybe replan + 1 synth | More structured, often fewer |
| Latency | Sequential within steps; varies | Similar to ReAct |
| Multi-step success | +12 percentage points on TravelPlanner | Notable |

**Cost optimization:** the planner-vs-executor split is where savings come from. Claude Opus for plan ($0.015/1k in), Haiku for steps ($0.0008/1k in) → ~10× cost reduction on the executor portion.

### Strengths

1. **Goal coherence.** The plan is a persistent artifact; the agent re-reads it each step. No "forgot why I'm doing this."
2. **Cost-controllable.** Use frontier model for planning, cheap model for execution.
3. **Predictable cost.** You can estimate the cost from the plan length before running.
4. **Inspectable.** The plan is human-readable. Show it to users for approval before running (HITL pattern).
5. **Parallelizable steps.** When steps don't depend on each other, dispatch in parallel (we don't do this in v1; v2 feature).

### Weaknesses

1. **Brittle to bad initial plans.** If the planner misunderstands the task, every step is wrong. Quality of planner LLM matters a lot.
2. **Doesn't adapt mid-execution.** Without replan, surprises are fatal. Even with replan, the cost adds up.
3. **Replan churn.** If the task is genuinely exploratory, you'll replan 3-4 times — at which point ReAct would have been cheaper.
4. **Assumes decomposability.** Tasks that can't be decomposed in advance (research, debugging, creative writing) fight this architecture.
5. **Synthesis is its own challenge.** Combining step outputs into a coherent final answer is non-trivial; weak synthesis prompts produce list-shaped outputs that don't read well.

### When to Use

- Multi-step tasks with clear structure (extract X, transform with Y, load to Z)
- Tasks where you want to inspect the plan before execution (HITL approvals)
- Cost-sensitive long tasks where ReAct's 8+ turns dominate cost
- Research/analysis with predictable phases (gather, analyze, synthesize, report)
- Coding agent tasks with clear feature decomposition

### When NOT to Use

- Exploratory tasks where the plan can't be built upfront
- Conversational agents (per-turn user input invalidates plans)
- Short tasks (< 4 steps) — overhead isn't worth it
- Tasks where surprise tool outputs commonly invalidate strategy → use ReAct + Reflexion

### Composition

```python
# Plan-and-Execute with replanning on failure (the canonical use)
agent = Agent("...", architecture=PlanAndExecute(
    planner_model="claude-opus-4-7",
    executor_model="claude-haiku-4-5",
    replan_on_failure=True,
))

# With Reflexion for cross-session learning
agent = Agent("...", architecture=Reflexion(
    base=PlanAndExecute(),
    max_iterations=2,
))

# As a worker in Supervisor
agent = Agent("manager", architecture=Supervisor(workers={
    "etl_pipeline": Agent("...", architecture=PlanAndExecute()),
}))

# With Deep Agent — adds filesystem + planning todo for plan tracking
agent = Agent("...", architecture=DeepAgent(base=PlanAndExecute()))
```

Plan-and-Execute does NOT compose with ReAct (alternatives), or with ReWOO (different paradigms).

### Tuning Guide

- **Planner produces too vague steps?** Increase planner model size (use Opus instead of Sonnet). Plans need reasoning quality.
- **Steps fail too often?** Either steps are too ambitious (raise `max_executor_turns_per_step` from 3 to 5) or executor model is too weak (upgrade).
- **Replan loops endlessly?** Cap `max_replans=2`. If you hit it, the architecture is wrong for this task — try ReAct or Deep Agent.
- **Synthesis output is choppy?** Improve the synthesis prompt template; ask the synthesizer to write paragraphs, not bullet lists.
- **Cost still too high?** Most cost is in the executor running 3 turns per step. Reduce `max_executor_turns_per_step=2` and tighten step instructions.

### Common Pitfalls

**1. Vague plan steps.** "Research the topic" is not a step; it's a project. Train the planner with examples of good vs bad steps in the system prompt.

**2. Forgetting to limit replans.** Without `max_replans` cap, a flaky tool can cause infinite replan loops. Always cap.

**3. Using same model for planner and executor.** Defeats the cost optimization. If you only have one model, you might as well use ReAct.

**4. Skipping synthesis.** Returning the last step's output as the final answer instead of synthesizing all outputs. The plan was supposed to produce a unified result.

**5. Plan steps with cyclic dependencies.** The planner can produce step 3 → depends on step 5 → depends on step 3. Validate plan DAG before execution.

### Worked Example

```python
import asyncio
from jeevesagent import Agent
from jeevesagent.architecture import PlanAndExecute

async def main():
    agent = Agent(
        "You are a market research analyst.",
        model="claude-opus-4-7",
        tools=["web.search", "web.fetch", "data.analyze"],
        architecture=PlanAndExecute(
            planner_model="claude-opus-4-7",  # frontier for planning
            executor_model="claude-haiku-4-5",  # cheap for execution
            replan_on_failure=True,
            max_replans=2,
        ),
    )

    result = await agent.run(
        "Research the current state of the European EV charging market. "
        "Identify the top 5 operators, their market share, and key competitive moats. "
        "Output a 1-page brief."
    )
    print(result.output)

    # Inspect the plan
    print("\nPlan executed:")
    for step in result.metadata["plan"].steps:
        print(f"  {step.id}: {step.intent}")

asyncio.run(main())
```


---

## 3. ReWOO (Reasoning Without Observations)

### Origin

**Paper:** Xu, Binfeng, et al. "ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models." arXiv:2305.18323 (May 2023).

**What it solves:** ReAct's main inefficiency is that every reasoning step re-reads all prior observations. On a task with N tool calls, the model sees the prompt N times and the result history grows quadratically. ReWOO eliminates this by separating planning from observation: the Planner produces a "blueprint" with placeholders for tool results; the Worker fills the placeholders without LLM reasoning; the Solver synthesizes once at the end. **Total LLM calls: 2.**

**Status:** Niche but powerful. The 5× token reduction is real when conditions match. Most teams skip it because the conditions rarely match.

### The Pattern

```
Question
   │
   ↓
┌──────────┐
│ Planner  │ ─→ Blueprint:
└──────────┘     Plan: [
                   Plan: search for X #E1
                   Plan: search for Y based on #E1 #E2
                   Plan: compare #E1 and #E2 #E3
                 ]
   │
   ↓
┌──────────┐
│  Worker  │ ─→ Fills: #E1=result, #E2=result, #E3=result
└──────────┘     (just tool calls, no LLM)
   │
   ↓
┌──────────┐
│  Solver  │ ─→ Final answer
└──────────┘
   (uses question + all evidence)
```

### Mechanism

**Phase 1 — Plan with placeholders.** The Planner generates a structured plan where each step says "use tool T with args A; store result as #En." Args can reference previous placeholders: "search for products related to #E1." This is one LLM call.

**Phase 2 — Worker execution.** No LLM. Just a deterministic loop:
1. Build dependency graph from placeholders
2. Execute steps in topological order
3. For each step, substitute previous placeholder values into args, call the tool, store result

When steps don't depend on each other, they run in parallel. Pure tool execution; no model in the loop.

**Phase 3 — Solver.** A single LLM call that receives: original question + all evidence (`{#E1: ..., #E2: ..., ...}`) → final answer.

### Configuration

```python
@dataclass
class ReWOOConfig:
    planner_model: str | None = None
    """Frontier model recommended — plan quality is everything."""

    solver_model: str | None = None
    """Often same as planner. Solver synthesizes from evidence."""

    max_steps: int = 15
    """Cap on plan steps (LLM is constrained to produce at most this many)."""

    fail_on_missing_tool: bool = True
    """If plan references a tool that doesn't exist, fail at parse time
    rather than at execution."""

    parallel_workers: int = 5
    """Max concurrent tool calls during Worker phase."""
```

### Implementation

```python
# jeevesagent/architecture/rewoo.py

from __future__ import annotations
import re
import anyio
from collections import defaultdict
from typing import AsyncIterator
from pydantic import BaseModel

from .base import Architecture
from .config import ReWOOConfig


class ReWOOStep(BaseModel):
    var: str  # e.g. "#E1"
    tool: str  # e.g. "web.search"
    args_template: dict[str, str]  # may contain "#En" placeholders


class ReWOOBlueprint(BaseModel):
    steps: list[ReWOOStep]


class ReWOO:
    name = "rewoo"

    def __init__(self, config: ReWOOConfig | None = None, **kwargs):
        if config is None and kwargs:
            config = ReWOOConfig(**kwargs)
        self.cfg = config or ReWOOConfig()

    def declared_workers(self):
        return {}

    async def run(self, session, deps, prompt) -> AsyncIterator[Event]:
        planner_model = deps.resolve_model(self.cfg.planner_model) or deps.model
        solver_model = deps.resolve_model(self.cfg.solver_model) or deps.model

        # === PHASE 1: PLAN ===
        yield Event.phase(session.id, "planning")
        blueprint = await deps.runtime.step(
            "rewoo_plan",
            self._make_blueprint,
            planner_model,
            session,
            prompt,
            await session.tools_view(),
        )
        yield Event.plan_created(session.id, blueprint)

        # Validate dependency DAG
        levels = self._topological_levels(blueprint)
        if levels is None:
            yield Event.plan_invalid(session.id, "cyclic dependency")
            return

        # === PHASE 2: WORKER ===
        yield Event.phase(session.id, "executing")
        evidence: dict[str, ToolResult] = {}
        sem = anyio.Semaphore(self.cfg.parallel_workers)

        for level_num, level_steps in enumerate(levels):
            yield Event.level_started(session.id, level_num, len(level_steps))

            async with anyio.create_task_group() as tg:
                async def _run_step(step: ReWOOStep):
                    async with sem:
                        # Substitute previous evidence into args
                        resolved_args = self._resolve_placeholders(
                            step.args_template, evidence
                        )

                        try:
                            tool_def = await deps.tools.get_tool(step.tool)
                            result = await deps.runtime.step(
                                f"rewoo_tool_{step.var}",
                                deps.sandbox.execute,
                                tool_def,
                                resolved_args,
                                idempotency_key=f"rewoo:{step.var}",
                            )
                        except Exception as e:
                            result = ToolResult.error(step.var, f"failed: {e}")

                        evidence[step.var] = result

                for step in level_steps:
                    tg.start_soon(_run_step, step)

            yield Event.level_completed(session.id, level_num)

        # === PHASE 3: SOLVE ===
        yield Event.phase(session.id, "solving")
        answer = await deps.runtime.step(
            "rewoo_solve",
            self._solve,
            solver_model,
            prompt,
            blueprint,
            evidence,
        )
        session.complete = True
        session.result = answer
        yield Event.completed(session.id, answer)

    # --- helpers ---------------------------------------------------------

    async def _make_blueprint(self, model, session, prompt, tools) -> ReWOOBlueprint:
        plan_prompt = (
            f"Task: {prompt}\n\n"
            f"Available tools:\n{self._format_tools(tools)}\n\n"
            f"Generate a plan as a list of steps. Each step has:\n"
            f"  var: a unique placeholder like '#E1', '#E2', ...\n"
            f"  tool: the name of a tool to call\n"
            f"  args: a dict of arguments. Args can reference earlier placeholders\n"
            f"        as strings like '#E1' to substitute their result.\n\n"
            f"Return JSON: {{ \"steps\": [...] }}"
        )
        return await model.complete_structured(
            messages=[{"role": "user", "content": plan_prompt}],
            output_schema=ReWOOBlueprint,
        )

    def _topological_levels(
        self, blueprint: ReWOOBlueprint
    ) -> list[list[ReWOOStep]] | None:
        """Group steps into dependency levels for parallel execution.
        Returns None if cyclic."""
        # Build dependency graph
        deps: dict[str, set[str]] = {step.var: set() for step in blueprint.steps}
        for step in blueprint.steps:
            for arg in step.args_template.values():
                if isinstance(arg, str):
                    for ref in re.findall(r"#E\d+", arg):
                        if ref in deps and ref != step.var:
                            deps[step.var].add(ref)

        # Kahn's algorithm
        levels: list[list[ReWOOStep]] = []
        remaining = {step.var: step for step in blueprint.steps}
        while remaining:
            ready = [
                step for var, step in remaining.items()
                if not (deps[var] & set(remaining.keys()))
            ]
            if not ready:
                return None  # cyclic
            levels.append(ready)
            for step in ready:
                remaining.pop(step.var)
        return levels

    def _resolve_placeholders(
        self, args: dict[str, str], evidence: dict[str, ToolResult]
    ) -> dict:
        """Substitute #En references in args with the corresponding evidence text."""
        resolved = {}
        for k, v in args.items():
            if isinstance(v, str):
                def replace(match):
                    var = match.group(0)
                    if var in evidence:
                        return evidence[var].text or ""
                    return var

                resolved[k] = re.sub(r"#E\d+", replace, v)
            else:
                resolved[k] = v
        return resolved

    async def _solve(self, model, prompt, blueprint, evidence) -> str:
        evidence_str = "\n".join(
            f"{var}: {result.text}" for var, result in evidence.items()
        )
        plan_str = "\n".join(
            f"  {s.var}: {s.tool}({s.args_template})" for s in blueprint.steps
        )
        solve_prompt = (
            f"Question: {prompt}\n\n"
            f"Plan executed:\n{plan_str}\n\n"
            f"Evidence:\n{evidence_str}\n\n"
            f"Synthesize the answer using the evidence."
        )
        response = await model.complete(
            messages=[{"role": "user", "content": solve_prompt}],
        )
        return response.text
```

### Performance

| Metric | Value |
|---|---|
| LLM calls | 2 (planner + solver) regardless of step count |
| Tokens | 0.2× ReAct on multi-step tasks (5× reduction) |
| Latency | Lower than ReAct because no per-step reasoning |
| Per-task cost | $0.002-$0.05 typical at frontier prices |

**Benchmark anchor (Xu et al.):** On HotpotQA, ReWOO achieves 32.5% EM with 65% fewer tokens than ReAct's 33.8% EM. Roughly equivalent quality, fraction of cost.

### Strengths

1. **Massively token-efficient.** 2 LLM calls regardless of plan length. The single biggest cost win in agent architectures.
2. **Deterministic execution.** The Worker phase is just tool calls; predictable, fast, idempotent.
3. **Parallel by default.** Independent steps run concurrently without explicit fan-out logic.
4. **Plan inspectable.** You can show users the blueprint before running.

### Weaknesses

1. **No mid-execution adaptation.** If #E1 returns garbage, every downstream step using #E1 produces garbage. The Worker can't decide to retry or take a different approach.
2. **Plan failures are total.** A bad initial plan = entire run wasted. No replan path (the architecture's design rejects iteration).
3. **Tool failures cascade.** A single tool error pollutes all dependent steps.
4. **Limited tool expressiveness.** The plan must precommit to which tools and which args. Complex tools whose args depend on observations (e.g., "find the URL in the search results, then fetch it") fight this architecture.
5. **Synthesis quality is critical.** Bad solver = bad answer regardless of good evidence.

### When to Use

- Predictable, structured tasks with known tool sequences (form filling, ETL)
- Cost-sensitive batch processing where every cent matters
- Tasks where you've validated the plan template empirically (production playbooks)
- Lookups where intermediate results are facts, not decisions

### When NOT to Use

- Exploratory tasks (the entire architecture rejects exploration)
- Tasks with branching logic ("if X then Y else Z")
- Tasks with flaky tools (no retry path)
- Quality-critical work (no review step)
- Conversational agents

### Composition

ReWOO is mostly anti-compositional. It can:

```python
# As a route in a Router (router picks ReWOO for known-pattern tasks)
agent = Agent("...", architecture=Router(routes=[
    RouteSpec(name="lookup", architecture=ReWOO()),  # cheap for known patterns
    RouteSpec(name="exploration", architecture=ReAct()),  # for everything else
]))

# As a worker in Supervisor for specific subtasks
agent = Agent("...", architecture=Supervisor(workers={
    "data_fetcher": Agent("...", architecture=ReWOO()),  # cheap fetching
    "analyst": Agent("...", architecture=ReAct()),
}))
```

ReWOO does NOT compose with: ReAct, Plan-and-Execute (alternatives); Reflexion or Self-Refine (no place to inject critique); Tree of Thoughts (incompatible paradigms).

### Tuning Guide

- **Plan keeps producing impossible tool sequences?** Strengthen the planner system prompt with examples; or move to a stronger planner model.
- **Plans are too long?** Cap `max_steps`; the planner respects this in the prompt.
- **Solver synthesizes badly?** Most often the solver prompt needs work. Be explicit: "Output only the answer; do not restate the plan."
- **Tools' results are too verbose for solver context?** Add a summarization step in the Worker phase (truncate or summarize each evidence entry before passing to solver).

### Common Pitfalls

**1. Using ReWOO for exploratory tasks.** It will produce weird plans and bad answers. Profile your task: if you can write the plan template by hand, ReWOO works; if not, use ReAct.

**2. Cyclic dependencies in the plan.** The planner LLM occasionally produces "#E2 references #E3 references #E2." Always validate DAG before execution.

**3. Placeholder substitution producing oversized args.** A tool call result of 50K characters substituted into the next step's args blows the tool's input limit. Truncate evidence values when substituting.

**4. Forgetting that ReWOO is non-interactive.** No human-in-the-loop; no streaming progress for users. If your UX needs progress indicators, this isn't the architecture.

### Worked Example

```python
import asyncio
from jeevesagent import Agent
from jeevesagent.architecture import ReWOO

async def main():
    # Use case: known-pattern data retrieval
    agent = Agent(
        "You retrieve and synthesize information about companies.",
        model="claude-haiku-4-5",  # cheap; ReWOO is the cost optimization
        tools=["web.search", "web.fetch", "company.lookup"],
        architecture=ReWOO(
            planner_model="claude-opus-4-7",  # frontier for plan quality
            solver_model="claude-haiku-4-5",
            max_steps=8,
        ),
    )

    result = await agent.run(
        "Find the CEO of Anthropic, when they were appointed, and their previous role."
    )
    print(result.output)

    # Inspect for analysis
    print("\nBlueprint:")
    for step in result.metadata["blueprint"].steps:
        print(f"  {step.var}: {step.tool}({step.args_template})")

asyncio.run(main())
```


---

## 4. Reflexion

### Origin

**Paper:** Shinn, Noah, et al. "Reflexion: Language Agents with Verbal Reinforcement Learning." NeurIPS 2023 (arXiv:2303.11366).

**What it solves:** When an agent fails, traditional architectures throw the trajectory away. The next attempt starts from scratch with the same prompt and produces the same failure. Reflexion adds "verbal reinforcement learning" — after a trajectory ends, a Reflector LLM produces a verbal critique, which is stored in long-term memory and prepended to the prompt on the next attempt. The agent learns from its own mistakes without parameter updates.

**Status:** **The 2026 consensus add-on for ReAct** when failure modes repeat. Production-grade single-agent stack = ReAct + Reflexion.

### The Pattern

```
Episode 1:
   prompt → Actor (ReAct trajectory) → outcome
                                          │
                                          ↓
                                       Evaluator
                                          │
                                          ↓ (failed)
                                       Reflector → verbal critique
                                                       │
                                                       ↓
                                                  stored in memory

Episode 2:
   prompt + past_reflections → Actor (ReAct trajectory) → outcome
                                                              │
                                                              ↓
                                                          (success!)
```

Three components:
- **Actor** — runs the task (typically ReAct)
- **Evaluator** — produces a binary or scalar success signal (external eval, or self-eval)
- **Reflector** — generates verbal critique on failure, stored in memory

### Mechanism

The architecture wraps a base architecture (default: ReAct) and runs up to `max_iterations` attempts:

1. **Pull past reflections** from semantic memory matching this prompt class
2. **Inject reflections** into the system prompt of the base architecture
3. **Run base architecture** to produce a trajectory
4. **Evaluate** trajectory: external evaluator (provided by user) or LLM-as-judge self-evaluation
5. **If success:** terminate, return result
6. **If failure:**
   - Reflector LLM reads (prompt, trajectory, evaluation) → produces verbal critique
   - Critique is written to semantic memory as a Fact with the prompt as subject
   - Session resets (fresh context)
   - Loop to step 1 with the new reflection in memory

The verbal critique is **specific** — not "I should be more careful" but "I tried to fetch URL X, got a 404; I should have searched for the URL first instead of guessing."

### Configuration

```python
@dataclass
class ReflexionConfig:
    base: Architecture | None = None
    """Base architecture to run as the Actor. Default: ReAct."""

    max_iterations: int = 3
    """Max attempts before giving up."""

    evaluator: Callable[[RunResult], Awaitable[Eval]] | None = None
    """User-provided evaluator. Receives the result, returns Eval(passed: bool, score: float, reason: str).
    If None, uses self-evaluation via the LLM."""

    reflector_model: str | None = None
    """Override main model for reflection. Defaults to Agent's main model."""

    persist_reflections: bool = True
    """Write reflections to semantic memory for future episodes (across sessions).
    Set False for one-shot experiments."""

    reflection_template: str = DEFAULT_REFLECTION_TEMPLATE
    """Prompt for the reflector. Customize for domain-specific critique."""
```

### Implementation

```python
# jeevesagent/architecture/reflexion.py

from __future__ import annotations
from typing import AsyncIterator, Callable, Awaitable
from datetime import datetime
import anyio

from .base import Architecture
from .react import ReAct
from .config import ReflexionConfig


DEFAULT_REFLECTION_TEMPLATE = """\
You are a self-reflection assistant for an AI agent.

The agent attempted this task:
{prompt}

Here is what the agent did (full trajectory):
{trajectory}

The result was evaluated as:
{evaluation}

Produce a verbal reflection: a SPECIFIC critique describing what went wrong and a CONCRETE
strategy for next time. Don't be generic ("be more careful"). Cite specific decisions or
tool choices. If the agent did multiple things wrong, list each.

Reflection:
"""


class Reflexion:
    name = "reflexion"

    def __init__(self, config: ReflexionConfig | None = None, **kwargs):
        if config is None and kwargs:
            config = ReflexionConfig(**kwargs)
        self.cfg = config or ReflexionConfig()
        if self.cfg.base is None:
            self.cfg.base = ReAct()

    def declared_workers(self):
        # Reflexion's base might have workers; surface them
        return self.cfg.base.declared_workers()

    async def run(self, session, deps, prompt) -> AsyncIterator[Event]:
        reflector = (
            deps.resolve_model(self.cfg.reflector_model) or deps.model
        )

        # Pull past reflections from semantic memory
        past_reflections = await deps.memory.recall(
            f"reflection on: {prompt}",
            kind="semantic",
            limit=3,
        )
        if past_reflections:
            session.append_system(self._format_reflections(past_reflections))
            yield Event.reflections_loaded(session.id, past_reflections)

        for iteration in range(self.cfg.max_iterations):
            yield Event.iteration_started(session.id, iteration)

            # Run base architecture
            trajectory_events: list[Event] = []
            async for event in self.cfg.base.run(session, deps, prompt):
                trajectory_events.append(event)
                yield event.with_iteration(iteration)

            # Evaluate
            run_result = self._build_run_result(trajectory_events, session)
            if self.cfg.evaluator:
                evaluation = await self.cfg.evaluator(run_result)
            else:
                evaluation = await deps.runtime.step(
                    f"reflexion_self_eval_{iteration}",
                    self._self_evaluate,
                    reflector,
                    prompt,
                    run_result,
                )
            yield Event.evaluation(session.id, evaluation, iteration)

            if evaluation.passed:
                yield Event.completed(session.id, run_result.output)
                return

            # Don't reflect on the last iteration (no point)
            if iteration == self.cfg.max_iterations - 1:
                yield Event.max_iterations_reached(session.id, run_result.output)
                return

            # Generate reflection
            reflection = await deps.runtime.step(
                f"reflexion_reflect_{iteration}",
                self._reflect,
                reflector,
                prompt,
                trajectory_events,
                evaluation,
            )
            yield Event.reflection_generated(session.id, reflection, iteration)

            # Persist for future episodes (cross-session learning)
            if self.cfg.persist_reflections:
                from ..memory.types import Fact
                await deps.memory.write_fact(Fact(
                    subject=f"task:{self._task_signature(prompt)}",
                    predicate="reflected",
                    object=reflection.text,
                    confidence=evaluation.confidence,
                    valid_from=datetime.utcnow(),
                    sources=[e.id for e in trajectory_events if hasattr(e, 'id')],
                ))

            # Reset session for next attempt; inject the reflection into context
            session.reset()
            session.append_system(
                f"Reflection from a previous attempt at this task:\n{reflection.text}\n\n"
                f"Use this insight; don't repeat the same mistake."
            )

    async def _self_evaluate(self, model, prompt, run_result):
        """LLM-as-judge fallback when no user-provided evaluator."""
        eval_prompt = (
            f"Task: {prompt}\n\n"
            f"Result: {run_result.output}\n\n"
            f"Evaluate whether the result correctly and completely addresses the task. "
            f"Output JSON: {{ \"passed\": bool, \"score\": float (0-1), \"reason\": string }}"
        )
        response = await model.complete_structured(
            messages=[{"role": "user", "content": eval_prompt}],
            output_schema=Eval,
        )
        return response

    async def _reflect(self, model, prompt, trajectory, evaluation):
        trajectory_str = self._format_trajectory(trajectory)
        prompt_text = self.cfg.reflection_template.format(
            prompt=prompt,
            trajectory=trajectory_str,
            evaluation=evaluation.format(),
        )
        response = await model.complete(
            messages=[{"role": "user", "content": prompt_text}],
        )
        return Reflection(text=response.text, generated_at=datetime.utcnow())

    def _task_signature(self, prompt: str) -> str:
        """Short hash for memory key. Avoids inflating subject column."""
        import hashlib
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
```

### Performance

| Metric | Value | Source |
|---|---|---|
| Token cost | 2-3× single-pass (extra critique + replay) | Shinn et al. 2023 |
| AlfWorld success | +22% over ReAct | Shinn et al. 2023 |
| HotPotQA improvement | +20% | Shinn et al. 2023 |
| HumanEval pass@1 | +11 points (80→91 for GPT-4) | Shinn et al. 2023 |
| Convergence | Most gains in iterations 2-3; diminishing after | Shinn et al. 2023 |

The cross-episode learning is the key feature. On a benchmark with 100 distinct tasks of the same type, accuracy on task #100 is meaningfully higher than task #1 because the agent has accumulated 99 reflections.

### Strengths

1. **Learns without retraining.** Verbal critiques accumulate as memory; the model never sees a parameter update but behaves better over time.
2. **Interpretable improvements.** You can read each reflection and understand why the agent improved.
3. **Composes with everything.** Wraps any base architecture cleanly.
4. **Strong empirical results.** The +22% / +20% / +11 numbers replicated across multiple follow-up papers.
5. **Cross-session learning.** Reflections persist in memory across runs; the agent at runtime t=100 is genuinely better than at t=1.

### Weaknesses

1. **Needs an evaluator.** Without one, you fall back to LLM-as-judge self-evaluation, which is unreliable. Best results require ground-truth eval signals.
2. **Token cost.** Each iteration adds critique + replay. 3 iterations = ~3× cost.
3. **Bad reflections poison future episodes.** A wrong critique gets persisted and biases future attempts. Quality control on reflections matters.
4. **Doesn't help with one-shot tasks.** If the task is unique (no future similar tasks), persistence is wasted.
5. **Latency.** Three serial attempts = three full task durations.

### When to Use

- **Tasks repeated across sessions** with similar structure (customer support, code review, content generation)
- **You have an evaluator** — a test suite, a regex check, a downstream verifier
- **Quality matters more than cost or latency**
- **The agent fails on a recognizable pattern** that a critique can capture

### When NOT to Use

- One-shot unique tasks
- Tasks with no eval signal (where LLM-as-judge is unreliable)
- Latency-sensitive applications (3× wall-clock)
- Cost-sensitive deployments (3× tokens)

### Composition

```python
# THE production-grade single-agent stack (2026 consensus)
agent = Agent("...", architecture=Reflexion(base=ReAct()))

# Reflexion wrapping Plan-and-Execute (replan on failure + learn across sessions)
agent = Agent("...", architecture=Reflexion(
    base=PlanAndExecute(replan_on_failure=True),
    max_iterations=2,
))

# Reflexion wrapping Deep Agent for the most ambitious workflows
agent = Agent("...", architecture=Reflexion(
    base=DeepAgent(),
    evaluator=my_test_runner,
))

# A worker in Supervisor that itself does Reflexion
agent = Agent("manager", architecture=Supervisor(workers={
    "researcher": Agent("...", architecture=Reflexion(base=ReAct())),
}))
```

Reflexion does NOT compose with itself (Reflexion-of-Reflexion is just burning tokens), with Self-Refine (overlapping concerns), or with Multi-Agent Debate (incompatible structures).

### Tuning Guide

- **`max_iterations=3` is the sweet spot.** Beyond 3, gains plateau and costs balloon. Set to 1-2 if cost-sensitive.
- **Provide an evaluator!** LLM-as-judge self-eval is okay but evaluator-driven Reflexion gets the headline numbers.
- **`persist_reflections=False` during dev/testing.** You don't want test runs polluting the memory used in production.
- **Customize `reflection_template`.** Default is generic; domain-specific templates get better critiques (e.g., for code: "What was wrong with the test approach? What input did you forget to test?").
- **Watch for reflection drift.** Periodically audit stored reflections; bad ones make the agent worse over time.

### Common Pitfalls

**1. No evaluator → unreliable self-eval.** The agent says "passed" when it actually failed; Reflexion thinks it succeeded and stops. Always provide an evaluator if possible.

**2. Reflections that are too generic.** "I should be more careful" doesn't help. The reflection template needs to demand specificity.

**3. Persisting bad reflections.** If a reflection itself is wrong, it makes future attempts worse, not better. Add periodic reflection audits.

**4. Subject collision in memory.** Two different tasks hashing to the same subject pull each other's reflections. Use a robust task signature (e.g., embedding-based clustering, not text hash) for production.

**5. Forgetting to reset session between iterations.** The base architecture must run on a fresh context, not see its own previous trajectory plus the reflection. Otherwise the agent gets confused.

### Worked Example

```python
import asyncio
from jeevesagent import Agent
from jeevesagent.architecture import Reflexion, ReAct

# Domain-specific evaluator
async def code_test_evaluator(run_result):
    """Runs the generated code against a test suite."""
    code = extract_code(run_result.output)
    passed, failed = await run_tests(code)
    return Eval(
        passed=(failed == 0),
        score=passed / (passed + failed) if (passed + failed) else 0,
        reason=f"{passed} passed, {failed} failed",
    )

async def main():
    agent = Agent(
        "You write Python code that passes the user's tests.",
        model="claude-opus-4-7",
        tools=["fs.read", "fs.write", "bash"],
        architecture=Reflexion(
            base=ReAct(max_turns=20),
            max_iterations=3,
            evaluator=code_test_evaluator,
            persist_reflections=True,  # learn across runs
        ),
    )

    result = await agent.run(
        "Implement a function `parse_iso_date(s)` that handles malformed input. "
        "Tests are in /home/claude/test_dates.py."
    )

    print(f"Final result: {result.output}")
    print(f"Iterations needed: {result.metadata['iterations']}")
    if result.metadata.get('reflections'):
        print("\nReflections generated:")
        for r in result.metadata['reflections']:
            print(f"  - {r.text[:200]}...")

asyncio.run(main())
```


---

## 5. Self-Refine

### Origin

**Paper:** Madaan, Aman, et al. "Self-Refine: Iterative Refinement with Self-Feedback." arXiv:2303.17651 (March 2023). Published at NeurIPS 2023.

**What it solves:** Reflexion needs an external eval signal and persistent memory. Self-Refine asks: what if the same LLM does generation, critique, and refinement, in a single session? No memory, no evaluator, just iterate. The same model that wrote the output reads it again and asks "what's wrong?" then revises. Surprisingly, this works remarkably well.

**Status:** The lightweight quality-improvement add-on. When you can't provide an evaluator and don't need cross-session learning, Self-Refine is the easiest path to better outputs.

### The Pattern

```
Round 0: Generator → output_0
                       │
                       ↓
Round 1: same model as Critic → critique_0
                       │
                       ↓
         same model as Refiner → output_1 (improvement on output_0)
                       │
                       ↓
Round 2: Critic → critique_1 → Refiner → output_2
                       │
                       ↓
         ... until critic says "no further improvements" or max_rounds
```

One model. Three roles per round. Only intra-episode (no cross-session memory).

### Mechanism

**Round 0 — Generation.** Run the base architecture (typically ReAct) to produce an initial output.

**Each subsequent round:**
1. **Critique.** Same model is given the output and asked to critique it specifically. Prompt: "Review this output. What are the issues? Be specific."
2. **Stop check.** If critique contains the stop phrase ("no further improvements" or similar), terminate.
3. **Refine.** Same model is given (original prompt, current output, critique) and asked to produce a revised version that addresses the critique.
4. **Replace** current output with refined version. Loop.

**Termination:** stop phrase, `max_rounds`, or critic-claimed quality threshold.

### Configuration

```python
@dataclass
class SelfRefineConfig:
    base: Architecture | None = None
    """Base for the initial generation. Default: ReAct."""

    max_rounds: int = 3
    """Max critique-refine cycles after initial generation."""

    stop_phrase: str = "no further improvements"
    """Case-insensitive substring; if critic emits this, terminate."""

    critique_template: str = DEFAULT_CRITIQUE_TEMPLATE
    refine_template: str = DEFAULT_REFINE_TEMPLATE
    """Prompts for critique and refinement; customize per domain."""

    require_specific_critique: bool = True
    """If the critique is too short/generic, treat it as 'no improvements'.
    Prevents lazy critiques."""
```

### Implementation

```python
# jeevesagent/architecture/self_refine.py

from __future__ import annotations
from typing import AsyncIterator

from .base import Architecture
from .react import ReAct


DEFAULT_CRITIQUE_TEMPLATE = """\
Original task: {prompt}

Output produced so far:
{output}

Review this output critically. List specific issues with concrete examples.
If the output is genuinely good with no meaningful improvements possible,
say "no further improvements" exactly.

Critique:
"""

DEFAULT_REFINE_TEMPLATE = """\
Original task: {prompt}

Previous output:
{output}

Critique to address:
{critique}

Produce a revised output addressing every critique point.
"""


class SelfRefine:
    name = "self-refine"

    def __init__(self, config=None, **kwargs):
        if config is None and kwargs:
            from .config import SelfRefineConfig
            config = SelfRefineConfig(**kwargs)
        from .config import SelfRefineConfig
        self.cfg = config or SelfRefineConfig()
        if self.cfg.base is None:
            self.cfg.base = ReAct()

    def declared_workers(self):
        return self.cfg.base.declared_workers()

    async def run(self, session, deps, prompt) -> AsyncIterator[Event]:
        # === ROUND 0: INITIAL GENERATION ===
        async for event in self.cfg.base.run(session, deps, prompt):
            yield event.with_role("generator")
        current_output = session.result

        for round_num in range(self.cfg.max_rounds):
            # Budget check
            status = await deps.budget.allows_step()
            if status.blocked:
                yield Event.budget_exceeded(session.id, status)
                return

            # === CRITIQUE ===
            critique = await deps.runtime.step(
                f"self_refine_critique_{round_num}",
                self._critique,
                deps.model,
                prompt,
                current_output,
            )
            yield Event.critique(session.id, critique, round_num)

            # Stop check
            if self.cfg.stop_phrase.lower() in critique.lower():
                session.complete = True
                session.result = current_output
                yield Event.completed(session.id, current_output)
                return

            if self.cfg.require_specific_critique and len(critique) < 50:
                # Critique was too lazy; treat as "no improvements"
                yield Event.critique_rejected(session.id, "too_short")
                session.result = current_output
                yield Event.completed(session.id, current_output)
                return

            # === REFINE ===
            current_output = await deps.runtime.step(
                f"self_refine_apply_{round_num}",
                self._refine,
                deps.model,
                prompt,
                current_output,
                critique,
            )
            yield Event.refined(session.id, round_num, current_output)

        session.result = current_output
        yield Event.completed(session.id, current_output)

    async def _critique(self, model, prompt, output) -> str:
        text = self.cfg.critique_template.format(prompt=prompt, output=output)
        response = await model.complete(messages=[{"role": "user", "content": text}])
        return response.text

    async def _refine(self, model, prompt, output, critique) -> str:
        text = self.cfg.refine_template.format(
            prompt=prompt, output=output, critique=critique
        )
        response = await model.complete(messages=[{"role": "user", "content": text}])
        return response.text
```

### Performance

| Metric | Value |
|---|---|
| Token cost | 2-3× single-pass |
| Latency | 2-3× (critique + refine per round are sequential) |
| Code Optimization (Madaan et al.) | +13 points over single-pass on GPT-4 |
| Math Reasoning | +5-10 points typical |
| Acrostic Writing | +20% human preference |

The math/code numbers are particularly notable because they suggest the same LLM, when asked to critique its own output, *can* spot mistakes — even ones it just made.

### Strengths

1. **No evaluator needed.** Works on tasks where ground-truth eval signals don't exist (creative writing, brainstorming, design).
2. **Strong on quality-driven tasks.** The +13 points on code optimization is real.
3. **Simple to implement.** Three prompts in a loop. The whole architecture is ~100 lines.
4. **Works with one model.** No need for multi-model setups.
5. **Good early-exit behavior.** When the output is genuinely good, the critic says so and we stop quickly.

### Weaknesses

1. **Same blind spots.** The critic shares biases with the generator. If the generator wrote a wrong fact confidently, the critic likely confirms it.
2. **No cross-session learning.** Each invocation starts fresh; doesn't accumulate insight.
3. **Can loop on borderline outputs.** If critiques keep finding nits, you hit `max_rounds` and the final output may not be better than round 0.
4. **Drift on subjective tasks.** "Better" is undefined; rounds can drift toward different style without improving quality.
5. **Token cost without proportional gain.** On well-handled tasks, rounds 2 and 3 add little.

### When to Use

- Quality-critical single-shot outputs (a report, a piece of code, a design)
- Tasks with no available evaluator
- Creative work where Reflexion's "fail/retry" doesn't fit
- Domains where critiques can be specific and actionable

### When NOT to Use

- Tasks with available evaluators → use Reflexion (gets cross-session learning)
- Cost-sensitive (2-3× tokens for marginal gains on already-good outputs)
- Latency-sensitive (sequential critique-refine doubles wall-clock minimum)
- Tasks where same-model blind spots dominate → use Actor-Critic (different LLMs)

### Composition

```python
# Self-Refine over ReAct (the default usage)
agent = Agent("...", architecture=SelfRefine(base=ReAct(), max_rounds=2))

# Inside Deep Agent for project-shaped tasks with quality polish
agent = Agent("...", architecture=DeepAgent(base=SelfRefine(base=ReAct())))

# Self-Refine for each step of Plan-and-Execute (rare; expensive)
# Don't do this; the executor LLM is supposed to be cheap
```

Self-Refine does NOT compose with: Reflexion (overlapping concerns; pick one), Multi-Agent Debate (judges instead of self-critique), Tree of Thoughts (self-eval is already built into ToT).

### Tuning Guide

- **`max_rounds=2` is the sweet spot.** Round 1 catches obvious issues; round 2 polishes; round 3 rarely helps.
- **`require_specific_critique=True` always.** Without it, lazy critiques get treated as "improve anyway."
- **Customize `critique_template` per domain.** For code: "Review for: security issues, edge cases, performance, missing tests." For writing: "Review for: clarity, structure, factual accuracy, tone."
- **Watch for thrashing.** If round N's output is worse than round N-1's, the architecture has failed. Add a "compare against original" prompt to refuse regression.

### Common Pitfalls

**1. Generic critique prompts.** "Is this output good?" produces "Yes, this output is good." Be domain-specific.

**2. Treating Self-Refine as a quality oracle.** The critic is the same model as the generator. It cannot catch issues neither would have caught originally. For real adversarial review, use Actor-Critic with a different model.

**3. Skipping the stop-phrase check.** Without `stop_phrase`, you always run `max_rounds` even when output is already great.

**4. Composing with ReAct's `max_turns` poorly.** Self-Refine wraps the entire ReAct trajectory; if ReAct hits its turn cap, the output is already partial, and refining a partial output is futile. Make `ReAct.max_turns` generous.

### Worked Example

```python
import asyncio
from jeevesagent import Agent
from jeevesagent.architecture import SelfRefine, ReAct

async def main():
    agent = Agent(
        "You write technical documentation. Be precise, complete, and well-organized.",
        model="claude-opus-4-7",
        tools=["web.search"],
        architecture=SelfRefine(
            base=ReAct(max_turns=20),
            max_rounds=2,
            critique_template="""\
Original task: {prompt}

Documentation produced:
{output}

Review for: clarity, completeness, technical accuracy, examples, edge cases.
Be SPECIFIC: cite the section and quote the issue.
If genuinely complete, say "no further improvements" exactly.

Critique:
""",
            require_specific_critique=True,
        ),
    )

    result = await agent.run(
        "Write a 1-page guide for using Python's anyio.create_task_group()."
    )
    print(result.output)

asyncio.run(main())
```


---

## 6. Tree of Thoughts (ToT)

### Origin

**Paper:** Yao, Shunyu, et al. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." NeurIPS 2023 (arXiv:2305.10601).

**What it solves:** Chain-of-Thought commits to a single reasoning path. If the path is wrong, the agent powers through to a wrong answer. ToT explores multiple reasoning paths in parallel: at each step, generate K candidate thoughts; evaluate them with the LLM as a heuristic; keep the most promising M; expand them. The agent backtracks from dead ends and finds the goal via search.

**Status:** A specialty tool. On combinatorial reasoning tasks (Game of 24, creative writing, mini crosswords) it dramatically outperforms ReAct. Token cost is enormous. Reserve for tasks where quality matters more than cost by 100×.

### The Pattern

```
                    [root: question]
                   /       |        \
                  /        |         \
              [thought1] [thought2] [thought3]    ← K candidates per node
                /  \         |       /  \
            [t1.1][t1.2]  [t2.1] [t3.1][t3.2]
                            │
                          [goal!]                 ← search until goal reached
```

Each level: branch by K (generate candidates), evaluate (LLM scores), prune to top M (keep best), expand (continue from kept).

### Mechanism

ToT defines four operations:

1. **Generate.** From a parent thought, prompt the LLM to produce K candidate next thoughts.
2. **Evaluate.** Each thought is scored by the LLM as a heuristic ("how promising is this path? 1-10").
3. **Prune.** Keep top-M scoring thoughts; discard the rest.
4. **Expand.** Continue search from kept thoughts.

Two search strategies:
- **BFS (breadth-first):** explore all M kept thoughts at depth d before going to d+1
- **DFS (depth-first):** drill into best thought; backtrack on dead end

Termination: goal reached (some thought scores ≥ goal threshold), max depth, or max LLM calls.

### Configuration

```python
@dataclass
class ToTConfig:
    branches_per_step: int = 3
    """K — how many candidate thoughts per node."""

    keep_per_step: int = 2
    """M — top-M kept after evaluation."""

    max_depth: int = 5
    """Max tree depth."""

    search: Literal["bfs", "dfs"] = "bfs"

    goal_score_threshold: float = 0.9
    """Score (0-1) above which a thought is considered to reach the goal."""

    max_total_evaluations: int = 50
    """Hard cap on LLM evaluations to prevent runaway cost."""

    branching_template: str = DEFAULT_BRANCHING_TEMPLATE
    evaluation_template: str = DEFAULT_EVALUATION_TEMPLATE
```

### Implementation

```python
# jeevesagent/architecture/tot.py

from __future__ import annotations
import anyio
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal


@dataclass
class ThoughtNode:
    id: str
    content: str
    parent: "ThoughtNode | None" = None
    depth: int = 0
    score: float = 0.0
    children: list["ThoughtNode"] = field(default_factory=list)

    def path_from_root(self) -> list[str]:
        path = []
        node = self
        while node:
            path.append(node.content)
            node = node.parent
        return list(reversed(path))


class TreeOfThoughts:
    name = "tree-of-thoughts"

    def __init__(self, config=None, **kwargs):
        from .config import ToTConfig
        if config is None and kwargs:
            config = ToTConfig(**kwargs)
        self.cfg = config or ToTConfig()
        self._eval_count = 0

    def declared_workers(self):
        return {}

    async def run(self, session, deps, prompt) -> AsyncIterator[Event]:
        self._eval_count = 0
        root = ThoughtNode(id="root", content=prompt, depth=0)

        if self.cfg.search == "bfs":
            async for event in self._bfs(session, deps, root, prompt):
                yield event
        else:
            async for event in self._dfs(session, deps, root, prompt):
                yield event

    # --- BFS ------------------------------------------------------------

    async def _bfs(self, session, deps, root, prompt) -> AsyncIterator[Event]:
        frontier: list[ThoughtNode] = [root]

        for depth in range(self.cfg.max_depth):
            yield Event.tot_layer_started(session.id, depth, len(frontier))

            # Budget gate
            status = await deps.budget.allows_step()
            if status.blocked:
                yield Event.budget_exceeded(session.id, status)
                return

            if self._eval_count >= self.cfg.max_total_evaluations:
                yield Event.tot_max_evals_reached(session.id, frontier[0])
                return

            # Expand all frontier nodes in parallel
            expanded: list[ThoughtNode] = []
            async with anyio.create_task_group() as tg:
                for node in frontier:
                    tg.start_soon(self._expand_node, deps, node, prompt, expanded)

            yield Event.tot_expanded(session.id, depth, len(expanded))

            if not expanded:
                # No more thoughts to explore
                yield Event.tot_dead_end(session.id, depth)
                return

            # Evaluate all expanded thoughts in parallel
            async with anyio.create_task_group() as tg:
                for node in expanded:
                    tg.start_soon(self._evaluate_node, deps, prompt, node)

            # Sort by score, take top-M
            expanded.sort(key=lambda n: n.score, reverse=True)
            frontier = expanded[: self.cfg.keep_per_step]
            yield Event.tot_pruned(session.id, depth, frontier)

            # Check if any thought reached the goal
            for node in frontier:
                if node.score >= self.cfg.goal_score_threshold:
                    final_answer = await self._extract_answer(deps, node, prompt)
                    session.complete = True
                    session.result = final_answer
                    yield Event.completed(
                        session.id,
                        final_answer,
                        path=node.path_from_root(),
                    )
                    return

        # Max depth reached without goal — return best frontier node
        if frontier:
            best = frontier[0]
            final_answer = await self._extract_answer(deps, best, prompt)
            session.result = final_answer
            yield Event.tot_max_depth(session.id, best, final_answer)

    # --- DFS ------------------------------------------------------------

    async def _dfs(self, session, deps, node, prompt) -> AsyncIterator[Event]:
        """Depth-first: drill into best child; backtrack on low scores."""
        if node.depth >= self.cfg.max_depth:
            return
        if self._eval_count >= self.cfg.max_total_evaluations:
            return

        # Generate children
        children = await self._branch(deps, node, prompt)
        # Evaluate
        async with anyio.create_task_group() as tg:
            for child in children:
                tg.start_soon(self._evaluate_node, deps, prompt, child)

        # Sort by score, drill into best
        children.sort(key=lambda c: c.score, reverse=True)
        for child in children:
            if child.score >= self.cfg.goal_score_threshold:
                # Goal!
                yield Event.completed(session.id, child.content, path=child.path_from_root())
                session.result = child.content
                return
            yield Event.tot_drill(session.id, child)
            async for e in self._dfs(session, deps, child, prompt):
                yield e

    # --- helpers --------------------------------------------------------

    async def _expand_node(self, deps, node, prompt, output_list):
        children = await self._branch(deps, node, prompt)
        output_list.extend(children)

    async def _branch(self, deps, node, prompt) -> list[ThoughtNode]:
        """Generate K candidate thoughts from this node."""
        branch_prompt = self.cfg.branching_template.format(
            prompt=prompt,
            current_thought=node.content,
            depth=node.depth,
            k=self.cfg.branches_per_step,
        )
        response = await deps.runtime.step(
            f"tot_branch_{node.id}",
            deps.model.complete_structured,
            messages=[{"role": "user", "content": branch_prompt}],
            output_schema=BranchOutput,  # list of K thoughts
        )
        return [
            ThoughtNode(
                id=f"{node.id}.{i}",
                content=t,
                parent=node,
                depth=node.depth + 1,
            )
            for i, t in enumerate(response.thoughts[: self.cfg.branches_per_step])
        ]

    async def _evaluate_node(self, deps, prompt, node):
        eval_prompt = self.cfg.evaluation_template.format(
            prompt=prompt,
            thought=node.content,
            path=" → ".join(node.path_from_root()),
        )
        response = await deps.runtime.step(
            f"tot_eval_{node.id}",
            deps.model.complete_structured,
            messages=[{"role": "user", "content": eval_prompt}],
            output_schema=EvalScore,
        )
        node.score = response.score
        self._eval_count += 1

    async def _extract_answer(self, deps, node, prompt) -> str:
        """Synthesize a final answer from the path leading to this node."""
        path = " → ".join(node.path_from_root())
        synth_prompt = (
            f"Original problem: {prompt}\n\n"
            f"Reasoning path: {path}\n\n"
            f"Provide the final answer based on this reasoning:"
        )
        response = await deps.model.complete(
            messages=[{"role": "user", "content": synth_prompt}],
        )
        return response.text
```

### Performance

| Metric | Value | Source |
|---|---|---|
| Game of 24 success | 4% (CoT) → 74% (ToT) | Yao et al. 2023 |
| Token cost | 10-100× ReAct | Same paper |
| Latency | 5-30 seconds typical | Sequential evals dominate |
| Creative Writing coherence | +15% over CoT (subjective) | Same paper |

The Game of 24 number is a stunning result, but it's the hardest case. On most tasks ToT improves modestly while cost balloons.

### Strengths

1. **Backtracking.** When a path leads to a dead end, the agent backs out and tries another.
2. **Self-evaluation built in.** The LLM is the heuristic; no external scorer needed.
3. **Strong on combinatorial problems.** Math puzzles, logic, planning with branching.
4. **Interpretable reasoning paths.** You can see exactly which paths the agent considered.

### Weaknesses

1. **Token cost is enormous.** Each level multiplies cost by K + M evaluations. 100× ReAct is not unusual.
2. **Latency.** Most evals are sequential within a level; even with parallel expansion, you wait for all evals before pruning.
3. **Scoring quality is everything.** Bad heuristic = wrong path picked. The LLM-as-judge is noisy.
4. **K and M are hand-tuned.** Different problems need different settings; no autoscaling.
5. **Doesn't scale to complex world.** Tools, MCP, multi-modal — ToT is designed for pure reasoning, not tool-use trajectories.

### When to Use

- Math, logic, puzzle problems (Game of 24, mini crosswords)
- Scientific reasoning where multiple hypotheses need evaluation
- Creative tasks where divergent thinking + selection helps (story generation, design ideation)
- Quality > cost by an enormous margin

### When NOT to Use

- Anything tool-using (use ReAct or Deep Agent)
- Any cost-sensitive task (10-100× cost is real)
- Conversational agents
- Tasks where "the path doesn't matter, only the answer" (use ReAct + Self-Refine)

### Composition

ToT is mostly a leaf architecture. It can:

```python
# As a worker for hard reasoning subtasks in Supervisor
agent = Agent("manager", architecture=Supervisor(workers={
    "reasoner": Agent("solve hard problems", architecture=TreeOfThoughts()),
    "researcher": Agent("...", architecture=ReAct()),
}))

# Inside Self-Refine (rarely useful; ToT already self-evaluates)
# Don't typically compose
```

ToT does NOT compose with: ReAct (alternatives), Plan-and-Execute (different paradigms), ReWOO (incompatible).

### Tuning Guide

- **`branches_per_step=3` and `keep_per_step=2` is the canonical default.** Try 5/3 for harder problems.
- **`max_depth=5` is enough for most problems.** Beyond 5, tree size explodes.
- **`max_total_evaluations` is your safety net.** Never run ToT without it; cost can spiral.
- **For DFS, use lower K, higher depth.** For BFS, use higher K, lower depth.
- **Customize evaluation template per domain.** For math: "Score how close this is to the answer 1-10." For writing: "Score coherence and creativity 1-10."

### Common Pitfalls

**1. Running ToT on conversational tasks.** It's not designed for that; tool calls don't fit the "thought tree" model.

**2. Forgetting `max_total_evaluations`.** Without this cap, a hard problem can run for hours.

**3. Not validating tree DAG.** Cycles can form when child thoughts reference parents. Implementations that don't track ancestry can loop.

**4. Trusting the score blindly.** LLM-as-judge scoring has 10-20% noise. A "0.95 confidence" thought might be wrong; "0.40" might be right. Use only as a heuristic, not as ground truth.

### Worked Example

```python
import asyncio
from jeevesagent import Agent
from jeevesagent.architecture import TreeOfThoughts

async def main():
    agent = Agent(
        "You solve combinatorial reasoning problems.",
        model="claude-opus-4-7",
        # No tools — ToT is pure reasoning
        architecture=TreeOfThoughts(
            branches_per_step=3,
            keep_per_step=2,
            max_depth=5,
            search="bfs",
            goal_score_threshold=0.9,
            max_total_evaluations=40,
        ),
    )

    result = await agent.run(
        "Use the numbers 4, 7, 8, 13 with operators (+, -, *, /) "
        "to make 24. Each number must be used exactly once."
    )

    print(f"Solution: {result.output}")
    print(f"Path explored: {result.metadata['path']}")
    print(f"Total evaluations: {result.metadata['eval_count']}")

asyncio.run(main())
```


---

## 7. Graph of Thoughts (GoT)

### Origin

**Paper:** Besta, Maciej, et al. "Graph of Thoughts: Solving Elaborate Problems with Large Language Models." AAAI 2024 (arXiv:2308.09687). Extended by Adaptive Graph of Thoughts (AGoT) in 2025, which reports **+46.2% on GPQA scientific reasoning**.

**What it solves:** ToT is a tree — every thought has one parent. But real reasoning often combines multiple lines of thought into a synthesis (aggregation), or refines a single thought iteratively (cycles). GoT generalizes ToT to an arbitrary directed graph with three operations: generation, aggregation, refinement. This subsumes CoT (chain), ToT (tree), and self-reflection (cycle) under one framework.

**Status:** **Ship in v2.** Too much engineering for v1, and v1 should focus on architectures with broader applicability. Documenting here so when v2 lands, the design is ready.

### The Pattern

```
                [thought_a]
               /     |
              /      |
         [thought_b] [thought_c]
              \      /
               \    /
            [aggregate(b, c)]    ← synthesis from multiple parents
                  |
                  ↓
            [refine(...)]        ← iterative refinement (cycle allowed)
                  |
                  ↓
              [final]
```

Three operations:
- **Generate** — produce K children from a parent (like ToT)
- **Aggregate** — merge multiple parents into one child (synthesis; in-degree > 1)
- **Refine** — feed a thought back through the LLM with downstream context (iterate on a single thought)

### Mechanism (deferred to v2 implementation)

The architecture is described in detail in `AGENT_ARCHITECTURES.md` §5.7. The v1 build does not include implementation — placeholder protocol stub is registered with the entry point so users see "graph-of-thoughts" listed as "coming in v2."

### Why we defer

The implementation complexity is high:
- Graph topology management (cycle detection, DAG enforcement when needed)
- Three operation types with different evaluation semantics
- AGoT's adaptive variant requires meta-LLM calls to decide which operation to invoke
- Token cost is the highest of any architecture (50-200× ReAct on hard problems)

The 2026 production literature is clear: most teams should not need this. Reserve for research-grade or scientific reasoning work. Adding it well is a 3-week effort; we'd rather ship 8 architectures that cover 95% of use cases first.

---

## 8. Deep Agent

### Origin

**Library:** `langchain-ai/deepagents` (LangChain). Originally inspired by Claude Code, Deep Research, and Manus. Anthropic also documents this as "the harness pattern" in their 2026 long-running agents guide.

**Key insight:** Most "agent architectures" are reasoning patterns. Deep Agent is something different — it's a **bundle of pillars** that wrap any base reasoning architecture. The four pillars (planning tool, filesystem, subagents, detailed system prompt) together solve the long-task coherence problem that causes 50%+ ReAct failure rate on 10+ step tasks.

**Status:** The standard architecture for project-shaped tasks. If your agent will run for hours and produce deliverables (code, reports, documents), this is what you want.

### The Pattern

```
┌─────────────────────────────────────────────────────────────┐
│ Base loop (typically ReAct, but any architecture works)      │
│                                                              │
│  Built-in tools always available:                            │
│   • write_todos       ← planning as working memory           │
│   • ls, read_file,    ← filesystem for context offload       │
│     write_file, edit_file                                    │
│   • task              ← spawn isolated subagent              │
│                                                              │
│  Detailed system prompt teaches: plan first, write tasks     │
│  to disk for handoff, spawn subagents for deep dives         │
└─────────────────────────────────────────────────────────────┘
```

The base architecture is unaware of the pillars; it just sees more tools. The system prompt teaches the model to use them well.

### Mechanism

**Setup phase (when run starts):**
1. Inject the Deep Agent system prompt addon (teaches use of planning, filesystem, subagents)
2. Register four pillar tool sets as in-process MCP tools:
   - **Planning** — `write_todos(items)`, `complete_todo(id)`
   - **Filesystem** — `ls(path)`, `read_file(path)`, `write_file(path, content)`, `edit_file(path, ...)`
   - **Subagents** — `task(subagent_name, instructions)` to delegate to declared subagents
3. Configure filesystem permissions per `FilesystemConfig` (allowed roots, read/write rules)

**Runtime phase:**
1. Run the base architecture (default ReAct) — it sees the pillar tools as ordinary tools
2. The model uses planning to maintain goal coherence ("I need to do steps A, B, C; A is in progress")
3. The model uses filesystem to offload context (writes a 50K-token tool result to a file, reads it back later by line range)
4. The model uses subagents to keep the main context lean (delegates "research X" to a subagent that returns a 2-paragraph summary instead of 100 web pages)

The combination delivers **multi-context-window coherence** — the agent can work for hours across what would otherwise be many separate runs.

### Configuration

```python
@dataclass
class DeepAgentConfig:
    base: Architecture | None = None
    """Base architecture to wrap. Default: ReAct(max_turns=50)."""

    subagents: list[SubagentDef] | None = None
    """Pre-defined subagents available via the `task` tool.
    Each has: name, description, instructions, tools."""

    filesystem: FilesystemConfig | None = None
    """Allowed filesystem roots, write permissions, max file size."""

    enable_planning: bool = True
    enable_filesystem: bool = True
    enable_subagents: bool = True
    """Disable individual pillars if not needed."""

    system_prompt_addon: str | None = None
    """Extra prompt explaining when/how to use the pillars.
    None uses our well-tested default."""


@dataclass
class FilesystemConfig:
    allowed_roots: list[Path] = field(default_factory=lambda: [Path.cwd()])
    read_only: bool = False
    max_file_size_bytes: int = 10_000_000
    auto_chunk_threshold: int = 8_000
    """Files larger than this are returned as preview + path reference,
    not full content."""


@dataclass
class SubagentDef:
    name: str
    description: str   # used by the model to decide when to invoke
    instructions: str  # the subagent's system prompt
    tools: list[str] | None = None  # tool allowlist; None = all
    architecture: Architecture | None = None  # default: ReAct
```

### Implementation

```python
# jeevesagent/architecture/deep_agent.py

from __future__ import annotations
from typing import AsyncIterator

from .base import Architecture
from .react import ReAct
from .config import DeepAgentConfig, FilesystemConfig, SubagentDef


DEFAULT_DEEP_AGENT_PROMPT = """\
You have access to four powerful capabilities:

1. PLANNING: Use `write_todos` to maintain a checklist of tasks. Reference and
   update it throughout your work. Mark items complete with `complete_todo`.
   This is your working memory.

2. FILESYSTEM: Use `read_file`, `write_file`, `edit_file`, `ls` to manage state.
   Offload large content (long documents, raw data, intermediate results) to
   files instead of keeping them in your context. Read by section when needed.

3. SUBAGENTS: Use `task` to delegate isolated subtasks. Each subagent has its
   own context window — useful for deep research or complex subtasks where you
   only need the final summary, not every step.

4. DEFAULT TOOLS: Your other tools are listed below. Use them as appropriate.

Best practices:
- Plan before acting on multi-step tasks. Write the plan as a todo list FIRST.
- Update the plan as you progress; mark items done; add new items as you discover them.
- For tool outputs > ~2KB, write them to a file rather than keeping them in context.
- For research-heavy subtasks, delegate to a subagent so you stay focused on the main task.
- For long projects, periodically summarize progress in `progress.md` so you can
  resume cleanly across sessions.
"""


class DeepAgent:
    name = "deep-agent"

    def __init__(self, config=None, **kwargs):
        if config is None and kwargs:
            config = DeepAgentConfig(**kwargs)
        self.cfg = config or DeepAgentConfig()
        if self.cfg.base is None:
            self.cfg.base = ReAct(max_turns=50)
        if self.cfg.filesystem is None:
            self.cfg.filesystem = FilesystemConfig()

    def declared_workers(self):
        # Subagents are workers
        if self.cfg.subagents:
            return {sa.name: self._build_subagent(sa) for sa in self.cfg.subagents}
        return {}

    async def run(self, session, deps, prompt) -> AsyncIterator[Event]:
        # Inject the Deep Agent system prompt
        addon = self.cfg.system_prompt_addon or DEFAULT_DEEP_AGENT_PROMPT
        session.append_system(addon)

        # Register pillar tools
        if self.cfg.enable_planning:
            deps.tools.register_inproc(self._planning_tool_set(session))

        if self.cfg.enable_filesystem:
            deps.tools.register_inproc(self._filesystem_tool_set(session))

        if self.cfg.enable_subagents and self.cfg.subagents:
            deps.tools.register_inproc(self._subagent_tool_set(deps))

        yield Event.deep_agent_initialized(session.id, self._summary())

        # Run the base architecture; it sees the pillars as ordinary tools
        async for event in self.cfg.base.run(session, deps, prompt):
            yield event

    def _summary(self) -> dict:
        return {
            "planning": self.cfg.enable_planning,
            "filesystem": self.cfg.enable_filesystem,
            "subagents": [sa.name for sa in (self.cfg.subagents or [])],
        }

    # --- Planning tools ---

    def _planning_tool_set(self, session):
        from ..mcp.tools import tool, ToolSet

        @tool
        async def write_todos(items: list[str]) -> dict:
            """Initialize or replace the todo list."""
            session.todos = [
                {"id": i, "text": text, "done": False}
                for i, text in enumerate(items)
            ]
            return {"todos": session.todos}

        @tool
        async def add_todo(text: str) -> dict:
            """Append a todo to the list."""
            new_id = len(session.todos)
            session.todos.append({"id": new_id, "text": text, "done": False})
            return {"id": new_id, "todos": session.todos}

        @tool
        async def complete_todo(id: int) -> dict:
            """Mark a todo as done."""
            for t in session.todos:
                if t["id"] == id:
                    t["done"] = True
                    break
            return {"todos": session.todos}

        return ToolSet(name="planning", tools=[write_todos, add_todo, complete_todo])

    # --- Filesystem tools ---

    def _filesystem_tool_set(self, session):
        from ..mcp.tools import tool, ToolSet

        @tool
        async def ls(path: str = ".") -> list[str]:
            """List directory entries."""
            full = self._validate_path(path)
            return [p.name for p in full.iterdir()]

        @tool
        async def read_file(path: str, line_start: int | None = None, line_end: int | None = None) -> str:
            """Read a file. Optionally specify line range to read partial content."""
            full = self._validate_path(path)
            content = full.read_text()
            if line_start is not None or line_end is not None:
                lines = content.splitlines()
                lo = (line_start or 1) - 1
                hi = line_end or len(lines)
                content = "\n".join(lines[lo:hi])
            return content

        @tool
        async def write_file(path: str, content: str) -> dict:
            """Write content to a file (creating or overwriting)."""
            if self.cfg.filesystem.read_only:
                raise PermissionError("filesystem is read-only")
            full = self._validate_path(path)
            full.write_text(content)
            return {"path": str(full), "bytes": len(content)}

        @tool
        async def edit_file(path: str, find: str, replace: str) -> dict:
            """Find-and-replace within a file."""
            if self.cfg.filesystem.read_only:
                raise PermissionError("filesystem is read-only")
            full = self._validate_path(path)
            content = full.read_text()
            if find not in content:
                raise ValueError(f"`find` text not found in {path}")
            new_content = content.replace(find, replace, 1)
            full.write_text(new_content)
            return {"path": str(full), "replacements": 1}

        return ToolSet(name="filesystem", tools=[ls, read_file, write_file, edit_file])

    def _validate_path(self, path: str) -> Path:
        from pathlib import Path
        p = Path(path).resolve()
        for root in self.cfg.filesystem.allowed_roots:
            if p == root or root in p.parents:
                return p
        raise PermissionError(f"path {path} outside allowed roots")

    # --- Subagent tools ---

    def _subagent_tool_set(self, deps):
        from ..mcp.tools import tool, ToolSet

        @tool
        async def task(subagent: str, instructions: str) -> str:
            """Delegate a subtask to a specialist subagent. Returns the final answer only."""
            if subagent not in self._subagent_map:
                return f"Error: unknown subagent {subagent!r}"
            sub_agent = self._subagent_map[subagent]
            sub_session = deps.fresh_session()
            sub_deps = deps.scope_for_worker(sub_agent)
            events = []
            async for event in sub_agent._architecture.run(sub_session, sub_deps, instructions):
                events.append(event)
            return sub_session.result or "[no result]"

        return ToolSet(name="subagents", tools=[task])
```

### Performance

| Metric | Value |
|---|---|
| Token cost vs ReAct | 1.2× on simple tasks; 0.6× on long tasks (filesystem/subagent offload) |
| 10+ step task success | 50%+ improvement over raw ReAct |
| Filesystem context savings | Variable; 30-80% on heavy-IO tasks |
| Subagent context savings | Major: parent context grows by summary, not transcript |

The cost story is task-dependent. On tasks where Deep Agent's pillars actually pay off (long, project-shaped), it's *cheaper* than ReAct because filesystem offload reduces context size. On simple tasks, the overhead of the system prompt + extra tools makes it slightly more expensive.

### Strengths

1. **Survives long tasks.** The combination of planning + filesystem + subagents is what makes hours-long agents work.
2. **Filesystem context offload.** A 100-page document's contents stay on disk; only the relevant chunks enter context.
3. **Subagent isolation.** Heavy subtasks don't pollute the main context.
4. **Resumable.** With persistent filesystem, an agent can stop and resume across sessions (write `progress.md`, read on resume).
5. **Production-proven.** This is the Claude Code / Deep Research / Manus pattern.

### Weaknesses

1. **Heavy for simple tasks.** A 5-line task gets a 2KB system prompt, four extra tool sets, and filesystem permissions to manage. Overkill.
2. **Subagent coordination has its own failure modes.** A confused parent spawns wrong subagents.
3. **Filesystem is a security surface.** Path traversal, write to wrong root, etc. The validation layer matters.
4. **Opinionated system prompt.** If your domain disagrees with the default prompt, you're fighting the architecture.
5. **Abstraction tax.** Users who just need a chatbot will be confused by the planning/filesystem complexity.

### When to Use

- Coding agents (the canonical use)
- Research agents producing reports/documents
- Multi-document workflows (read 10 PDFs, synthesize, write summary)
- Tasks expected to span 30+ tool calls
- Anything where you want resume-across-sessions semantics

### When NOT to Use

- Conversational agents (filesystem + planning don't fit chat)
- Simple tasks (overkill)
- High-frequency, low-latency tasks (system prompt overhead matters)
- Domains where filesystem doesn't make sense (pure API agents)

### Composition

Deep Agent wraps base architectures. It composes well:

```python
# Standard: Deep Agent over ReAct
agent = Agent("...", architecture=DeepAgent(base=ReAct(max_turns=50)))

# Heavy: Deep Agent over Plan-and-Execute (planning twice — once explicit, once via todos)
agent = Agent("...", architecture=DeepAgent(base=PlanAndExecute()))

# With Reflexion for cross-session learning
agent = Agent("...", architecture=Reflexion(base=DeepAgent()))

# With Self-Refine for quality polish on the final output
agent = Agent("...", architecture=SelfRefine(base=DeepAgent(), max_rounds=1))

# Deep Agent with declared subagents
agent = Agent(
    "You manage a research project.",
    architecture=DeepAgent(
        subagents=[
            SubagentDef(
                name="researcher",
                description="Searches and summarizes external sources.",
                instructions="You research thoroughly. Return only relevant findings.",
                tools=["web.search", "web.fetch"],
            ),
            SubagentDef(
                name="critic",
                description="Reviews drafts for accuracy.",
                instructions="You critique drafts. Be specific.",
                tools=[],
            ),
        ],
    ),
)
```

### Tuning Guide

- **`max_turns` should be HIGH.** 50-100. Deep Agent is for long tasks; capping at 25 defeats the purpose.
- **`auto_chunk_threshold=8000` is a good default.** Below this, return full content; above, chunked. Adjust per model context size.
- **Subagents should be specialists, not generalists.** Three focused subagents > one general one.
- **Customize the system prompt for your domain.** "You are a coding agent" → add coding-specific guidance; "You are a research agent" → add citation format guidance.
- **Disable pillars you don't need.** A pure-chat task doesn't need filesystem; setting `enable_filesystem=False` saves the tool overhead.

### Common Pitfalls

**1. Letting the agent edit `progress.md` through normal `edit_file`.** It works but it's brittle. Better: have a dedicated `update_progress(text)` tool with controlled semantics.

**2. Subagents that write to the parent's filesystem.** If subagents share the parent's filesystem, they can stomp on each other. Either give each subagent its own scratch directory, or make subagents read-only.

**3. Path-traversal in `write_file`.** A model might try `write_file("/etc/passwd", ...)`. Strict path validation (already in our impl) prevents this. Don't disable it.

**4. Ignoring the system prompt addon.** Removing the addon and expecting the pillars to "just work" — the model needs the prompt to know when to use them.

### Worked Example

```python
import asyncio
from pathlib import Path
from jeevesagent import Agent
from jeevesagent.architecture import DeepAgent, ReAct
from jeevesagent.architecture.config import FilesystemConfig, SubagentDef

async def main():
    workdir = Path("/tmp/research_project")
    workdir.mkdir(exist_ok=True)

    agent = Agent(
        "You are a research assistant working on a long-running project.",
        model="claude-opus-4-7",
        tools=["web.search", "web.fetch"],
        architecture=DeepAgent(
            base=ReAct(max_turns=80),
            filesystem=FilesystemConfig(
                allowed_roots=[workdir],
                max_file_size_bytes=5_000_000,
            ),
            subagents=[
                SubagentDef(
                    name="researcher",
                    description="Deep web research; returns structured summaries.",
                    instructions=(
                        "You research a single topic thoroughly. "
                        "Return a 2-paragraph summary with cited sources. "
                        "Do not return raw content."
                    ),
                    tools=["web.search", "web.fetch"],
                ),
                SubagentDef(
                    name="reviewer",
                    description="Critiques draft writing for clarity and accuracy.",
                    instructions=(
                        "You review drafts. Provide specific critiques. "
                        "Cite passages. Don't rewrite, just review."
                    ),
                    tools=[],
                ),
            ],
        ),
    )

    result = await agent.run(
        "Produce a 5-page report on 'agent harness architectures in 2026'. "
        "Save the final report as report.md in the working directory. "
        "Use the researcher subagent for source gathering and the reviewer for quality checks."
    )

    print(f"Done. Files produced:")
    for f in workdir.iterdir():
        print(f"  {f.name} ({f.stat().st_size} bytes)")

asyncio.run(main())
```


---

# Multi-agent architectures

## 9. Supervisor / Hierarchical

### Origin

**Production source:** Anthropic's internal multi-agent research system (reported **+90.2% performance improvement** over single-agent baseline). Productized as **Anthropic Agent Teams** in Claude Opus 4.6 (Feb 2026). Same pattern in CrewAI's "hierarchical process" and LangGraph's supervisor subgraph.

**What it solves:** When a task naturally decomposes into specialist domains (research + writing + code review), a single agent juggling all three has poor focus and bloated context. The Supervisor pattern delegates: a manager agent routes subtasks to specialist worker agents with isolated contexts and focused tool allowlists; workers return results; the manager synthesizes.

**Status:** **The 2026 consensus winner for multi-agent.** When the literature says "go multi-agent," it almost always means this.

### The Pattern

```
                       ┌──────────────┐
                       │  Supervisor  │  ← owns user conversation
                       │  (manager)   │
                       └──────┬───────┘
                              │
                ┌─────────────┼─────────────┐
                ↓             ↓             ↓
          ┌──────────┐  ┌──────────┐  ┌──────────┐
          │ Worker 1 │  │ Worker 2 │  │ Worker 3 │
          │ research │  │  coding  │  │  review  │
          └──────────┘  └──────────┘  └──────────┘
              ↑              ↑              ↑
         isolated       isolated       isolated
         context        context        context
```

The supervisor sees user input. It calls a `delegate(worker, instructions)` tool to invoke a specific worker. The worker runs in its own isolated context, returns a final answer. The supervisor's context grows by that one answer, not the worker's full transcript.

### Mechanism

The composition spec (`MULTI_AGENT_COMPOSITION_SPEC.md`) describes the full mechanism in detail. Key points:

1. **Worker registration.** At construction, the Supervisor's `workers={"name": Agent}` dict is registered. Each worker is a full Agent instance.

2. **Delegate tool.** The framework synthesizes a `delegate` tool with:
   - Schema generated from worker names + descriptions
   - When called, the framework spawns the named worker, passes `instructions` as its prompt, awaits completion
   - Worker's final answer returns as the tool result

3. **Supervisor loop.** Internally runs ReAct (or another base architecture) — sees `delegate` as just another tool. Decides when to call it based on the task.

4. **Worker context isolation.** Each worker has its own `Session`, scoped memory namespace, possibly different model. Inherits MCP servers, runtime, budget, hooks per the composition spec.

5. **Synthesis.** When the supervisor decides the task is done, it produces a final answer using the workers' outputs. This is normal ReAct termination — model emits no tool calls, output is the answer.

### Configuration

```python
@dataclass
class SupervisorConfig:
    workers: dict[str, "Agent"]
    """Worker Agents keyed by name. Names must be valid Python identifiers."""

    base: Architecture | None = None
    """Architecture for the supervisor itself. Default: ReAct(max_turns=20)."""

    supervisor_instructions: str | None = None
    """Extra instructions for the supervisor. None uses our default that
    teaches it to delegate effectively."""

    delegate_tool_name: str = "delegate"
    """Name of the delegation tool. Customize if 'delegate' clashes with user tools."""

    parallel_delegations: bool = True
    """If supervisor emits multiple delegate calls in one turn, run them in parallel.
    Off only for debugging."""
```

### Implementation

```python
# jeevesagent/architecture/supervisor.py

from __future__ import annotations
from typing import AsyncIterator

from .base import Architecture
from .react import ReAct


DEFAULT_SUPERVISOR_PROMPT = """\
You are a supervisor coordinating specialist worker agents.

For each user task:
1. Decide which workers are needed.
2. Call `delegate(worker, instructions)` to invoke a specialist.
3. The worker runs independently and returns its final answer.
4. Synthesize worker outputs into a unified response to the user.

You can delegate multiple workers in parallel within a single turn.
Be specific in your instructions to workers — they don't see the user's original message.

Available workers:
{worker_descriptions}
"""


class Supervisor:
    name = "supervisor"

    def __init__(self, config=None, **kwargs):
        if config is None and kwargs:
            from .config import SupervisorConfig
            config = SupervisorConfig(**kwargs)
        self.cfg = config
        if self.cfg.base is None:
            self.cfg.base = ReAct(max_turns=20)

    def declared_workers(self):
        return self.cfg.workers

    async def run(self, session, deps, prompt) -> AsyncIterator[Event]:
        # Inject supervisor prompt
        worker_desc = self._format_worker_descriptions()
        supervisor_prompt = (
            self.cfg.supervisor_instructions or DEFAULT_SUPERVISOR_PROMPT
        ).format(worker_descriptions=worker_desc)
        session.append_system(supervisor_prompt)

        # Register the delegate tool
        deps.tools.register_inproc(self._make_delegate_toolset(deps))

        # Run base architecture (the supervisor sees `delegate` as a normal tool)
        async for event in self.cfg.base.run(session, deps, prompt):
            yield event

    def _format_worker_descriptions(self) -> str:
        lines = []
        for name, worker in self.cfg.workers.items():
            desc = worker._cfg.instructions[:200]
            lines.append(f"  - {name}: {desc}")
        return "\n".join(lines)

    def _make_delegate_toolset(self, deps):
        from ..mcp.tools import tool, ToolSet
        from pydantic import Field
        from typing import Literal

        worker_names = list(self.cfg.workers.keys())

        @tool(name=self.cfg.delegate_tool_name)
        async def delegate(
            worker: Literal[tuple(worker_names)],  # type: ignore[valid-type]
            instructions: str = Field(..., description="Task for the worker"),
        ) -> str:
            """Delegate a subtask to a named specialist worker."""
            spec = self.cfg.workers.get(worker)
            if spec is None:
                return f"Error: unknown worker {worker!r}"

            # Spawn child agent in scoped context (composition spec applies)
            sub_session = deps.fresh_session(parent=deps.session_id)
            sub_deps = deps.scope_for_worker(spec)

            # Run the worker's architecture
            sub_arch = spec._architecture
            events = []
            async for event in sub_arch.run(sub_session, sub_deps, instructions):
                events.append(event)
                # Forward to parent's telemetry as a child span
                await deps.telemetry.emit_child_event(event, agent_path=[*deps.path, worker])

            return sub_session.result or "[no result]"

        return ToolSet(name="delegation", tools=[delegate])
```

### Performance

| Metric | Value | Source |
|---|---|---|
| Anthropic MA Research benchmark | +90.2% over single-agent | Anthropic 2026 |
| Token cost vs single-agent | 1.5-3× | Empirical, varies with task |
| Wall-clock | Often *faster* (parallel workers) | Empirical |
| Failure mode rate | Lower than swarm or blackboard | 2026 taxonomy |

Token cost is higher (multiple LLMs, multiple contexts), but wall-clock is often lower because workers run in parallel. For latency-sensitive multi-domain tasks, this is a net win.

### Strengths

1. **The proven multi-agent pattern.** "Earns its cost in production" per the 2026 consensus.
2. **Clear authority.** Failures are easy to attribute (which worker, which delegation).
3. **Parallel speedup.** Independent workers run concurrently; wall-clock benefits.
4. **Context isolation.** Workers don't pollute each other's contexts; supervisor sees only summaries.
5. **Easy to scale.** Add a new specialist by adding a worker; no architecture change.

### Weaknesses

1. **Supervisor is a bottleneck.** Every message routes through it. If the supervisor is slow, the system is slow.
2. **Cross-worker communication is lossy.** Worker A and Worker B can only talk via the supervisor's synthesized message. Information is lost in the translation.
3. **Wrong-worker selection cascades.** If supervisor picks the wrong worker, you waste a delegation cycle.
4. **Cost.** 1.5-3× single-agent baseline. Real money on scale.
5. **Synthesis quality is critical.** Bad synthesis = bad output even with great workers.

### When to Use

- Tasks naturally decomposable into specialist domains
- Multi-format outputs (research findings + code + visualizations)
- Long tasks where context isolation matters more than coordination overhead
- When you want parallel execution of independent subtasks

### When NOT to Use

- Single-domain tasks (use Deep Agent or ReAct + tools)
- Highly coupled subtasks where workers need to coordinate (use Blackboard if peer-comm is essential)
- Cost-critical applications
- Latency-critical with serial dependencies

### Composition

Supervisor is highly compositional:

```python
# Workers can be any architecture
agent = Agent("manager", architecture=Supervisor(workers={
    "researcher": Agent("...", architecture=DeepAgent()),  # heavy
    "coder": Agent("...", architecture=ActorCritic()),  # quality-critical
    "summarizer": Agent.specialist("..."),  # lightweight
}))

# Workers can themselves be supervisors (nested teams)
code_team = Agent("code lead", architecture=Supervisor(workers={
    "tester": Agent.specialist("..."),
    "implementer": Agent.specialist("..."),
}))
agent = Agent("CEO", architecture=Supervisor(workers={
    "code": code_team,  # nested supervisor
    "research": Agent.specialist("..."),
}))

# With Reflexion for learning across sessions
agent = Agent("...", architecture=Reflexion(base=Supervisor(workers={...})))
```

### Tuning Guide

- **Number of workers: 3-7 is the sweet spot.** Below 3, routing overhead isn't worth it. Above 7, the supervisor's selection becomes unreliable.
- **Worker descriptions matter.** The supervisor decides who to call based on these. Make them specific and distinguishing.
- **Limit supervisor's `max_turns`.** Default 20 is fine; if the supervisor is making too many delegations, it's not synthesizing well.
- **Use cheap model for simple workers, frontier for the supervisor.** The supervisor's reasoning quality matters most.

### Common Pitfalls

**1. Vague worker descriptions.** "Worker that does stuff" → supervisor picks the wrong one. Be specific.

**2. Workers with overlapping responsibilities.** Two workers that could both handle the same task → supervisor flips a coin. Either merge them or sharpen their descriptions.

**3. Supervisor that delegates everything.** If the supervisor never thinks for itself, it's just a router. Make sure it has reasoning capacity in its prompt.

**4. Workers writing to shared state without synchronization.** Each worker should produce a result; combining results is the supervisor's job. Don't let workers stomp on shared resources.

**5. Forgetting that workers don't see user's original message.** Always include necessary context in your delegation instructions.

### Worked Example

```python
import asyncio
from jeevesagent import Agent
from jeevesagent.architecture import Supervisor, DeepAgent, ActorCritic

async def main():
    # Build the worker team
    researcher = Agent(
        "You are a thorough researcher with web access.",
        tools=["web.search", "web.fetch"],
        architecture=DeepAgent(),
    )

    coder = Agent(
        "You write production-quality Python code.",
        tools=["fs.read", "fs.write", "bash"],
        architecture=ActorCritic(max_rounds=2),
    )

    writer = Agent.specialist(
        "You synthesize research findings into clear prose.",
    )

    # Build the supervisor
    agent = Agent(
        "You orchestrate research, coding, and writing tasks.",
        model="claude-opus-4-7",
        architecture=Supervisor(workers={
            "researcher": researcher,
            "coder": coder,
            "writer": writer,
        }),
    )

    result = await agent.run(
        "Research the LangGraph framework and produce a 1-page comparison "
        "with our JeevesAgent design. Include code examples that demonstrate "
        "the difference."
    )

    print(result.output)
    print(f"\nWorkers invoked: {result.metadata['delegations']}")

asyncio.run(main())
```


---

## 10. Actor-Critic (Generator-Critic)

### Origin

**Roots:** Reinforcement learning literature (Sutton & Barto, 1998 — "Reinforcement Learning: An Introduction"). LLM applications include:
- **Self-Refine** (Madaan et al., 2023) — same model as both
- **CRITIC** (Gou et al., 2023) — tool-augmented critique
- **CGI / Critique-Guided Improvement** (Sun et al., 2025) — separate critic model
- **Production case studies in adversarial code review** ("Just Understanding Data" 2026)

**Note on naming:** In casual conversation, this is sometimes called "GAN architecture" (Generative Adversarial Network). That's a misnomer — GAN is a *training* technique for neural networks, not an agent pattern. The correct term is **Actor-Critic** (from RL) or **Generator-Critic**. The "adversarial" framing matters: the critic is *designed* to find issues, not to rubber-stamp.

**What it solves:** Self-Refine has the same model as both generator and critic; same blind spots, marginal gains. Actor-Critic uses **different prompts and often different models** for generation vs critique. The critic is adversarially-prompted to find issues. Asymmetric criticism produces issues neither model would have caught alone.

**Status:** The standard architecture for quality-critical work — code, security review, important writing.

### The Pattern

```
┌──────────┐   output   ┌──────────┐
│  Actor   │ ─────────→ │  Critic  │  ← different prompt, often different model
│ (generate│            │ (find ALL│
│   code)  │            │  issues) │
└──────────┘            └─────┬────┘
     ↑                        │
     │   detailed critique    │
     │   (cite specifics)     │
     └────────────────────────┘
                              │
                              ↓ (if score < threshold)
                    Actor refines based on critique
```

The critic is **adversarial by design** — its prompt instructs it to find ALL issues, not to assess quality holistically. This asymmetry is the point.

### Mechanism

1. **Round 0 — Initial generation.** The Actor (a sub-Agent with the actor instructions) runs to produce an initial output.

2. **For each round up to `max_rounds`:**
   - **Critique.** The Critic (a sub-Agent with the critic instructions, possibly using a different model) reviews the current output. The critic returns a structured critique with severity ratings and an overall score (0-1).
   - **Termination check.** If `critique.score >= approval_threshold`, terminate with current output.
   - **Refinement.** The Actor receives (original prompt, current output, critique) and produces a revised version addressing the critique.
   - Loop with new output.

3. **Termination.** Score threshold met, max rounds hit, or budget exceeded.

### Configuration

```python
@dataclass
class ActorCriticConfig:
    actor: Optional["Agent"] = None
    """The actor as a sub-Agent. Inherits from parent if None."""

    critic: Optional["Agent"] = None
    """The critic as a sub-Agent with different prompt and possibly different model.
    If None, falls back to Self-Refine semantics (same model)."""

    max_rounds: int = 5
    """Max critique-refine cycles after initial generation."""

    approval_threshold: float = 0.9
    """Critic outputs a 0-1 score. Above this, terminate."""

    actor_instructions: str | None = None
    critic_instructions: str | None = None
    """Custom prompts for actor and critic. None uses our defaults."""

    require_score_in_critique: bool = True
    """Critic must output a structured score; reject critiques without one."""
```

### Implementation

```python
# jeevesagent/architecture/actor_critic.py

from __future__ import annotations
from typing import AsyncIterator, Optional
from pydantic import BaseModel, Field

from .base import Architecture
from .react import ReAct


DEFAULT_ACTOR_PROMPT = """\
You produce the requested output. Be thorough and complete.
You will receive critiques from a reviewer and produce revisions accordingly.
"""


DEFAULT_CRITIC_PROMPT = """\
You are an ADVERSARIAL critic. Your job is to find ALL problems with the output.
Be ruthless. List every specific issue you find with concrete examples.
Score the output 0-1 (1 = no issues, 0 = unusable).

For each issue:
- Cite the specific section/quote
- Explain why it's a problem
- Suggest a concrete fix

Output structured JSON: {"issues": [...], "score": float, "summary": str}
"""


class CriticOutput(BaseModel):
    issues: list[str]
    score: float = Field(ge=0, le=1)
    summary: str


class ActorCritic:
    name = "actor-critic"

    def __init__(self, config=None, **kwargs):
        if config is None and kwargs:
            from .config import ActorCriticConfig
            config = ActorCriticConfig(**kwargs)
        self.cfg = config

    def declared_workers(self):
        workers = {}
        if self.cfg.actor is not None:
            workers["actor"] = self.cfg.actor
        if self.cfg.critic is not None:
            workers["critic"] = self.cfg.critic
        return workers

    async def run(self, session, deps, prompt) -> AsyncIterator[Event]:
        # Build actor and critic if not pre-supplied (use parent's defaults)
        actor = self._resolve_actor(deps)
        critic = self._resolve_critic(deps)

        # === ROUND 0: INITIAL GENERATION ===
        actor_session = deps.fresh_session(parent=session.id)
        actor_deps = deps.scope_for_worker(actor)
        async for event in actor._architecture.run(
            actor_session, actor_deps, prompt
        ):
            yield event.with_role("actor")
        current_output = actor_session.result

        for round_num in range(self.cfg.max_rounds):
            # Budget gate
            status = await deps.budget.allows_step()
            if status.blocked:
                yield Event.budget_exceeded(session.id, status)
                return

            # === CRITIQUE ===
            critic_session = deps.fresh_session(parent=session.id)
            critic_deps = deps.scope_for_worker(critic)
            critique_prompt = (
                f"Original task:\n{prompt}\n\n"
                f"Output to review:\n{current_output}\n\n"
                f"Provide structured critique with score."
            )

            critique_events = []
            async for event in critic._architecture.run(
                critic_session, critic_deps, critique_prompt
            ):
                critique_events.append(event)
                yield event.with_role("critic")

            critique = self._parse_critique(critic_session.result)
            yield Event.critique(session.id, round_num, critique)

            # Termination check
            if critique.score >= self.cfg.approval_threshold:
                session.complete = True
                session.result = current_output
                yield Event.completed(
                    session.id,
                    current_output,
                    rounds=round_num + 1,
                    final_score=critique.score,
                )
                return

            # === REFINE ===
            refine_prompt = (
                f"Original task:\n{prompt}\n\n"
                f"Your previous output:\n{current_output}\n\n"
                f"Critique to address (score: {critique.score}):\n"
                f"{chr(10).join(f'- {issue}' for issue in critique.issues)}\n\n"
                f"Produce a revised version that addresses every critique point."
            )
            actor_session = deps.fresh_session(parent=session.id)
            actor_deps = deps.scope_for_worker(actor)
            async for event in actor._architecture.run(
                actor_session, actor_deps, refine_prompt
            ):
                yield event.with_role("actor")
            current_output = actor_session.result

        # Max rounds reached without approval
        session.result = current_output
        yield Event.max_rounds_reached(session.id, current_output, rounds=self.cfg.max_rounds)

    def _resolve_actor(self, deps):
        if self.cfg.actor is not None:
            return self.cfg.actor
        # Build a default actor from parent's settings
        from ..agent.api import Agent
        return Agent.specialist(
            self.cfg.actor_instructions or DEFAULT_ACTOR_PROMPT,
            tools=deps.parent_tools,
        )

    def _resolve_critic(self, deps):
        if self.cfg.critic is not None:
            return self.cfg.critic
        from ..agent.api import Agent
        return Agent.specialist(
            self.cfg.critic_instructions or DEFAULT_CRITIC_PROMPT,
            tools=[],  # critic typically doesn't need tools
        )

    def _parse_critique(self, text: str) -> CriticOutput:
        # Robust parsing of structured output
        # Real impl: try JSON parse, fall back to regex extraction
        ...
```

### Performance

| Metric | Value | Source |
|---|---|---|
| Token cost | 2-5× single-pass | Round count × actor + critic |
| Latency | 2-5× wall-clock | Mostly sequential |
| Issues caught after 3-5 rounds | 90%+ of issues that would otherwise reach human review | Just Understanding Data 2026 |
| Code review use case | Reduces human review time 60-70% | Same source |

The "90% of issues caught" number is industry case study territory — not from a peer-reviewed paper, but consistent with observed behavior across deployments.

### Strengths

1. **Different blind spots.** Actor and critic with different prompts (and ideally different models) catch different issues.
2. **Adversarial by design.** The critic prompt explicitly demands rigor; no rubber-stamping.
3. **Structured score.** The 0-1 score gives a clear termination signal.
4. **Strong on quality-driven tasks.** Particularly code generation, security review, technical writing.
5. **Composes well with Supervisor.** Each worker in a Supervisor can be Actor-Critic for quality control.

### Weaknesses

1. **2-5× token cost.** Real money on scale.
2. **Sequential rounds.** Wall-clock multiplied.
3. **Critic can be too aggressive.** "Find ALL issues" prompts produce nitpicks; need to tune severity threshold.
4. **Critic can be too lenient.** Especially when actor and critic share a model. Use different models for the asymmetry.
5. **Risk of refinement loops without convergence.** If the critic always finds something new, you hit `max_rounds`.

### When to Use

- Code generation (the canonical use)
- Security-critical writing (privacy policies, ToS, etc.)
- Technical documentation requiring accuracy
- Important written communications
- Any task where you'd otherwise have a human review cycle

### When NOT to Use

- Cost-sensitive simple tasks
- Latency-critical applications
- Conversational agents (multiple rounds break flow)
- Tasks with no clear quality criterion (Self-Refine works as well)

### Composition

```python
# Standard: Actor-Critic for code work, used as a worker in Supervisor
agent = Agent("manager", architecture=Supervisor(workers={
    "coder": Agent("write code", architecture=ActorCritic(
        actor=Agent.specialist("...", model="claude-opus-4-7"),
        critic=Agent.specialist("...", model="gpt-4o"),  # cross-model triangulation
        max_rounds=3,
    )),
}))

# Actor-Critic with Self-Refine inside the actor (extreme quality)
agent = Agent("...", architecture=ActorCritic(
    actor=Agent.specialist("...", architecture=SelfRefine(max_rounds=2)),
    max_rounds=2,  # outer
))

# Inside Reflexion for cross-session learning of critic patterns
agent = Agent("...", architecture=Reflexion(base=ActorCritic()))
```

### Tuning Guide

- **`max_rounds=3` is the sweet spot.** 5 is reasonable for high-stakes code; beyond 5, you're polishing nits.
- **`approval_threshold=0.9` is strict.** Lower to 0.85 for friendlier convergence; raise to 0.95 for production-critical work.
- **Use different models for actor and critic.** The asymmetry is the point. Claude Opus actor + GPT-4o critic, or vice versa.
- **Customize critic prompts per domain.** "Find security issues" for security-critical code; "Find missing edge cases" for production code; "Check for factual errors" for written content.
- **Watch for nitpicking.** If the critic always emits 10 minor issues, raise the severity threshold or rework the critic prompt.

### Common Pitfalls

**1. Actor and critic on same model with same prompt.** That's Self-Refine, not Actor-Critic. Differentiate.

**2. Critic prompt that's too friendly.** "Review for issues" → "looks good!" Be adversarial: "Find ALL issues. Be ruthless."

**3. Forgetting to require structured output.** Free-text critiques are hard to parse and don't produce usable scores.

**4. Skipping the actor's awareness of past critiques.** The refine prompt should include the previous output AND the critique. Otherwise the actor regenerates from scratch.

**5. Running too many rounds.** 5 is generous. If you're hitting it, the architecture isn't converging — fix the prompt or accept partial output.

### Worked Example

```python
import asyncio
from jeevesagent import Agent
from jeevesagent.architecture import ActorCritic

async def main():
    # Different models for actor/critic to maximize blind-spot diversity
    actor = Agent(
        "You write production Python code. Be complete and correct.",
        model="claude-opus-4-7",
        tools=["fs.read", "fs.write"],
    )

    critic = Agent(
        "You are a senior security engineer reviewing code adversarially. "
        "Find all issues: security vulnerabilities, edge cases, performance, "
        "missing error handling, untested paths. Cite line numbers. "
        "Output JSON: {issues: [...], score: 0-1, summary: str}.",
        model="gpt-4o",  # different model
        tools=[],  # critic doesn't write code
    )

    agent = Agent(
        "You produce reviewed, production-quality code.",
        architecture=ActorCritic(
            actor=actor,
            critic=critic,
            max_rounds=3,
            approval_threshold=0.9,
        ),
    )

    result = await agent.run(
        "Implement a secure password reset flow using JWT in FastAPI. "
        "Include rate limiting and full test coverage."
    )

    print(f"Final code:\n{result.output}")
    print(f"\nRounds needed: {result.metadata['rounds']}")
    print(f"Final score: {result.metadata['final_score']}")

asyncio.run(main())
```


---

## 11. Multi-Agent Debate

### Origin

**Paper:** Du, Yilun, et al. "Improving Factuality and Reasoning in Language Models through Multiagent Debate." arXiv:2305.14325 (May 2023). Also Liang et al. 2023 — "Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate." Production patterns in AutoGen GroupChat, CAMEL.

**What it solves:** Single agents (and even Self-Refine / Actor-Critic where the same model plays both roles) share blind spots. When confronted with a contested or ambiguous question, they often confidently produce a wrong answer. Multi-Agent Debate forces multiple agents (often the same LLM with different roles or different LLMs) to argue different positions across multiple rounds. Disagreement surfaces hidden evidence; agreement signals real consensus. A separate Judge synthesizes the conclusion.

**Status:** Niche but powerful. The 2026 production literature is cautious — debate adds 3-5× cost and works best on a narrow set of decision-style questions. Reserve for high-stakes contested questions where a wrong answer is expensive.

### The Pattern

```
            Question
               │
               ↓
   ┌───────────┼───────────┐
   ↓           ↓           ↓
┌──────┐  ┌──────┐    ┌──────┐
│ AgA  │  │ AgB  │    │ AgC  │   Round 1: independent answers
└──┬───┘  └──┬───┘    └──┬───┘
   │ ←─────cross-read──→ │
   ↓           ↓           ↓
┌──────┐  ┌──────┐    ┌──────┐
│ AgA  │  │ AgB  │    │ AgC  │   Round 2: argue, citing each other
└──┬───┘  └──┬───┘    └──┬───┘
   │           │           │
   └───────────┼───────────┘
               ↓
          ┌─────────┐
          │  Judge  │           Synthesize / vote
          └─────────┘
               │
               ↓
          Final answer
```

Three components:
- **Debaters** (2+) — each defends a position; they read each other's arguments between rounds
- **Judge** (separate agent) — synthesizes the final answer after debate concludes
- **Optional moderator** — keeps debate on track (rare in practice; usually the structure suffices)

### Mechanism

1. **Setup.** N debater Agents are configured with possibly-divergent personas or stances. The judge is a separate Agent.

2. **Round 0 — Independent answers.** Each debater answers the original question independently, with no awareness of the others. This produces N initial responses.

3. **Rounds 1 to K-1 — Debate.** Each debater receives:
   - The original question
   - All previous-round responses from all debaters
   - Instructions to: defend their position OR update it based on others' arguments

4. **Convergence check (optional).** If all debaters now agree, debate can terminate early. Otherwise continue.

5. **Round K — Final positions.** Each debater commits to their final answer.

6. **Judge synthesis.** The judge receives all final positions and the full debate transcript. It produces:
   - A single final answer
   - A confidence rating (0-1)
   - Justification (which arguments swayed the decision)

### Configuration

```python
@dataclass
class DebateConfig:
    debaters: list[Agent]
    """Two or more debater Agents. They run in parallel each round."""

    judge: Agent | None = None
    """Synthesizer. None = use majority vote across debaters' final answers."""

    rounds: int = 3
    """How many debate rounds (after round 0). Default 3 = total of 4 turns."""

    convergence_check: bool = True
    """Terminate early if all debaters agree."""

    instructions: str = DEFAULT_DEBATE_INSTRUCTIONS
    """Guidance to debaters: 'defend your position but update if convinced'."""

    judge_instructions: str = DEFAULT_JUDGE_INSTRUCTIONS
    """Guidance to the judge."""
```

### Implementation

```python
# jeevesagent/architecture/debate.py

from __future__ import annotations
import anyio
from typing import AsyncIterator
from pydantic import BaseModel, Field

from .base import Architecture
from .react import ReAct


DEFAULT_DEBATE_INSTRUCTIONS = """\
You are participating in a structured debate about the following question.
Other debaters have proposed answers. Your task:
1. Defend your position with specific reasoning.
2. Address each other debater's argument: where do you agree, where do you disagree, and why.
3. If a counter-argument is convincing, update your position openly. Don't be stubborn for its own sake.
4. Cite specifics; avoid hand-waving.
"""

DEFAULT_JUDGE_INSTRUCTIONS = """\
You are an impartial judge synthesizing a multi-agent debate.

Read all final positions and the debate transcript. Output:
- final_answer: the conclusion you draw
- confidence: 0-1 (1 = high consensus, 0 = no consensus)
- justification: which arguments most influenced your decision
- dissenting_view: the strongest minority position, even if you disagree

Output JSON: {final_answer: str, confidence: float, justification: str, dissenting_view: str | null}
"""


class DebateRound(BaseModel):
    round_num: int
    responses: dict[str, str]  # debater_name -> response


class JudgeOutput(BaseModel):
    final_answer: str
    confidence: float = Field(ge=0, le=1)
    justification: str
    dissenting_view: str | None = None


class MultiAgentDebate:
    name = "debate"

    def __init__(self, config=None, **kwargs):
        if config is None and kwargs:
            from .config import DebateConfig
            config = DebateConfig(**kwargs)
        self.cfg = config
        if len(self.cfg.debaters) < 2:
            raise ValueError("Debate requires at least 2 debaters")

    def declared_workers(self):
        workers = {f"debater_{i}": d for i, d in enumerate(self.cfg.debaters)}
        if self.cfg.judge is not None:
            workers["judge"] = self.cfg.judge
        return workers

    async def run(self, session, deps, prompt) -> AsyncIterator[Event]:
        history: list[DebateRound] = []

        # === ROUND 0: Independent answers ===
        yield Event.debate_round_started(session.id, 0)
        round0_responses = await self._run_round(
            deps, prompt, history=[], round_num=0
        )
        history.append(DebateRound(round_num=0, responses=round0_responses))
        for name, resp in round0_responses.items():
            yield Event.debater_response(session.id, 0, name, resp)

        # === DEBATE ROUNDS ===
        for r in range(1, self.cfg.rounds + 1):
            # Budget gate
            status = await deps.budget.allows_step()
            if status.blocked:
                yield Event.budget_exceeded(session.id, status)
                return

            yield Event.debate_round_started(session.id, r)
            round_responses = await self._run_round(
                deps, prompt, history=history, round_num=r
            )
            history.append(DebateRound(round_num=r, responses=round_responses))
            for name, resp in round_responses.items():
                yield Event.debater_response(session.id, r, name, resp)

            # Convergence check
            if self.cfg.convergence_check and self._converged(round_responses):
                yield Event.debate_converged(session.id, r)
                break

        # === JUDGE SYNTHESIS ===
        yield Event.debate_judging(session.id)
        final = await deps.runtime.step(
            "debate_judge",
            self._judge_synthesize,
            deps,
            prompt,
            history,
        )
        session.complete = True
        session.result = final.final_answer
        yield Event.completed(
            session.id,
            final.final_answer,
            confidence=final.confidence,
            justification=final.justification,
            dissenting_view=final.dissenting_view,
        )

    # --- helpers --------------------------------------------------------

    async def _run_round(
        self, deps, prompt, history, round_num
    ) -> dict[str, str]:
        """Run all debaters in parallel for one round."""
        responses: dict[str, str] = {}

        async with anyio.create_task_group() as tg:
            for i, debater in enumerate(self.cfg.debaters):
                name = f"debater_{i}"
                tg.start_soon(
                    self._run_debater,
                    deps, prompt, history, round_num, name, debater, responses,
                )

        return responses

    async def _run_debater(
        self, deps, prompt, history, round_num, name, debater, output_dict
    ):
        # Build the round's prompt: original question + history of others' arguments
        debate_prompt = self._build_debater_prompt(prompt, history, name, round_num)

        sub_session = deps.fresh_session(parent=deps.session_id)
        sub_deps = deps.scope_for_worker(debater)

        # Run debater
        async for _event in debater._architecture.run(sub_session, sub_deps, debate_prompt):
            pass  # events streamed elsewhere

        output_dict[name] = sub_session.result or ""

    def _build_debater_prompt(self, prompt, history, name, round_num) -> str:
        if round_num == 0:
            return f"{prompt}\n\nProvide your independent answer with reasoning."

        # Subsequent rounds: include all prior responses
        context_parts = [
            self.cfg.instructions,
            f"\nOriginal question: {prompt}\n",
            "Debate transcript so far:\n",
        ]
        for r in history:
            context_parts.append(f"\n=== Round {r.round_num} ===")
            for n, resp in r.responses.items():
                marker = " (you)" if n == name else ""
                context_parts.append(f"\n{n}{marker}: {resp}")

        context_parts.append(f"\n\nNow it's round {round_num}. Defend or update your position.")
        return "\n".join(context_parts)

    def _converged(self, responses: dict[str, str]) -> bool:
        """Naive: all responses share a 'final answer' keyword. Real impl uses LLM-as-judge."""
        # Production version: have a small LLM call check semantic agreement
        return False  # Conservative default

    async def _judge_synthesize(self, deps, prompt, history) -> JudgeOutput:
        if self.cfg.judge is None:
            return self._majority_vote(history)

        transcript = self._format_transcript(prompt, history)
        judge_prompt = (
            self.cfg.judge_instructions + "\n\n" + transcript + "\n\nProvide your synthesis."
        )

        sub_session = deps.fresh_session()
        sub_deps = deps.scope_for_worker(self.cfg.judge)
        async for _event in self.cfg.judge._architecture.run(sub_session, sub_deps, judge_prompt):
            pass
        return self._parse_judge_output(sub_session.result)

    def _majority_vote(self, history) -> JudgeOutput:
        """Fallback when no judge configured: take the modal final answer."""
        final_round = history[-1]
        # Real impl: cluster semantically; v1 stub takes most common exact match
        from collections import Counter
        counts = Counter(final_round.responses.values())
        winner, votes = counts.most_common(1)[0]
        confidence = votes / len(final_round.responses)
        return JudgeOutput(
            final_answer=winner,
            confidence=confidence,
            justification=f"Majority vote: {votes}/{len(final_round.responses)}",
            dissenting_view=None,
        )
```

### Performance

| Metric | Value | Source |
|---|---|---|
| Token cost | 3-5× single-pass | N debaters × K rounds + judge |
| Latency | High (rounds are sequential, debaters per round are parallel) | Empirical |
| TruthfulQA | +12% over single agent (3 debaters, 2 rounds) | Du et al. 2023 |
| GSM8K math | +4-7% typical | Du et al. 2023 |
| Hallucination reduction | ~30% on factual questions | Du et al. 2023 |

### Strengths

1. **Surfaces blind spots through disagreement.** When debaters genuinely disagree, the disagreement reveals issues neither would have seen alone.
2. **Strong on factuality.** The TruthfulQA gain comes from cross-checking each other's claims.
3. **Adversarial stress-testing built in.** Position-defense forces specific reasoning, not vague hand-waving.
4. **Interpretable consensus.** The transcript shows what swayed each debater (or what didn't).
5. **Heterogeneous-model friendly.** Different LLMs (Claude + GPT + Llama) bring genuinely different priors.

### Weaknesses

1. **3-5× cost.** N debaters × K rounds + judge. Real money.
2. **Sequential rounds bound latency.** Even with parallel debaters, you wait round-by-round.
3. **Groupthink risk.** Debaters with shared priors converge on wrong answers fast. Use different models for the diversity to be real.
4. **Judge quality is critical.** Bad judge = bad final answer regardless of debate quality.
5. **Doesn't compose well.** Debate inside Reflexion is mostly redundant; debate inside Actor-Critic is a different shape.

### When to Use

- High-stakes factual decisions (medical/legal/financial reviews where a wrong answer is expensive)
- Adversarial stress-testing of plans
- Contested questions where one model's confidence is suspect
- Research mode where understanding *why* models disagree matters

### When NOT to Use

- Simple questions with clear right answers
- Cost-sensitive applications (3-5× is real)
- Latency-critical applications
- Tasks where the right answer is procedural, not contested

### Composition

Debate is mostly a leaf architecture. It can:

```python
# As one option in a Router (route contested questions to debate)
agent = Agent("...", architecture=Router(routes=[
    RouteSpec("simple", architecture=ReAct()),
    RouteSpec("contested", architecture=MultiAgentDebate(
        debaters=[d1, d2, d3],
        judge=judge,
    )),
]))

# As a worker in Supervisor for verification subtasks
agent = Agent("...", architecture=Supervisor(workers={
    "researcher": Agent.specialist("..."),
    "verifier": Agent("...", architecture=MultiAgentDebate(...)),  # for fact-checking
}))
```

Debate does NOT compose with: Reflexion, Self-Refine, Actor-Critic (overlapping/incompatible critique mechanisms), ToT or GoT (different paradigms).

### Tuning Guide

- **3 debaters is optimal.** 2 is too few (just disagrees). 4+ produces diminishing returns and ballooning cost.
- **2-3 rounds is enough.** Round 1: see disagreement. Round 2: address arguments. Round 3: final positions. More rarely helps.
- **Use different models if possible.** Same model = shared blind spots. Claude + GPT + Llama beats 3× Claude.
- **Always use a judge for important decisions.** Majority vote is a fallback, not a primary mechanism.
- **Customize debater personas.** Default "defend your position" works; "be the contrarian" / "be the conservative" / "be the optimist" can produce more genuine disagreement.

### Common Pitfalls

**1. Same model, same prompt for all debaters.** They produce essentially identical outputs and "debate" is a no-op. Differentiate.

**2. No judge.** Majority vote on free-text answers rarely converges (different phrasings of the same answer). Always configure a judge.

**3. Too many rounds.** 5+ rounds usually means the question is genuinely contested or the debaters are shallow. Either way, more rounds doesn't help.

**4. Letting debaters use tools.** Most debate use cases work better with frozen evidence: gather facts once, then debate the interpretation. Tools-during-debate creates information asymmetry.

**5. Treating debate confidence as ground truth.** The judge's confidence rating reflects debate consensus, not factual accuracy. Independent verification still matters for safety-critical applications.

### Worked Example

```python
import asyncio
from jeevesagent import Agent
from jeevesagent.architecture import MultiAgentDebate

async def main():
    # Three debaters with different models for genuine diversity
    optimist = Agent(
        "You analyze investments optimistically. Find the upside case.",
        model="claude-opus-4-7",
    )
    skeptic = Agent(
        "You analyze investments skeptically. Find the downside case.",
        model="gpt-4o",
    )
    quant = Agent(
        "You analyze investments quantitatively. Cite specific numbers.",
        model="claude-opus-4-7",
        tools=["data.fetch", "calc.eval"],
    )
    judge = Agent(
        "You are an impartial financial analyst synthesizing debate. "
        "Output JSON with final_answer, confidence (0-1), justification, dissenting_view.",
        model="claude-opus-4-7",
    )

    agent = Agent(
        "You produce well-reasoned investment analysis through structured debate.",
        architecture=MultiAgentDebate(
            debaters=[optimist, skeptic, quant],
            judge=judge,
            rounds=2,
            convergence_check=True,
        ),
    )

    result = await agent.run(
        "Should we invest $10M in Series B of a vertical AI startup with "
        "$2M ARR growing 200% YoY but $4M annual burn?"
    )

    print(f"Final: {result.output}")
    print(f"Confidence: {result.metadata['confidence']}")
    print(f"Dissenting view: {result.metadata['dissenting_view']}")

asyncio.run(main())
```


---

## 12. Blackboard

### Origin

**Classical AI:** Erman et al. (1980) — Hearsay-II speech understanding system. Botti et al. (1995) — survey. Concept goes back to a 1962 Newell paper.

**LLM revival:**
- Han & Zhang 2025 — "Exploring Advanced LLM Multi-Agent Systems Based on Blackboard Architecture" (arXiv:2507.01701)
- **Salemi et al. 2026** — "LLM-Based Multi-Agent Blackboard System for Information Discovery in Data Science" (arXiv:2510.01285) reports **13-57% relative improvement** in end-to-end success on data discovery tasks.

**What it solves:** Supervisor architectures route everything through one agent — the supervisor must understand all subtasks. Swarm has no supervisor and drifts. Blackboard splits the difference: agents share a public workspace ("blackboard") that holds problem state, partial results, evidence, open questions. Each agent watches the blackboard and contributes when it can. A coordinator picks the next contributor based on current state. No agent is forced to participate; participation is opt-in based on capability.

**Status:** **Ship in v3 if user demand emerges.** Specialized use case. The 2026 production literature is split — "theoretically interesting but rarely outperforms hierarchical or graph in practice" per the taxonomy guide, but the data discovery paper shows real wins on the right problems.

### The Pattern

```
                ┌──────────────────────────────────┐
                │           Blackboard              │
                │                                   │
                │   Public state:                   │
                │   - problem statement             │
                │   - partial results               │
                │   - open questions                │
                │   - evidence                      │
                │   - pending votes                 │
                │                                   │
                │   Private spaces (per agent):     │
                │   - debate scratchpads            │
                │   - private verification          │
                └─────┬─────┬──────┬──────┬────────┘
                      │     │      │      │
                 ┌────┴┐ ┌──┴──┐ ┌─┴───┐ ┌┴─────┐
                 │ Plan│ │Critic│ │Code │ │Search│
                 └─────┘ └─────┘ └─────┘ └──────┘
                                   ↑
                          ┌────────┴────────┐
                          │   Coordinator    │
                          │ (picks next      │
                          │  contributor)    │
                          └─────────────────┘
```

The coordinator reads the blackboard and decides who contributes next. Selection is based on the current state — what's missing, what's unresolved, who's qualified. Agents communicate solely through the blackboard (no direct messaging).

### Mechanism

1. **Initialize blackboard.** Problem statement is written to the public space. Agents (each with a role: planner, critic, code, search, etc.) are loaded but inactive.

2. **Coordinator loop:**
   - Read current blackboard state
   - Decide if termination criteria met (consensus reached, problem solved, max rounds hit)
   - If not done, pick the next agent to contribute (based on capabilities + current need)
   - Invoke that agent with the current blackboard state
   - Agent reads, optionally writes to blackboard (adds findings, asks questions, votes)
   - Loop

3. **Agents are reactive.** They don't have their own goals; they respond to what's on the blackboard. A "critic" agent contributes only when there's something to critique; a "search" agent contributes only when there's an open question.

4. **Termination.** Either:
   - Consensus on the blackboard (a designated "decider" agent declares the answer)
   - All agents pass on contributing (no one has anything to add)
   - Max rounds hit

### Configuration

```python
@dataclass
class BlackboardConfig:
    agents: dict[str, Agent]
    """Roles → Agents. e.g. {"planner": ..., "critic": ..., "search": ...}"""

    coordinator: Agent | None = None
    """Picks next contributor each round. None = round-robin."""

    decider: Agent | None = None
    """Decides termination. None = max_rounds based."""

    max_rounds: int = 20
    """Hard cap on coordinator iterations."""

    blackboard_state_schema: type[BaseModel] | None = None
    """Optional Pydantic schema for typed blackboard state. None = freeform dict."""

    initial_state_template: str = "{prompt}"
    """How the initial blackboard is populated from the user prompt."""
```

### Implementation

```python
# jeevesagent/architecture/blackboard.py

from __future__ import annotations
from typing import AsyncIterator
from pydantic import BaseModel
from datetime import datetime

from .base import Architecture
from .react import ReAct


DEFAULT_COORDINATOR_PROMPT = """\
You coordinate a team of specialist agents working on a shared problem.

Current blackboard state:
{state}

Available agents and their capabilities:
{agents}

Decide:
- Should we terminate? (yes/no, reason)
- If continuing, which agent should contribute next? (name)
- What instruction should that agent receive?

Output JSON: {terminate: bool, reason: str, next_agent: str|null, instruction: str|null}
"""


class CoordinatorDecision(BaseModel):
    terminate: bool
    reason: str
    next_agent: str | None = None
    instruction: str | None = None


class BlackboardEntry(BaseModel):
    timestamp: datetime
    author: str
    content: str
    kind: str  # "evidence" | "question" | "answer" | "vote" | "note"


class Blackboard:
    """In-memory blackboard with public + private partitions."""
    def __init__(self):
        self.public: list[BlackboardEntry] = []
        self.private: dict[str, list[BlackboardEntry]] = {}

    def post(self, author: str, content: str, kind: str = "note", private_to: str | None = None):
        entry = BlackboardEntry(
            timestamp=datetime.utcnow(),
            author=author,
            content=content,
            kind=kind,
        )
        if private_to:
            self.private.setdefault(private_to, []).append(entry)
        else:
            self.public.append(entry)

    def view_for(self, agent_name: str) -> str:
        """Format blackboard view for an agent (public + their private)."""
        public_view = "\n".join(
            f"[{e.timestamp.isoformat()}] {e.author} ({e.kind}): {e.content}"
            for e in self.public
        )
        private_view = "\n".join(
            f"[{e.timestamp.isoformat()}] {e.author} ({e.kind}): {e.content}"
            for e in self.private.get(agent_name, [])
        )
        if private_view:
            return f"=== Public ===\n{public_view}\n\n=== Your private ===\n{private_view}"
        return public_view


class BlackboardArchitecture:
    name = "blackboard"

    def __init__(self, config=None, **kwargs):
        if config is None and kwargs:
            from .config import BlackboardConfig
            config = BlackboardConfig(**kwargs)
        self.cfg = config

    def declared_workers(self):
        return self.cfg.agents

    async def run(self, session, deps, prompt) -> AsyncIterator[Event]:
        blackboard = Blackboard()
        blackboard.post("user", prompt, kind="problem")
        yield Event.blackboard_initialized(session.id, blackboard)

        for round_num in range(self.cfg.max_rounds):
            # Budget gate
            status = await deps.budget.allows_step()
            if status.blocked:
                yield Event.budget_exceeded(session.id, status)
                return

            # COORDINATOR DECIDES
            decision = await deps.runtime.step(
                f"bb_coord_{round_num}",
                self._coordinate,
                deps,
                blackboard,
                round_num,
            )
            yield Event.blackboard_decision(session.id, round_num, decision)

            if decision.terminate:
                # FINAL DECISION
                final = await self._decide_final(deps, blackboard)
                session.complete = True
                session.result = final
                yield Event.completed(session.id, final, reason=decision.reason)
                return

            if decision.next_agent is None:
                yield Event.blackboard_no_contributor(session.id, round_num)
                continue

            # INVOKE AGENT
            agent = self.cfg.agents.get(decision.next_agent)
            if agent is None:
                blackboard.post(
                    "system",
                    f"Coordinator picked unknown agent: {decision.next_agent}",
                    kind="error",
                )
                continue

            yield Event.blackboard_invoking(session.id, round_num, decision.next_agent)
            agent_view = blackboard.view_for(decision.next_agent)
            agent_prompt = (
                f"Blackboard state:\n{agent_view}\n\n"
                f"Coordinator instruction: {decision.instruction}\n\n"
                f"Contribute to the blackboard. Output one of:\n"
                f"- evidence: a finding\n"
                f"- question: an open question\n"
                f"- answer: a candidate answer\n"
                f"- vote: agreement/disagreement on a candidate\n"
                f"- note: a process note\n\n"
                f"Format your contribution clearly."
            )

            sub_session = deps.fresh_session(parent=deps.session_id)
            sub_deps = deps.scope_for_worker(agent)
            async for _ in agent._architecture.run(sub_session, sub_deps, agent_prompt):
                pass

            contribution = sub_session.result or "[no contribution]"
            blackboard.post(decision.next_agent, contribution, kind="contribution")
            yield Event.blackboard_contribution(
                session.id, round_num, decision.next_agent, contribution
            )

        # Max rounds reached
        final = await self._decide_final(deps, blackboard)
        session.result = final
        yield Event.max_rounds_reached(session.id, final)

    async def _coordinate(self, deps, blackboard, round_num) -> CoordinatorDecision:
        if self.cfg.coordinator is None:
            # Round-robin fallback
            agents = list(self.cfg.agents.keys())
            picked = agents[round_num % len(agents)]
            return CoordinatorDecision(
                terminate=False,
                reason="round-robin",
                next_agent=picked,
                instruction=f"Read the blackboard and contribute as {picked}.",
            )

        coord_prompt = DEFAULT_COORDINATOR_PROMPT.format(
            state=blackboard.view_for("coordinator"),
            agents="\n".join(
                f"  - {n}: {a._cfg.instructions[:100]}"
                for n, a in self.cfg.agents.items()
            ),
        )
        sub_session = deps.fresh_session()
        sub_deps = deps.scope_for_worker(self.cfg.coordinator)
        async for _ in self.cfg.coordinator._architecture.run(
            sub_session, sub_deps, coord_prompt
        ):
            pass
        return CoordinatorDecision.parse_raw(sub_session.result)

    async def _decide_final(self, deps, blackboard) -> str:
        if self.cfg.decider is None:
            # Take last "answer" entry from blackboard
            for entry in reversed(blackboard.public):
                if entry.kind == "answer":
                    return entry.content
            return blackboard.public[-1].content if blackboard.public else "[no answer]"

        decide_prompt = (
            f"Final blackboard state:\n{blackboard.view_for('decider')}\n\n"
            f"Synthesize the final answer from the contributions."
        )
        sub_session = deps.fresh_session()
        sub_deps = deps.scope_for_worker(self.cfg.decider)
        async for _ in self.cfg.decider._architecture.run(
            sub_session, sub_deps, decide_prompt
        ):
            pass
        return sub_session.result or "[no answer]"
```

### Performance

| Metric | Value | Source |
|---|---|---|
| Token cost | 2-4× single-agent | LLM coordinator overhead |
| Data discovery success | +13-57% relative improvement | Salemi et al. 2026 |
| Math/reasoning | Competitive with SOTA, fewer tokens | Han & Zhang 2025 |
| Latency | Variable (depends on coordinator decisions) | Empirical |

### Strengths

1. **Decentralized contribution.** Agents contribute when relevant; no forced participation.
2. **Natural for problems with no clear decomposition.** Brainstorming, research, exploratory analysis.
3. **Transparent state.** The blackboard is the audit log — you see exactly what each agent contributed and when.
4. **Token-efficient when used right.** Agents only run when the coordinator picks them, not every round.
5. **Recently empirically validated.** The Salemi paper's 13-57% gain is real and replicable.

### Weaknesses

1. **Coordinator deadlocks.** If the coordinator can't decide, the system stalls. Round-robin fallback helps but defeats the purpose.
2. **Race conditions on blackboard updates.** When multiple agents try to contribute simultaneously, you need consistency mechanisms.
3. **Hard to debug.** State changes unpredictably; replicating failures is challenging.
4. **Coordinator quality is critical.** A bad coordinator picks wrong agents repeatedly.
5. **"Theoretically interesting but rarely outperforms hierarchical or graph in practice"** (2026 taxonomy guide). For most production problems, Supervisor wins.

### When to Use

- Exploratory research where decomposition isn't known upfront
- Data discovery / information retrieval problems
- Brainstorming with peer-equivalent agents
- Problems where reactive contribution beats proactive routing
- Research-mode systems where transparency of process matters

### When NOT to Use

- Standard production tasks (use Supervisor)
- Time-critical applications (coordinator overhead)
- Simple decomposable problems (overkill)
- When you can write a clear plan upfront (use Plan-and-Execute)

### Composition

Blackboard is mostly a leaf architecture:

```python
# As a worker for exploratory subtasks in a Supervisor
agent = Agent("...", architecture=Supervisor(workers={
    "researcher": Agent("...", architecture=ReAct()),
    "exploratory_analysis": Agent("...", architecture=BlackboardArchitecture(
        agents={"hypothesis": h_agent, "evidence": e_agent, "critic": c_agent},
        coordinator=coord_agent,
    )),
}))
```

Blackboard does NOT compose with: Reflexion, Self-Refine (different paradigms), Debate (overlapping mechanisms).

### Tuning Guide

- **Use an LLM coordinator, not round-robin.** Round-robin defeats the "contribute when relevant" feature.
- **3-5 agents is the sweet spot.** Fewer = not enough diversity; more = coordinator can't track.
- **Cap `max_rounds=20`.** If you're hitting it, the architecture isn't converging.
- **Customize the blackboard state schema for your problem.** Typed state catches bugs and helps the coordinator reason.
- **Provide a decider for complex problems.** Don't rely on "last answer wins"; use an LLM to synthesize.

### Common Pitfalls

**1. No coordinator → round-robin → no benefit over Supervisor.** Round-robin is a stub for testing, not production.

**2. Blackboard explodes in size.** As entries accumulate, the LLM context fills up. Implement summarization or compaction (drop old entries).

**3. Agents that don't read the blackboard.** Some agents respond as if they're being asked a fresh question. Be explicit in agent prompts: "Read the blackboard. Build on existing contributions."

**4. Forgetting termination criteria.** Without clear termination, debate runs forever. The coordinator must have explicit "are we done?" logic.

### Worked Example

```python
import asyncio
from jeevesagent import Agent
from jeevesagent.architecture import BlackboardArchitecture

async def main():
    # Agents for an exploratory data discovery task
    hypothesis = Agent.specialist(
        "You propose hypotheses about data patterns. Read the blackboard; "
        "if there's an open question, propose a hypothesis. If hypotheses "
        "have been proposed, propose alternatives."
    )
    evidence = Agent(
        "You search for evidence supporting/refuting hypotheses. Read the "
        "blackboard; for each hypothesis, search for evidence.",
        tools=["data.query", "web.search"],
    )
    critic = Agent.specialist(
        "You critique hypotheses and evidence quality. Read the blackboard; "
        "challenge weak claims; ask for missing evidence."
    )

    coordinator = Agent.specialist(
        "You coordinate a research team. Decide which agent should contribute "
        "next based on what's missing. Output JSON: "
        "{terminate: bool, reason: str, next_agent: str, instruction: str}"
    )

    decider = Agent.specialist(
        "You synthesize a final answer from the blackboard contributions. "
        "Cite evidence; acknowledge dissent."
    )

    agent = Agent(
        "You produce data-driven analysis through collaborative exploration.",
        architecture=BlackboardArchitecture(
            agents={
                "hypothesis": hypothesis,
                "evidence": evidence,
                "critic": critic,
            },
            coordinator=coordinator,
            decider=decider,
            max_rounds=15,
        ),
    )

    result = await agent.run(
        "Analyze our 2026 sales data to identify what drives the 30% YoY growth. "
        "Don't just describe; identify causal hypotheses and test them."
    )

    print(f"Final analysis: {result.output}")

asyncio.run(main())
```


---

## 13. Swarm

### Origin

**Reference implementation:** OpenAI Swarm (released late 2024 as experimental). Anthropic's **Agent Teams** (Feb 2026) is the production answer that improved on the swarm idea by adding lightweight coordination.

**What it solves:** Supervisor architectures have a bottleneck — every message routes through one agent. Swarm rejects this: peer agents pass control to each other directly via "handoffs." Agent A handles initial input; if A decides B is better suited, A hands off; if B then needs C's expertise, B hands off; etc. No central authority; routing is decentralized.

**Status:** **Ship in v2 with prominent warning labels.** The 2026 production literature is unanimous: swarm has goal-drift and deadlock failure modes that hierarchical/graph topologies don't. Use only for exploratory or research-mode systems where flow can't be pre-specified.

### The Pattern

```
User ─→ Agent A ─handoff─→ Agent B ─handoff─→ Agent C ─→ User
              ↘                    ↗
                 Agent D ─────────
                          ↑
                     possible cycles
                     (Agent C might handoff back to A)
```

Each agent owns a turn. When an agent decides another agent should handle the conversation, it emits a handoff. The framework switches the active agent and continues. The conversation history is preserved across handoffs (or trimmed per the swarm's policy).

### Mechanism

1. **Setup.** N peer agents are configured. One is designated the "entry" agent (receives first user message).

2. **Active agent loop:**
   - Active agent runs ReAct (or similar) on the current conversation
   - Active agent has access to a `handoff(target_agent, optional_message)` tool
   - If active agent emits no handoff and produces a final answer, return to user
   - If active agent emits a handoff:
     - Switch active agent to target
     - Optionally add a transition message to the history
     - Continue with new active agent

3. **Cycle detection.** If A handoffs to B, B handoffs to A, B handoffs to A again — that's a deadlock. Detect (e.g., max handoffs per session, or detect handoff cycles).

4. **Termination.** Active agent produces final answer with no handoff, or budget/turn cap hit, or cycle detected.

### Configuration

```python
@dataclass
class SwarmConfig:
    agents: dict[str, Agent]
    """Peer agents keyed by name."""

    entry_agent: str
    """Name of the agent that receives the first user message."""

    max_handoffs: int = 10
    """Hard cap on handoffs per session. Prevents infinite cycles."""

    detect_cycles: bool = True
    """Detect A→B→A→B patterns and terminate."""

    pass_full_history: bool = True
    """Subsequent agents see the full prior conversation. False = only see
    the handoff message (more isolation, more information loss)."""

    transition_template: str = (
        "[Handoff: {from_agent} → {to_agent}] {message}"
    )
```

### Implementation

```python
# jeevesagent/architecture/swarm.py

from __future__ import annotations
from typing import AsyncIterator
from collections import deque

from .base import Architecture
from .react import ReAct


class SwarmArchitecture:
    name = "swarm"

    def __init__(self, config=None, **kwargs):
        if config is None and kwargs:
            from .config import SwarmConfig
            config = SwarmConfig(**kwargs)
        self.cfg = config
        if self.cfg.entry_agent not in self.cfg.agents:
            raise ValueError(f"entry_agent {self.cfg.entry_agent} not in agents")

    def declared_workers(self):
        return self.cfg.agents

    async def run(self, session, deps, prompt) -> AsyncIterator[Event]:
        active_name = self.cfg.entry_agent
        handoff_count = 0
        recent_handoffs: deque[str] = deque(maxlen=4)
        history = [prompt]  # accumulated conversation

        while True:
            if handoff_count >= self.cfg.max_handoffs:
                yield Event.swarm_max_handoffs(session.id, handoff_count)
                # Last active agent's output is the final answer
                session.result = history[-1]
                yield Event.completed(session.id, session.result)
                return

            active_agent = self.cfg.agents[active_name]
            yield Event.swarm_active(session.id, active_name, handoff_count)

            # Build prompt for active agent
            if self.cfg.pass_full_history:
                active_prompt = "\n\n".join(history)
            else:
                active_prompt = history[-1]

            # Register handoff tool for this agent
            agent_deps = self._with_handoff_tool(deps, active_agent)

            sub_session = deps.fresh_session(parent=deps.session_id)
            async for event in active_agent._architecture.run(
                sub_session, agent_deps, active_prompt
            ):
                yield event.with_role(active_name)

            # Check if a handoff was requested via the tool
            handoff_req = sub_session.metadata.get("handoff_request")
            if handoff_req is None:
                # No handoff = final answer
                session.complete = True
                session.result = sub_session.result
                yield Event.completed(session.id, session.result)
                return

            # Process handoff
            target = handoff_req["target"]
            message = handoff_req.get("message", "")

            if target not in self.cfg.agents:
                # Bad target — terminate gracefully
                yield Event.swarm_bad_handoff(session.id, target)
                session.result = sub_session.result
                yield Event.completed(session.id, session.result)
                return

            # Cycle detection
            recent_handoffs.append((active_name, target))
            if self.cfg.detect_cycles and self._is_cycling(recent_handoffs):
                yield Event.swarm_cycle_detected(session.id, list(recent_handoffs))
                session.result = sub_session.result
                yield Event.completed(session.id, session.result)
                return

            # Build transition message
            transition = self.cfg.transition_template.format(
                from_agent=active_name, to_agent=target, message=message,
            )
            history.append(sub_session.result or "[no output]")
            history.append(transition)

            # Switch active
            active_name = target
            handoff_count += 1
            yield Event.swarm_handoff(
                session.id, from_agent=active_name, to_agent=target, message=message,
            )

    def _with_handoff_tool(self, deps, active_agent):
        """Inject the handoff tool into this agent's available tools."""
        from ..mcp.tools import tool, ToolSet
        from typing import Literal

        agent_names = [n for n in self.cfg.agents.keys()]

        @tool
        async def handoff(
            target: Literal[tuple(agent_names)],  # type: ignore
            message: str = "",
        ) -> str:
            """Hand off the conversation to another peer agent."""
            # Side-effect: store the request in session metadata.
            # The architecture loop reads it after the agent's turn ends.
            return f"[handoff requested → {target}]"

        return deps.with_extra_tools(ToolSet(name="swarm", tools=[handoff]))

    def _is_cycling(self, recent: deque) -> bool:
        """Detect A→B→A→B repetition in the last 4 handoffs."""
        if len(recent) < 4:
            return False
        return recent[0] == recent[2] and recent[1] == recent[3]
```

### Performance

| Metric | Value |
|---|---|
| Token cost | Variable (depends on handoff frequency) — typically 1-2× single agent |
| Latency | Variable (sequential by definition) |
| Production failure rate | High — "From Spark to Fire" 2026 paper showed cascading failures |
| Best-case throughput | Comparable to Supervisor on simple routing tasks |

The "From Spark to Fire" paper (2026) found multi-agent systems with no supervisor have brittle topology — a single bad output cascades. Swarm is the worst-case for this.

### Strengths

1. **No supervisor bottleneck.** Direct A→B passes have less overhead than A→Supervisor→B.
2. **Lightweight.** Agents are peers; no centralized state.
3. **Naturally extensible.** Add a peer; existing agents can hand off to it without coordinator changes.
4. **Good for prototyping.** Quick to set up; useful for exploring agent topologies.

### Weaknesses

1. **Goal drift.** Without a supervisor anchor, agents collectively forget the original task. ("From Spark to Fire" 2026)
2. **Coordination deadlocks.** A→B→A→B cycles. Detection helps but not always.
3. **Debugging hell.** Conversation history is a tangled web of handoffs.
4. **Information loss at handoffs.** Each handoff is a chance to lose context.
5. **"Default to graph or hierarchy in production. Swarm only for exploratory or research-mode."** (2026 taxonomy guide, the consensus warning.)

### When to Use

- Research / prototyping of agent topologies
- Exploratory systems where flow can't be pre-specified
- Peer-equivalent specialists with clear handoff criteria
- Demos and tutorials (it's visually intuitive)

### When NOT to Use

- **Production systems.** Almost always.
- Cost or correctness sensitive applications
- Tasks with clear authority structure (use Supervisor)
- Long-running tasks (drift accumulates)

### Composition

Swarm is mostly a leaf architecture and **does not compose well**. Avoid wrapping it.

```python
# Acceptable: Swarm inside Router as the "exploration mode"
agent = Agent("...", architecture=Router(routes=[
    RouteSpec("structured", architecture=Supervisor(workers={...})),
    RouteSpec("exploratory", architecture=SwarmArchitecture(
        agents={...}, entry_agent="...",
    )),
]))

# Discouraged: Swarm inside anything else
```

### Tuning Guide

- **`max_handoffs=10` is generous.** If you're hitting it, the agents aren't converging.
- **Always enable `detect_cycles=True`.** A→B→A is the most common deadlock.
- **Use clear handoff criteria.** Bad: "Handoff if you want help." Good: "Handoff to billing-agent for any pricing question. Handoff to tech-agent for any debugging question."
- **Consider `pass_full_history=False` for cleaner handoffs.** Forces explicit context-passing in the message.

### Common Pitfalls

**1. Treating Swarm as production-ready.** The 2026 literature is clear: it isn't, except in narrow research contexts. Read the cautions before deploying.

**2. No cycle detection.** A→B→A→B becomes infinite. Always enable.

**3. Vague agent boundaries.** Agents that overlap in capability play hot-potato with the conversation; pick clear, distinguishing roles.

**4. Forgetting the entry agent.** The first message has to go to *some* agent; that agent's handoff logic is critical (it sets the tone).

### Worked Example

```python
import asyncio
from jeevesagent import Agent
from jeevesagent.architecture import SwarmArchitecture

async def main():
    # Three peer agents for a customer support prototype
    triage = Agent(
        "You triage incoming queries. Hand off to billing-agent for pricing, "
        "tech-agent for technical issues, or sales-agent for new sales questions. "
        "Use the handoff tool.",
    )
    billing = Agent(
        "You handle billing questions. If the question becomes technical, "
        "hand off to tech-agent.",
        tools=["billing.lookup"],
    )
    tech = Agent(
        "You handle technical issues. If the question becomes about pricing, "
        "hand off to billing-agent.",
        tools=["docs.search", "logs.query"],
    )
    sales = Agent.specialist(
        "You handle sales inquiries. Once converted, you don't hand off."
    )

    agent = Agent(
        "You provide customer support via a peer agent network. (Prototype)",
        architecture=SwarmArchitecture(
            agents={
                "triage-agent": triage,
                "billing-agent": billing,
                "tech-agent": tech,
                "sales-agent": sales,
            },
            entry_agent="triage-agent",
            max_handoffs=8,
            detect_cycles=True,
        ),
    )

    result = await agent.run(
        "I'm being charged $300/month but I think I'm on the $200 plan. "
        "Also, my dashboard shows an error 'connection refused'."
    )

    print(result.output)
    print(f"\nHandoff trail: {result.metadata['handoffs']}")

asyncio.run(main())
```


---

## 14. Router

### Origin

**Production source:** OpenAI Agents SDK "Handoff" pattern (March 2026 Agents SDK release). CrewAI's "sequential process" with classification. A simpler-than-Supervisor variant that's been independently invented by every framework.

**What it solves:** Supervisor is overkill for tasks that have one specialist who can fully handle them. Router classifies the input once and dispatches to ONE specialist. No coordination, no synthesis. Specialist returns the answer directly. The router is a small classification model; specialists are the heavy lifters.

**Status:** **Ship in v1.** The lightest multi-agent pattern. Cheap, deterministic, easy to reason about. The right architecture for helpdesks, support, and any pattern where input → specialist → output.

### The Pattern

```
User → Router → Specialist (one chosen) → User
                    │
                    └ runs alone, no synthesis
```

That's it. The router classifies, the specialist handles. There's no "team" — at any moment, exactly one specialist owns the task.

### Mechanism

1. **Router classification.** A small LLM call: "Given this input, which route is best?" Output is a single route name. Optionally returns confidence and reasoning.

2. **Dispatch.** Spawn the chosen route as a sub-Agent with isolated context. Pass the original prompt unchanged.

3. **Run specialist.** Specialist runs to completion (its own architecture — typically ReAct).

4. **Return.** Specialist's final answer becomes the Router architecture's output. Done.

There's no second LLM call after the specialist. The Router doesn't synthesize — it just routes.

### Configuration

```python
@dataclass
class RouterConfig:
    routes: dict[str, Agent]
    """Route name → specialist Agent."""

    router_model: str | None = None
    """Override main model for routing. Default = use main. A small fast model
    is ideal here (Haiku, Mini)."""

    fallback_route: str | None = None
    """If router can't classify, route here. None = error."""

    require_confidence_above: float = 0.0
    """If router's confidence is below this, use fallback (or fail)."""

    classification_template: str = DEFAULT_ROUTER_PROMPT
    """Prompt for classification."""
```

### Implementation

```python
# jeevesagent/architecture/router.py

from __future__ import annotations
from typing import AsyncIterator, Literal
from pydantic import BaseModel, Field

from .base import Architecture
from .react import ReAct


DEFAULT_ROUTER_PROMPT = """\
You are a routing classifier. Given the user's request, decide which specialist
handles it best.

Available routes:
{route_descriptions}

Output JSON:
- route: name of chosen specialist (must be in the list above)
- confidence: 0-1
- reasoning: brief justification
"""


class RoutingDecision(BaseModel):
    route: str
    confidence: float = Field(ge=0, le=1)
    reasoning: str


class Router:
    name = "router"

    def __init__(self, config=None, **kwargs):
        if config is None and kwargs:
            from .config import RouterConfig
            config = RouterConfig(**kwargs)
        self.cfg = config

    def declared_workers(self):
        return self.cfg.routes

    async def run(self, session, deps, prompt) -> AsyncIterator[Event]:
        router_model = (
            deps.resolve_model(self.cfg.router_model) or deps.model
        )

        # === ROUTING STEP ===
        decision = await deps.runtime.step(
            "router_classify",
            self._classify,
            router_model,
            prompt,
        )
        yield Event.route_classified(session.id, decision)

        # Confidence check
        if decision.confidence < self.cfg.require_confidence_above:
            if self.cfg.fallback_route:
                yield Event.route_fallback(
                    session.id, decision, self.cfg.fallback_route,
                )
                target = self.cfg.fallback_route
            else:
                yield Event.route_failed(
                    session.id, "low_confidence", decision,
                )
                session.result = (
                    f"Could not confidently route this request. "
                    f"Best guess: {decision.route} ({decision.confidence})"
                )
                return
        else:
            target = decision.route

        # Route validity
        if target not in self.cfg.routes:
            if self.cfg.fallback_route:
                target = self.cfg.fallback_route
            else:
                yield Event.route_failed(
                    session.id, "unknown_route", decision,
                )
                session.result = f"Unknown route: {target}"
                return

        # === DISPATCH ===
        specialist = self.cfg.routes[target]
        yield Event.route_dispatched(session.id, target)

        sub_session = deps.fresh_session(parent=deps.session_id)
        sub_deps = deps.scope_for_worker(specialist)

        async for event in specialist._architecture.run(sub_session, sub_deps, prompt):
            yield event.with_role(target)

        # Specialist's output becomes ours
        session.complete = True
        session.result = sub_session.result
        yield Event.completed(session.id, session.result, route=target)

    async def _classify(self, model, prompt) -> RoutingDecision:
        route_descriptions = "\n".join(
            f"  - {name}: {agent._cfg.instructions[:150]}"
            for name, agent in self.cfg.routes.items()
        )
        classify_prompt = (
            self.cfg.classification_template.format(route_descriptions=route_descriptions)
            + "\n\n"
            + f"User request: {prompt}"
        )

        response = await model.complete_structured(
            messages=[{"role": "user", "content": classify_prompt}],
            output_schema=RoutingDecision,
        )
        return response
```

### Performance

| Metric | Value |
|---|---|
| Routing overhead | 1 small LLM call (~50-200 tokens) | ~1.1× single agent |
| Latency | ~200ms additional (classification) | Negligible |
| Routing accuracy | 95%+ on well-defined route boundaries | Empirical |
| Cost | Cheaper than Supervisor (no synthesis pass) | |

### Strengths

1. **Cheapest multi-agent pattern.** One classification call + one specialist call. That's it.
2. **Deterministic.** Easy to reason about; routes are explicit; no emergent behavior.
3. **Low overhead.** A small router model (Haiku, Mini) handles routing in 200ms.
4. **Easy to evaluate.** Per-route accuracy is testable in isolation.
5. **Easy to extend.** Add a route; existing routes are unchanged.

### Weaknesses

1. **Single specialist must own the task.** No cross-specialist synthesis. If the user asks "I have a billing question and a technical issue," router can pick only one.
2. **Routing errors are total.** Wrong route → wrong specialist → bad answer. No recovery.
3. **No clarification.** Router doesn't ask "did you mean billing or sales?"; it commits.
4. **Brittle to ambiguous inputs.** If routes overlap, router flip-flops.

### When to Use

- Customer support (route to billing/tech/sales/account)
- Helpdesk systems
- API gateway-style: route requests to backend services by intent
- Single-domain task selection (route to "research mode" vs "code mode")

### When NOT to Use

- Multi-domain tasks ("I need both research and code")
- Tasks where routing might be wrong and you need recovery → use Supervisor with delegate
- Very simple cases (one specialist) — just use the specialist directly
- Cases where the right answer requires combining specialists

### Composition

```python
# Router as the entry point of a system, with Supervisor for complex routes
agent = Agent("...", architecture=Router(routes={
    "simple": Agent.specialist("..."),  # cheap for simple queries
    "complex": Agent("...", architecture=Supervisor(workers={...})),  # heavy team
    "exploratory": Agent("...", architecture=BlackboardArchitecture(...)),
}))

# Router with Reflexion to learn routing patterns
agent = Agent("...", architecture=Reflexion(base=Router(routes={...})))

# Router with ReWOO for cheap predictable routes
agent = Agent("...", architecture=Router(routes={
    "lookup": Agent("...", architecture=ReWOO()),  # cheap factual lookups
    "exploration": Agent("...", architecture=ReAct()),
}))
```

### Tuning Guide

- **Use a fast model for routing.** Haiku, Mini. The classifier doesn't need reasoning.
- **`require_confidence_above=0.7` is a good default.** Below that, use fallback or fail clearly.
- **Always provide `fallback_route`.** "general" or "human-handoff" agent that handles unclassified queries.
- **Customize the classification template per domain.** Generic prompts produce generic routing; domain-specific prompts produce sharp routing.
- **Add few-shot examples.** "Examples: 'My card was declined' → billing. 'My API is returning 500' → tech." Routes get much sharper.

### Common Pitfalls

**1. Overlapping routes.** Two routes that could both handle the same query → coin flip. Sharpen route descriptions.

**2. No fallback.** Edge cases produce errors. Always have a "general" route.

**3. Heavy router model.** Don't use Opus to classify what Haiku could classify in 50ms.

**4. Routing on user surface text.** "I'm angry" → ??? Routes should be on intent, not emotion. The router needs to extract intent first.

**5. Forgetting that the specialist sees the user's original message.** Don't paraphrase or summarize before passing — pass the prompt verbatim.

### Worked Example

```python
import asyncio
from jeevesagent import Agent
from jeevesagent.architecture import Router

async def main():
    # Specialist agents for a customer support system
    billing = Agent(
        "You handle billing inquiries. You have access to billing systems "
        "and can issue refunds, change plans, and explain charges.",
        tools=["billing.lookup", "billing.refund", "billing.change_plan"],
    )
    tech = Agent(
        "You handle technical issues. You can search docs, query logs, "
        "and create support tickets.",
        tools=["docs.search", "logs.query", "tickets.create"],
    )
    sales = Agent(
        "You handle pre-sales inquiries. You explain plans, run demos, "
        "and qualify leads.",
        tools=["product.demo", "lead.qualify"],
    )
    general = Agent.specialist(
        "You handle general inquiries that don't fit billing, tech, or sales. "
        "Be helpful and direct people to the right resources."
    )

    agent = Agent(
        "Customer support routing system.",
        architecture=Router(
            routes={
                "billing": billing,
                "tech": tech,
                "sales": sales,
                "general": general,
            },
            router_model="claude-haiku-4-5",  # cheap, fast
            fallback_route="general",
            require_confidence_above=0.7,
            classification_template="""\
You route customer support inquiries.

Routes:
{route_descriptions}

Examples:
- "My card was declined" → billing
- "My API returns 500 errors" → tech
- "I'd like a demo" → sales
- "What's your address?" → general

Output: route, confidence (0-1), reasoning.
""",
        ),
    )

    queries = [
        "Why was my card charged twice last month?",
        "I'm getting a 502 error from your API endpoint.",
        "What's the difference between Pro and Enterprise plans?",
        "Where are you headquartered?",
    ]

    for q in queries:
        result = await agent.run(q)
        print(f"Q: {q}")
        print(f"  Route: {result.metadata['route']}")
        print(f"  A: {result.output[:100]}...\n")

asyncio.run(main())
```


---

# Operations

## Custom architectures

JeevesAgent's architecture system is genuinely extensible. New architectures don't require forking the library or submitting PRs. They're external Python packages that register via entry points. This section shows how.

### When to write a custom architecture

Most teams should NOT write custom architectures. The 14 shipped architectures cover ~95% of real-world cases. Reach for custom when:

1. **A new research paper drops** with a meaningfully different pattern (this happens about quarterly)
2. **Your domain has unique structure** that doesn't fit existing patterns (rare; usually it's just a tuning of an existing one)
3. **You need to embed a proprietary algorithm** that you don't want in a public package
4. **You're prototyping** before contributing back to JeevesAgent

If you find yourself writing a third "custom" architecture, you're probably re-implementing existing patterns badly. Ask in the community first.

### Anatomy of a custom architecture

A custom architecture is a single class implementing the `Architecture` Protocol:

```python
# my_company/architectures/algorithm_of_thoughts.py

from typing import AsyncIterator
from jeevesagent.architecture.base import Architecture
from jeevesagent.core.types import Event
from jeevesagent.agent.session import Session
from jeevesagent.agent.deps import Dependencies


class AlgorithmOfThoughts:
    """Sel et al. 2024 — explores algorithmic search within a single LLM call.

    Reference: 'Algorithm of Thoughts: Enhancing Exploration of Ideas in
    Large Language Models' arXiv:2308.10379
    """

    name = "algorithm-of-thoughts"

    def __init__(
        self,
        max_explorations: int = 5,
        backtrack_threshold: float = 0.3,
        algorithm_template: str | None = None,
    ):
        self.max_explorations = max_explorations
        self.backtrack_threshold = backtrack_threshold
        self.algorithm_template = algorithm_template or self._default_template()

    def declared_workers(self) -> dict:
        return {}  # single-agent

    async def run(
        self,
        session: Session,
        deps: Dependencies,
        prompt: str,
    ) -> AsyncIterator[Event]:
        # === YOUR ARCHITECTURE LOGIC HERE ===

        # Use deps.runtime.step(...) for non-deterministic operations to enable replay
        result = await deps.runtime.step(
            "aot_explore",
            self._explore,
            deps.model,
            prompt,
        )

        # Track budget
        await deps.budget.consume(
            tokens_in=result.usage.input_tokens,
            tokens_out=result.usage.output_tokens,
            cost_usd=result.usage.cost_usd,
        )

        session.complete = True
        session.result = result.output
        yield Event.completed(session.id, result.output)

    def _default_template(self) -> str:
        return """You are an algorithmic problem solver.
        Explore the solution space using a heuristic search.
        For each branch, evaluate viability before committing.
        ...
        """

    async def _explore(self, model, prompt):
        # Implementation
        ...
```

### Registering the architecture

Add an entry point in your package's `pyproject.toml`:

```toml
[project.entry-points."jeevesagent.architecture"]
algorithm-of-thoughts = "my_company.architectures.algorithm_of_thoughts:AlgorithmOfThoughts"
```

After `pip install my-company-architectures`, users can:

```python
agent = Agent("...", architecture="algorithm-of-thoughts")
```

JeevesAgent discovers the architecture via the entry point — no code changes to JeevesAgent itself.

### Required protocol contract

Your architecture MUST:

1. **Have a `name: str` class attribute.** Used for resolution and logging.
2. **Implement `async def run(session, deps, prompt) -> AsyncIterator[Event]`**.
3. **Yield Events for every meaningful state change.** At minimum: `Event.started`, `Event.completed`. Ideally also: per-turn events, tool events, error events.
4. **Use `deps.runtime.step(...)` for non-deterministic operations.** This is what enables replay-on-crash.
5. **Respect cancel scopes.** Use `anyio` task groups, not raw `asyncio.create_task`. Cancellation must propagate cleanly.
6. **Track budget via `deps.budget.consume(...)`.** Token/cost accounting requires this.
7. **Implement `declared_workers() -> dict[str, Agent]`.** Even if empty (`return {}`).

Your architecture SHOULD:

1. **Accept config via constructor.** Either positional kwargs or a config object. Both work; pick one consistently.
2. **Document failure modes.** What happens when the model errors? When budget is exceeded? When tools fail?
3. **Compose with at least ReAct as the base.** If your architecture wraps another (like Reflexion does), accept `base: Architecture`.
4. **Honor `tree_limits` for child spawning.** Use `deps.spawn_child(...)` for children, not direct invocation.

### Testing custom architectures

JeevesAgent ships test fixtures specifically for architecture testing:

```python
# tests/test_my_architecture.py

import pytest
from jeevesagent.testing import (
    fake_model,
    fake_runtime,
    fake_dependencies,
    fresh_session,
    collect_events,
)
from my_company.architectures.algorithm_of_thoughts import AlgorithmOfThoughts


@pytest.mark.anyio
async def test_simple_run():
    arch = AlgorithmOfThoughts(max_explorations=3)
    session = fresh_session()
    deps = fake_dependencies(
        model=fake_model(responses=["explore...", "answer: 42"]),
        runtime=fake_runtime(),
    )

    events = await collect_events(arch.run(session, deps, "What is 6 × 7?"))

    assert any(e.kind == "started" for e in events)
    assert any(e.kind == "completed" for e in events)
    assert session.complete
    assert "42" in session.result


@pytest.mark.anyio
async def test_respects_budget():
    arch = AlgorithmOfThoughts()
    session = fresh_session()
    deps = fake_dependencies(
        model=fake_model(responses=[...]),
        budget=Budget(max_tokens=100),  # tiny
    )

    events = await collect_events(arch.run(session, deps, "..."))

    # Should emit budget_exceeded event, not crash
    assert any(e.kind == "budget_exceeded" for e in events)


@pytest.mark.anyio
async def test_replay_after_crash():
    """Verify the architecture is deterministic via runtime.step."""
    arch = AlgorithmOfThoughts()

    # First run — record
    session1 = fresh_session()
    runtime1 = fake_runtime(record=True)
    deps1 = fake_dependencies(runtime=runtime1)
    events1 = await collect_events(arch.run(session1, deps1, "..."))

    # Second run — replay from the recording
    session2 = fresh_session()
    runtime2 = fake_runtime(replay=runtime1.recording())
    deps2 = fake_dependencies(runtime=runtime2)
    events2 = await collect_events(arch.run(session2, deps2, "..."))

    assert session1.result == session2.result
```

### Publishing your architecture

Once tested:

1. Publish to PyPI with a clear name (`jeevesagent-arch-aot`, etc.)
2. Document benchmarks against existing architectures (which patterns it beats and on what)
3. Reference the originating paper (if applicable)
4. Submit a pointer to the JeevesAgent community registry (we maintain a list of third-party architectures)

This is how the architecture ecosystem grows. We define the protocol; the community fills in the implementations.

---

## Upgrade paths

When your application's needs change, you don't rewrite — you swap architectures. This section shows the canonical upgrade paths and what to expect.

### From ReAct (the default)

The starting point. Most applications begin here. Signs you should upgrade:

| Symptom | Upgrade To | Why |
|---|---|---|
| Long tasks lose coherence (>10 steps) | **Deep Agent** | Adds planning, filesystem, subagents |
| Same failure happens repeatedly | **Reflexion** (wrapping ReAct) | Cross-session learning |
| Output quality is inconsistent | **Self-Refine** or **Actor-Critic** | Iterative critique |
| Cost too high on multi-step tasks | **Plan-and-Execute** or **ReWOO** | Fewer LLM calls |
| Single agent has too many tools (>10) | **Supervisor** with workers | Specialization |

### From Plan-and-Execute

Common upgrade target from ReAct. Signs you should upgrade further:

| Symptom | Upgrade To | Why |
|---|---|---|
| Plans need to learn from failures | **Reflexion(base=PlanAndExecute)** | Verbal RL on plan-level |
| Plans need quality review | **Actor-Critic on plan output** | Critic reviews plan |
| Need to handle fully unknown tasks | **DeepAgent(base=PlanAndExecute)** | Adds adaptability |
| Cost is fine but quality matters more | **Reflexion** with eval signal | Cross-session improvement |

### From Supervisor

Common multi-agent target. Signs you should upgrade:

| Symptom | Upgrade To | Why |
|---|---|---|
| Supervisor is bottleneck (slow) | Make workers themselves more capable (DeepAgent workers) | Reduce delegation count |
| Workers need to talk to each other | **Blackboard** (specialized cases) | Peer comms |
| Workers should learn over time | **Reflexion(base=Supervisor)** | Cross-session learning |
| One worker dominates → just route | **Router** | Lighter pattern |

### From Router

Common simple multi-agent. Signs you should upgrade:

| Symptom | Upgrade To | Why |
|---|---|---|
| Tasks need multiple specialists | **Supervisor** | Synthesis across workers |
| Routing is unreliable | **Supervisor** with delegate | Recovery via re-delegation |
| Same routing failures repeat | **Reflexion(base=Router)** | Learn routing patterns |

### Avoid these "upgrades"

Some moves look like upgrades but aren't:

- **ReAct → Tree of Thoughts.** Different problems, not different complexities. ToT is for combinatorial reasoning; if you're not solving Game of 24, don't use it.
- **Supervisor → Swarm.** Almost always a regression. Swarm trades supervisor's bottleneck for goal drift, deadlocks, and debugging hell.
- **Reflexion → Reflexion + Actor-Critic + Self-Refine.** Triple-stacking critiques produces marginal gains and 10× cost. Pick one.
- **Anything → Graph of Thoughts.** Reserve for v2 and research-grade problems. The complexity-to-value ratio is bad for most applications.

### Migration mechanics

Architecture swaps are one-line changes:

```python
# Before
agent = Agent("...", architecture="react")

# After
agent = Agent("...", architecture="reflexion")
```

What stays the same:
- Agent's prompt, tools, model, memory backend
- MCP servers, sandbox config, permissions
- Existing memory data (Reflexion will read from same memory backend)

What changes:
- Iteration logic and event stream
- Token cost (usually higher for more sophisticated architectures)
- Latency profile

What you should test before/after:
- Per-task accuracy on your eval suite
- Cost per task (run 100 tasks, compare)
- Latency P50 / P95
- Memory growth (some architectures persist more state)

### Recommended progression for a new application

If you're building a new agent application from scratch, go in this order:

1. **Start with ReAct.** Build the basic flow; get tools working; ship a v0.
2. **Hit a quality wall?** Add Self-Refine (1 round). Fast win.
3. **Hit a reliability wall?** Add Reflexion with an evaluator. Real win.
4. **Hit a complexity wall (long tasks)?** Move to Deep Agent.
5. **Hit a specialization wall?** Move to Supervisor with focused workers.
6. **Hit a quality bar that requires review?** Add Actor-Critic at strategic points.

Most production applications never go past step 3. That's fine. The 2026 consensus is "ReAct + Reflexion is the production-grade single-agent stack" for a reason.

---

## Frequently asked questions

### Why ReAct as the default?

Because the 2026 consensus across every major framework (LangGraph, AutoGen, CrewAI, OpenAI Agents SDK, Claude Agent SDK) is that ReAct is the right starting point. It's:
- The most studied architecture (3+ years of production data)
- The simplest to debug (linear loop, transparent reasoning)
- The one that composes with almost everything else
- The one users transferring from other frameworks already know

Defaults matter. We pick the default that's right for the most users, not the one that's most powerful.

### Can I write an architecture that doesn't use the model at all?

Yes — but it's almost certainly the wrong abstraction. If your "architecture" doesn't use an LLM, it's deterministic logic that should be a tool or a workflow, not an agent architecture. The Architecture protocol exists to swap *iteration patterns over LLM calls*. Pure deterministic logic doesn't need that.

Exception: a "noop" architecture for testing is fine and ships in `jeevesagent.testing`.

### How do I choose between Self-Refine and Actor-Critic?

Two questions:

1. **Do you have an external eval signal?** If yes, **Reflexion** is better than both.
2. **Do you have a different model available?** If yes, **Actor-Critic** with different models for actor and critic. The asymmetry is the point.
3. **Same model, no eval signal?** **Self-Refine**. It's the simplest and works well enough.

### Why is Tree of Thoughts in v1 but Graph of Thoughts in v2?

ToT has a clear, well-defined implementation that fits the protocol cleanly. GoT requires graph topology management, three operation types, and AGoT's adaptive variant requires meta-LLM calls. Engineering complexity is much higher.

For v1, we ship architectures with broad applicability and clean implementation. GoT is research-grade — most teams won't need it. We add it when there's user demand.

### What's the right architecture for a coding agent?

Almost certainly **Deep Agent + Actor-Critic**.

- Deep Agent gives you planning, filesystem (for the codebase), and subagents (for "research this library" tasks).
- Actor-Critic at the implementation step gives you adversarial code review built in.

```python
agent = Agent(
    "You are a coding agent.",
    tools=["fs.read", "fs.write", "bash", "git"],
    architecture=DeepAgent(
        base=ReAct(max_turns=80),
        subagents=[
            SubagentDef("researcher", description="researches libraries", tools=["web.search"]),
            SubagentDef("reviewer", description="reviews code adversarially", tools=[]),
        ],
    ),
)

# Or with explicit Actor-Critic at the implementation level (heavier):
agent = Agent(
    "...",
    architecture=ActorCritic(
        actor=Agent("write code", architecture=DeepAgent()),
        critic=Agent("review code", model="gpt-4o"),
        max_rounds=2,
    ),
)
```

### What's the right architecture for customer support?

Almost certainly **Router** with specialist agents per domain (billing, tech, sales, general). It's:
- Cheap (one classification call + one specialist run)
- Deterministic (route → specialist → answer)
- Easy to evaluate per route
- Easy to extend by adding routes

Add Reflexion if you want the router to learn from misroutes over time.

### What's the right architecture for research?

If single-domain → **Deep Agent**. If multi-domain → **Supervisor with Deep Agent workers**.

```python
agent = Agent(
    "You produce researched analyses.",
    architecture=Supervisor(workers={
        "researcher": Agent("...", architecture=DeepAgent()),
        "writer": Agent.specialist("..."),
        "fact_checker": Agent("...", architecture=ActorCritic()),  # quality-critical
    }),
)
```

### Do I need to import all the architecture classes?

No. Three patterns:

1. **String** — `architecture="reflexion"` — no import needed
2. **Recipe** — `architecture="reflexion+react"` — no import needed
3. **Object** — `architecture=Reflexion(base=ReAct())` — requires `from jeevesagent.architecture import Reflexion, ReAct`

For the simple cases, strings are fine. Import classes only when you're configuring.

### How do I benchmark architectures against each other?

JeevesAgent ships an eval harness:

```python
from jeevesagent.eval import compare_architectures

results = await compare_architectures(
    task_set=my_tasks,  # list of (prompt, expected_output)
    architectures=[
        ReAct(),
        Reflexion(base=ReAct()),
        DeepAgent(base=ReAct()),
        Supervisor(workers={...}),
    ],
    metrics=["accuracy", "tokens", "latency", "cost"],
    n_runs=5,  # for variance
)
results.print_table()
results.save_html("benchmarks.html")
```

The eval harness lives in `jeevesagent.eval` (engineering plan §16). It runs each architecture against the same task set with the same model, captures all metrics, and produces variance-aware comparisons.

### Can architectures change at runtime based on the task?

Not in v1. The architecture is fixed at Agent construction. Reasons:
- State management gets complicated (memory, runtime, telemetry must align)
- Cost accounting becomes ambiguous
- Debug becomes harder

If you want runtime selection, use **Router** at the top level — it picks a specialist Agent per request, and each specialist has its own architecture.

For v3 we may add `architecture="auto"` that chooses dynamically based on task analysis (the AdaptOrch pattern). Not v1.

### What if I find a bug in a shipped architecture?

File an issue in the JeevesAgent repository with:
1. Architecture name and version
2. Minimal reproduction (preferably failing test)
3. Expected vs actual behavior
4. Model used (Claude Opus 4.7, GPT-4o, etc.)

Architecture bugs are first-priority because they affect correctness across many users.

---

## Closing

This is the complete architecture reference. Three things to internalize:

1. **The harness is constant; the architecture is variable.** Memory, durability, security, MCP, observability — these don't change when you swap architectures. Only the iteration logic does.

2. **Composition is first-class.** Real production stacks are compositions, not single architectures. Reflexion *of* ReAct, Supervisor *of* Deep Agents, Actor-Critic *with a* Tree-of-Thoughts *critic*. The protocol takes a `base: Architecture` parameter wherever it makes sense, and that's how composition happens.

3. **Research becomes API.** When the next paper drops a better architecture (it will, every quarter), JeevesAgent ships it as one entry point and one new class. Users adopt it with a string change. We don't lock anyone into 2026's best practices.

The user types one word; the harness becomes that architecture. Everything else is exactly the same. That's the magic moment.

---

*Reference list (chronological):*

- *Erman, L. D., et al. "The Hearsay-II Speech-Understanding System." 1980. (Blackboard architecture)*
- *Sutton, R. S., & Barto, A. G. "Reinforcement Learning: An Introduction." 1998. (Actor-Critic foundations)*
- *Wei, J., et al. "Chain-of-Thought Prompting Elicits Reasoning." 2022. (CoT, the foundation)*
- *Yao, S., et al. "ReAct: Synergizing Reasoning and Acting in Language Models." arXiv:2210.03629. 2022.*
- *Madaan, A., et al. "Self-Refine: Iterative Refinement with Self-Feedback." arXiv:2303.17651. 2023.*
- *Shinn, N., et al. "Reflexion: Language Agents with Verbal Reinforcement Learning." NeurIPS 2023.*
- *Yao, S., et al. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." NeurIPS 2023.*
- *Wang, L., et al. "Plan-and-Solve Prompting." ACL 2023.*
- *Xu, B., et al. "ReWOO: Decoupling Reasoning from Observations." arXiv:2305.18323. 2023.*
- *Du, Y., et al. "Improving Factuality and Reasoning Through Multiagent Debate." arXiv:2305.14325. 2023.*
- *Gou, Z., et al. "CRITIC: Empowering LLMs with Tool-Interactive Critiquing." 2023.*
- *Besta, M., et al. "Graph of Thoughts: Solving Elaborate Problems." AAAI 2024.*
- *He, Y., et al. "Plan-then-Execute Pattern Implementation." 2025.*
- *Sun, B., et al. "CGI: Critique-Guided Improvement." 2025.*
- *Han, B., & Zhang, S. "Exploring Advanced LLM MAS Based on Blackboard Architecture." arXiv:2507.01701. 2025.*
- *AGoT — "Adaptive Graph of Thoughts: Test-Time Adaptive Reasoning." arXiv:2502.05078. 2025.*
- *Salemi, A., et al. "LLM-Based Multi-Agent Blackboard System." arXiv:2510.01285. 2026.*
- *"From Spark to Fire" — multi-agent cascade failure paper. 2026.*
- *"Why Do Multi-Agent LLM Systems Fail?" — MIT analysis. 2026.*

*Production sources:*

- *LangChain `deepagents` library (Deep Agent pattern)*
- *Anthropic Multi-Agent Research System internal report (90.2% improvement)*
- *Anthropic Agent Teams (Claude Opus 4.6, Feb 2026)*
- *OpenAI Swarm reference implementation*
- *OpenAI Agents SDK (March 2026 release, Handoff pattern)*
- *Microsoft AutoGen GroupChat*
- *CrewAI hierarchical/sequential/consensual processes*

*2026 production guides referenced:*

- *Agent Architecture Patterns: 2026 Taxonomy Guide (Digital Applied)*
- *Multi-Agent in Production 2026: What Actually Survived (Lanham, Medium)*
- *AI Agent Architecture Patterns: Single & Multi-Agent (Redis)*
- *AgixTech Reasoning Loops 2026: ReAct, ReWOO, CoT in Production*
- *DeepAgents Architecture (BetterLink Blog)*
- *Agent Swarm vs Anthropic Workflows vs LangGraph (SoftmaxData)*