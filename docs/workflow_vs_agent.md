# Workflow vs Agent

Loom ships two peer primitives. Picking between them is an
engineering decision, not a stylistic one. Wrong pick → wrong cost,
wrong latency, wrong debug story.

This page is the rubric. Read it once, save the link, settle the
question on every project before you write code.

## The two patterns

| | **Workflow** | **Agent** |
|---|---|---|
| Who controls the path | Developer (Python `if`s, edges) | LLM (tool calls + loop) |
| Predictability | Same input → same path | LLM picks the path each time |
| Cost | Low — no LLM call to decide what's next | High — every decision is an LLM call |
| Latency | Lower (deterministic) | Higher (loop iterations) |
| Debuggability | Easy — set a breakpoint at step N | Hard — replay the conversation |
| Compliance / audit | Easy — graph IS the spec | Hard — emergent behaviour to prove |
| Open-ended tasks | Poorly | Well |
| Adapts to new info | Only at branches you anticipated | Yes |

Neither column is universally better. Real production systems
usually need both, often in the same request path.

## Pick a Workflow if…

* You can draw the graph on a whiteboard before writing code.
* The branching depends on **data**, not on conversation.
* You need an auditor to read the steps without running the system.
* Latency matters and an extra LLM call to "decide what's next" is
  wasted work.
* The same input must produce the same path (compliance, billing,
  approval flows).
* You're using an LLM to decide between two hardcoded branches
  ("if user wants A do X, if user wants B do Y") — that's a
  classifier step inside a workflow, not an agent.

## Pick an Agent if…

* The user can ask anything; the model has to figure out what
  matters.
* The next step depends on what was found in the last step.
* You'd need a long `if/elif` chain to enumerate the cases and
  most of them are speculative.
* You need ReAct-style "think → act → observe → think" loops over
  a tool surface.

## Use both (the most common shape)

Most production systems combine the two:

* **Outer workflow for the deterministic skeleton** — triage,
  routing, audit, compliance.
* **Inner agent for the open-ended specialist** — the part where
  reasoning matters.
* Or the inverse: **outer agent calls a workflow as a tool** when
  it needs to perform a deterministic compliance flow ("submit
  refund request" with strict steps) inside an otherwise
  open-ended conversation.

The framework makes both directions free:

```python
from loomflow import Agent, Workflow

# Agent inside a Workflow — drop the Agent in as a node.
specialist = Agent("billing specialist", model="...")
wf = Workflow.route(
    classify,
    {"billing": specialist, "tech": tech_specialist, "general": general},
)
result = await wf.run("I can't see my invoice", user_id="alice")

# Workflow inside an Agent — expose the workflow as a tool.
refund = Workflow.chain([validate, create_record, notify_customer])
agent = Agent(
    "Customer support",
    model="...",
    tools=[refund.as_tool(name="process_refund")],
)
```

Both share the same observability spine — `RunContext`,
`user_id` partition, telemetry spans, audit-log entries — so a
single trace shows exactly which decisions were workflow-
deterministic and which were LLM-driven.

## Within Workflow: which constructor for which case?

Once you've decided you want a workflow, the next question is
which *shape*. Pick the simplest that fits — most users only ever
need the sugar constructors.

| Situation | Use | Why |
|---|---|---|
| Two or three sequential `await` calls, no branching, no observability needs | **Plain Python** (no framework) | The Workflow primitive earns its weight in observability + audit + cycles. Without those, a plain `async def` is shorter and clearer. |
| Linear sequence, you want telemetry / audit / per-step events | `Workflow.chain([fn_a, fn_b, fn_c])` | One call. Each step's return is the next step's input. |
| Classify input, dispatch to one of N specialists, terminate | `Workflow.route(classifier, {"a": h_a, "b": h_b}, default=...)` | One classifier, one fork, that's it — sugar for the most common branching pattern. |
| Fan-out: run all N steps with the same input, merge results | `Workflow.parallel([s_1, s_2, s_3], merge=combine)` | `anyio` task group runs them concurrently. |
| Anything with cycles, mid-graph branching, multi-stage routing, conditional entry | Explicit `add_node` + `add_edge` + `add_router` (+ `START` / `END` sentinels) | The graph becomes the artifact. Use `wf.to_mermaid()` to inspect. |

**The hidden option is plain Python.** If you have three steps,
no branching, and don't need audit / streaming / per-step
telemetry, `await step_c(await step_b(await step_a(x)))` is a
better fit than any framework call. Workflow earns its weight on
*observability*, *cycles*, and *graph-as-artifact* — not on
sequential composition alone.

## Three ways to write a workflow, ordered by ceremony

### 1. Plain Python with `@step` (recommended for short flows)

No DSL. Decorate the functions, write the workflow as regular
async Python:

```python
from loomflow import step

@step
async def classify(text: str) -> str:
    return (await classifier_agent.run(text)).output

@step
async def respond(text: str, label: str) -> str:
    return (await get_specialist(label).run(text)).output

async def triage(text: str, user_id: str) -> str:
    label = await classify(text)
    return await respond(text, label)
```

The `@step` decorator is transparent when called outside a
workflow context — runs the function with zero overhead. Inside a
workflow it adds telemetry spans + audit entries automatically.

### 2. Sugar constructors (most common shapes)

One call makes a fully-wired workflow with no graph builder:

```python
# Linear sequence
wf = Workflow.chain([step_a, step_b, step_c])

# Classify-then-dispatch
wf = Workflow.route(classify, {"a": handler_a, "b": handler_b}, default=fallback)

# Fan-out, run all, merge
wf = Workflow.parallel([s1, s2, s3], merge=combine)
```

Each step can be an `async def`, a sync function, an `Agent`
instance, or a nested `Workflow`. Pick whatever returns the right
shape; the framework coerces.

### 3. Explicit graph builder (for the cases where the graph IS the artifact)

When you need conditional edges, multiple branches, or the graph
itself is a deliverable (compliance, BPMN-like flows):

```python
from loomflow import Workflow, START, END

wf = Workflow("triage")
wf.add_node("billing", billing_agent)
wf.add_node("tech", tech_agent)
wf.add_node("fallback", fallback_handler)

# Branch directly from START — classifier picks the first node.
wf.add_router(
    START,
    fn=lambda q: classify(q).lower(),
    routes={"billing": "billing", "tech": "tech"},
    default="fallback",
)
wf.add_edge("billing", END)
wf.add_edge("tech", END)
wf.add_edge("fallback", END)
```

`START` and `END` are sentinels that work the same way in any
edge-builder call:

| Want to | Write |
|---|---|
| Mark a node as the entry | `wf.add_edge(START, "first")` (alias for `set_start`) |
| Branch at the entry | `wf.add_router(START, fn=..., routes={...})` |
| Terminate after a node | `wf.add_edge("last", END)` |
| Branch then terminate one path | `routes={"done": END, "more": "next_node"}` |

Visualise any workflow inline in Jupyter — just type `wf` in a
cell, or call `wf.to_mermaid()` for the diagram source you can
paste into GitHub or [mermaid.live](https://mermaid.live).

## Cycles and feedback loops

Workflows support cycles — `A → B → classify → (C | D | END) → B`
is a first-class pattern, useful for refinement loops, retry
loops, multi-pass review, anything where a step decides "this
isn't good enough, go back."

```python
wf = Workflow("refinement")
wf.add_node("draft", drafter)
wf.add_node("review", reviewer)
wf.add_node("classify", judge)
wf.add_node("revise", revisor)
wf.add_edge("draft", "review")
wf.add_edge("review", "classify")
wf.add_router(
    "classify",
    lambda verdict: verdict,
    {"good_enough": END, "needs_work": "revise"},
)
wf.add_edge("revise", "review")  # loop back
wf.set_start("draft")
```

Two safety caps stop a buggy router from looping forever:

* `max_steps` (default 100) — total step executions in one run.
* `max_visits_per_node` (default 25) — single-node revisits.

Hit either cap and the workflow raises `RuntimeError` naming the
node that exceeded it. Override on the constructor:
`Workflow("name", max_steps=500, max_visits_per_node=50)` for
longer loops, or tighten for stricter contracts.

For inspection, `WorkflowResult.visited` preserves the full
iteration trace (with repeats) so you can see exactly how many
loops ran. Use `Counter(result.visited)` for per-node visit counts.

## What the framework does for you

You write Python; the framework wires up everything else:

* **Per-step telemetry spans** — `jeeves.workflow.step` tagged
  with `step.name`, `user_id`, `pattern="workflow"`. Nested agent
  runs land under the parent step's span.
* **Audit entries per step** — `step_started` / `step_completed`
  / `step_failed` actions, attributed to the active `user_id`.
* **`RunContext` propagation** — `user_id`, `session_id`, and
  `metadata` set on `Workflow.run` flow into every nested
  `Agent.run` automatically.
* **Streaming events** — `WORKFLOW_STARTED`, per-step
  `_STARTED` / `_COMPLETED` / `_FAILED`, `WORKFLOW_COMPLETED`.
  Consumers can break out of the iterator to cancel.
* **Cycle detection** — a workflow that visits the same node
  twice raises with the cycle node named, instead of looping
  forever (a real risk with conditional routers).

## Anti-patterns

These are signals that you've picked the wrong primitive:

* **Using an Agent because you might add open-ended behaviour
  later.** Don't. Start with a workflow; upgrade specific steps
  to agents when reality forces it. Agents cost ~10× more in
  tokens and latency than workflows for routing decisions.
* **Using a Workflow because Agent feels too magical.** If the
  next step actually depends on what the model just learned,
  forcing a workflow makes you write an `if/elif` chain that's a
  worse version of the agent loop.
* **Using both for the same control-flow decision.** Either the
  developer decides or the LLM decides. Don't have a workflow
  branch followed by an agent that re-decides the same thing.

## Quick chooser

Answer these in order; first "yes" wins:

1. **Can you list all the possible paths now, before any user
   interacts?** → Workflow.
2. **Is the next step a function of structured data the previous
   step produced?** → Workflow with a router.
3. **Is the next step a function of natural language the user
   typed?** → Workflow with a classifier step (still a workflow!).
4. **Could the system surprise you with a useful action you
   didn't anticipate?** → Agent.
5. **Will an auditor be reading the source to certify the system?**
   → Workflow at the top level; agents only inside named leaves.

If you got to "Agent" via #4 and you also need #5, you want the
hybrid: outer workflow for the audited skeleton, agent inside the
single open-ended leaf.
