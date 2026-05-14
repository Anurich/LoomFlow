# Changelog

All notable changes to Loom will be documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/).

For development-history detail (per-slice notes, file maps, gate
counts), see [`BUILD_LOG.md`](BUILD_LOG.md).

## [0.10.0] — unreleased

### Added — workspace self-improvement loop: citation tracking + outcome attribution + relevance-aware search

Three additions that turn the workspace from a passive notebook
into the substrate for an agent that gets smarter with every run:

1. **Citation tracking via contextvar.** A new
   ``_ambient_citations_var`` is installed by ``Agent._loop`` for
   the duration of every run. ``Workspace.read_note`` and
   ``read_version`` log the slug they returned into this set.
   Best-effort: outside a run (tests, direct tool calls) the set
   doesn't exist and the loggers no-op. No author check — citation
   is OBSERVATION, not authorship.

2. **Outcome attribution via ``Workspace.attribute_outcome(success,
   user_id)``.** Drains the per-run citation set and updates each
   cited note's metadata: ``cited_count`` += 1, ``success_count``
   += 1 when ``success=True``, ``last_cited_at`` = now. Returns
   the number of notes whose metadata was updated. User code
   calls this after ``agent.run()`` returns, passing a
   pass/fail signal — the workspace then knows WHICH past notes
   actually led to working outcomes.

3. **Relevance-aware search ranking.** ``search_notes`` gains an
   opt-in ``boost_relevance: bool = False`` kwarg. When True,
   each result's base score is multiplied by a citation-history
   boost: ``1 + log(1+cited_count) + 2*log(1+success_count)`` —
   log scaling so a single runaway-popular note doesn't drown
   out fresh ones, success-citations weighted 2x mere citations
   so "verified useful" outranks "merely read." Default off
   preserves back-compat. Available in all four modes (``auto``
   / ``bm25`` / ``semantic`` / ``hybrid``).

Public additions: three optional fields on ``Note`` and
``NoteSummary`` (``cited_count: int = 0``, ``success_count: int = 0``,
``last_cited_at: datetime | None = None``). Legacy notes (no
citation history in frontmatter) default to 0 and start
accumulating from first cite.

This is the **scaffold for self-improvement**, not the full
loop. The framework now KNOWS which past notes were useful;
follow-up work would add an offline "dreaming" pass that uses
that signal to consolidate / promote / archive notes. But just
the relevance signal alone — usable today — means search results
improve as the agent's history grows. Search results in run N+1
are biased toward notes that worked in runs 1..N.

### Added — workspace v2: namespacing, archive, versioning, questions, semantic search, shape-aware prompt

Six additions to `loomflow/workspace/`, all backward-compatible. Legacy `.md`
files load cleanly; existing Agent / Team calls keep working unchanged.

1. **Namespacing** — `write_note(..., namespace="...")` scopes writes to a
   sub-bucket. `list_notes` / `search_notes` ignore namespace by default
   (teammates' work in adjacent namespaces stays visible); pass
   `namespace=` to filter explicitly. `make_workspace_tools(workspace,
   namespace=...)` scopes a member's WRITES to one namespace.

2. **Archive** — `Workspace.archive_note(slug)` sets `archived_at`.
   Default behavior of `list_notes` / `search_notes` EXCLUDES archived
   notes; pass `include_archived=True` to see them. Archived notes
   remain readable by slug via `read_note`. New `archive_note` tool is
   on by default in `make_workspace_tools` (opt out via
   `include_archive=False`).

3. **Revision history** — every `update_note` snapshots the prior body
   into a `.history/<slug>/0001.md` sequence (4-digit monotonic
   counter). New methods `list_versions(slug)` / `read_version(slug, n)`
   on the Workspace protocol. Excluded from `list_notes` via a
   `_walk_note_files` filter so historical revisions never surface as
   live notes.

4. **Ask / answer pattern** — opt-in via `make_workspace_tools(...,
   questions=True)`. Adds `ask_question(title, content)` (writes
   `kind="question"` with `answered=False`), `answer_question(slug,
   content)` (writes a child finding + flips the question's
   `answered=True`), and `list_open_questions()` (lists unanswered).
   Cross-author safe via a new `mark_answered` carve-out on
   `update_note` — non-owners can flip the answered flag without
   touching the question body.

5. **Semantic search** — optional `embedder=` ctor param on
   `LocalDiskWorkspace` and `InMemoryWorkspace`. When wired,
   `write_note` also persists an embedding sidecar (with model name
   for stale-vector detection); `search_notes(query, mode=...)` accepts
   `"auto"` / `"bm25"` / `"semantic"` / `"hybrid"`. Hybrid mode uses
   reciprocal rank fusion (RRF) of BM25 + cosine. Default `auto`
   = hybrid when an embedder is wired, BM25 otherwise.

6. **Shape-aware prompt section** — `workspace_prompt_section` now
   produces TWO variants. When `teammates` is non-empty, the copy
   emphasises team coordination ("share findings with teammates"). When
   `teammates` is None / empty (single-agent cross-run mode), the copy
   emphasises persistent knowledge across runs ("this is YOUR
   persistent notebook"). Empirical motivation: Sonnet was skipping
   the notebook in single-agent mode because the team-language confused
   it about whether the notebook applied to its situation.

Public additions: `NoteVersion` (Tier 1). New optional fields on `Note` /
`NoteSummary`: `namespace`, `archived_at`, `answered`, `answered_by`,
`parent_slug`. `extra="ignore"` on `Note.model_config` for forward-compat
on future schema bumps.

Protocol additions (breaking for duck-typed `Workspace` implementations
— accept and document since workspace is young): `archive_note`,
`list_versions`, `read_version` are now part of `Workspace`. The
in-tree backends `LocalDiskWorkspace` and `InMemoryWorkspace` implement
all of them.

### Added — TodoWrite-style living plan via `living_plan=`

`Agent(living_plan=True)` wires two structured-plan tools onto the
agent: `plan_write(goal, steps)` (atomic full-list rewrite — every
call returns the rendered plan back as markdown, so the plan
becomes load-bearing in the conversation) and `plan_read()`. Each
step is `{description, status, finding}` where status is one of
`todo` | `doing` | `done` | `blocked` | `skipped`. Synonyms like
`in_progress` / `WIP` / `failed` are auto-normalised. The `steps`
argument accepts four serialisation shapes (native list, JSON
string of list, JSON string of `{"steps":[…]}` wrapper, free-form
numbered text) because providers serialise complex args differently.

When `workspace=` is also wired, the plan mirrors to a `kind="plan"`
note in the shared notebook — the first `plan_write` creates the
note, subsequent calls `update_note` the same slug. A
`recall_past_plans(query)` tool is auto-added in this case, so
future runs can search past plans by free-text query and bootstrap
from ones that match. Cross-task plan lineage is multi-tenant by
`user_id`, partitioned via the standard `RunContext`.

Per-run plan state lives in a new `_ambient_living_plan_var`
contextvar (mirroring `_ambient_workspace_var`), so concurrent
`agent.run()` invocations on the same `Agent` instance have
isolated plans. Custom architectures / hooks read the active plan
via `loomflow.tools.plan.get_active_plan()`.

Public exports: `LivingPlan` and `LivingPlanStep` at the top level;
`make_plan_tools`, `make_recall_past_plans_tool`,
`living_plan_prompt_section`, `resolve_living_plan`,
`ResolvedLivingPlan` under `loomflow.tools`.

Default is **opt-in** for v0.10.0 (`living_plan=None` → disabled).
v0.11 will flip the default to "auto" — on when `tools=` is
non-empty, off otherwise — after the primitive has been dogfooded.

Empirical motivation: Terminal-Bench 2.0 tasks where the agent
previously failed 3× in a row (across 3 different architectures)
were resolved on the first attempt after adding a structured plan
tool. The pattern is mainstream in 2026 (Claude Code's TodoWrite,
Devin's plan-mode, OpenHands' task graph) — empirical SWE-bench
data shows scaffolding moved scores from 1.96% to 78.4% holding
the model constant.



This release turns Loom from a working agent harness into a
**production framework**. Five themed milestones (M1–M5) close the
gaps that separate a demo loop from something you put in front of
paying users — multi-tenant safety, conversation continuity, typed
outputs, retry/error taxonomy, and a fast path that skips no-op
integrations. Plus an audit pass that aligns the public surface,
a Sphinx docs site, migration guides for LangGraph and the raw
OpenAI SDK, and a live integration suite that hits paid endpoints.

Every change is **additive**; existing code keeps working unchanged.
Multi-tenancy and structured outputs are opt-in by passing
`user_id=` / `output_schema=`. Retry-on-transient is auto-applied
to in-tree network adapters (OpenAI, Anthropic, LiteLLM); custom
models opt in.

### Added — verbatim audit capture via `audit_log={...}` dict

The default audit log is compliance-friendly: prompts are truncated
to 500 chars, the model's final output isn't recorded, and tool
results carry only `ok` / `denied` / `error` / `reason`. That's
right for regimes that prohibit logging customer PII verbatim.

For **debugging**, **post-incident replay**, or **internal
investigations**, opt into verbatim capture by passing the existing
`audit_log=` parameter a config dict — no extra class to import:

```python
agent = Agent(
    "...",
    audit_log={
        "name": "./audit.jsonl",   # path; omit for in-memory
        "scope_full": True,         # capture prompts + outputs + tool bodies
        "secret": "hmac-key",       # optional signing key
    },
)
```

The same dict works on `Workflow(...)`. With `scope_full: True` the
audit payload includes the full prompt, the model's final output
(and `parsed` for structured runs), the full tool result body, and
tool call duration. Signatures still verify cleanly.

`Agent(audit_log=...)` and `Workflow(audit_log=...)` now share one
resolver — accepts `None`, an `AuditLog` instance, a `str` / `Path`
sugar for `FileAuditLog`, or the new dict form. The class
`FullTranscriptAuditLog` is still available for power users who
want to wrap a hand-built backend; `isinstance(log,
FullTranscriptAuditLog)` is the audit reviewer's signal that PII
may be in the log.

### Added — dict-form `model=` config

The `model=` parameter now accepts a config dict, mirroring the
`audit_log={...}` shape — one parameter carries the model spec and
related agent-level defaults (no need for separate `effort` /
`strict_effort` kwargs at the call site):

```python
agent = Agent(
    "Plan the migration in detail.",
    model={
        "name": "claude-opus-4-7",
        "effort": "high",
        "strict_effort": True,
    },
)
```

Recognised dict keys: `name` (the model spec — required; `model`
is an alias), `effort`, `strict_effort`. Top-level kwargs win when
both are specified, so `Agent(model={"effort": "low"},
effort="high")` uses "high". Unknown keys raise `ConfigError`.

Equivalent to the explicit form `Agent(model="claude-opus-4-7",
effort="high", strict_effort=True)` — pick whichever reads better
at the call site.

### Added — `effort` dial for reasoning-capable models

One enum, one kwarg, every provider's shape — agents can now
request "more thinking" without learning each lab's API.

```python
agent = Agent(
    "Plan the migration in detail.",
    model="claude-opus-4-7",
    effort="xhigh",       # Loom translates per provider
)
# Per-call wins over the agent default:
await agent.run("...", effort="low")
```

`effort` accepts ``"minimal" | "low" | "medium" | "high" | "xhigh"
| "max"`` (exported as ``loomflow.Effort``). Each adapter
translates into the provider's native shape:

* **OpenAI** o-series / GPT-5 — `reasoning_effort` (`xhigh` / `max`
  clamp to `high` — OpenAI's enum tops out there).
* **Anthropic Opus 4.7 / Mythos** — adaptive `thinking` +
  `output_config.effort`, the only regime where `xhigh` and `max`
  pass through unclamped.
* **Anthropic Opus 4.6 / Sonnet 4.6** — adaptive + effort enum
  (xhigh/max clamp to high).
* **Anthropic Sonnet 3.7 / 4 / 4.5** — legacy
  `thinking.budget_tokens` integer (1024 → 32768 across the
  range).
* **LiteLLM** — `reasoning_effort` forwarded; LiteLLM handles
  per-provider routing.

Adapters whose model doesn't support reasoning effort emit a
one-time warning per `(model, effort)` pair and drop the kwarg —
opt into hard-fail with `Agent(..., strict_effort=True)` to catch
typos or capability mismatches loudly during development.

### Added — `FileTelemetry` — JSONL telemetry on disk

A no-collector-required sink for users who want spans + metrics
persisted to disk in a parseable format. Each completed span
and metric emit becomes a single JSON line, with a ``"kind"``
discriminator so downstream pipelines can split them.

```python
from loomflow.observability import FileTelemetry

agent = Agent(..., telemetry=FileTelemetry("./traces.jsonl"))
await agent.run("...")
```

Output:

```jsonl
{"kind":"span","name":"loom.turn","trace_id":"...","parent_span_id":"...","duration_ms":380,"attributes":{"turn":1},"exception":null}
{"kind":"span","name":"loom.run","parent_span_id":null,"duration_ms":420,...}
{"kind":"metric","name":"loom.tokens.input","value":42,"instrument_kind":"counter","attributes":{},"emitted_at":"..."}
```

Query offline with `jq`:

```shell
jq -c 'select(.kind=="span" and .duration_ms > 1000)' traces.jsonl
jq -c 'select(.attributes.session_id=="sess_xyz")'     traces.jsonl
```

Mirrors `FileAuditLog`'s pattern: parent dir auto-created,
writes through `anyio.to_thread.run_sync` so the event loop
never blocks on disk I/O, internal lock serialises concurrent
emits from parallel tool dispatches, file is append-only so
restart-recovery is free.

**Complements** `FileAuditLog`; doesn't replace it. Audit log =
business events for compliance; telemetry = performance /
diagnostic spans. Run both together in production for full
offline forensics. No rotation built in — use `logrotate` /
`journald`.

5 new tests: span JSONL format, metric kind tag, parent dir
auto-creation, exception recording, append-across-restart.

### Added — `InMemoryTelemetry` + `ConsoleTelemetry` + `MultiTelemetry`

Three new telemetry sinks that need no OTel collector deploy.
Previously the only "see what spans my agent emits" path was
`OTelTelemetry` plus ~10 lines of OTel SDK boilerplate
(``TracerProvider`` + ``SpanProcessor`` + ``InMemorySpanExporter``).
That's the right primitive for production, but pure friction
for tests, exploration, and dev "tail my agent" workflows.

```python
from loomflow.observability import (
    InMemoryTelemetry, ConsoleTelemetry, MultiTelemetry,
)

# Assert on spans + metrics in unit tests
tel = InMemoryTelemetry()
agent = Agent(..., telemetry=tel)
await agent.run(...)
assert any(s.name == "loom.tool" for s in tel.spans())

# Watch a flow live in stderr while developing
agent = Agent(..., telemetry=ConsoleTelemetry())

# Fan out — see live AND inspect after
in_mem = InMemoryTelemetry()
agent = Agent(..., telemetry=MultiTelemetry([
    ConsoleTelemetry(), in_mem,
]))
```

`InMemoryTelemetry` records each span as a
:class:`~loomflow.observability.CapturedSpan` (name, trace_id,
span_id, parent_span_id, started_at, ended_at, duration_ms,
attributes, optional exception repr). Metrics become
:class:`~loomflow.observability.CapturedMetric` with the
auto-detected instrument kind (counter vs histogram) — same
suffix-based dispatch rule as `OTelTelemetry`.

`ConsoleTelemetry` prints one line per span completion + one
line per metric emit. Indented by parent depth so the trace
tree is visible. Default stream is `sys.stderr`; `show_metrics=
False` available for span-only output.

`MultiTelemetry` enters every sink's ``trace()`` contextmanager
via `AsyncExitStack`, so cleanup runs in reverse order even
when one sink raises mid-emit. ``MultiTelemetry([])`` is rejected
with a clear "use NoTelemetry() for a no-op" error.

Production path unchanged — `OTelTelemetry()` with no args still
picks up the globally-configured OTel `TracerProvider` /
`MeterProvider`.

10 new tests across span hierarchy, metric kind dispatch, clear-
state reset, exception recording, console stream output, multi
fan-out, empty-sink rejection.

Updated `examples/13_telemetry.py` — was 80 lines of OTel SDK
plumbing, now 30 lines of Loom-native sink usage.

### Added — `response_tone=` on Agent + Workflow

A new optional kwarg that steers *how* the agent phrases its
output — not *what* it answers. Pass a preset name or any
free-form string; the framework appends a one-line style
directive to the system prompt. Default is ``None`` (no
directive, no behaviour change).

```python
# Preset
agent = Agent("...", model="gpt-4.1-mini", response_tone="legal")

# Free-form passthrough — preset map is convenience, not gatekeeper
agent = Agent("...", response_tone="warm but precise, like a doctor")

# Per-call override beats agent default
result = await agent.run("...", response_tone="casual")

# Workflow ambient flows to nested agents that didn't set their own
wf = Workflow.chain([agent_a, agent_b], response_tone="executive")
```

**Shipped presets** (one sentence each, intentionally tight):
``casual``, ``professional``, ``technical``, ``legal``,
``finance``, ``executive``, ``academic``.

**Resolution order**: per-call > agent default > workflow ambient
> ``None``. Same propagation pattern as ``Workflow(memory=...)``
— via a contextvar in :mod:`loomflow.core.context` that the
workflow installs in ``stream()`` and resets in ``finally``.

**Tone vs persona vs instructions** — three orthogonal levers:

* **Instructions** (``Agent(instructions=...)``) — *what* the
  agent does.
* **Persona** (text inside instructions, e.g. "You are a tax
  lawyer...") — *who* the agent is.
* **Tone** (``response_tone=``) — *how* the agent phrases its
  output.

Free-form strings let users pin a custom org voice without
registering it as a preset. The framework treats the value as
opaque text appended to the system prompt — the model handles
the actual styling.

**Interaction with structured output:** when both
``response_tone`` and ``output_schema`` are set on a non-native
model adapter (so the schema directive does get injected), the
tone is appended AFTER the schema so it's the last thing the
model reads. For native-structured-output adapters (OpenAI,
Anthropic), only the tone is appended; the schema is the
API-level constraint as before.

### Fixed — Don't double-inject schema for native structured output

When the model adapter declares
``supports_native_structured_output = True`` (OpenAI, Anthropic,
LiteLLM-passthrough), the agent loop no longer ALSO appends the
JSON Schema as text into the system prompt. The native API-level
constraint (``response_format=json_schema, strict=True`` /
forced ``__output__`` tool call) is sufficient on its own;
duplicating the schema in the prompt was just dead tokens —
~2k extra input tokens per structured-output call.

Concrete impact, measured on ``gpt-4.1-mini`` against the
benchmark scenario in ``test/bench/`` (RAG + Pydantic
``PdfSummary`` schema):

* Input tokens: **~3,091 → ~1,100** (≈64% reduction).
* Cost per call: **$1.535m → ~$0.55m** (Loom now cheaper than
  LangGraph's $0.729m on the same workload).
* Reliability: unchanged — validate-with-retry still injects
  the schema into the retry message if the model ever produces
  invalid JSON.

Custom user-supplied model adapters that don't set the flag keep
the prompt-augmentation safety net (default off). Two tests guard
against regression: one for the native-skip path, one for the
non-native still-injects path.

### Added — `Memory.recall_scored()` — hybrid BM25+vector retrieval with score breakdown

A protocol-evolution change so adding rerankers / MMR / hybrid
weighting later isn't a breaking surface change. The new
``recall_scored`` method returns a list of :class:`EpisodeMatch`
— each carrying the raw episode plus a fused score and the
component scores (BM25, vector cosine, optional reranker) used
to rank it. Callers that want to apply a downstream reranker, an
MMR diversifier, or a score-threshold filter can now do so
without re-running recall.

```python
matches = await agent.memory.recall_scored(
    "postgres replication",
    user_id="alice",
    alpha=0.5,        # 0=BM25 only, 1=vector only, 0.5=balanced (RRF)
)
for m in matches:
    print(m.episode.input, m.score, m.bm25_score, m.vector_score)
```

**Native hybrid implementations:**

* :class:`InMemoryMemory` now ships a real BM25 ranker (replaces
  the prior substring-match-then-recency behaviour). Episodes
  whose ``input`` / ``output`` lexically match the query rank
  ahead of unrelated recent episodes — a real recall-quality
  upgrade for the default backend.
* :class:`VectorMemory` does the full BM25 + cosine + Reciprocal
  Rank Fusion (RRF) hybrid that the framework's vectorstore
  module already used for RAG, now extended to agent memory.

**Other backends (Chroma, Postgres, Redis, Sqlite, AutoExtract,
Lazy)** ship a thin shim via the new
:func:`loomflow.memory.default_recall_scored` helper that wraps
their existing recall results with neutral scores. The protocol
stays coherent; native hybrid implementations for those backends
can land later without breaking callers.

**Why this matters competitively:** before this change, Loom's
recall was cosine + token-overlap fallback only — weaker than
Zep (BM25 + vector + graph BFS + reranker) and CrewAI (composite
+ deep mode). After this change, Loom matches the hybrid-recall
baseline of the field for in-process memory, with the protocol
shape ready for a reranker / MMR / cross-encoder layer when
someone needs one. The full bi-temporal + auto-extract +
multi-tenant + hybrid-recall combination doesn't exist anywhere
else open-source.

### Changed — Skills `tools.py` now imports lazily + `build_tools(ctx)` factory

Two related improvements to how Loom skills load their Python
tools — both fixing real friction and adding a capability that
beats deepagents' "no tools.py at all" punt.

**Lazy `tools.py` import (breaking, but fixes a footgun).**
Previously, ``Skill("path/")`` imported the skill's ``tools.py``
*at construction time*. If that file did module-level event-loop
work — e.g., ``asyncio.run(setup_vectorstore())`` — and the
caller was already inside a Jupyter event loop, the import
crashed with ``RuntimeError: asyncio.run() cannot be called from
a running event loop``.

Now the import is deferred to first-use:

* ``Skill(...)`` only *detects* ``tools.py``; doesn't import it.
* ``Skill.materialize_tools(ctx)`` does the import on first call
  and caches the result. Subsequent calls reuse the cache.
* The framework's built-in ``load_skill`` tool calls
  ``materialize_tools`` from inside the running agent loop, where
  doing event-loop work is fine.

**Behaviour change:** import errors in ``tools.py`` now surface
when the model first calls ``load_skill(name)``, not at
``Skill(...)`` construction. Most skill code paths are unaffected;
the Mode C subprocess-tool path stays eager. Migration: if your
tests asserted "construction raises on bad ``tools.py``", call
``skill.materialize_tools()`` explicitly to trigger the import.

**`build_tools(ctx)` factory protocol (new feature).** Skills
can now ship tools that close over caller-supplied state without
globals or module-level setup:

```python
# skills/pdf-retrieval/tools.py
from loomflow import tool

def build_tools(ctx):
    vectorstore = ctx.metadata["vectorstore"]
    @tool
    async def retreiver(query: str) -> list:
        return await vectorstore.search_hybrid(query=query)
    return [retreiver]
```

```python
# In your script:
agent = Agent(..., skills=[Skill("skills/")])
result = await agent.run(
    "...",
    metadata={"vectorstore": vectorstore},
)
# load_skill('pdf-retrieval') passes the live RunContext to
# build_tools(ctx); the resulting `retreiver` is registered with
# the agent's tool host and visible on the next turn.
```

When ``tools.py`` *doesn't* export ``build_tools``, the framework
falls back to discovering module-level ``@tool``-decorated
globals (the prior behaviour). Back-compatible.

**Bonus: clearer error on the asyncio gotcha.** When a skill's
``tools.py`` does ``asyncio.run(...)`` at module level and the
import does fire inside a running loop, the resulting
``SkillError`` now includes a hint pointing at the
``build_tools(ctx)`` pattern instead of just the raw asyncio
traceback.

### Fixed — Workflow router classifier can now be `async def`

The classifier passed to ``add_router`` (both ``add_router(node,
fn=...)`` and ``add_router(START, fn=...)``) is now awaited when
it's an async function. Previously the framework called it
synchronously, so an ``async def`` classifier returned a coroutine
object that ``str()`` rendered as ``<coroutine object …>`` —
which never matched any route key, causing:

> ``RuntimeError: entry router on '...' produced key
> <coroutine object ... at 0x...> with no matching route and no
> default``

Common case this hits: a classifier that calls a model to decide
the branch. New ``_eval_classifier`` helper detects coroutine
functions via ``inspect.iscoroutinefunction`` and awaits them;
also handles sync wrappers around async inner calls
(``lambda v: some_async_fn(v)``) by awaiting the returned
coroutine.

### Added — `add_router(START, ...)` — branch at the entry of the graph

You can now classify the workflow's input and route directly to
one of N first nodes, without an artificial passthrough "entry"
step. Mirrors LangGraph's ``add_conditional_edges(START, ...)``:

```python
from loomflow import Workflow, START, END

wf = Workflow()
wf.add_node("step_1", step_1)
wf.add_node("step_3", step_3)
wf.add_router(
    START,
    fn=lambda q: "step_1" if "work" in q else "step_3",
    routes={"step_1": "step_1", "step_3": "step_3"},
    default=END,                         # optional fallback
)
wf.add_edge("step_1", END)
wf.add_edge("step_3", END)
```

Validation at build time: route targets must be registered nodes
or ``END``, so typos raise a clear error before the run starts.
``set_start`` and ``add_router(START, ...)`` are mutually
exclusive — calling one resets the other. The mermaid / DOT
diagrams show ``START -->|key| node`` directly so the visual
matches what you wrote.

Ships alongside fix to ``WorkflowResult`` reconstruction so the
zero-step ``default=END`` path returns the original input as
``result.output`` (it was returning ``None`` before).

### Added — `Workflow(memory=...)` shared agent memory across the graph

Workflows can now own a single :class:`~loomflow.Memory` and
propagate it to every nested :class:`~loomflow.Agent` step that
didn't specify its own — so episodes / facts written by one
agent are recall-able by the next without per-agent wiring:

```python
from loomflow import Workflow, Agent, InMemoryMemory

mem = InMemoryMemory()
agent_a = Agent(model="gpt-4.1-mini", instructions="...")
agent_b = Agent(model="gpt-4.1-mini", instructions="...")

wf = Workflow.chain([agent_a, agent_b], memory=mem)
await wf.run("hi", user_id="alice", session_id="conv-1")
# Both agents wrote to / read from `mem`.
```

Resolution order is **explicit always wins**:

1. ``Agent(memory=my_mem)`` keeps using ``my_mem`` even inside
   a workflow with ``memory=``.
2. Otherwise ``Workflow(memory=mem)`` is used as the fallback.
3. Otherwise the agent's per-instance default (in-memory).

Implemented via a contextvar in :mod:`loomflow.core.context`
that the workflow installs at the start of ``run`` / ``stream``
and resets in ``finally``, so memory does not leak across
workflow runs. Available on the explicit ``Workflow(memory=...)``
constructor and on all sugar constructors (``chain``, ``route``,
``parallel``).

### Added — `add_edge(START, "node")` as alias for `set_start("node")`

The ``START`` sentinel was previously inert (just a name). It now
works as an edge source so graphs read symmetrically with the
``END`` sentinel:

```python
wf.add_edge(START, "first")     # alias for set_start("first")
wf.add_edge("first", "second")
wf.add_edge("second", END)
```

Matches the pattern users coming from LangGraph expect, and lets
the entry / exit show up in mermaid diagrams via the same edge
syntax. ``set_start`` continues to work — no breaking change.
``add_edge(START, END)`` and ``add_edge(END, ...)`` are rejected
with messages that point at the right method.

### Added — Workflow visualisation: `to_mermaid()` / `to_dot()` / Jupyter

* **`Workflow.to_mermaid() -> str`** — returns a Mermaid
  ``flowchart TD`` diagram of the graph. Pastes directly into
  GitHub Markdown (renders inline) or https://mermaid.live for
  PNG / SVG export. Solid arrows are unconditional edges,
  labelled solid arrows are router branches, dotted arrows are
  router *defaults*. ``START`` and ``END`` are stadium-shaped.
* **`Workflow.to_dot() -> str`** — same picture in Graphviz DOT
  for users who prefer the Graphviz toolchain (`dot -Tpng -o
  graph.png`). Optional — Mermaid is the recommended path since
  it needs no install.
* **`Workflow._repr_markdown_`** — Jupyter / VS Code / JupyterLab
  auto-render the diagram inline when you type ``wf`` into a cell.
  No imports, no extra calls.

Tests cover linear chains, routers (labelled + default branches),
empty workflows, DOT shape declarations, and the markdown wrapper.

### Changed — Boundary-input errors now name the fix, not just the failure

A pass over the most-hit user entry points: when the framework
rejects a wrong-shape argument, the error message now (1) names
the offending value, (2) lists the accepted forms, and (3) shows
a working example. The goal is "fail fast at the point of
mis-use, with the next step in the message" — replacing several
errors that previously surfaced deep inside the runtime as
generic Python exceptions (`'str' can't be used in 'await'
expression`, `list.append() takes no keyword arguments`, etc.).

* **`Workflow(audit_log=...)` accepts `str` / `Path`** — auto-
  wraps as :class:`~loomflow.security.FileAuditLog` so users can
  write `Workflow.chain(..., audit_log="run.log")` without
  importing the backend. Anything else (e.g. a bare `list`) is
  rejected at construction time with a `TypeError` listing the
  four valid forms (`InMemoryAuditLog`, `FileAuditLog`, path
  string/`Path`, `None`).
* **`Agent(tools=...)` rejection** — non-list / non-Tool / non-
  callable / non-`ToolHost` values now fail with a message that
  names every accepted form (`tools=None`, list, single tool,
  single callable, `ToolHost` instance).
* **`tools=[entry]` rejection** — non-callable list entries now
  fail with a `@tool`-decorated function example in the error
  text, so the user sees the fix inline.

### Changed — Workflow `@step` decorator: clearer error on sync functions

* **`@step` now raises at decoration time** when applied to a
  synchronous `def`. Previously, wrapping a sync function silently
  succeeded and the workflow only failed deep inside the runner
  with `'str' can't be used in 'await' expression` — a cryptic
  message that gave no hint that the user's function was the
  cause. The new `TypeError` names the offending function and
  spells out both fixes:
    1. Add `async` to the `def` (gets telemetry / audit / journaling).
    2. Drop `@step` and pass the plain function directly —
       `Workflow.chain` and `.route` already accept sync callables
       and dispatch them to a worker thread.
  Failure now surfaces at module import, not after a workflow
  has started running.

### Added — Multi-tenancy by default (M1)

* **`RunContext`** (frozen `dataclass(slots=True)`) — typed,
  immutable per-run scope with `user_id`, `session_id`, `run_id`,
  and `metadata`. First-class framework primitive, not strings in
  a `configurable` dict. `with_overrides(...)` for sub-agent
  inheritance.
* **`get_run_context()`** — read the live context inside any tool /
  hook / sub-agent. Backed by a `ContextVar` set in `Agent._loop`;
  `anyio` task groups propagate it across parallel tool dispatch
  and spawned sub-agents automatically. Returns the empty default
  outside an active run — never raises, so direct `@tool` calls in
  tests keep working.
* **`set_run_context(ctx)`** — async context manager for installing
  a context outside an active run (background workers that share
  tool implementations with the agent).
* **`Agent.run(user_id=, session_id=, metadata=, context=)`** —
  flat kwargs; LangGraph-style `config={"configurable": {...}}`
  nesting deliberately avoided. Same kwargs added to
  `Agent.stream` and `Agent.resume`.
* **`Episode.user_id` + `Fact.user_id`** — Pydantic fields, optional.
  Backends partition on these as a hard namespace boundary:
  episodes / facts stored under one `user_id` are never visible to
  a recall scoped to a different one. `None` is its own
  ("anonymous / single-tenant") bucket.
* **`Memory.recall(user_id=)`, `Memory.recall_facts(user_id=)`,
  `FactStore.query(user_id=)`, `FactStore.recall_text(user_id=)`** —
  partition filter wired through every implementation:
  `InMemoryMemory`, `VectorMemory`, `ChromaMemory`,
  `PostgresMemory`, `RedisMemory`, plus all four fact stores.
  Postgres / SQLite use `IS NOT DISTINCT FROM` for safe
  NULL-bucket comparisons; Chroma uses native `where` filters;
  Redis stores `user_id` as a Hash field and post-filters.
* **Schema migrations** — `episodes` and `facts` Postgres tables
  gain a `user_id TEXT` column with idempotent
  `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` for in-place
  upgrades; SQLite gets the same with a duplicate-column-tolerant
  `try`/`except`. New `(namespace, user_id, occurred_at DESC)`
  index on Postgres episodes; `(user_id, subject, predicate)` on
  the facts tables.
* **Namespace-scoped supersession** — fact-store supersession (the
  bi-temporal `valid_until = new.valid_from` write when a new
  fact replaces an old one) is now scoped by `user_id`: alice's
  new claim never invalidates bob's currently-valid claim on the
  same `(subject, predicate)`. Across all four fact-store
  backends.

### Added — Conversation continuity (M2)

* **`Memory.session_messages(session_id, *, user_id=, limit=)`** —
  new protocol method returning prior user/assistant turns from
  the named conversation, scoped to the `user_id` partition,
  oldest-first. Implemented across all five memory backends.
* **`_build_seed_messages` rehydration** — when `session_id` was
  reused, the agent loop loads the prior turns as real `Message`
  history (proper USER / ASSISTANT roles) before the current
  prompt. Cross-session episodic recall filters out the current
  session's episodes to avoid duplication. Reusing the same
  `session_id` across `agent.run()` calls now genuinely continues
  the conversation — no reducer protocol, no `add_messages` magic.

### Added — Footgun protection + multi-agent inheritance (M3)

* **`IsolationWarning`** (subclass of `UserWarning`) — fires when
  `Memory.recall(user_id=None)` runs against a store whose
  episodes / facts include a non-None `user_id`. The partition is
  still safe, but the developer probably forgot to pass
  `user_id=` somewhere; loud failure beats silent confusion. Goes
  through Python's `warnings` machinery — apps promote to error
  with `warnings.simplefilter("error", IsolationWarning)`.
* **Multi-agent context inheritance** — `SubagentInvocation` (used
  by `Supervisor`, `Debate`, `Swarm`, `Router`, `ActorCritic`,
  `Blackboard`) now reads `get_run_context()` automatically when
  no explicit `context=` is passed. The parent's `user_id` and
  `metadata` propagate to every sub-agent without each
  architecture having to plumb them by hand. Sub-agents get a
  fresh `session_id` so each worker has its own thread while
  inheriting the parent's namespace.

### Added — Structured outputs (M4)

* **`Agent.run(output_schema=, output_validation_retries=)`** —
  pass any Pydantic `BaseModel` and the framework augments the
  per-run system prompt with a `STRUCTURED OUTPUT REQUIRED`
  directive embedding the schema's JSON Schema, parses the final
  assistant text, and returns a validated typed instance on
  `RunResult.parsed`. Static `agent._instructions` is not
  mutated; the augmentation is per-run.
* **Retry-with-feedback** — on parse failure, the framework gives
  the model up to `output_validation_retries` (default 1) extra
  single-shot turns to fix the output, feeding the validation
  error back as a USER message. After the retry budget is
  exhausted, raises `OutputValidationError`.
* **`OutputValidationError`** — carries the raw text (`raw`), the
  schema being targeted (`schema`), and the underlying Pydantic
  `ValidationError` (`cause`, also linked via `__cause__`).
* **`RunResult.parsed: Any | None`** — typed, validated instance
  when `output_schema=` was supplied; `None` otherwise.
  `RunResult.output` keeps the raw (cleaned) JSON text for
  logging / audit.
* **Markdown-fence tolerance** — strips `` ```json `` / `` ``` ``
  fences before parsing, since real models occasionally wrap
  output despite being told not to.
* **`RunResult.value`** — smart accessor that returns `parsed`
  when a schema validated, else the raw `output` string. Removes
  the `result.output` vs `result.parsed` "did the schema even
  fire?" footgun: `result.value` is always "the answer" in the
  shape the caller expects. The original `parsed` / `output`
  fields stay untouched for code that reads them directly.
* **`Agent(output_schema=...)`** — agent-bound default schema.
  Pass it once on construction and every `agent.run()` /
  `agent.stream()` applies it; a per-call `output_schema=` still
  overrides for one-off shapes. Mirrors Pydantic AI's
  `output_type=` ergonomics.
* **Tagged-union output schemas** — `output_schema=A | B` (or
  `Union[A, B]`) lets an agent return one of multiple shapes per
  call. Validation tries each member in declaration order and
  accepts the first that fits, so callers can model
  "valid result vs structured error" without a discriminator
  field.

### Added — Production hardening (M10)

The "best-in-class" pass: closes seven holes a Google reviewer
called out (unbounded per-user state, Postgres empty-string-for-
NULL hack, silent auto-extract, silent ``ask`` bypass, missing
deprecation infrastructure, hard-coded API-key resolution, no
multi-tenant load proof). Every change is additive; default
behaviour is preserved unless a caller opts in.

* **Bounded per-user state (M10.1).** ``StandardBudget._by_user``
  and ``InMemoryMemory._blocks`` now use a new
  :class:`loomflow.core._eviction.BoundedDict` with LRU +
  idle-TTL eviction. Defaults: ``max_users=100_000``,
  ``user_idle_ttl_seconds=86_400`` (24h). Pass ``None`` to
  either constructor kwarg to disable bounding for single-tenant
  deployments. Eviction *drops* a user's bucket — callers needing
  durable spill-to-disk should pick :class:`SqliteMemory` /
  :class:`PostgresMemory` instead of relying on the in-process
  bound.
* **Postgres anonymous-bucket sentinel (M10.2).** The empty-
  string-for-NULL hack on ``memory_blocks.user_id`` is replaced
  with a reserved sentinel ``__jeeves_anon_user__``. Schema DDL
  includes an idempotent migration that rewrites legacy ``''``
  rows. Callers that try to use the sentinel as a real
  ``user_id`` get a ``ValueError`` — defense against impersonating
  the anonymous bucket.
* **Auto-extract observability (M10.3).** :class:`AutoExtractMemory`
  now emits ``jeeves.auto_extract.duration_ms`` (histogram) and
  ``jeeves.auto_extract.invocations`` (counter) per extraction,
  tagged with ``user_id`` and ``status``. A one-time-per-process
  ``INFO`` log notice fires when the wrapper is enabled by the
  default-on heuristic, so ops teams learn about it before the
  LLM bill arrives.
* **Permission "ask" approval handler (M10.4).**
  ``Agent(approval_handler=callable)`` resolves
  ``Decision.ask_(...)`` outcomes from the permissions layer.
  Without one, ``ask`` falls back to deny — closes a security
  hole where the fast-hooks default-allow was silently bypassing
  the approval gate.
* **Deprecation infrastructure (M10.5).** New
  :class:`LoomDeprecationWarning` subclass + ``warn_legacy_
  protocol(...)`` helper; every protocol-evolution
  ``except TypeError`` shim now warns once-per-process pointing
  at the v1.0 removal target so callers can migrate.
* **Secrets protocol wired into model resolution (M10.6).** New
  concrete :class:`EnvSecrets` (default) and
  :class:`DictSecrets` impls plus a ``lookup_sync(ref)`` method
  on the Secrets protocol. ``Agent(secrets=...)`` flows through
  to model adapters; ``OpenAIModel`` / ``AnthropicModel`` /
  ``LiteLLMModel`` resolve API keys via ``api_key=`` →
  ``secrets.lookup_sync`` → ``os.environ`` precedence.
  ``redact()`` masks common API-key shapes (OpenAI / Anthropic /
  AWS / GitHub) so audit logs don't leak credentials.
* **Multi-tenant load benchmark (M10.7).** New
  ``bench/multi_tenant.py`` simulates N concurrent users × M turns
  on one shared Agent and reports p50 / p99 latency, RSS growth,
  isolation violations, budget mismatches. Smoke-test variant in
  ``tests/test_multi_tenant_load.py`` runs as part of the regular
  suite.
* **Tests + docs.** 980 tests pass (up from 933 at the start of
  M10); mypy ``--strict`` clean across 112 source files; ruff
  clean. CHANGELOG entry, capability matrix updated.

### Added — Multi-tenant by default *everywhere* (M9)

Closes the remaining gaps so every stateful primitive partitions by
``user_id``. Memory was already done (M1–M8); M9 covers working
blocks, budget, audit log, permissions, hooks.

* **Working memory blocks** — ``Memory.working(user_id=)`` /
  ``update_block(name, content, user_id=)`` /
  ``append_block(name, content, user_id=)``. All six backends
  partition: in-memory dicts re-keyed to ``(user_id, name)`` tuples;
  SQLite + Postgres got migrations adding a ``user_id`` PK column
  (idempotent, with table-rebuild fallback for SQLite which can't
  ``ALTER`` a PK). Pinned-order is per-user — adding bob's first
  block doesn't bump alice's slots. The agent loop reads
  ``deps.memory.working(user_id=deps.context.user_id)``; legacy
  custom Memory impls without the kwarg fall back gracefully.
* **Per-user budget accounting** — ``StandardBudget`` now tracks
  tokens / cost per ``user_id``. New ``BudgetConfig`` fields
  ``per_user_max_tokens``, ``per_user_max_input_tokens``,
  ``per_user_max_output_tokens``, ``per_user_max_cost_usd``,
  ``per_user_max_wall_clock`` enforce per-user caps alongside (or
  instead of) the global ones. ``Budget.allows_step(user_id=)`` /
  ``consume(user_id=)`` are the new protocol shape.
  ``StandardBudget.usage_for(user_id)`` snapshots a single user's
  totals for ops dashboards.
* **Audit log: top-level ``user_id``** — ``AuditEntry.user_id`` is
  now a first-class field (was buried in payload). ``AuditLog.append``
  + ``AuditLog.query`` gain the kwarg. The HMAC signature covers
  ``user_id`` so a tampered entry that swaps user identity won't
  verify. Both ``InMemoryAuditLog`` and ``FileAuditLog`` updated.
* **Permissions: ``user_id`` kwarg** — ``Permissions.check(call,
  context=, user_id=)``. ``StandardPermissions`` accepts and ignores
  it; the new ``PerUserPermissions`` routes to per-user policies::

      perms = PerUserPermissions(
          policies={
              "admin_alice": StandardPermissions(mode=Mode.BYPASS),
              "paid_user_42": StandardPermissions(mode=Mode.ACCEPT_EDITS),
          },
          default=StandardPermissions(
              mode=Mode.DEFAULT, denied_tools=["bash"]
          ),
      )
      Agent(..., permissions=perms)

* **Hooks: ``user_id`` kwarg** — ``HookHost.pre_tool`` /
  ``post_tool`` accept ``user_id`` so custom hook hosts can dispatch
  per-user. The bundled ``HookRegistry`` accepts and ignores the
  kwarg; individual hook callables continue to receive only
  ``(call,)`` / ``(call, result)`` for API stability.
* **Backwards-compatible everywhere** — every protocol change is a
  keyword-only add. The agent loop wraps every kwarg-bearing call in
  a ``try / except TypeError`` fallback to the legacy signature, so
  custom implementations users wrote pre-M9 keep working unchanged.
* **Public exports** — ``PerUserPermissions`` is exported from both
  ``loomflow.security`` and the top-level ``loomflow`` package.
* **Tests** — 14 new in ``tests/test_user_id_isolation_full.py``
  covering every primitive: working-block partition (in-memory +
  SQLite-persistent), pinned-order per-user, budget per-user
  totals + per-user caps + global+per-user combination, audit
  top-level user_id + filter + combined filter, permissions
  routing + default fallback, full end-to-end with one Agent
  serving alice and bob through Memory + Budget + AuditLog all at
  once.

### Added — Auto fact extraction (M8)

* **`AutoExtractMemory`** — a :class:`Memory` wrapper that runs the
  bundled :class:`Consolidator` on every persisted episode,
  extracting structured ``(subject, predicate, object)`` claims
  into the inner backend's fact store. Implements the full
  :class:`Memory` protocol; forwards every method through to the
  inner backend; only ``remember`` adds the extraction pass.
* **`Agent(auto_extract=...)`** — new kwarg, default-picked by
  model class. ON for in-tree network adapters (``OpenAIModel`` /
  ``AnthropicModel`` / ``LiteLLMModel``); OFF for in-process fakes
  (``ScriptedModel`` / ``EchoModel``) and unrecognised custom
  Models. Pass ``auto_extract=True``/``False`` to override.
* **Internal split: `_memory` vs `_wrapped_memory`** —
  ``agent.memory`` (the public accessor) keeps returning the
  user-supplied / resolver-built backend so introspection and
  ``agent.memory.profile(...)`` style code work transparently.
  ``_wrapped_memory`` is the loop-facing view that runs through
  the auto-extract layer. Same dual-attribute pattern as
  ``_model`` / ``_wrapped_model`` for the retry layer.
* **Best-effort by design** — extraction failures (model errors,
  malformed JSON, rate limits) never break the run. The episode
  write succeeds first; extraction runs after and either appends
  facts or logs and moves on. The agent's primary contract
  (return a result, persist the episode) is preserved unchanged.
* **End-to-end UX** — a single ``agent.run("I prefer dark mode",
  user_id="alice")`` against a real model now leaves a
  ``Fact(user_id="alice", subject="alice", predicate="prefers",
  object="dark_mode")`` in the store, partition-respecting, ready
  for future ``recall_facts`` queries to surface.

### Added — Memory inspection + GDPR helpers (M7)

* **`Memory.profile(user_id=)`** — returns a `MemoryProfile`
  carrying episode count, fact count, last-seen timestamp, the 10
  most-recent sessions touched, and a sample of the most-recently-
  recorded facts. Per-user, partition-respecting; suitable for
  rendering "what does the bot know about me?" views to end users
  or ops dashboards.
* **`Memory.forget(*, user_id=, session_id=, before=)`** —
  right-to-erasure. With ``user_id`` only, erases all episodes +
  facts for that user. With ``session_id``, narrows to that
  conversation. With ``before``, narrows to a retention window.
  Filters AND together. Returns the count of records deleted.
  ``user_id=None`` erases the anonymous bucket only — same hard
  partition rule as `recall`. Erasing every user is deliberately
  per-user-explicit so it can't happen by accident.
* **`Memory.export(user_id=)`** — full data dump for portability /
  DSAR responses. Returns a `MemoryExport` with every episode and
  fact for the user; serialise with `.model_dump_json()` for
  download.
* **`MemoryProfile`, `MemoryExport`** — new Pydantic types in
  `core/types.py`, exported from both `loomflow.core` and the
  top-level `loomflow` package.
* **Cross-backend implementations** — all six backends
  (`InMemoryMemory`, `SqliteMemory`, `VectorMemory`, `ChromaMemory`,
  `PostgresMemory`, `RedisMemory`) honour the new methods.
  Postgres uses native `IS NOT DISTINCT FROM`-aware DELETEs;
  Chroma uses native `where` filters; SQLite uses `DELETE` against
  the same `.db` file the FactStore lives in; in-memory backends
  filter dicts directly. `LazyMemory` forwards through to the
  inner backend on first use, same as the other protocol methods.
* **Bug fix in `Consolidator._build_fact`** — extracted facts now
  inherit the source episode's ``user_id``. Prior to this, every
  consolidator-extracted fact landed in the anonymous bucket
  regardless of which user the episode belonged to, breaking
  multi-tenant fact recall.
* **Example** — `examples/05_memory_showcase.py` walks every
  backend (Postgres + Redis skip gracefully without DSNs),
  exercises the resolver in all three tiers (string / dict /
  instance), demonstrates profile / forget / export across
  backends, and runs the `Consolidator` to extract structured
  facts from raw episodes. Single runnable file.

### Added — Memory string resolver + SqliteMemory (M6)

* **`memory=` URL/dict/instance resolver** — `Agent(...)` accepts:
  * `None` → default `InMemoryMemory()`
  * `"inmemory"` / `"sqlite:./bot.db"` / `"sqlite"` /
    `"chroma:./vec"` / `"chroma"` / `"postgres://..."` /
    `"redis://..."` (URL scheme picks the backend)
  * `{"backend": ..., "path": ..., "namespace": ...,
    "embedder": ..., "with_facts": ...}` (config dict)
  * any already-constructed `Memory` instance (today's API, unchanged)
  Mirrors the design of the existing `model=` resolver.
* **`resolve_memory(spec)`** — public helper for the same resolution
  logic. Used internally by `Agent.__init__`; exposed so external
  config systems (TOML, YAML, env-driven configs) can drive memory
  picks.
* **`SqliteMemory`** — new backend at
  `loomflow.memory.sqlite.SqliteMemory`. Episodes, working
  blocks, session messages, and the bi-temporal fact store all in
  one sqlite file. Single-file persistence, no server, idempotent
  schema migrations (`CREATE TABLE IF NOT EXISTS`,
  `ALTER TABLE ADD COLUMN`-with-duplicate-tolerant exception).
  Honours the M1 `user_id` partition contract; emits
  `IsolationWarning` on mixed-bucket recall, same as
  `InMemoryMemory`. Use `SqliteMemory(":memory:")` for an
  ephemeral in-process database.
* **`LazyMemory`** — wraps async-construct backends (Postgres /
  Redis) so `Agent(...)` stays synchronous. Connection opens on
  first protocol method call; concurrent first-uses serialise
  through an `anyio.Lock`; backend exceptions get normalised to
  `MemoryStoreError` with the original on `__cause__`. The proxy
  forwards every Memory protocol method (`working`,
  `update_block`, `append_block`, `remember`, `recall`,
  `recall_facts`, `session_messages`, `consolidate`) and exposes
  `.facts` once resolved.
* **Auto-picked embedder** — string and dict specs pick
  `OpenAIEmbedder("text-embedding-3-small")` when
  `OPENAI_API_KEY` is set, `HashEmbedder()` otherwise. Override
  via `embedder=` in the dict form, taking either an `Embedder`
  instance or one of `"hash"`, `"openai"`, `"openai-large"`,
  `"voyage"`, `"cohere"`.
* **Auto-attached fact store on resolver path** —
  `with_facts=True` is the default for string and dict specs;
  semantic-recall layer is on out of the box. Pass
  `with_facts=False` in the dict form to skip it. Explicit
  `Memory(...)` instances keep their existing per-backend
  defaults so today's call sites are unchanged.
* **Public exports** — `SqliteMemory`, `LazyMemory`,
  `resolve_memory` exported from both `loomflow.memory` and
  the top-level `loomflow` package.

### Added — Resilient model calls (M5)

* **Error taxonomy** — `ModelError` base + `TransientModelError`
  (retryable; carries `retry_after`) + `RateLimitError` (subclass
  of transient) + `PermanentModelError` + `AuthenticationError` +
  `InvalidRequestError` + `ContentFilterError`. All inherit from
  `LoomError`; existing `except LoomError` catches
  keep working. Each carries a `cause` slot and chains through
  `__cause__` so debug code can still inspect the raw SDK error.
* **`RetryPolicy`** (frozen dataclass) — `max_attempts`,
  `initial_delay_s`, `max_delay_s`, `multiplier`, `jitter`. Plus
  `RetryPolicy.disabled()` and `RetryPolicy.aggressive()` factories.
  Sensible default: 3 attempts, 1 → 2 → 4 s with ±10% jitter,
  capped at 30 s.
* **`compute_backoff(policy, attempt, retry_after=)`** — exponential
  growth, capped at `max_delay_s`, jittered. Provider-supplied
  `Retry-After` is a **floor**: can exceed the cap because the
  provider is more authoritative than our heuristic.
* **`classify_model_error(exc)`** — maps OpenAI / Anthropic /
  httpx exceptions to the taxonomy via lazy imports (no hard
  dependency on any SDK). Returns `None` for unrecognised
  exceptions — the framework refuses to silently retry errors it
  doesn't understand.
* **`RetryingModel`** — wraps any `Model`; auto-applied to in-tree
  network adapters (`OpenAIModel`, `AnthropicModel`,
  `LiteLLMModel`) by default. Custom Models are not auto-wrapped
  (we can't reason about their error types); pass
  `retry_policy=RetryPolicy()` to `Agent(...)` to opt in.
  Streaming retries fire only **before** the first chunk — once
  tokens are flowing, errors propagate.
* **`Agent(retry_policy=RetryPolicy.disabled())`** — opt out of
  retries when handling errors at a higher layer.

### Added — Documentation + migration

* **Sphinx docs site** at <https://loomflow.readthedocs.io>
  (`docs/conf.py`, Furo theme, `sphinx-autoapi` for the full API
  reference, `myst-parser` so existing `.md` content mounts
  cleanly). Build locally with `pip install -e ".[docs]"` and
  `sphinx-build -b html docs docs/_build/html`. ReadTheDocs
  integration via `.readthedocs.yaml`.
* **`docs/migrations/from-langgraph.md`** — concrete side-by-side
  translations: hello world, tools, multi-tenant memory (the
  `user_id`-as-convention vs `user_id`-as-primitive contrast),
  session continuity, structured output, streaming, multi-agent.
  Plus a "things Loom does NOT have" section.
* **`docs/migrations/from-openai-sdk.md`** — translation guide for
  hand-rolled-loop users: tool definitions, multi-turn state,
  structured output, retries, streaming, parallel tool calls.
* **`pyproject.toml` `docs` extras** — `sphinx`, `furo`,
  `sphinx-autoapi`, `myst-parser`, `linkify-it-py`.
* **Live integration tests** (`tests/test_live_openai.py`) — 9
  tests, ~16 s wall clock, marked `live` (deselected by default).
  Run with `pytest -m live` once `OPENAI_API_KEY` is set. Covers
  the differentiating contracts end-to-end against `gpt-4.1-mini`:
  basic round-trip, tool dispatch, `user_id` partition, `session_id`
  continuity, tool reads `user_id` via `get_run_context()`,
  structured output, structured output retry-on-failure, real
  auth-error → `AuthenticationError` classification, streaming.
* **CHANGELOG.md** — version-by-version release notes (this file).

### Added — Examples

* `examples/01_rag_pdf.py` — single-agent RAG over a folder of
  PDFs. Loader → `RecursiveChunker` → `ChromaVectorStore` →
  `@tool` retriever → `Agent`.
* `examples/02_specialist_debate.py` — five domain specialists
  (IT / physics / medicine / finance / law), each with their own
  Chroma collection, composed via `Team.debate(...)` with a
  synthesising judge.
* `examples/03_multi_user_sessions.py` — live demo of M1+M2 on one
  shared `Agent` + `InMemoryMemory`. Two users, distinct sessions,
  no cross-contamination, tool reads `user_id` via
  `get_run_context()`.
* `examples/04_structured_outputs.py` — extracts a `MeetingSummary`
  (with nested `ActionItem` lists, ISO dates, sentiment enum)
  from a raw transcript.

### Changed — Public API surface

* **`Agent.resume` signature aligned with `Agent.run`** — gained
  `context=`, `output_schema=`, `output_validation_retries=` kwargs.
  The three call methods (`run`, `stream`, `resume`) now have the
  same kwarg surface in the same order.
* **`OutputValidationError.cause` typed as `BaseException | None`**
  (was `Exception | None`) to match `ModelError.cause`.
* **README intro rewritten** — quickstart now demonstrates `user_id`
  partitioning, `session_id` continuity, and structured outputs in
  a single ~25-line example. The "Why pick this over LangGraph"
  framing is preserved but the differentiating bullets reference
  the actual M1–M5 work.
* **README "API stability" section** — four-tier table (Stable /
  Stable backends / Experimental / Internal) so adopters know what
  they can pin against in production.
* **`Dependencies.context: RunContext`** — new field on the
  per-run dependency bundle architectures receive. Existing
  architectures read `deps.context.user_id` to scope memory recall.

### Performance

* **Fast path by default** — every layer (audit, telemetry,
  permissions, hooks, runtime, budget) is detected as no-op or
  production-wired at construction time. The hot path skips the
  integration when the layer is no-op, so a barebones `Agent`
  runs at LangChain-class latency (parity ±2 % on tool-using
  scenarios; see `bench/jeeves_vs_langchain.py`). The moment any
  of those layers is wired up, the integration becomes active —
  no flag, no constructor split.
* **Non-streaming `Model.complete()`** — every model adapter now
  has a single-shot `complete(...)` method alongside `stream(...)`.
  `agent.run` (no consumer reading from `stream()`) prefers
  `complete()` and skips per-chunk yield + Event construction.
  About 100–200 ms saved per turn on token-heavy responses.

### Quality

* **866 offline tests pass** in ~6 s (5 env-gated integrations
  skip without `JEEVES_TEST_PG_DSN` / `JEEVES_TEST_REDIS_URL`).
* **9 live tests pass** against real OpenAI in ~16 s.
* **mypy `--strict`** clean across **105 production source files**.
* **ruff** clean across `loomflow`, `tests`, `examples`,
  including `flake8-async` lints.

### Compatibility

* All new fields on `Episode`, `Fact`, `RunResult`, and
  `Dependencies` are optional with safe defaults. Existing
  pickled / JSON-serialised records load unchanged.
* All new kwargs on `Agent.run` / `stream` / `resume` are
  keyword-only with `None` / sensible defaults. Existing call
  sites keep working unchanged.
* All new memory-protocol methods (`session_messages`) ship with
  default implementations on every shipped backend; custom
  `Memory` implementations gain a graceful `[]` fallback in
  `_build_seed_messages` so old backends keep working too.

---

## [0.3.0] — unreleased

### Added — Architectures

* **`MultiAgentDebate`** — N debater Agents argue across rounds with
  optional judge synthesis. Round 0 is independent (parallel via
  anyio task group); rounds 1..K each debater sees the full prior
  transcript and defends or updates its position. Naive convergence
  check (whitespace-normalized exact match) terminates early when
  all debaters agree; pass `convergence_check=False` to disable.
  When `judge=None`, falls back to majority vote on the final
  round (modal answer wins, original casing preserved). Each
  debater + judge invocation uses deterministic session ids
  (`{parent}__debater_<i>_round_<r>` / `{parent}__judge`) for
  replay correctness. Min 2 debaters; min 1 round. Useful for
  high-stakes contested questions; reserve for cases where a wrong
  answer is expensive (3-5× cost over single-agent).
* **`TreeOfThoughts`** + **`ThoughtNode`** — branching exploration
  with per-node evaluation (Yao et al. 2023). BFS beam search:
  each level generates `branch_factor` candidates per frontier
  node, evaluator scores each, top `beam_width` survive to the
  next level. Best leaf wins. Early-exit on
  `score >= solved_threshold` (default 1.0). Per-call uses
  `text_only_model_call` so every propose / evaluate is journaled.
  Architecture events surface tree state at each step (`tot.proposed`,
  `tot.evaluated`, `tot.pruned`, `tot.solved`, `tot.completed`).
  Full search tree is stashed on `session.metadata["tot_nodes"]`
  for post-hoc rendering. Resolver string `"tree-of-thoughts"`.
* **`ActorCritic`** — generator + adversarial critic. Both the
  actor and critic are required, separate `Agent` instances —
  same-model self-iteration is what `SelfRefine` is for; ActorCritic
  earns its complexity only when there's actual asymmetry (different
  models, different prompts, different blind spots). Round 0 is
  actor; each round is critic → approve check → actor refine. Critic
  output parsed as JSON (with markdown-fence stripping) into
  `CriticOutput(issues, score, summary)`; regex-only fallback when
  JSON parsing fails. Each actor / critic invocation uses a
  deterministic session id (`{parent}__actor_<round>` /
  `{parent}__critic_<round>`) for replay correctness. Constructor:
  `ActorCritic(actor=..., critic=..., max_rounds=3,
  approval_threshold=0.9)`. Composes inside Supervisor (per-worker
  quality control) and inside Reflexion (cross-session learning of
  effective critique patterns).
* **`Supervisor`** — second multi-agent architecture; the
  hierarchical pattern. Workers (dict of `Agent` instances) +
  a base architecture (default `ReAct`). The supervisor's
  ToolHost is wrapped to inject one extra tool —
  `delegate(worker, instructions)` — that routes calls to the
  named worker `Agent` and returns its output as the tool result.
  Multiple `delegate` calls in a single supervisor turn run in
  parallel for free (ReAct's tool dispatch is already a
  task-group). Custom `delegate_tool_name=` to avoid clashes.
  Worker session ids are uniquely generated per delegation, so
  the same worker can be invoked multiple times in one turn
  without journal collisions. Composition: workers can be any
  architecture themselves (DeepAgent for research, Reflexion for
  cross-session learning). The agent's own `instructions` are
  preserved and the supervisor template is appended.
* **`Agent.instructions`** public property — symmetric with
  `model`/`memory`/`runtime`/`architecture`/etc. Surfaced so
  multi-agent architectures (Supervisor, future Actor-Critic) can
  read each worker's role description when composing supervising
  prompts.
* **`Router`** + **`RouterRoute`** — first multi-agent architecture.
  Classify input → dispatch to ONE specialist `Agent`. Each route is
  a fully-constructed `Agent` (its own model / memory / tools /
  architecture). Specialist runs with a deterministic session_id
  (`{parent}__route_{route_name}`) so replay flows through both the
  parent's classifier journal and the specialist's own journal.
  Optional `fallback_route` + `require_confidence_above` for graceful
  handling of ambiguous inputs and unknown routes from the
  classifier. `declared_workers()` exposes routes by name for
  introspection. NOT registered as a resolver string — Router needs
  config; pass an instance:
  `architecture=Router(routes=[RouterRoute(name="billing",
  agent=billing_agent), ...], fallback_route="general")`.
* **`Reflexion`** — verbal reinforcement learning via memory
  (Shinn et al. 2023). Wraps any base architecture (default
  `ReAct`); each attempt, an evaluator scores the output (0-1) and
  if below `threshold` (default 0.8) a reflector produces a
  one-sentence lesson. Lessons are appended via
  `memory.append_block(lessons_block_name, ...)` so the base
  architecture's own `memory.working()` recall picks them up on
  the next attempt — zero plumbing on the base side. With a
  persistent memory backend (Sqlite / Postgres / Redis), lessons
  carry across process restarts (cross-session learning).
  Constructor: `architecture=Reflexion(base=ReAct(),
  threshold=0.8, max_attempts=3, lessons_block_name="...")` or
  `architecture="reflexion"`.
* **`SelfRefine`** — iterative refinement via critique
  (Madaan et al. 2023). Wraps any `base` architecture (default
  `ReAct`); each round, the same model plays critic and refiner.
  Stops on `stop_phrase` (default `"no issues"`) or after
  `max_rounds`. Composable: `architecture=SelfRefine(base=ReAct(...),
  max_rounds=3)` or `architecture="self-refine"`.
* **`EventKind.ARCHITECTURE_EVENT`** + `Event.architecture_event(
  session_id, name, **data)` factory — generic progress event for
  architecture-specific milestones. Each architecture uses a
  namespaced name (`"self_refine.critique"`,
  `"self_refine.refined"`, `"self_refine.converged"`,
  `"self_refine.max_rounds_reached"`) so consumers can pattern-match
  without expanding `EventKind` per architecture.
* **`loomflow.architecture.helpers`** — shared utilities
  architectures reuse: `text_only_model_call(deps, step_name,
  messages) -> (text, usage)` (one-shot text-only model call,
  journaled for replay) and `add_usage(a, b)` (sum two `Usage`
  records).
* **12 new SelfRefine tests** covering protocol satisfaction,
  stop-phrase early exit, full critique → refine cycles,
  `max_rounds` enforcement, budget gating, progress events, and
  `architecture="self-refine"` resolver string.
* **19 new Reflexion tests** covering protocol, constructor
  validation, score parsing (`"score: X"` patterns +
  fallbacks + clamping), threshold-met early exit, full
  evaluate → reflect → retry cycles, `max_attempts` enforcement,
  lesson persistence into the memory block, lesson visibility on
  the next attempt's seed_context, end-to-end via `ReAct` base,
  and resolver-string construction.
* **21 new Router tests** covering protocol, constructor validation
  (empty / duplicate / invalid fallback / confidence range),
  classification regex (`route:` + `confidence:` + `=`-separator +
  defaults + clamping), successful dispatch, fallback paths
  (low confidence, unknown route), specialist interruption
  propagation, deterministic specialist session_id, architecture
  events surfacing through `Agent.stream`.
* **13 new Supervisor tests** covering protocol, constructor
  validation, single delegation, parallel delegations (two
  delegate calls in one turn), unknown-worker error handling,
  instructions composition (user prompt + template + worker
  descriptions), unique worker session ids per delegation, custom
  delegate tool name, and the helper `_make_delegate_tool`
  building a valid `Tool`.
* **19 new ActorCritic tests** covering protocol, constructor
  validation, critique-parsing (pure JSON / markdown-fenced JSON /
  regex fallback / empty-string default), single-round approval
  (no refine call when critic approves on round 1), full
  refine-then-approve cycles, max_rounds enforcement, full event
  sequence emission, deterministic actor/critic session ids per
  round, and interruption propagation from sub-agents.
* **16 new TreeOfThoughts tests** covering protocol, constructor
  validation (branch_factor / max_depth / beam_width /
  solved_threshold ranges), single-level beam pruning, multi-level
  expansion with deterministic top-by-score selection, early
  termination on solved_threshold, max_depth enforcement, helper
  `_chain_to_root`, full event sequence emission, and beam pruning
  to top-N per level.
* **20 new MultiAgentDebate tests** covering protocol, constructor
  validation (≥2 debaters, ≥1 rounds), helpers (`_normalize`,
  `_converged`, `_majority_vote` including casing preservation),
  parallel round 0, multi-round with history visibility, convergence
  early-exit, judge synthesis path, majority-vote fallback,
  deterministic debater + judge session ids, and full event
  sequence emission.
* **7 new examples** — `examples/09_self_refine.py`,
  `10_reflexion.py`, `11_router.py`, `12_supervisor.py`,
  `13_actor_critic.py`, `14_tree_of_thoughts.py`, `15_debate.py`.
  Each runs deterministically with `ScriptedModel` (no API key) and
  prints a streaming event view plus the final answer.
* **`parse_score(text) -> float`** promoted to
  `loomflow.architecture.helpers` (was private in `reflexion.py`).
  Used by Reflexion, Tree of Thoughts, and any future architecture
  with an evaluator step.
* **32 new ReWOO tests** covering protocol, resolver string,
  constructor validation, placeholder helpers (`_extract_placeholders` /
  `_substitute_placeholders` recursion + dedupe + non-string
  passthrough + unresolved-placeholder leave-as-is), topological
  helpers (`_topological_levels` linear chain / collapsed
  independent-steps level / cycle detection / unknown-dep
  treatment), plan parser (clean JSON / markdown fences /
  malformed-step skipping / auto-id assignment), full
  end-to-end loop with real tools, parallel level execution,
  step-error path that doesn't crash the architecture,
  `max_steps` cap, cyclic plan handling, full event sequence,
  and Pydantic round-trip.
* Total tests: **560** (was 341 in v0.2.0; +219 across the v0.3
  architecture work — 15 foundation + 12 SelfRefine + 19 Reflexion +
  21 Router + 13 Supervisor + 19 ActorCritic + 16 TreeOfThoughts +
  20 MultiAgentDebate + 17 Swarm + 18 BlackboardArchitecture +
  17 PlanAndExecute + 32 ReWOO).

### Added — Architecture protocol foundation

* **`loomflow.architecture`** package — pluggable agent-loop
  strategies. The `Architecture` protocol lets users swap iteration
  patterns (ReAct, Plan-and-Execute, Reflexion, Router, Supervisor,
  ...) without touching memory / runtime / tools / governance. See
  `Subagent.md` in the repo root for the design rationale and
  catalogue of architectures.
* **`Architecture` protocol** (`runtime_checkable`) — every
  architecture implements `name: str`, `async def run(session, deps,
  prompt) -> AsyncIterator[Event]`, and `declared_workers() ->
  dict[str, Agent]`. Architectures are async generators that yield
  `Event` values for milestones; setup / teardown stays in `Agent`.
* **`AgentSession`** — mutable per-run state (id, instructions,
  messages, turns, output, cumulative_usage, interrupted,
  interruption_reason, metadata). Architectures mutate this as they
  iterate; `Agent` reads the final state to build a `RunResult`.
* **`Dependencies`** — bundles every protocol implementation an
  architecture might need (model, memory, runtime, tools, budget,
  permissions, hooks, telemetry, audit_log, max_turns) into one
  struct so `run()` signatures stay short.
* **`ReAct`** — the canonical observe/think/act loop, lifted out of
  `Agent._loop` verbatim. Now the framework's default architecture.
  Constructor takes optional `max_turns` override (useful when
  composing inside other architectures, e.g. `Reflexion(base=ReAct(max_turns=10))`).
* **`Agent(architecture=...)`** kwarg — accepts an `Architecture`
  instance, a known string (`"react"`), or `None` (defaults to
  `ReAct()`). Public `agent.architecture` property exposes it.
* **`resolve_architecture(spec)`** — string / instance / None →
  concrete `Architecture`. Unknown strings raise `ConfigError` with
  the list of known names.
* **15 new tests** (`tests/test_architecture.py`) covering the
  Protocol surface, the resolver, Agent integration, and a custom
  architecture driving an end-to-end run.
* **`Subagent.md`** committed as the architecture reference manual
  for upcoming v0.4+ work (Reflexion, Self-Refine, Plan-and-Execute,
  Router, Supervisor, ...).

### Changed (non-breaking)

* `Agent._loop` is now ~100 lines (was ~330): setup wraps the
  runtime/telemetry context and audits the run boundary, then
  delegates iteration to `self._architecture.run(session, deps,
  prompt)`, then teardown persists the episode and builds the
  `RunResult`. Helpers (`_seed_context`, `_take_one_turn`,
  `_dispatch_tools`, `_run_single_tool`) moved into
  `loomflow/architecture/react.py`.
* The `jeeves.run` telemetry span carries an `architecture` attribute
  alongside `model` / `max_turns` / `session_id`. The audit
  `run_started` payload also includes the architecture name.
* All 341 v0.2.0 tests still pass without modification — the refactor
  is behaviour-preserving.

---

## [0.2.0] — 2026-05-06

### Changed (breaking)

* **`Agent(...)` requires `model=`**. The previous behaviour silently
  defaulted to `EchoModel`, which produced `"Echo: ..."` output that
  users misread as a real LLM response. Forgetting the kwarg now
  raises `ConfigError` with a suggestion list. Migration: pass
  `model="echo"` for tests / zero-key dev, or one of the real
  strings (`"claude-opus-4-7"`, `"gpt-4o"`, `"mistral-large"`, ...).
* **Resolver errors harmonised to `ConfigError`**. `_resolve_model`
  used to raise `ValueError` for unknown specs; now it's
  `ConfigError` with a message that lists every supported prefix and
  the explicit `litellm/` opt-in.

### Added

* **`LiteLLMModel`** — single adapter for ~100 providers via the
  `litellm` SDK (Cohere, Mistral, Bedrock, Vertex, Together, Ollama,
  Gemini, Groq, Replicate, Azure, …). Inherits from `OpenAIModel`
  since LiteLLM normalises every provider's chunks to OpenAI's
  shape — zero new chunk-aggregation code, just a different
  underlying client.
* **String resolver dispatches more prefixes** to `LiteLLMModel`:
  `mistral-`, `command-`, `bedrock/`, `vertex_ai/`,
  `together_ai/`, `ollama/`, `gemini/`, `groq/`, `replicate/`,
  `azure/`. Plus `litellm/<spec>` as an explicit opt-in that strips
  the prefix and forces the LiteLLM path even for specs the direct
  adapters would otherwise grab.
* **`VoyageEmbedder`** — embeddings via Voyage AI's `voyageai` SDK.
  Models: `voyage-3` / `voyage-3-large` / `voyage-code-3` (1024
  dim), `voyage-3-lite` (512 dim). Configurable `input_type`
  (``"document"`` / ``"query"``).
* **`CohereEmbedder`** — embeddings via Cohere's `cohere` SDK.
  Models: `embed-english-v3.0` / `embed-multilingual-v3.0` (1024),
  `embed-english-light-v3.0` / `embed-multilingual-light-v3.0`
  (384). Required `input_type` (``"search_document"`` /
  ``"search_query"``) plus `embedding_types=["float"]` baked in.
* **`Agent.__repr__()`** — concise dev-time inspection:
  ``Agent(model='claude-opus-4-7', memory=InMemoryMemory,
  runtime=InProcRuntime, tools=InProcessToolHost, max_turns=50)``.
* **`RunResult.total_tokens`** (`tokens_in + tokens_out`) and
  **`RunResult.duration`** (`finished_at - started_at`) convenience
  properties.
* **`Agent.consolidate()` returns the count of new facts** extracted.
  ``0`` when no consolidator is configured or the memory backend
  doesn't expose a `.facts` store. Useful for batch consolidation
  loops that want to know whether anything changed.
* **`tools=` accepts a single callable or `Tool`**. Previously you
  had to wrap one tool in a list (`tools=[my_fn]`); now
  `tools=my_fn` and `tools=Tool(...)` both work. List form is
  unchanged.
* **30 new tests** for embedders + polish + tool ergonomics,
  bringing total tests to 279 from 244.
* New `voyage` and `cohere` extras in `pyproject.toml`
  (`pip install 'loomflow[voyage,cohere]'`).
* **`ConsolidationWorker`** — long-running anyio task that calls
  `memory.consolidate()` every N seconds. For long-lived agents
  where per-run `auto_consolidate=True` is wasteful. Surfaces new
  fact counts via `on_consolidated(count)` and consolidator failures
  via `on_error(exc)` so a transient LLM hiccup doesn't kill the
  worker. Doubles as an async-context-manager
  (`async with worker: ...` runs in the background, exiting cancels
  cleanly).
* **`agent.add_tool(fn)`** — register tools after construction.
  Friendly for plugin patterns. Works with the default
  `InProcessToolHost`; raises `ConfigError` with a clear message for
  custom hosts that don't support post-hoc registration.
* **`examples/07_litellm.py`** — runnable LiteLLM dispatch demo,
  picks a model based on which provider key is in the environment.
* **CI install line** — adds `litellm`, `voyage`, `cohere` extras so
  mypy resolves the SDK type hints in their lazy-import paths.
* **Plugin API**: `agent.remove_tool(name)` and
  `agent.tools_list()` round out post-construction tool management.
* **Public introspection** properties on `Agent`: `model`, `memory`,
  `runtime`, `tool_host`, `budget`, `permissions`. Use these instead
  of the `_model` / `_memory` / etc. private attributes.
* **`agent.recall(query, kind=, limit=)`** — convenience wrapper
  around `agent.memory.recall(...)`.
* **Migration guide** at `docs/migration_0.1_to_0.2.md` covering
  both breaking changes and the stale-install-shadowing pitfall.
* **`SubprocessSandbox`** — runs each tool call in a fresh child
  Python process via `multiprocessing` (spawn). Process isolation,
  hard timeout, memory boundary. Wraps any `InProcessToolHost`;
  rejects other host types with a clear error. Picklable
  module-level functions are required.
* **`Memory.recall_facts(query, *, limit, valid_at)` protocol
  method** — formalises the previously-duck-typed `.facts` access
  path. Every memory backend implements it: backends with a fact
  store forward to `self.facts.recall_text`; backends without one
  return `[]`. The agent loop calls `memory.recall_facts(...)`
  directly now.
* **Batch embedding in `InMemoryFactStore.append_many`** — coalesces
  the per-fact `embed()` calls into a single `embed_batch()`
  round-trip when an embedder is configured. The `Consolidator`
  uses `append_many` internally so multi-fact extraction from one
  episode hits the embedder API once instead of N times. Falls back
  to per-fact `append` for stores without `append_many`.
* **`Agent.from_config(toml_path)`** — load an `Agent` from a TOML
  file. Supports declarative `instructions`, `model`, `max_turns`,
  `auto_consolidate`, and a `[budget]` block. Concrete instances
  for `memory` / `runtime` / `tools` / model overrides can be passed
  as kwargs.
* **`Agent.from_dict(cfg)`** — same shape as `from_config` but skips
  the file read. Useful when config comes from env vars, Pydantic
  settings, YAML, an HTTP API, etc. `from_config` now delegates to
  `from_dict` for the parsing logic.
* **`@agent.with_tool` decorator** — register a tool inline:
  ```python
  @agent.with_tool
  async def search(q: str) -> str: ...
  ```
  Returns the function unchanged so it stays directly callable;
  registers it on the underlying `InProcessToolHost`.
* **`PostgresJournalStore` + `PostgresRuntime`** — Phase 5
  production durable runtime. Same `JournaledRuntime` architecture
  as `SqliteRuntime`, but the journal lives in two Postgres tables
  (`journal_steps`, `journal_streams`). Lazy `asyncpg` import.
  Idempotent `init_schema()`. **Note**: this isn't a DBOS-specific
  adapter — DBOS Python's workflow model requires
  `@DBOS.workflow()` / `@DBOS.communicator()` decoration at
  module-load time, which doesn't compose with our generic
  `runtime.step(name, fn, *args)` API. `PostgresJournalStore` gives
  the same durability guarantee with no decorator intrusion; users
  who want the full DBOS workflow surface can layer DBOS on top of
  their own tool functions.
* **`examples/08_from_config.py`** + companion `examples/agent.toml`
  — runnable demo of `from_config` / `from_dict` / `@agent.with_tool`.

---

## [0.1.0] — 2026-05-06

First public release.

### Added

* **Phase 1 — Protocols + types.** 18 Pydantic value objects, 14
  module-boundary `Protocol` definitions, exception hierarchy, ULID
  helpers.
* **Phase 2 — Agent loop.** `Agent` class with `run()` / `stream()` /
  `resume()`. Parallel tool dispatch via `anyio.create_task_group`.
  Streaming events through bounded `anyio.create_memory_object_stream`
  with backpressure + clean cancellation.
* **Phase 3 — MCP spine.** `MCPClient` (lazy `mcp` SDK), `MCPRegistry`
  (auto name-disambiguation across servers), `JeevesGateway`
  one-line wrapper.
* **Phase 4 — Memory + bi-temporal facts.** Five backends:
  `InMemoryMemory`, `VectorMemory`, `ChromaMemory`, `PostgresMemory`,
  `RedisMemory`. Two embedders: `HashEmbedder` (zero-deps,
  deterministic), `OpenAIEmbedder`. **Bi-temporal `FactStore` in
  every backend** with supersession, `valid_at` historical queries,
  embedder-driven cosine recall. LLM-driven `Consolidator` for fact
  extraction with `auto_consolidate=True` opt-in.
* **Phase 5 — Durable runtime.** `JournaledRuntime` + `SqliteRuntime`
  for crash-recovery replay across process restarts. Session-id
  tracking via `contextvars`.
* **Phase 6 — Security + governance + observability.**
  `StandardPermissions` (mode + allow/deny), `HookRegistry`
  (timeout-shielded), `StandardBudget` (token/cost/wall-clock with
  soft warnings), `OTelTelemetry` (spans + metrics for every
  milestone), `FilesystemSandbox` (path-arg validation, symlink
  resolution), `InMemoryAuditLog` + `FileAuditLog` (HMAC-signed),
  `FreshnessPolicy` + `LineagePolicy` validators.
* **Provider adapters.** `AnthropicModel` (`claude-opus-4-7`, etc.),
  `OpenAIModel` (`gpt-4o`, etc.). String-based resolver:
  `Agent(model="claude-opus-4-7")` dispatches by prefix.
* **Resume API.** `agent.resume(session_id, prompt)` + optional
  `session_id` kwarg on `run()` / `stream()`. Pairs with
  `SqliteRuntime` for cross-process replay.

### Documentation

* `README.md` — value prop, capability matrix, install matrix,
  30-second quickstart.
* `docs/quickstart.md` — 14 copy-pasteable examples.
* `docs/recipes.md` — 8 production patterns + 24-item production
  checklist.
* `docs/architecture.md` — module map, lifecycle walkthrough,
  extension-points table.
* `examples/` — 7 runnable scripts mirroring the recipes.

### Engineering

* **Async-only** — anyio everywhere; zero raw `asyncio.create_task`
  / `gather` calls.
* **mypy `--strict` clean** across 53 production source files.
* **243 tests passing** + 4 env-gated live-integration skips.
* **CI** on Python 3.11 + 3.12 (ruff + mypy + pytest + examples
  smoke).
* **Release** workflow via PyPI trusted publishing on `v*` tags.
