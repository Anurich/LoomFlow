# Examples

Nineteen end-to-end examples that exercise Loom's own
primitives — loader, vector store, retriever-as-tool pattern,
multi-agent architectures, multi-user / session-continuity
primitives, the workflow + agent composition story, observability
sinks, the reasoning-effort dial, TOML / dict-form declarative
config, the shared-notebook workspace for multi-agent coordination,
provider-aware prompt caching, the TodoWrite-style living plan
primitive, and the workspace lifecycle / self-improvement surface.
Nothing pulled in from outside the framework.

## Agent + retrieval + memory

| File | What it shows |
|---|---|
| [`01_rag_pdf.py`](01_rag_pdf.py) | Single-agent RAG over a folder of PDFs. Loader → `RecursiveChunker` → `ChromaVectorStore` → `@tool` retriever → `Agent`. Demonstrates the new `unstructured` / `docling` PDF backends via `--backend` flag. |
| [`02_specialist_debate.py`](02_specialist_debate.py) | Five domain specialists (IT / physics / medicine / finance / law), each with their own folder of PDFs and their own Chroma collection, composed via `Team.debate(...)` with a synthesising judge agent. |
| [`03_multi_user_sessions.py`](03_multi_user_sessions.py) | Multi-user namespacing + conversation continuity on **one** shared `Agent` + `InMemoryMemory`. Demonstrates that `user_id` is a hard partition (Alice's history never surfaces in Bob's recall) and that reusing `session_id` rehydrates prior turns as real chat history. Also shows tools reading scope via `get_run_context()`. |
| [`04_structured_outputs.py`](04_structured_outputs.py) | Type-safe structured outputs. Define a Pydantic `BaseModel`, pass it as `output_schema=`, get a validated typed instance back on `result.parsed`. Demonstrates schema-driven extraction (a `MeetingSummary` with nested `ActionItem`s, ISO dates, sentiment enum) from a raw meeting transcript. |
| [`05_memory_showcase.py`](05_memory_showcase.py) | Every memory backend behind one parameter. Walks through `inmemory` / `sqlite` / `chroma` / `postgres` / `redis` (Postgres/Redis skip gracefully without a DSN), demonstrates `profile(user_id=)` / `forget(user_id=)` / `export(user_id=)` GDPR ops, and shows the `Consolidator` extracting structured facts from raw chat episodes. The `memory=` parameter is the only thing that changes between backends. |

## Workflow primitives

Each file is small (50–200 lines) and demonstrates one workflow
pattern in isolation. Read them in order — each builds on the
previous one's vocabulary.

| File | What it shows | Needs OpenAI? |
|---|---|---|
| [`06_workflow_chain.py`](06_workflow_chain.py) | Linear `Workflow.chain([...])` of plain async functions. The simplest possible workflow shape — no LLM involved, no API key required. Touches `RunContext` propagation, `WorkflowResult.visited`, `per_step` introspection. | No |
| [`07_workflow_route.py`](07_workflow_route.py) | `Workflow.route(classifier, {"a": agent_a, ...})` — classify the question with a tiny model, dispatch to a specialist Agent. Demonstrates "Agent as a workflow node" composition with developer-controlled branching. | Yes |
| [`08_workflow_loop.py`](08_workflow_loop.py) | Refinement loop with cycles: `draft → review → judge → (revise → review → ... → END)`. Shows `add_router` with `END` sentinels, `max_visits_per_node` safety cap, and graceful cap-exceeded handling via `try/except RuntimeError` + the in-place state dict. | Yes |
| [`09_workflow_as_tool.py`](09_workflow_as_tool.py) | `wf.as_tool()` — the opposite composition direction. An open-ended customer-support `Agent` has a deterministic refund workflow available as a tool. Unified audit log shows agent's `tool_call` AND workflow's per-step entries under one `user_id`. | Yes |
| [`10_workflow_architecture.py`](10_workflow_architecture.py) | Agent with `architecture="self-refine"` inside a workflow chain. Demonstrates that workflow shape and agent architecture are orthogonal axes — the architecture is encapsulated inside the agent step; the workflow doesn't see the internal draft → critique → refine iteration. | Yes |
| [`11_workflow_custom_step.py`](11_workflow_custom_step.py) | Agent wrapped in a custom `async def` step. For when "just call agent.run(prev_output)" isn't enough — multi-field prompt formatting, capturing `RunResult` metadata (tokens, turns) into workflow state, post-processing the agent's output. | Yes |

## Observability

| File | What it shows | Needs OpenAI? |
|---|---|---|
| [`12_audit_log.py`](12_audit_log.py) | `InMemoryAuditLog` + `FileAuditLog` — HMAC-signed audit entries written from both Agents (`run_started` / `tool_call` / `tool_result`) and Workflows (`step_started` / `step_completed`). Demonstrates tamper detection via `verify_signature` and seq-counter recovery across process restart. | No |
| [`13_telemetry.py`](13_telemetry.py) | `InMemoryTelemetry` + `ConsoleTelemetry` + `FileTelemetry` + `MultiTelemetry` — four "no collector required" sinks. Inspect the full trace tree (`loom.run` → `loom.turn` → `loom.model.complete` + `loom.tool`) via `.spans()` / `.metrics()`, see them live in stderr, append structured JSONL to disk for `jq` queries, or fan-out across all of them at once. Production path swaps in `OTelTelemetry` with no other code changes. | No |

## Model-tuning knobs

| File | What it shows | Needs API key? |
|---|---|---|
| [`14_effort_dial.py`](14_effort_dial.py) | One enum, every provider's reasoning-effort shape. Runs the same question at each effort tier on Claude Opus 4.7 (the only regime that takes the full `low → xhigh → max` range), shows the agent-default + per-call override pattern, and demonstrates `strict_effort=True` raising `EffortNotSupportedError` when wired to a model that can't honour it. | Anthropic |

## Declarative config

| File | What it shows | Needs API key? |
|---|---|---|
| [`15_config_file.py`](15_config_file.py) | `Agent.from_config("agent.toml")` and `Agent.from_dict({...})` — wire memory, runtime, telemetry, audit log, permissions, budget, architecture, skills, and MCP servers in one declarative file. Each backend block goes through the same resolver Agent uses for its kwargs, so anything you can build inline you can also declare in TOML / YAML / settings. Kwargs override matching cfg entries for things TOML can't express (callable tools, hooks, custom secret stores). | No |

## Multi-agent coordination

| File | What it shows | Needs API key? |
|---|---|---|
| [`16_shared_workspace.py`](16_shared_workspace.py) | `Workspace` — a shared notebook primitive for multi-agent teams. Agents call `note(title, content)` to share findings; teammates read via `list_notes()` / `read_note(slug)` / `search_notes(query)`. Author attribution, slug generation, multi-tenant `user_id` partitioning, and an auto-regenerated `WORKSPACE.md` index are all handled by the framework. Three demos: 5 specialists in `Workflow.parallel` writing concurrently, `Team.supervisor` threading the workspace to its workers, and cross-run persistence (second run sees the first run's notes). | No |

## Cost optimization

| File | What it shows | Needs API key? |
|---|---|---|
| [`17_prompt_caching.py`](17_prompt_caching.py) | `Agent(prompt_caching=True)` — one boolean enables provider-aware prompt caching. **OpenAI**: parses `cached_tokens` from response, applies 0.5x discount automatically. **Anthropic**: injects `cache_control:{type:"ephemeral"}` on the last system block + last tool definition (2 of 4 breakpoints), parses `cache_read_input_tokens` / `cache_creation_input_tokens` from response, applies 0.1x read discount + 1.25x write premium (5m TTL) or 2x (1h TTL). Dict form for advanced: `{"enabled": True, "ttl": "1h", "cache_key": "user_42"}`. Live demo runs the same prompt twice and shows the cache-hit cost drop in `result.cached_tokens_in` + `result.cost_usd`. | OpenAI or Anthropic |

## Living plan (TodoWrite-style structured plan)

| File | What it shows | Needs API key? |
|---|---|---|
| [`18_living_plan.py`](18_living_plan.py) | `Agent(living_plan=True)` — wires `plan_write` + `plan_read` tools that maintain a structured `LivingPlan` the agent atomically rewrites each turn. Steps have `{description, status, finding}` where status is `todo`/`doing`/`done`/`blocked`/`skipped`. The plan tool returns the rendered plan back as markdown so it becomes load-bearing in the conversation — drift becomes structurally hard. When `workspace=` is also wired, the plan mirrors to a `kind="plan"` note and `recall_past_plans(query)` is auto-added for cross-run plan lineage. Plan auto-inherits the workspace via ambient when an `Agent` doesn't set its own. Per-run state via contextvar (concurrent runs on the same Agent have isolated plans). Example uses `ScriptedModel` so it runs offline — no API key needed. | No |

## Workspace lifecycle + self-improvement

| File | What it shows | Needs API key? |
|---|---|---|
| [`19_workspace_lifecycle.py`](19_workspace_lifecycle.py) | The workspace v0.10 lifecycle surface — eight features in one offline run. **Namespacing** (sub-buckets in one workspace; `list_notes` sees all by default). **Versioning** (every `update_note` snapshots `.history`; `list_versions` / `read_version` walk it). **Archive** (`archive_note` soft-hides; still readable by slug). **Questions** (`ask_question` / `answer_question` / `list_open_questions`, opt-in via `questions=True`; cross-author `mark_answered` carve-out). **Semantic search** (optional `embedder=` on the backend; `mode="semantic"|"hybrid"` with RRF). **Citation tracking** (`read_note` logs into a per-run set; `attribute_outcome(success=)` updates `cited_count` / `success_count` / `last_cited_at`). **Relevance-aware search** (`search_notes(boost_relevance=True)` ranks proven notes higher). **Retention** (`prune()` citation-aware GC — keeps what's been *used*, not just what's *recent*). Uses `InMemoryWorkspace` + a deterministic stub embedder, so no API key. Where 16 shows the workspace as multi-agent COORDINATION, 19 shows it as the substrate an agent gets smarter on, run over run. | No |
| [`22_run_until_goal.py`](22_run_until_goal.py) | `Agent(run_until=...)` — the run-until-done loop (the `/goal` pattern). After each architecture pass a small, fast **checker model** (`run_until={"checker": ...}`, falls back to the main model) judges whether a *measurable* stop condition holds; if not, the agent is re-prompted and runs again. Wraps any architecture by hanging a `GoalStopHook` off the existing framework stop-hook loop. Three first-class guardrails — `max_iterations`, `max_no_progress` (bail when N passes change nothing), and `max_cost_usd` — because an unbounded run-until loop is the #1 autonomous-agent failure mode. The hook records why it stopped under `session.metadata["run_until.exit"]` (`condition_met` / `max_iterations` / `no_progress` / `cost_cap` / `budget:*`). Accepts a `str` condition or a dict; also TOML-expressible via `from_config`. Runs offline with `ScriptedModel` worker + checker. | No |

The image-bearing examples (01, 02) generate small sample PDFs on
first run via `reportlab` and cache them under `examples/data/`.
The on-disk Chroma indices are also cached, so subsequent runs only
re-execute the agent loop against OpenAI.

## Run

```bash
# .env should contain OPENAI_API_KEY=sk-...
python examples/01_rag_pdf.py
python examples/02_specialist_debate.py
```

## What's wired up

```
01_rag_pdf.py
─────────────
  examples/data/general/
    company_handbook.pdf
    engineering_guide.pdf
    security_policy.pdf
    support_runbook.pdf
        │
        ▼  loomflow.loader.load(...)
    Document(content=<markdown>)
        │
        ▼  RecursiveChunker(chunk_size=600).split(...)
    list[Chunk]
        │
        ▼  ChromaVectorStore.add(chunks)   (persisted on disk)
    indexed collection 'general_docs'
        │
        ▼  @tool search_docs(query): wraps store.search(query, k=4)
    Agent(model="gpt-4.1-mini", tools=[search_docs])

02_specialist_debate.py
───────────────────────
  examples/data/it/         examples/data/physics/    ...
    it_runbook.pdf            physics_notes.pdf       ...
        │                         │
        ▼                         ▼
  Chroma 'it_docs'         Chroma 'physics_docs'      ...
        │                         │
        ▼                         ▼
  search_it_docs           search_physics_docs        ...
        │                         │
        ▼                         ▼
  Agent (IT tech)          Agent (Physicist)         ...

  Team.debate(
    debaters=[it, phys, med, fin, law],
    judge=Agent("...synthesis judge..."),
    rounds=1,
  )
```
