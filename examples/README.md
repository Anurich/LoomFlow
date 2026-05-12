# Examples

Fifteen end-to-end examples that exercise Loom's own
primitives — loader, vector store, retriever-as-tool pattern,
multi-agent architectures, multi-user / session-continuity
primitives, the workflow + agent composition story, observability
sinks, the reasoning-effort dial, and TOML / dict-form declarative
config. Nothing pulled in from outside the framework.

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
