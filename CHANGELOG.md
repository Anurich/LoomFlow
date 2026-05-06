# Changelog

All notable changes to JeevesAgent will be documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/).

For development-history detail (per-slice notes, file maps, gate
counts), see [`BUILD_LOG.md`](BUILD_LOG.md).

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
  (`pip install 'jeevesagent[voyage,cohere]'`).
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
