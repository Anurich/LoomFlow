# Changelog

All notable changes to JeevesAgent will be documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/).

For development-history detail (per-slice notes, file maps, gate
counts), see [`BUILD_LOG.md`](BUILD_LOG.md).

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
* **`jeevesagent.architecture.helpers`** — shared utilities
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
  `jeevesagent.architecture.helpers` (was private in `reflexion.py`).
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

* **`jeevesagent.architecture`** package — pluggable agent-loop
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
  `jeevesagent/architecture/react.py`.
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
