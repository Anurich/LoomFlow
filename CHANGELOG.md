# Changelog

All notable changes to JeevesAgent will be documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/).

For development-history detail (per-slice notes, file maps, gate
counts), see [`BUILD_LOG.md`](BUILD_LOG.md).

## [0.10.0] — unreleased

This release turns JeevesAgent from a working agent harness into a
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

### Added — Resilient model calls (M5)

* **Error taxonomy** — `ModelError` base + `TransientModelError`
  (retryable; carries `retry_after`) + `RateLimitError` (subclass
  of transient) + `PermanentModelError` + `AuthenticationError` +
  `InvalidRequestError` + `ContentFilterError`. All inherit from
  `JeevesAgentError`; existing `except JeevesAgentError` catches
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

* **Sphinx docs site** at <https://jeevesagent.readthedocs.io>
  (`docs/conf.py`, Furo theme, `sphinx-autoapi` for the full API
  reference, `myst-parser` so existing `.md` content mounts
  cleanly). Build locally with `pip install -e ".[docs]"` and
  `sphinx-build -b html docs docs/_build/html`. ReadTheDocs
  integration via `.readthedocs.yaml`.
* **`docs/migrations/from-langgraph.md`** — concrete side-by-side
  translations: hello world, tools, multi-tenant memory (the
  `user_id`-as-convention vs `user_id`-as-primitive contrast),
  session continuity, structured output, streaming, multi-agent.
  Plus a "things JeevesAgent does NOT have" section.
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
* **ruff** clean across `jeevesagent`, `tests`, `examples`,
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
