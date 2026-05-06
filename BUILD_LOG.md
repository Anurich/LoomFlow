# JeevesAgent — Build Log

A running record of every slice we've shipped. Each section captures
what was added, the files touched, the verification gates we ran, and
the test count after the slice landed. Newest at the top.

---

## Slice 17 — Examples + CI workflows

**What landed.** Seven runnable example scripts that mirror the
recipes, plus GitHub Actions workflows for CI and PyPI release.

### `examples/`

* `00_hello.py` — smallest possible agent (no keys, no infra). Echoes
  the prompt and prints the `RunResult` summary.
* `01_real_model.py` — string-based model resolver with graceful
  fallback. Uses the first available of
  `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`; falls back to echo when
  neither is set.
* `02_tools_parallel.py` — `@tool` decorator + parallel dispatch.
  `ScriptedModel` deterministically asks for two tool calls in one
  turn; example reports elapsed time so you can see ~0.2s instead of
  ~0.4s (parallel vs serial).
* `03_streaming.py` — `agent.stream()` event loop. Prints each
  `STARTED` / `MODEL_CHUNK` (text) / `TOOL_CALL` / `TOOL_RESULT` /
  `COMPLETED` event as it arrives.
* `04_facts.py` — bi-temporal fact store. Pre-seeds a "lives in
  Tokyo" fact, supersedes it with "lives in Paris", queries at two
  historical timestamps, then runs the agent with
  `auto_consolidate=True` so additional facts get extracted from new
  conversations.
* `05_durable.py` — `SqliteRuntime` cross-instance replay. Calls a
  counter-incrementing function under one runtime instance; opens a
  fresh runtime against the same DB; verifies the counter didn't
  advance (cached replay) and that a different session DOES execute
  fresh.
* `06_production.py` — full production shape in ~120 lines: model
  + memory + facts + consolidator + runtime + tool host +
  permissions + budget + audit log + telemetry +
  auto-consolidation. Falls back to `EchoModel` when no API key is
  set; everything else is real and writes to a tempdir.
* `examples/README.md` — index table mapping each script to what it
  shows and what env vars it optionally reads.

All seven verified to run end-to-end in the conda dev env.

### `.github/workflows/`

* `ci.yml` — runs on push to `main` and on PRs.
  - Concurrency group cancels older runs on the same ref.
  - Two-job matrix:
    1. **test** (Python 3.11 + 3.12): `pip install -e
       '.[dev,anthropic,openai,otel,chroma,redis]'` then `ruff
       check`, `mypy --strict`, `pytest -v`.
    2. **examples** (Python 3.12, depends on `test`): runs every
       `examples/0*.py` end-to-end as a smoke test.
  - Pip cache keyed on `pyproject.toml` so installs stay fast.
* `release.yml` — runs on `v*` tag push.
  - **build** job: `python -m build`, uploads sdist + wheel as a
    GitHub Actions artifact.
  - **publish** job: downloads the artifact, publishes to PyPI via
    [trusted publishing](https://docs.pypi.org/trusted-publishers/).
    No API token needed once the project is configured at
    `pypi.org/manage/account/publishing/`.

### `pyproject.toml`

Added `chroma` and `redis` extras (those backends were lazy-imported
in source but missing from the optional-deps table). `dev` extras
unchanged; the new ones are opt-in.

### Source unchanged

All gates still green: ruff clean, mypy `--strict` clean across **53
source files**, **236 passed + 4 skipped in 2.29s**.

---

## Slice 16 — Documentation push

**What landed.** A real README plus three deeper docs covering the
public surface, end-to-end recipes, and the architecture map.

### `README.md` (rewritten — 204 lines)

* Hero + value proposition. Three-line code sample at the top so
  readers see the API before any prose.
* "Why this exists" — comparison against LangChain/LangGraph,
  Claude Agent SDK, OpenAI Assistants, CrewAI/AutoGen.
* Three governing principles: deterministic loop / non-deterministic
  core; trust boundary stays outside the sandbox; validate state on
  write not on read.
* Install matrix (extras per provider).
* 30-second quickstart (real code with `@tool` + Anthropic).
* Capability matrix table — every public class / convenience flag with
  a one-line description and the import path.
* Pointers to deeper docs.
* Test/CI status block (236 tests / mypy strict / ruff).

### `docs/quickstart.md` (470 lines)

Fourteen step-by-step examples, each runnable as-is:

1. Hello, agent (no API keys)
2. Real models — string resolver + explicit instances
3. `@tool` decorator (sync + async)
4. Streaming events
5. MCP servers (stdio + Streamable HTTP)
6. Jeeves Gateway one-liner + composition
7. All five memory backends
8. Bi-temporal facts (manual + auto-consolidate + historical query)
9. Durable replay with `SqliteRuntime`
10. OpenTelemetry wiring
11. Audit log
12. Permissions + hooks
13. Filesystem sandbox
14. Budget caps

Plus an "everything together" production-shaped agent in ~30 lines.

### `docs/recipes.md` (378 lines)

Eight production patterns:

1. Customer-support bot with persistent facts
2. Coding assistant with sandboxed FS access
3. Long-running research agent with durable replay
4. Multi-server MCP setup (Jeeves + git + filesystem)
5. Custom embedder (Cohere example)
6. Custom permissions policy (business-hours gate)
7. Streaming UI integration (FastAPI + SSE)
8. **Production checklist** — 24-item bullet list across reliability,
   observability, security, memory, testing.

### `docs/architecture.md` (243 lines)

* Module map matching the engineering plan's `§4` topology.
* Layer rules — strict downward import direction, drawn out.
* Lifecycle walkthrough: every step `agent.run()` takes through
  events, telemetry, audit, runtime journal.
* Streaming internals — the `anyio.create_memory_object_stream` +
  `task_group.cancel_scope.cancel()` pattern that gives backpressure
  + clean cancellation.
* Extension points table — Protocol → "implement this to add a new …".
* Mapping of engineering-plan phases to shipped modules.
* Test patterns by module + recommended reading order for new
  contributors (~1500 LOC for a full picture of the harness).

**No source changes.** All gates still green: 236 passed, 4 skipped
in 2.51s, mypy --strict clean across 53 source files, ruff clean.

---

## Slice 15 — Chroma + Redis fact-store wiring

**What landed.** Two more `FactStore` implementations and the
matching `with_facts=True` flags on `ChromaMemory` / `RedisMemory`.
Every memory backend in the harness now has integrated fact storage.

### `memory/chroma_facts.py` — `ChromaFactStore`

* Facts stored as Chroma documents with bi-temporal metadata:
  `valid_from_ts` / `valid_until_ts` / `recorded_at_ts` (epoch
  floats), `currently_valid` (bool — mirrors `valid_until_ts == 0`
  so it can drive `where` filters directly), JSON-encoded `sources`.
* **Supersession via `where` query**: a `coll.get` selects the prior
  currently-valid facts with matching `(subject, predicate)` and
  filters in Python to those with a different object; a single
  `coll.update` flips their flags.
* `recall_text` uses Chroma's `query()` with the same `where` clauses
  to filter by `valid_at`.
* `_build_where` composes single / multi-clause / `$and`-wrapped
  filters; handles the `currently_valid OR valid_until_ts > ts`
  shape for historical queries.
* `_decode_get` and `_decode_query` handle Chroma's two response
  shapes (flat lists for `get`, nested-per-query lists for `query`).
* `local()` and `ephemeral()` factories mirror `ChromaMemory`.

### `memory/redis_facts.py` — `RedisFactStore`

* Facts stored as Redis hashes at `{prefix}{id}` with the same
  bi-temporal field layout as Chroma's metadata; embeddings packed
  via the shared `pack_float32` util from slice 14.
* **Supersession via brute-force `SCAN`**: walk all `jeeves:fact:*`
  keys, find currently-valid facts with matching subject + predicate
  + different object, `HSET` each to flip the flag and stamp
  `valid_until_ts`. RediSearch indexes are a follow-up; for the
  modest fact volumes typical of agent memory, the scan is fine.
* `_scan_facts` is an async generator with a proper return-type
  annotation (`AsyncIterator[tuple[bytes, dict[bytes, Any]]]`) so
  mypy strict has no quibbles about callers.
* `recall_text` cosine-ranks against stored embeddings for facts
  that have them.

### Memory factory wiring

* `ChromaMemory.local(..., with_facts=True)` and
  `ChromaMemory.ephemeral(..., with_facts=True)` — both gained the
  flag plus `facts_collection_name=` to keep the facts collection
  isolated from episodes. Lazy import of `ChromaFactStore` inside
  the conditional.
* `RedisMemory.connect(..., with_facts=True)` — same pattern, with
  `fact_key_prefix="jeeves:fact:"` so facts and episodes don't
  collide on key namespace.
* Both `ChromaMemory` and `RedisMemory` gained an optional
  `fact_store=` constructor kwarg for callers who want to plumb in
  an explicit instance (matches the `PostgresMemory` shape from
  slice 14).

### Tests added (17)

- **ChromaFactStore (9):** roundtrip append+query, multi-field
  filter, sources round-trip through metadata, supersession closes
  off prior fact, same-object skip, `valid_at` historical query,
  embedding-based cosine recall, `ChromaMemory(with_facts=True)`
  attaches a `ChromaFactStore` round-trip works, default
  `ChromaMemory.facts is None`. Skipped automatically if
  `chromadb` isn't installed; each test gets its own
  UUID-suffixed collection because `EphemeralClient` shares state
  across instances.
- **RedisFactStore (8):** append writes a hash with all expected
  fields and `currently_valid=1`, query filters by subject /
  predicate / object, sources round-trip, supersession closes off
  prior fact, same-object skip, `valid_at` historical query,
  embedding-based cosine recall with a fake embedder. Plus a
  live-Redis test gated on `JEEVES_TEST_REDIS_URL`.

**Gates:** ruff clean, mypy `--strict` clean across **53 source
files**, **236 passed + 4 skipped in 2.47s** (4 skipped are the
live-Postgres / live-Postgres-facts / live-Redis-memory /
live-Redis-facts integration tests gated on env vars).

---

## Slice 14 — Persistent FactStore backends: Sqlite + Postgres

**What landed.** Two durable fact stores so the bi-temporal facts
shipped in slices 12-13 survive process restarts. Plus a tiny
refactor to share the float32-packing helper that already lived in
`memory/redis.py`.

### `memory/_embedding_util.py`

Two functions: `pack_float32` / `unpack_float32`. Used by `redis.py`
and the new `sqlite_facts.py`. The pre-existing `_pack_float32` /
`_unpack_float32` names in `redis.py` are retained as aliases so any
external import path keeps working.

### `memory/sqlite_facts.py` — `SqliteFactStore`

Same shape as `InMemoryFactStore` (supersession on append, `valid_at`
queries, optional embedder) but durable. Pattern mirrors
`SqliteJournalStore` from slice 11:

* sqlite3 + `anyio.to_thread.run_sync` (new connection per call —
  sqlite connections aren't thread-safe).
* `mkdir -p` on parent directory.
* Idempotent schema in `__init__` — `facts` table + two indexes.
* `valid_from` / `valid_until` / `recorded_at` stored as unix-epoch
  floats.
* `sources` stored as JSON-encoded TEXT.
* `embedding` stored as float32 BLOB (nullable).
* Supersession is a single SQL UPDATE that runs *before* the INSERT,
  so the new fact's row never gets caught by its own update.
* `recall_text` branches: cosine over stored embeddings when an
  embedder is configured; token-overlap with stop-word filter
  otherwise. Tokenisation duplicated from `memory/facts.py` rather
  than imported, to avoid a circular import.

### `memory/postgres_facts.py` — `PostgresFactStore`

* Lazy `asyncpg` + `pgvector.asyncpg` imports inside `connect()`.
* Idempotent `init_schema()`: CREATE EXTENSION vector, CREATE TABLE
  facts with `vector(N)` column, two btree indexes, plus an HNSW
  index `vector_cosine_ops` *only when an embedder is configured*
  (no point on a placeholder column).
* Same supersession-then-insert ordering inside an `acquire()` block.
* `recall_text` uses pgvector's `<=>` cosine distance when embedder
  set; falls back to OR'd `ILIKE` clauses across subject / predicate
  / object when not.
* `schema_sql()` exposed for tests and migration tooling.

### `PostgresMemory` integration

`PostgresMemory.__init__` gained `fact_store=` kwarg.
`PostgresMemory.connect(...)` gained `with_facts=True` convenience —
constructs a `PostgresFactStore` rooted at the same pool and attaches
it as `self.facts`. `init_schema()` now runs the fact-store schema in
the same call when `self.facts` is present.

### Tests added (23)

- **SqliteFactStore (12):** init creates table + indexes,
  append+query roundtrip, multi-field filter, token-overlap recall,
  sources JSON roundtrip, supersession closes off prior fact,
  same-triple skips invalidation, `valid_at` historical query, **facts
  survive a fresh instance against the same DB file**, **supersession
  persists across restart**, embedding-based cosine recall, embedding
  recall persists across instances.
- **PostgresFactStore (10):** schema includes pgvector + table +
  indexes, HNSW index appears only when embedder configured,
  dimensions track embedder, append emits supersede-then-insert in
  that order, append with embedder produces a vector arg in the
  INSERT, query assembles the right `$N` placeholder clauses,
  `recall_text` uses pgvector `<=>` with embedder / `ILIKE`
  without, time-window arg passed through, `PostgresMemory.init_schema`
  runs the fact-store schema too.
- **Live integration (1, gated):** `JEEVES_TEST_PG_DSN` triggers a
  real connect → init_schema → append → supersession → query roundtrip.

**Gates:** ruff clean, mypy `--strict` clean across **51 source
files**, **220 passed + 3 skipped in 2.27s** (3 skipped are the
live-Postgres / live-Redis / live-Postgres-facts integration tests
gated on env vars).

---

## Slice 13 — Embedding-based fact recall + auto-consolidation

**What landed.** Two paired upgrades that make slice 12's bi-temporal
fact memory drop-in usable: smarter retrieval via cosine similarity,
and an opt-in auto-consolidate flag so users don't have to call
``memory.consolidate()`` manually.

### Embedding-based recall

* `InMemoryFactStore` gained an optional ``embedder=`` kwarg
  (`Embedder | None`).
* On `append`, when an embedder is configured, the fact's triple
  (`"subject predicate object"`) is embedded and stored alongside.
  Embedding happens *outside* the lock so a network-bound embedder
  (OpenAI) doesn't serialise unrelated reads.
* `recall_text` branches:
  - With embedder: cosine-similarity ranking of the query's embedding
    against each candidate fact's stored embedding. Top-k by score.
  - Without embedder: existing token-overlap path (unchanged).
* `VectorMemory` now defaults its fact store to
  `InMemoryFactStore(embedder=self._embedder)` so the same embedder
  powers episode and fact recall — no extra wiring for users who
  already pass `embedder=OpenAIEmbedder(...)`.
* New helpers: `_triple_text(fact)` (canonical embedding string) and
  `_cosine(a, b)` (the dot-product / norm-product formula, duplicated
  from `vector.py` because the modules sit at the same layer and
  cross-importing for six lines isn't worth the dependency).

### Auto-consolidation

* `Agent.__init__` gained `auto_consolidate: bool = False`.
* When set, `_loop` calls `await self._memory.consolidate()` after
  the response is finalized but before the COMPLETED event, so
  observers see consolidation as part of the same run.
* Failures from the consolidator's LLM call surface as an
  `Event.error` but never break the run — consolidation is
  best-effort post-processing.
* Convenience method `Agent.consolidate()` exposes manual triggering
  in the same API style as `run` / `stream` / `resume`.

### Tests added (10)

- **Embedding-based recall (4):** cosine-sim picks the right fact
  with a fake embedder mapping specific texts to specific vectors;
  top-k returned in score order; absent embedder falls back to
  token-overlap (zero-result case); `VectorMemory` defaults the
  fact-store embedder to its own.
- **Auto-consolidation (5):** facts land in the store after
  `Agent.run()` when `auto_consolidate=True`; default-off doesn't
  run consolidation; consolidator failures emit ERROR events but
  the run still completes; `agent.consolidate()` triggers manually;
  `auto_consolidate=True` with no consolidator configured is a
  silent no-op.
- **Existing tests unchanged:** all 16 prior fact tests still pass.

**Gates:** ruff clean, mypy `--strict` clean across **48 source
files**, **198 passed + 2 skipped in 2.51s**.

---

## Slice 12 — Phase 4 differentiator: bi-temporal facts

**What landed.** The Zep-style memory wedge the engineering plan keeps
calling out. A fact store with bi-temporal validity, an LLM-driven
consolidator that extracts facts from episodes, and Agent-loop
integration that surfaces relevant facts to the model on every turn.

### `memory/facts.py`

* `FactStore` — runtime-checkable Protocol: `append`, `query`,
  `recall_text`, `all_facts`, `aclose`.
* `InMemoryFactStore` — dict-backed, `anyio.Lock`-coordinated.
* **Supersession on append**: a new fact with the same
  ``(subject, predicate)`` as an existing currently-valid fact but
  a different ``object`` causes the old fact's ``valid_until`` to be
  set to the new fact's ``valid_from``. Old beliefs aren't deleted —
  they're closed off so we can still answer "what was true on
  date X" via the ``valid_at=`` query parameter.
* Same-triple appends (subject + predicate + object all match) skip
  invalidation, so re-asserting an unchanged fact is idempotent.
* `query(subject=, predicate=, object_=, valid_at=, limit=)` filters
  by any combination of fields and (optionally) restricts to facts
  valid at a specific timestamp.
* `recall_text(query)` does token-overlap search: tokenize both the
  query and each fact's formatted ``subject predicate object`` triple,
  score by number of overlapping tokens, tie-break by haystack
  length so more-specific facts rank higher. A small stop-word
  list (`the`, `is`, `me`, `my`, `what`, `tell`, ...) keeps naive
  queries like "tell me the user's name" finding "user name_is Alice".

### `memory/consolidator.py`

* `Consolidator` — wraps a `Model` with a default extraction system
  prompt that asks for JSON arrays of
  ``{subject, predicate, object, confidence}``.
* Robust parsing: tolerates ``` json fences, skips records missing
  required fields, clamps `confidence` to `[0.0, 1.0]`, returns an
  empty list on invalid JSON instead of raising.
* `consolidate(episodes, *, store)` runs the model per episode,
  builds `Fact` instances with `valid_from = episode.occurred_at` and
  `recorded_at = utcnow()` and `sources = [episode.id]`, appends each
  to the store. Returns the new facts in extraction order.

### Memory backend integration

* `InMemoryMemory` and `VectorMemory` both gained:
  - Optional `consolidator=` constructor kwarg.
  - Public `.facts: FactStore` attribute (defaults to a fresh
    `InMemoryFactStore`).
  - Real `consolidate()` implementation: pulls episodes not yet in
    `_consolidated_ids`, runs them through the consolidator,
    marks them done. Idempotent — calling consolidate twice doesn't
    double-extract.

### Agent loop integration

`agent/api.py:_seed_context` now pulls facts before episodes:

```python
fact_store = getattr(self._memory, "facts", None)
if fact_store is not None:
    facts = await fact_store.recall_text(prompt, limit=5)
```

Backends without `.facts` are skipped silently — no breaking change
for `PostgresMemory`/`ChromaMemory`/`RedisMemory` which haven't
gained fact support yet. When facts are present, they're prepended
to the system message under a "Known facts:" header so the model
sees them with the highest priority.

### `Fact.format()`

Added on `core/types.py` so the loop can render facts uniformly:
``"user name_is Alice [confidence 0.95]"`` (with a "(until ...)"
suffix when `valid_until` is set).

### Tests added (17)

- **InMemoryFactStore basics (3):** append + query, multi-field
  filters (subject/predicate/object), `recall_text` substring match.
- **Bi-temporal supersession (4):** prior fact's `valid_until` gets
  closed off, same-triple append doesn't invalidate, `valid_at`
  returns the right fact for each historical moment, `valid_at`
  before `valid_from` excludes the fact.
- **Consolidator (5):** extracts facts from a single episode (sources
  list includes the episode id, valid_from inherits from
  `occurred_at`), parses ``` json fenced output, tolerates invalid
  JSON, skips records missing required fields, clamps confidence.
- **Memory backend integration (2):** `InMemoryMemory.consolidate`
  runs the extractor and is idempotent on a second call;
  `VectorMemory.consolidate` processes multi-episode history.
- **Agent loop (3):** facts surface in the system messages a model
  receives, no facts section when the store is empty, agent works
  with backends that don't carry `.facts` at all.

**Gates:** ruff clean, mypy `--strict` clean across **48 source
files**, **189 passed + 2 skipped in 2.32s**.

---

## Slice 11 — Phase 5: durable runtime with journal-based replay

**What landed.** A journal-backed runtime that records every step's
result against a session ID. Re-running with the same session against
the same journal store returns cached values without re-executing the
underlying functions. Survives process restarts when paired with the
sqlite-backed store.

### `runtime/journal.py`

* `JournalEntry` — frozen dataclass holding `(value, created_at)`.
* `JournalStore` — runtime-checkable Protocol: `get_step` /
  `put_step` / `get_stream` / `put_stream` / `aclose`.
* `InMemoryJournalStore` — dict-backed; coordinated by `anyio.Lock`.
  Lost on process exit; ideal for tests and short-lived runs.
* `SqliteJournalStore` — sqlite3 file with two tables (`journal_steps`
  and `journal_streams`); `mkdir -p` on the parent directory at
  construction. Each call opens a fresh connection inside an
  `anyio.to_thread.run_sync` block — sqlite connections aren't safe
  to share across threads, and we hop threads on every async call.
  Pickle for value serialization, since journals only ever hold
  return values from your own trusted tool/model/memory code.

### `runtime/journaled.py`

* `JournaledRuntime` — implements the full `Runtime` protocol on top
  of any `JournalStore`. Maintains current-session state via a
  module-level `contextvars.ContextVar`; anyio's structured
  concurrency propagates it correctly to spawned tasks (so parallel
  tool dispatches inside `_dispatch_tools` see the same session id
  without explicit threading).
* `JournaledSession` — the handle yielded by `runtime.session(id)`,
  with a `deliver(name, payload)` method for external signals.
* When `step()` is called outside any open session, the journal is
  bypassed and the function runs directly — graceful degradation to
  `InProcRuntime` semantics.

### `runtime/sqlite.py`

* `SqliteRuntime(path)` — convenience subclass of `JournaledRuntime`
  preconfigured with a `SqliteJournalStore` rooted at `path`. The
  one-line drop-in for users who want durable replay without
  external infrastructure.

### Agent loop change

`agent/api.py` — `_loop` now opens the runtime session before any
side effects:

```python
async with (
    self._runtime.session(session_id),
    self._telemetry.trace("jeeves.run", ...),
):
    ...
```

The wrap is a no-op for `InProcRuntime` and turns on journaling for
`JournaledRuntime`/`SqliteRuntime`. All existing tests pass
unchanged.

### Tests added (20)

- **JournaledRuntime step (4):** runs once per session and replays
  thereafter, runs every time outside any session, isolated across
  different sessions, args don't affect cache key (step_name does).
- **JournaledRuntime stream_step (2):** chunks replayed without
  re-executing the underlying generator, runs every time outside
  any session.
- **Contextvar propagation (1):** spawned tasks under a task group
  see the same session id; replay returns identical values across
  parallel + replay.
- **Agent integration (3):** `runtime.step("persist_episode_*")` and
  `runtime.step("tool_call_*_*")` keys land in the journal; full
  manual-replay flow proves a tool function never runs twice on
  replay.
- **SqliteJournalStore (5):** value/stream roundtrip, missing keys
  return `None`, replace existing, parent directory auto-created.
- **Cross-instance persistence (3):** new `SqliteRuntime` against
  the same DB file replays cached values from the prior instance,
  for both `step` and `stream_step`. Concurrent session ids in the
  same DB don't collide.
- **Agent + SqliteRuntime (2):** `path` property, end-to-end run
  writes a `persist_episode_1` row whose value is the episode id
  string returned by `InMemoryMemory.remember`.

**Gates:** ruff clean, mypy `--strict` clean across **46 source
files**, **172 passed + 2 skipped in 2.49s**.

DBOS and Temporal adapters deferred to follow-up slices — they need
real infrastructure (Postgres for DBOS, a Temporal cluster for
Temporal) for honest tests, and the current `JournaledRuntime` +
`SqliteRuntime` already covers the single-process durable use case.

---

## Slice 10 — Phase 6 essentials: sandbox + audit + certified values

**What landed.** The remaining Phase 6 deliverables in one slice:
filesystem-aware sandbox, append-only signed audit log wired into the
Agent loop, and freshness/lineage policies for certified values.

### Sandbox (`security/sandbox/`)

* `NoSandbox` — pass-through `ToolHost` wrapper. Documents the
  wrapping pattern; useful as a layer placeholder.
* `FilesystemSandbox` — wraps a `ToolHost` and rejects calls whose
  path-typed arguments resolve outside one or more declared roots.
  Detection has two modes:
  - **Auto (default)**: any string arg whose name is in
    `DEFAULT_PATH_ARG_NAMES` (path/file/directory/dir/folder/src/...)
    *or* whose value contains a path separator gets validated.
  - **Explicit**: pass `path_args=("path", "destination")` to
    validate exactly those names.
  Symlinks are resolved before the containment check so an attacker
  can't bypass the sandbox by symlinking `/etc/passwd` into a
  whitelisted directory. Multi-root configurations work; the
  resolved path must be inside at least one allowed root.

Both sandboxes implement `ToolHost` so they slot into
`Agent(tools=sandbox)` directly — no agent-side changes needed.

### Audit log (`security/audit.py`)

* `AuditLog` — runtime-checkable Protocol (`append`, `query`).
* `InMemoryAuditLog` — list-backed. Fast; great for tests and dev.
* `FileAuditLog` — JSONL append on disk, `mkdir -p` on the parent
  directory at construction. Recovers the highest `seq` from the
  file on startup so process restarts pick up where the last run
  left off.
* HMAC-SHA256 signatures over a canonicalised representation
  (sorted-keys JSON of seq/timestamp/session_id/actor/action/payload).
  `verify_signature(entry, secret)` recomputes and compares with
  `hmac.compare_digest`. Tampering with the payload invalidates the
  signature.
* Wired into `Agent` via the new `audit_log: AuditLog | None` param.
  Emits four entry types per run lifecycle:
  - `run_started` (actor=user, payload includes prompt[:500],
    model, max_turns)
  - `tool_call` (actor=model, payload includes tool name, call_id,
    args, destructive flag, turn)
  - `tool_result` (actor=system, payload includes tool name,
    call_id, ok, denied, error, reason, turn)
  - `run_completed` (actor=system, payload includes turns,
    interrupted, interruption_reason, tokens, cost, elapsed_ms)
* When `audit_log` is `None` (default), the agent skips emission
  with zero overhead.

### Certified values (`data/lineage.py`)

The `CertifiedValue` type already lived in `core/types.py` from
slice 1. This slice adds the *policy* layer:

* `FreshnessPolicy` — frozen dataclass mapping source-prefix tuples
  to `timedelta` max-ages, with an optional `default`. First
  matching prefix wins. The classmethod `from_dict(per_source=,
  default=)` is the conventional constructor.
* `LineagePolicy` — frozen dataclass holding an allow-list of
  source prefixes. The value's own `source` plus every entry in its
  `lineage` tuple must `startswith` an allowed prefix.
* Two helper styles for each: `check_*` returns bool;
  `require_*` raises `FreshnessError` / `LineageError` from
  `core.errors`.
* `valid_until` on the value itself always wins over the
  per-source freshness rule — a server that sets an explicit
  expiry overrides the policy's max-age.

### Files touched

Top-level `__init__.py` re-exports the eight new public names.
`security/__init__.py` re-exports the sandbox + audit additions.
`agent/api.py` gets the `audit_log` constructor param + four
`_audit(...)` calls in `_loop` and `_dispatch_tools`.

New modules: `security/sandbox/{__init__.py, base.py,
filesystem.py}`, `security/audit.py`, `data/__init__.py`,
`data/lineage.py`. New tests:
`tests/test_sandbox.py` (10), `tests/test_audit.py` (13),
`tests/test_lineage.py` (13).

### Tests added (37)

- **NoSandbox (2):** pass-through list_tools/call, end-to-end
  inside an Agent loop with a scripted tool call.
- **FilesystemSandbox (8):** allows paths inside root, blocks paths
  outside, **resolves symlink escapes**, explicit `path_args=`
  overrides auto-detection, skips non-path strings, requires at
  least one root, multi-root containment, introspection
  properties, relative paths resolved against cwd before check.
- **InMemoryAuditLog (4):** monotonic seq, HMAC verifies and
  rejects wrong secret, query filters by session_id and by action.
- **FileAuditLog (4):** JSONL written, seq recovered on restart,
  query reads back, parent directory created.
- **Agent wiring (5):** run_started + run_completed appear in
  order, tool_call + tool_result entries with right payload, no
  audit_log means no overhead, run_completed payload carries the
  run summary, signed entries verify and tamper-detect.
- **FreshnessPolicy (6):** no-rule = fresh, default rule applies
  when no prefix matches, per-source overrides default,
  `valid_until` overrides age, first matching prefix wins,
  `require_freshness` raises on stale.
- **LineagePolicy (7):** empty policy accepts everything, allows
  all-matching, denies disallowed ancestor, denies disallowed
  source, `require_lineage` raises and is silent when allowed.

**Gates:** ruff clean, mypy `--strict` clean across **43 source
files**, **152 passed + 2 skipped in 2.44s** (the 2 are live-
integration tests for Postgres and Redis gated on env vars).

---

## Slice 9 — Phase 4: embedders + 5 memory backends

**What landed.** Five storage backends, two embedders, all behind the
existing ``Memory`` protocol so users pick by passing
``Agent(memory=...)``.

### Embedders (`memory/embedder.py`)

* `HashEmbedder` — deterministic SHA256-seeded Gaussian unit vectors.
  Zero deps. Same text always produces the same vector. Default for
  tests, demos, and zero-key dev. Configurable dimension (default 384).
* `OpenAIEmbedder` — wraps `openai.AsyncOpenAI.embeddings.create`.
  Lazy SDK import. Defaults: `text-embedding-3-small` (1536 dim),
  `text-embedding-3-large` (3072), `text-embedding-ada-002` (1536).
  Optional `dimensions=` param projects 3-* models down (matrix
  mult done server-side).

### Memory backends

* `InMemoryMemory` — already there from slice 2; naive dict, no
  embeddings. Kept as the "no infrastructure" default.
* `VectorMemory` (`memory/vector.py`) — in-memory dict + cosine
  similarity recall. Pure Python, no external deps. Time-range
  filter, max-episodes FIFO eviction, blocks/episodes locked with
  `anyio.Lock`. Scales to a few thousand episodes.
* `ChromaMemory` (`memory/chroma.py`) — backed by `chromadb`,
  either `EphemeralClient` (in-memory) or `PersistentClient`
  (on-disk). Sync Chroma calls dispatched through
  `anyio.to_thread.run_sync` so the event loop stays free. Working
  blocks kept in-process (small, re-derivable). `_safe_list` helper
  tolerates Chroma's numpy-array embedding fields (which can't be
  truthy-tested with `or []`).
* `PostgresMemory` (`memory/postgres.py`) — Postgres + pgvector
  with HNSW index. `connect(dsn, ...)` opens an `asyncpg` pool
  and registers the pgvector codec. `init_schema()` is idempotent:
  CREATE EXTENSION, two tables (`memory_blocks`, `episodes` with
  `vector(N)` column), btree index on `(namespace, occurred_at)`,
  HNSW index on embedding with `vector_cosine_ops`. Lazy
  `asyncpg` + `pgvector.asyncpg` imports.
* `RedisMemory` (`memory/redis.py`) — two modes:
  - **vector mode** (default): tries `FT.CREATE ... HNSW` for
    RediSearch. KNN recall via `FT.SEARCH @embedding $vec ...`.
  - **brute-force mode**: scans every hash key under
    `jeeves:episode:*` and computes cosine in process. Used
    automatically when RediSearch isn't on the server (or
    `use_vector_index=False`).
  Float32 packing (`_pack_float32` / `_unpack_float32`) keeps the
  on-wire embedding compact. Lazy `redis.asyncio` import.

### Tests added (32)

- **HashEmbedder (5):** determinism, distinct-text-distinct-vector,
  unit-norm output, batch matches single, rejects 0 dimensions.
- **OpenAIEmbedder (5):** default dimensions per model, explicit
  dimensions propagate, embed/batch return correct shape, empty
  batch returns empty.
- **VectorMemory (7):** auto-attaches embedding on remember,
  exact-text query returns the matching episode, blank query
  falls back to recency, time-range filter, FIFO eviction at
  `max_episodes`, block update + append, full Agent.run() with
  `tools=ScriptedModel` persists 3 episodes and recall finds them.
- **ChromaMemory (4):** roundtrip, blank query recency, in-process
  blocks, full Agent.run() persistence. Skipped when `chromadb`
  isn't installed; uses unique UUID-suffixed collections per test
  because `EphemeralClient` shares state across instances.
- **PostgresMemory (6):** schema SQL contains CREATE EXTENSION /
  vector(N) / HNSW / vector_cosine_ops, dimensions track embedder,
  remember inserts with namespace + embedding, recall uses pgvector
  `<=>` operator, blank-query path uses `ORDER BY occurred_at`,
  time-range bounds. Plus a live-Postgres test gated on
  `JEEVES_TEST_PG_DSN` env var (skipped in CI).
- **RedisMemory (5):** float32 pack/unpack roundtrip, remember
  writes a hash with packed embedding, brute-force recall returns
  the best exact-text match (deterministic via empty `output`
  trick), blank-query recency, `ensure_index` issues `FT.CREATE`.
  Plus a live-Redis test gated on `JEEVES_TEST_REDIS_URL`.

### Pyproject

No changes — the existing `[chromadb]`, `[postgres]`,
`[redis]` extras (where defined) and lazy SDK imports already cover
the optional-dependency story.

**Files touched:** top-level `__init__.py` re-exports the new
backends and embedders, plus new `memory/embedder.py`,
`memory/vector.py`, `memory/chroma.py`, `memory/postgres.py`,
`memory/redis.py`, `tests/test_embedder.py`,
`tests/test_vector_memory.py`, `tests/test_chroma_memory.py`,
`tests/test_postgres_memory.py`, `tests/test_redis_memory.py`.

**Gates:** ruff clean, mypy `--strict` clean across **37 source
files**, **115/115 tests passing in 2.50s** (plus 2 skipped live-
integration tests gated on env vars).

---

## Slice 8 — OpenTelemetry telemetry

**What landed.** Real telemetry adapter — every milestone in the loop
emits an OTel span, every quantitative output emits a metric. Honeycomb,
Datadog, LangSmith, etc. dashboards just work without any custom
integration.

* `observability/tracing.py` — two implementations:
  - `NoTelemetry` is the default. `trace()` is an `@asynccontextmanager`
    that yields a stub `Span`; `emit_metric()` is a no-op. Both are
    intentionally as cheap as possible so wrapping every loop step
    costs essentially nothing when no exporter is configured.
  - `OTelTelemetry` (lazy `opentelemetry` import) wraps the SDK's
    `start_as_current_span` and routes metric writes by name suffix:
    names ending in `_ms`, `_seconds`, or `_bytes` go to histograms;
    everything else goes to counters. Counter and histogram instruments
    are cached per-name so repeated emissions don't re-create them.
    `None` attribute values are filtered out (OTel rejects them).
    Exceptions inside a `trace()` block are recorded on the span and
    set the status to `ERROR` before re-raising.
* `agent/api.py` — wired telemetry into the loop. Spans:
  - `jeeves.run` (root, attrs: `session_id`, `max_turns`, `model`)
  - `jeeves.turn` (attrs: `turn`, `session_id`)
  - `jeeves.model.stream` (attrs: `model`, `turn`, `session_id`,
    `tool_count`)
  - `jeeves.tool` (attrs: `tool`, `call_id`, `turn`)
  - Parent-child propagation works through `anyio.create_task_group`
    spawns because OTel uses contextvars; spans inside parallel tool
    dispatch correctly nest under `jeeves.turn`.
  Metrics:
  - `jeeves.tokens.input` / `jeeves.tokens.output` (counters,
    attrs: `session_id`, `model`)
  - `jeeves.cost.usd` (counter; only emitted when non-zero)
  - `jeeves.tool.duration_ms` (histogram, attrs: `tool`, `ok`,
    `denied`)
  - `jeeves.session.duration_ms` (histogram, attrs: `session_id`,
    `interrupted`, `turns`)
  - `jeeves.budget.exceeded` (counter, incremented when budget
    blocks a step)
* New `Agent(..., telemetry=...)` parameter; defaults to
  `NoTelemetry()` so existing tests pass unchanged.

**Files touched:** `agent/api.py`, top-level `__init__.py`, plus new
`observability/__init__.py`, `observability/tracing.py`,
`tests/test_telemetry.py`.

**Tests added (14):**

- `NoTelemetry.trace()` yields a span object with the right attrs.
- `NoTelemetry.emit_metric()` returns without error for both counter
  and histogram-suffix names.
- `OTelTelemetry.trace()` produces a span captured by the in-memory
  exporter with the expected name and attribute values; `trace_id`
  is the 32-hex 128-bit format.
- `None` attribute values are filtered before reaching OTel.
- Exceptions inside `trace()` are recorded on the span, status set
  to ERROR, then re-raised.
- `emit_metric()` routes `widget.count` to a counter and
  `widget.duration_ms` to a histogram; both readable through the
  `InMemoryMetricReader`.
- Metric attribute values propagate; counters with different
  attribute sets aggregate independently.
- A full `Agent.run()` produces `jeeves.run`, `jeeves.turn`,
  `jeeves.model.stream` spans plus `jeeves.tokens.input/output` and
  `jeeves.session.duration_ms` metrics.
- The `jeeves.run` span carries `session_id` matching the
  `RunResult` and `model` attribute matching the configured adapter.
- The `jeeves.turn` span's parent is the `jeeves.run` span (verified
  via parent context's `span_id`).
- Tool dispatch produces a `jeeves.tool` span with `tool` and
  `call_id` attributes plus a `jeeves.tool.duration_ms` histogram
  point with `tool` / `ok` / `denied` attributes.
- Two parallel tool calls produce two independent `jeeves.tool`
  spans with the right tool names.
- Budget exhaustion increments the `jeeves.budget.exceeded`
  counter and the run completes with `interrupted=True`.
- Two `Agent.run()` invocations produce two
  `jeeves.session.duration_ms` histogram counts.

**Gates:** ruff clean, mypy `--strict` clean across **32 source
files**, **83/83 tests passing in 1.08s**, no real telemetry
backend required (everything goes through the OTel SDK's in-memory
exporters).

---

## Slice 7 — Jeeves Gateway integration

**What landed.** First-party convenience wrapper for the public Jeeves
MCP gateway. Drops straight into ``Agent(tools=...)`` with one line.

* `jeeves/client.py` — `JeevesConfig` (frozen: api_key, base_url,
  server_name) and `JeevesGateway` class. The gateway is itself a
  `ToolHost`: it lazy-builds an `MCPRegistry` rooted at a single
  `MCPServerSpec.http(...)` whose URL is `{base_url}/{api_key}` (the
  Jeeves Gateway uses URL-path token auth). Constructors:
  - `JeevesGateway(JeevesConfig(...))` — direct.
  - `JeevesGateway.from_env(env_var=, base_url=, server_name=)` —
    reads `JEEVES_API_KEY` (default), strips whitespace, raises
    `ConfigError` with a helpful message if missing or empty.
* Composition methods:
  - `as_mcp_server()` returns the `MCPServerSpec` for inclusion in a
    multi-server `MCPRegistry` (e.g. Jeeves + git + filesystem).
  - `as_registry()` returns a fresh single-server `MCPRegistry`.
  - The class itself satisfies `ToolHost` by forwarding
    `list_tools` / `call` / `watch` / `aclose` to the lazy registry.
* `looks_like_jeeves_key()` — permissive prefix check for the
  `jm_sk_` token shape; helper for callers who want to validate
  inputs without blocking unconventional key formats.
* Injection seam for tests: `JeevesGateway(cfg, registry=...)` lets
  tests pass a pre-built registry with fake MCP clients, bypassing
  any network setup.

**Files touched:** top-level `__init__.py` re-exports
`JeevesConfig` / `JeevesGateway`; new `jeeves/__init__.py`,
`jeeves/client.py`, `tests/test_jeeves.py`.

**Tests added (16):**

- Constants match the documented values.
- Empty `api_key` raises `ConfigError` at construction time.
- `from_env()` reads `JEEVES_API_KEY`, strips whitespace, returns a
  configured gateway.
- `from_env()` raises `ConfigError` when env var unset.
- Custom `env_var=`, `base_url=`, `server_name=` kwargs are honored.
- `as_mcp_server()` produces an HTTP spec with URL-path auth
  (`base_url/api_key`).
- Custom base_url and server_name flow through into the spec.
- `as_registry()` produces an `MCPRegistry` with the right server
  list.
- `looks_like_jeeves_key()` accepts only `jm_sk_*` strings.
- ToolHost forwarding via injected fake registry: list_tools, query
  filter, call routing with call_id, aclose propagation.
- **End-to-end:** `Agent("...", tools=JeevesGateway.from_env())`
  pattern works — model emits a tool call, gateway routes it via
  the underlying registry to the fake MCP session, result feeds
  back into the next model turn.

**Gates:** ruff clean, mypy `--strict` clean across **30 source
files**, **69/69 tests passing in 0.94s**, no real Jeeves API hits.

---

## Slice 6 — MCP client + registry

**What landed.** A `ToolHost` implementation backed by Model Context
Protocol servers. This is the architectural spine the engineering plan
keeps emphasizing — the same agent code now talks to Jeeves Gateway,
Composio, or any MCP server with zero extra glue.

* `mcp/spec.py` — `MCPServerSpec` frozen dataclass. Two constructors:
  `MCPServerSpec.stdio(name, command, args=, env=)` for subprocess
  servers, `MCPServerSpec.http(name, url, headers=)` for Streamable
  HTTP. Args/env stored as tuples-of-tuples so the spec stays hashable
  and `frozen=True` works.
* `mcp/client.py` — `MCPClient` wrapping a single `mcp.ClientSession`.
  All `mcp` SDK imports are **lazy** and live inside
  `connect()` / `_open_transport()` so the module loads without the
  `mcp` extra installed. `connect()` opens the right transport
  (`stdio_client` or `streamablehttp_client`), wraps it in a
  `ClientSession`, calls `initialize()`, and stores everything in an
  `AsyncExitStack`. `aclose()` unwinds via `stack.aclose()`. Tests
  inject `session=` to bypass the connection entirely. Implements
  `__aenter__`/`__aexit__` for use with `async with`.
* `mcp/registry.py` — `MCPRegistry` implementing
  `ToolHost`. `connect()` connects all clients in parallel through an
  `anyio.create_task_group`, then `refresh()` pulls every client's
  tool list (also in parallel) and rebuilds the name index.
  Disambiguation: a name is keyed bare when unique across all
  servers, qualified as `server.name` when two servers expose the
  same name. Either form is accepted at `call()` time; the registry
  strips the prefix back off before forwarding to the underlying
  session. Tool errors (SDK exceptions, `isError=True` results,
  unknown names) all surface as `ToolResult.error_(...)` rather than
  raising. `_extract_output` prefers `structuredContent` (newer MCP
  spec), falls back to concatenated text content blocks, then to
  the raw block list.

**Files touched:** `core/errors.py` (no changes needed — `MCPError`
already there from slice 1), top-level `__init__.py`, plus new
`mcp/__init__.py`, `mcp/spec.py`, `mcp/client.py`, `mcp/registry.py`,
`tests/test_mcp.py`.

**Tests added (14):**

- `MCPServerSpec.stdio()` / `.http()` constructors store fields in
  the right shape.
- Client with injected fake session skips the real `connect()`
  (verified via `initialized=False` flag).
- Client `call_tool()` passes args through to the session.
- Registry aggregates tools from N servers; bare names when unique.
- Registry auto-qualifies `server.tool` when names collide across
  servers.
- Registry routes `call(tool, ...)` to exactly one server (others
  see no calls).
- Qualified `server.tool` form works and forwards the bare name to
  the underlying session.
- Unknown tool returns `ToolResult.error_(...)` with a descriptive
  message.
- A session that raises during `call_tool()` surfaces as an error
  result rather than propagating.
- `isError=True` from the SDK marks the result failed and uses the
  text-content as the error message.
- `structuredContent` is preferred over text blocks when both are
  present.
- `list_tools(query=...)` substring-filters on name and description.
- **End-to-end:** an `Agent` configured with `tools=registry`
  dispatches a model-emitted `ToolCall` through the registry to
  the right fake session and feeds the result back into the next
  model turn.

**Gates:** ruff clean, mypy `--strict` clean across **28 source
files**, **53/53 tests passing in 0.78s**, no `mcp` SDK ever invoked
(all tests use injected fake sessions).

---

---

## Slice 5 — Anthropic + OpenAI model adapters

**What landed.** Real provider adapters for Claude and GPT, plus a
string-based model resolver on `Agent`.

* `Message` gained `tool_calls: tuple[ToolCall, ...] = ()` so
  assistant turns carry the calls they emitted; real providers need
  this in conversation history. Fields reordered in `core/types.py` so
  `ToolCall` is defined before `Message`.
* `model/anthropic.py` — `AnthropicModel`. Lazy
  `anthropic.AsyncAnthropic` import inside `__init__` (so
  `from jeevesagent.model import AnthropicModel` works without the
  extra installed; ImportError fires only when constructing without
  a `client=`). Streams via `messages.stream` and normalises events:
  `text_delta` -> `ModelChunk(kind="text", ...)`,
  `input_json_delta` accumulated per `content_block.index`,
  `content_block_stop` flushes the assembled `tool_call`,
  `message_delta.stop_reason` becomes `finish_reason`. System
  messages collapse into Anthropic's top-level `system` field; tool
  results queue and flush as a user turn with `tool_result` blocks
  before the next non-tool message.
* `model/openai.py` — `OpenAIModel`. Lazy `openai.AsyncOpenAI`
  import. Streams via `chat.completions.create(stream=True,
  stream_options={"include_usage": True})`. `delta.content` flows
  through directly; `delta.tool_calls[*].index` keys the per-call
  accumulator (id, name, args JSON); final `tool_call` chunks emit
  after the stream ends. Message conversion serialises
  `tool_calls[].function.arguments` as JSON.
* `Agent.__init__` now accepts `model: Model | str | None`. A
  resolver `_resolve_model` dispatches by prefix:
  `claude-* -> AnthropicModel`, `gpt-*/o1-*/o3-* -> OpenAIModel`,
  `echo -> EchoModel`. Unknown strings raise `ValueError`.

**Files touched:** `core/types.py`, `agent/api.py`,
`model/__init__.py`, top-level `__init__.py`, plus new
`model/anthropic.py`, `model/openai.py`, `tests/test_anthropic.py`,
`tests/test_openai.py`.

**Tests added (12):** chunk normalization for both providers,
partial-JSON tool-arg accumulation, `Agent.run()` with fake clients
returning expected text, message-format conversion (system collapse,
tool_use/tool_result blocks for Anthropic; `tool_calls` array and
`tool_call_id` for OpenAI), tool-def conversion, resolver dispatch,
unknown-string error.

**Gates:** ruff clean, mypy `--strict` clean across 24 source files,
**39/39 tests passing in 0.89s**, no real API calls.

---

## Slice 4 — Streaming + Events

**What landed.** Two execution surfaces that share a single internal
loop: `run()` returns a `RunResult`; `stream()` yields `Event`s as
they happen.

* `Agent._loop(prompt, *, emit)` is the shared core. `emit` is a
  `Callable[[Event], Awaitable[None]]` that gets called at every
  milestone.
* `Agent.run()` passes `_noop_emit` and returns the `RunResult`
  directly.
* `Agent.stream()` runs `_loop` in a background task spawned inside
  `anyio.create_task_group`. Events go through an
  `anyio.create_memory_object_stream[Event](max_buffer_size=128)` —
  bounded buffer so a slow consumer applies backpressure to the
  producer instead of unbounded buffering. The generator yields from
  the receive end; on consumer break the `finally` block calls
  `tg.cancel_scope.cancel()` so the producer stops cleanly even
  mid-tool-call.
* Events emitted: `STARTED`, `MODEL_CHUNK` (per chunk),
  `TOOL_CALL` + `TOOL_RESULT` (per dispatch),
  `BUDGET_WARNING` / `BUDGET_EXCEEDED`, `ERROR` (shielded send before
  re-raise so the consumer always sees the error before the producer
  task fails), `COMPLETED` (carries
  `result.model_dump(mode="json")`).

**Files touched:** `agent/api.py`, plus new `tests/test_streaming.py`.

**Tests added (7):** ordering (STARTED first, COMPLETED last),
model-chunk emission count, TOOL_CALL/TOOL_RESULT pair with matching
call_ids, parity between `run()` output and the COMPLETED event
payload, **consumer-break cancels a 2-second tool in <0.5s** (proves
cancellation propagation), `BUDGET_EXCEEDED` precedes `COMPLETED`,
parallel tool calls each emit a call/result pair.

**Gates:** ruff clean, mypy strict clean, **27/27 tests passing in
0.28s**.

---

## Slice 3 — Tool dispatch with permissions, hooks, parallel fan-out

**What landed.** Multi-turn loop that dispatches tools in parallel,
gated by user hooks and a system permission policy.

* `tools/registry.py` — `Tool` dataclass + `@tool` decorator. The
  decorator derives a JSON-Schema-style `input_schema` from parameter
  type hints (primitives -> JSON types; anything else -> `string`).
  Both async and sync callables are supported; sync ones are
  dispatched to a worker thread via `anyio.to_thread.run_sync` so they
  don't block the event loop.
* `tools/registry.py` — `InProcessToolHost`: dict-backed
  `ToolHost`. `list_tools()` supports a `query=` substring filter;
  `call()` returns `ToolResult.error_` for unknown tools rather than
  raising. `watch()` is an async generator that yields nothing
  (using the `for ev in (): yield ev` idiom so mypy strict is happy).
* `security/permissions.py` — `Mode` enum (DEFAULT / ACCEPT_EDITS /
  BYPASS) mirroring the Claude Agent SDK so users don't relearn.
  `AllowAll` (default) and `StandardPermissions` with deny-list ⊂
  allow-list ⊂ mode precedence. `destructive=True` calls become
  `ask` in default mode (treated as deny when no hook overrides).
* `security/hooks.py` — `HookRegistry` implementing `HookHost`.
  Pre-tool callbacks: first deny wins. Post-tool callbacks:
  best-effort, exceptions absorbed. Both run inside
  `anyio.move_on_after(5.0)` so a buggy hook can't hang the loop.
* `model/scripted.py` — `ScriptedModel` / `ScriptedTurn` for
  multi-turn test fixtures (canned responses, no LLM dependency).
* `agent/api.py` — full multi-turn loop. `_dispatch_tools` fans calls
  into an `anyio.create_task_group` with a pre-allocated results
  list (no locks, order preserved by index). Each tool execution is
  its own journaled `runtime.step` keyed by `idempotency_key` and
  `call_id`. `before_tool` / `after_tool` decorator sugar on
  `Agent`. `max_turns=50` cap on runaway loops.

**Protocol break (intentional, pre-1.0):** `ToolHost.call(...)` now
takes `*, call_id: str = ""` so the host can stamp the right ID on
its `ToolResult`.

**Files touched:** `core/protocols.py`, `agent/api.py`, top-level
`__init__.py`, plus new `tools/__init__.py`, `tools/registry.py`,
`security/__init__.py`, `security/permissions.py`, `security/hooks.py`,
`model/scripted.py`, `tests/test_tools.py`.

**Tests added (14):** tool execution, sync function dispatched to
thread, decorator metadata, explicit Tool object, **parallel timing**
(two 0.10s tools in one turn finish in <0.18s — proves no
serialization), permission deny short-circuits, destructive default
asks then denies, BYPASS allows destructive, before-tool hook can
deny, after-tool hook observes results, buggy post-hook doesn't
break loop, max-turns kicks in on a 100-turn script with
`max_turns=3`, unknown tools return error and loop continues,
external `HookRegistry` injection.

**Gates:** ruff clean, mypy strict clean across 22 source files,
**20/20 tests passing in 0.27s**.

---

## Slice 2 — Phase 2 hello-world end-to-end

**What landed.** Smallest possible working agent. Zero config:
`await Agent("...").run("hi")` returns a `RunResult` with the
echoed prompt, no API keys needed.

* `runtime/inproc.py` — pass-through `InProcRuntime` and trivial
  `InProcSession` satisfying the full `Runtime`/`RuntimeSession`
  protocols (`step`, `stream_step`, `session`, `signal`). No
  durability — replays nothing, just runs.
* `memory/inmemory.py` — `InMemoryMemory` with `anyio.Lock`
  coordinated working blocks and episodes. `recall()` is naive:
  substring match against input/output text, recency-sorted, time
  range filter optional. Good enough for tests; real `PostgresMemory`
  comes in Phase 4.
* `model/echo.py` — `EchoModel` streaming chunk-per-word + a finish
  chunk with synthetic `Usage` (1 token per whitespace-separated word
  in input, same in output).
* `governance/budget.py` — `NoBudget` (always-ok stub) and full
  `StandardBudget` with hard limits on tokens (total + in/out
  separately), cost USD, wall clock, plus a soft warning at 80% by
  default. `anyio.Lock`-coordinated counters.
* `agent/api.py` — `Agent` class with seed-context (instructions +
  working blocks + recall), single-turn loop (tool dispatch
  intentionally absent — flagged `interrupted=True` if model emits
  tool calls), persist-episode at end, returns `RunResult`.

**Files touched:** top-level `__init__.py`, plus new directories
`runtime/`, `memory/`, `model/`, `governance/`, `agent/`, and new
`tests/__init__.py`, `tests/conftest.py`, `tests/test_smoke.py`.

**Tests added (6):** agent returns echoed output, episode persists,
distinct sessions across runs, recall finds prior episodes, zero
budget interrupts run, constructor wiring (model/runtime override).

**Gates:** ruff clean, mypy strict clean across 16 source files,
**6/6 tests passing in 0.47s**.

---

## Slice 1 — Phase 1 core types and protocols

**What landed.** The contract surface every other module is built
against: 18 Pydantic value objects, 12 Protocols, 12 exception
classes, ULID + deterministic-hash helpers.

* `pyproject.toml` — `anyio>=4.4`, `pydantic>=2.6`,
  `python-ulid>=2.2`. Optional extras for `anthropic`, `openai`,
  `litellm`, `mcp`, `postgres`, `dbos`, `temporal`, `otel`. Dev
  extras: pytest, anyio[trio], hypothesis, mypy, ruff,
  import-linter. Hatchling build, mypy `--strict`, ruff with E/F/I/B/UP/ASYNC.
* `core/types.py` — Pydantic models. The notable shapes:
  - `Message` (frozen) — chat message with role + content + optional
    name + optional tool_call_id. (`tool_calls` field added in
    slice 5.)
  - `ToolDef` (frozen), `ToolCall`, `ToolResult` (with
    `success` / `error_` / `denied_` class methods).
  - `Usage` (frozen) — input_tokens / output_tokens / cost_usd.
  - `ModelChunk` — discriminated by `kind` ("text" / "tool_call" /
    "finish"); exactly one optional payload field set per kind.
  - `MemoryBlock`, `Episode`, `Fact` (with bi-temporal
    `valid_from` / `valid_until` / `recorded_at`).
  - `PermissionDecision` (frozen) and `BudgetStatus` (frozen) with
    classmethod constructors.
  - `Event` — single class with `kind: EventKind` + `payload: dict`,
    classmethod constructors per kind.
  - `RunResult`, `CertifiedValue`, `AuditEntry`, `Span`, `ToolEvent`.
* `core/protocols.py` — every module-boundary contract as a
  `typing.Protocol`. `runtime_checkable` only on stable surfaces
  (Model, Memory, Runtime, ToolHost). Async-only: every I/O method
  is a coroutine, every stream is an `AsyncIterator`, every resource
  is an `AbstractAsyncContextManager`. Twelve protocols total: Model,
  Memory, Runtime, RuntimeSession, ToolHost, Sandbox, Permissions,
  HookHost, Budget, Telemetry, Embedder, Secrets.
* `core/errors.py` — `JeevesAgentError` base + 11 specific subclasses
  (ConfigError, BudgetExceeded, PermissionDenied, ToolError,
  SandboxError, RuntimeJournalError, MemoryStoreError, MCPError,
  FreshnessError, LineageError, CancelledByUser).
* `core/ids.py` — `new_id(prefix)` (prefixed ULIDs for readability:
  `sess_*`, `ep_*`, `tcall_*`, `fact_*`) and `deterministic_hash(*parts)`
  (canonical JSON + SHA256 for idempotency keys).

**Files touched (new):** `pyproject.toml`, `README.md`,
`jeevesagent/__init__.py`, `jeevesagent/core/__init__.py`,
`jeevesagent/core/types.py`, `jeevesagent/core/protocols.py`,
`jeevesagent/core/errors.py`, `jeevesagent/core/ids.py`.

**Gates:** ruff clean, mypy `--strict` clean across 6 source files,
package imports cleanly with 46 public names exported. No tests yet
(intentional — types and protocols have no behavior to test).

**Conda env:** All work happens in the user's `pro` conda env at
`~/miniconda3/envs/pro/` (Python 3.12.0). Activate with
`conda activate pro`.

---

## Cumulative state (after slice 17)

* **77 source files** under `jeevesagent/` and `tests/`
* **236 tests passing** + 4 skipped (live-integration) in ~2.5 seconds total
* **mypy `--strict` clean** across 53 production source files
* **4 user-facing docs** (README + quickstart + recipes + architecture) totaling ~1295 lines
* **7 runnable example scripts** + index, all verified end-to-end
* **CI** (ruff + mypy + pytest matrix on Python 3.11 / 3.12 + examples smoke job) and **release** (PyPI trusted publishing on `v*` tags) workflows configured
* **ruff clean** including ASYNC lints (no `asyncio.gather` /
  `create_task` anywhere — everything goes through anyio task groups
  and memory-object streams)
* **Zero raw `asyncio` imports in production code** — pure anyio
  per the engineering plan's foundational rule

## Architecture map (current)

```
jeevesagent/
  __init__.py          # Re-exports everything public
  core/                # Layer-free primitives
    types.py           # 18 Pydantic value objects
    protocols.py       # 12 module-boundary Protocols
    errors.py          # JeevesAgentError + 11 subclasses
    ids.py             # ULID + deterministic hash
  agent/
    api.py             # Agent class, _loop(emit) shared by run/stream
  runtime/
    inproc.py          # InProcRuntime (no durability)
  memory/
    inmemory.py        # Dict-backed Memory + facts + consolidator hookup
    embedder.py        # HashEmbedder + OpenAIEmbedder
    vector.py          # In-memory cosine similarity Memory + facts hookup
    chroma.py          # ChromaMemory (lazy chromadb)
    postgres.py        # PostgresMemory (lazy asyncpg + pgvector)
    redis.py           # RedisMemory (lazy redis, RediSearch optional)
    facts.py           # FactStore + InMemoryFactStore (bi-temporal)
    sqlite_facts.py    # SqliteFactStore (durable, optional embedder)
    postgres_facts.py  # PostgresFactStore (pgvector + HNSW)
    chroma_facts.py    # ChromaFactStore (Chroma where + supersession)
    redis_facts.py     # RedisFactStore (HSET + scan supersession)
    consolidator.py    # LLM-driven Fact extractor
    _embedding_util.py # shared float32 pack/unpack helpers
  model/
    echo.py            # Zero-key streaming model
    scripted.py        # Canned-turn model for tests
    anthropic.py       # AnthropicModel via official SDK (lazy import)
    openai.py          # OpenAIModel via official SDK (lazy import)
  tools/
    registry.py        # Tool dataclass, @tool decorator, InProcessToolHost
  security/
    permissions.py     # Mode enum, AllowAll, StandardPermissions
    hooks.py           # HookRegistry implementing HookHost
  governance/
    budget.py          # NoBudget + StandardBudget
  mcp/
    spec.py            # MCPServerSpec (stdio + http)
    client.py          # MCPClient (lazy mcp SDK import via AsyncExitStack)
    registry.py        # MCPRegistry implementing ToolHost
  jeeves/
    client.py          # JeevesGateway (ToolHost over single Jeeves spec)
  observability/
    tracing.py         # NoTelemetry + OTelTelemetry
  security/
    permissions.py     # Mode + AllowAll + StandardPermissions
    hooks.py           # HookRegistry implementing HookHost
    audit.py           # InMemoryAuditLog + FileAuditLog (HMAC signed)
    sandbox/
      base.py          # NoSandbox (pass-through)
      filesystem.py    # FilesystemSandbox (path-arg validation)
  data/
    lineage.py         # FreshnessPolicy + LineagePolicy + validators
  runtime/
    inproc.py          # InProcRuntime (no durability)
    journal.py         # JournalStore + InMemory + Sqlite stores
    journaled.py       # JournaledRuntime + JournaledSession
    sqlite.py          # SqliteRuntime (Sqlite-backed JournaledRuntime)
```

Phase 6 essentials feature-complete. Phase 5 essentials (in-process
replay + sqlite durability) feature-complete; DBOS / Temporal
adapters remain as platform-specific follow-ups.
