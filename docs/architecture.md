# Architecture

Light tour of how the harness is laid out and where to look for what.
For the design rationale and the priority order between competing
concerns, read [`project.md`](../project.md). This doc is the
shipped-code map.

## Module layout

```
loomflow/
  __init__.py        # Re-exports the public surface
  core/              # Layer-free primitives
    types.py         # 18 Pydantic value objects
    protocols.py     # 14 Protocols (Model, Memory, Runtime, ...)
    errors.py        # LoomError + 11 subclasses
    ids.py           # ULID + deterministic JSON hash
  agent/
    api.py           # The Agent class. Public API.
  runtime/
    inproc.py        # InProcRuntime (no durability)
    journal.py       # JournalStore + InMemoryJournalStore + SqliteJournalStore
    journaled.py     # JournaledRuntime (contextvar-tracked sessions)
    sqlite.py        # SqliteRuntime (durable replay, no infra)
  memory/
    inmemory.py      # Naive dict-backed Memory
    embedder.py      # HashEmbedder + OpenAIEmbedder
    vector.py        # In-memory cosine similarity
    chroma.py        # Chroma-backed
    postgres.py      # Postgres + pgvector
    redis.py         # Redis (with optional RediSearch HNSW)
    facts.py         # InMemoryFactStore (bi-temporal)
    sqlite_facts.py  # SqliteFactStore
    postgres_facts.py # PostgresFactStore
    chroma_facts.py  # ChromaFactStore
    redis_facts.py   # RedisFactStore
    consolidator.py  # LLM-driven Fact extractor
    _embedding_util.py # shared float32 pack/unpack
  model/
    echo.py          # Zero-key streaming model
    scripted.py      # Canned-turn model for tests
    anthropic.py     # AnthropicModel via official SDK
    openai.py        # OpenAIModel via official SDK
  tools/
    registry.py      # Tool dataclass, @tool, InProcessToolHost
  security/
    permissions.py   # Mode + AllowAll + StandardPermissions
    hooks.py         # HookRegistry
    audit.py         # InMemoryAuditLog + FileAuditLog
    sandbox/
      base.py        # NoSandbox (pass-through)
      filesystem.py  # FilesystemSandbox (path-arg validation)
  governance/
    budget.py        # NoBudget + StandardBudget
  observability/
    tracing.py       # NoTelemetry + OTelTelemetry
  data/
    lineage.py       # FreshnessPolicy + LineagePolicy + validators
  mcp/
    spec.py          # MCPServerSpec
    client.py        # MCPClient (lazy mcp SDK)
    registry.py      # MCPRegistry implementing ToolHost
  jeeves/
    client.py        # JeevesGateway (Jeeves MCP gateway wrapper)
```

## Layer rules

Modules import strictly downward. From top to bottom:

```
agent/                          # Public API
  ↓
governance/, observability/     # Cross-cutting concerns
  ↓
security/, data/                # Policies + provenance
  ↓
mcp/, tools/                    # Tool dispatch surfaces
  ↓
runtime/, memory/, model/       # Execution + storage + I/O
  ↓
core/                           # Types + protocols (no deps)
```

A module never imports from a layer above it. Tests can fake any
layer because every cross-layer call goes through a Protocol from
`core/protocols.py`.

## Lifecycle: what happens during `agent.run("hi")`

1. `Agent.run()` → `_loop(prompt, emit=_noop_emit)`.
2. **Open runtime session**: `async with self._runtime.session(session_id):`
   sets a contextvar that journaled runtimes use to key their cache.
3. **Open root span**: `async with self._telemetry.trace("loom.run"):`.
4. **Audit**: write `run_started` entry.
5. **Seed context**: pull working blocks, recall recent episodes,
   recall facts (when memory exposes `.facts`), build the
   `messages: list[Message]` to send to the model.
6. **Loop**:
   - Check budget. Block / warn as needed.
   - Open `jeeves.turn` span.
   - Open `jeeves.model.stream` span. Stream chunks through
     `runtime.stream_step("model_call_<turn>", model.stream, messages)`.
     Journaled runtimes cache the chunk list keyed by
     ``(session_id, "model_call_<turn>")``.
   - Aggregate text + tool_calls + usage from chunks. Emit each chunk
     as `MODEL_CHUNK` event.
   - If no tool calls, append assistant message and break.
   - Otherwise, dispatch tools in parallel inside an
     `anyio.create_task_group`. Each `_run_single_tool` opens its own
     `jeeves.tool` span, runs hooks → permissions → sandboxed
     `runtime.step("tool_call_<turn>_<slot>", tool_host.call, ...)`.
     Audit `tool_call` and `tool_result` per call.
   - Append tool result messages to the conversation; loop again.
7. **Persist episode**: `runtime.step("persist_episode_<turns>",
   memory.remember, episode)`.
8. **Compute session-duration metric**.
9. If `auto_consolidate=True`: `await memory.consolidate()`. Failures
   surface as ERROR events but don't break the run.
10. **Audit**: write `run_completed` entry with token / cost / elapsed
    payload.
11. **Emit**: `Event.completed(...)`.

Every milestone hits four boundaries: events, telemetry, audit,
runtime journal. Each can be independently configured; `_noop_emit`,
`NoTelemetry`, `audit_log=None`, `InProcRuntime` all let you turn off
anything you don't need.

## Streaming: how `agent.stream()` works

`stream()` runs `_loop` in a background task:

```python
send, receive = anyio.create_memory_object_stream[Event](max_buffer_size=128)

async def _produce():
    try:
        await self._loop(prompt, emit=send.send)
    except Exception as exc:
        # Shielded send so the consumer always sees ERROR before producer fails
        with anyio.CancelScope(shield=True):
            await send.send(Event.error("", exc))
        raise
    finally:
        send.close()

async with anyio.create_task_group() as tg:
    tg.start_soon(_produce)
    try:
        async with receive:
            async for event in receive:
                yield event
    finally:
        tg.cancel_scope.cancel()  # break in consumer ⇒ kill producer
```

Backpressure is automatic: a slow consumer blocks `send.send(...)`
inside the producer until the buffer drains. Breaking out of the
iteration triggers the `finally` clause; the cancel scope kills the
producer task even if it's mid-tool-call.

## Extension points

Every cross-layer interface is a `Protocol` in `core/protocols.py`.
To add a new backend, just satisfy the relevant protocol — no
inheritance required:

| Protocol | Implement to add a new… |
|---|---|
| `Model` | LLM provider (LiteLLM, Ollama, Together, ...) |
| `Memory` | Storage backend (DuckDB, Pinecone, Weaviate, ...) |
| `Embedder` | Embedding model (Cohere, Voyage, sentence-transformers) |
| `Runtime` | Durable executor (DBOS, Temporal, custom replay) |
| `ToolHost` | Tool registry (LangChain bridge, custom protocol) |
| `Sandbox` | Isolation backend (Bubblewrap, Seatbelt, Docker, gVisor) |
| `Permissions` | Permission policy (RBAC, geofencing, ...) |
| `HookHost` | Lifecycle hook aggregator |
| `Budget` | Resource governance (per-org limits, prepaid pools) |
| `Telemetry` | Observability backend (custom span/metric exporter) |
| `Secrets` | Secret resolution (Vault, AWS Secrets Manager, 1Password) |

The harness internals consume only the protocol surface, so you can
plug in any implementation without forking the harness.

## Where the engineering plan landed

| Phase | Plan section | Modules shipped |
|---|---|---|
| 1 — protocols + types | §5 | `core/types.py`, `core/protocols.py`, `core/errors.py`, `core/ids.py` |
| 2 — basic agent loop | §6, §7 | `agent/api.py`, `runtime/inproc.py`, `memory/inmemory.py`, `model/echo.py`, `model/scripted.py`, `governance/budget.py` |
| 3 — MCP spine | §11 | `mcp/spec.py`, `mcp/client.py`, `mcp/registry.py`, `jeeves/client.py` |
| 4 — memory + facts | §9 | All `memory/` modules; bi-temporal facts in every backend |
| 5 — durable runtime | §8 | `runtime/journal.py`, `runtime/journaled.py`, `runtime/sqlite.py` (DBOS / Temporal pending) |
| 6 — security + governance + observability | §10, §13, §14 | `security/`, `governance/budget.py`, `observability/tracing.py`, `data/lineage.py` |
| Provider adapters | §1, §15 | `model/anthropic.py`, `model/openai.py` (LiteLLM pending) |

## Testing the harness

```bash
ruff check loomflow
mypy --strict loomflow
pytest tests/
```

All three must pass. The CI gate is non-negotiable.

* **236 tests** in 16 test files
* **~2.5s** wall-clock for the full suite
* **mypy --strict** clean across 53 production source files
* **4 tests skip** without env vars: `JEEVES_TEST_PG_DSN`,
  `JEEVES_TEST_REDIS_URL` (live integration tests for the Postgres /
  Redis backends)

Test patterns by module:

* `tests/test_smoke.py` — `Agent.run()` end-to-end
* `tests/test_streaming.py` — backpressure + cancellation
* `tests/test_tools.py` — parallel dispatch, hook denials, max turns
* `tests/test_journaled_runtime.py` + `test_sqlite_runtime.py` — replay
* `tests/test_facts.py` — supersession, valid_at, consolidator
* `tests/test_*_memory.py` + `tests/test_*_facts.py` — per-backend
* `tests/test_telemetry.py` — span hierarchy, metric routing
* `tests/test_audit.py` — HMAC verify, file persistence
* `tests/test_sandbox.py` — symlink escape detection
* `tests/test_lineage.py` — freshness / lineage policies
* `tests/test_anthropic.py` + `test_openai.py` — chunk normalization
  with fake clients
* `tests/test_mcp.py` + `test_jeeves.py` — fake MCP sessions

## Reading the source

Recommended reading order if you want to understand the harness:

1. `core/types.py` — the value objects everything else moves around.
2. `core/protocols.py` — the contracts.
3. `agent/api.py` — the loop. Read top-to-bottom; about 600 lines
   total.
4. `runtime/journaled.py` — the replay mechanism. Small but central.
5. `memory/facts.py` — the bi-temporal supersession logic.
6. `memory/consolidator.py` — the LLM-extraction prompt + parser.

That's about 1500 lines of code; you'll have a complete picture of
the harness in an afternoon.
