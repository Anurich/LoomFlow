# Production hardening

Everything in this page is *opt-in*: the framework's defaults already
work for single-tenant scripts and demos. The settings below are
what you flip on when you're putting the agent in front of real
users on real infrastructure.

The whole page is organised around a single theme — **multi-tenancy
without footguns**. One shared `Agent` (and one `Memory`, one
`Budget`, one `AuditLog`) backing N users requires more than just
passing `user_id=` everywhere; it requires bounded state, per-user
caps, scoped permissions, observable extraction, and pluggable
secret resolution. Each section below maps a production concern
to the framework primitive that closes it.

## Per-user budget caps

`StandardBudget` tracks tokens / cost / wall clock both globally and
per `user_id`. The global caps catch runaway aggregate usage; the
per-user caps stop one tenant from exhausting another's quota.

```python
from datetime import timedelta
from loomflow import Agent
from loomflow.governance.budget import BudgetConfig, StandardBudget

agent = Agent(
    "...",
    budget=StandardBudget(BudgetConfig(
        # Global: applies to all users combined.
        max_tokens=10_000_000,
        max_cost_usd=100.0,
        max_wall_clock=timedelta(hours=1),
        # Per-user: applies to each user_id's bucket independently.
        per_user_max_tokens=100_000,
        per_user_max_cost_usd=2.0,
        per_user_max_wall_clock=timedelta(minutes=10),
        soft_warning_at=0.8,  # warn at 80% of either cap
    )),
)
```

A run is blocked when *either* its user's cap or the global cap is
exceeded — whichever fires first. `result.interrupted = True` and
`result.interruption_reason = "budget:per_user_max_tokens"` (or the
matching global field) on a blocked run.

Inspect a single user's running totals at any time:

```python
usage = agent._budget.usage_for("alice")
# {"tokens_in": 1240, "tokens_out": 880, "tokens_total": 2120,
#  "cost_usd": 0.0042}
```

Both the agent loop and direct callers pass `user_id` through to the
budget automatically — you don't need to thread it manually.

## Per-user permission policies

`PerUserPermissions` routes the policy decision per `user_id`. One
`Agent` can run in BYPASS mode for staff while still gating
destructive tools for end users.

```python
from loomflow import Agent, Mode, StandardPermissions
from loomflow.security import PerUserPermissions

policies = {
    "admin_alice": StandardPermissions(mode=Mode.BYPASS),
    "service_account": StandardPermissions(
        mode=Mode.DEFAULT,
        allowed_tools=["read", "search"],
    ),
}

perms = PerUserPermissions(
    policies=policies,
    default=StandardPermissions(
        mode=Mode.DEFAULT,
        denied_tools=["delete_account", "send_email"],
    ),
)

agent = Agent("...", permissions=perms)
```

Unknown `user_id`s fall through to `default`. The framework forwards
the live `user_id` from the active `RunContext` into every
`permissions.check(...)` call — you don't need to wire it manually.

## Approval handler for `Decision.ask_`

`StandardPermissions(mode=Mode.DEFAULT)` returns `Decision.ask_(...)`
when a tool is marked `destructive=True`. Without an approval
handler, `ask` falls back to **deny** — the agent never silently
bypasses the gate. Wire a handler to surface the decision to a
human / Slack / ticket queue:

```python
from loomflow import Agent
from loomflow.core.types import ToolCall

async def approve_via_slack(call: ToolCall, user_id: str | None) -> bool:
    """Return True to allow the tool call, False to deny."""
    msg_id = await slack.post(
        channel="#approvals",
        text=f"User {user_id} wants to run {call.tool}({call.args!r}). React 👍 to approve.",
    )
    reactions = await slack.wait_for_reaction(msg_id, timeout=300)
    return "👍" in reactions

agent = Agent(
    "...",
    permissions=StandardPermissions(mode=Mode.DEFAULT),
    approval_handler=approve_via_slack,
    tools=[delete_user_data],  # @tool(destructive=True)
)
```

Failure-mode contract:

* **Handler returns `False`** → tool result is `denied` with
  `reason="approval declined"`; the run continues, the model sees
  the denial in the next turn.
* **No handler wired (`approval_handler=None`)** → `denied` with
  `reason="approval required; no approver"`. Same UX as before
  M10.4 — single-tenant code that didn't want approvals keeps
  working.
* **Handler raises** → treated as deny + warning logged. A buggy
  approval flow must NOT silently green-light a gated tool.

## Bounded per-user state

`StandardBudget._by_user` and `InMemoryMemory._blocks` hold per-`user_id`
state in process. Without bounds, a runaway tenant or one-shot
`user_id` explosion (e.g. someone sending one request per random
UUID) grows the dict until the process OOMs. Both primitives now
default to bounded state with LRU + idle-TTL eviction:

```python
from loomflow import InMemoryMemory
from loomflow.governance.budget import BudgetConfig, StandardBudget

# Defaults: 100k users, 24h idle TTL.
budget = StandardBudget(BudgetConfig())  # implicit bounds
memory = InMemoryMemory()                # implicit bounds

# Tune for your workload:
budget = StandardBudget(
    BudgetConfig(),
    max_users=10_000,                # smaller cap for known tenant size
    user_idle_ttl_seconds=3_600,     # drop idle users after 1h
)
memory = InMemoryMemory(
    max_users=10_000,
    user_idle_ttl_seconds=3_600,
)

# Or disable bounding entirely (single-tenant or fixed N tenants):
budget = StandardBudget(BudgetConfig(), max_users=None, user_idle_ttl_seconds=None)
```

What eviction does:

* **LRU** — when `max_users` is exceeded, the least-recently-touched
  user's bucket is dropped. For `StandardBudget` that means the
  user's running totals reset; for `InMemoryMemory` their working
  blocks are deleted.
* **TTL** — a user idle longer than `user_idle_ttl_seconds` has
  their bucket dropped on the next access. Lazy: no background
  thread; the sweep runs on each touch.

Eviction is **destructive in process** — the bucket's data is gone,
not flushed elsewhere. Callers needing durable spill-to-disk should
use `SqliteMemory` or `PostgresMemory` (which persist working blocks
across restarts) instead of relying on the in-memory bound.

## Auto-extract observability

`AutoExtractMemory` runs a small LLM extraction pass after every
`agent.run()` to pull structured `(subject, predicate, object)` facts
into the bi-temporal store. It's **on by default** for real network
adapters, which means it's also on your bill — you should know when
it fires and how long it takes.

Two telemetry signals are emitted when `Agent(telemetry=...)` is wired:

| Metric | Type | Tags | Use |
|---|---|---|---|
| `jeeves.auto_extract.duration_ms` | histogram | `user_id`, `status` (`ok`/`error`) | Latency budget per tenant |
| `jeeves.auto_extract.invocations` | counter | `user_id`, `status` | Failure rate; cost attribution |

A one-time-per-process `INFO` log notice tells you when the
default-on heuristic fires:

```
INFO  loomflow.memory.auto_extract: AutoExtractMemory enabled
by default for this model class. Each remembered episode triggers
a small extraction call to pull (subject, predicate, object)
facts. Pass Agent(auto_extract=False) to disable, or
Agent(auto_extract=True) to silence this notice.
```

To disable for cost reasons, pass `auto_extract=False`:

```python
agent = Agent("...", model="gpt-4o", auto_extract=False)
```

To enable explicitly (and silence the startup notice):

```python
agent = Agent("...", model="gpt-4o", auto_extract=True)
```

## Secrets resolution

Production agents shouldn't hard-code API keys; they shouldn't
depend on `os.environ` either (process-level env vars leak across
threads, get logged in crashes, and don't rotate). The `Secrets`
protocol gives you a pluggable resolver:

```python
from loomflow import Agent
from loomflow.security import DictSecrets, EnvSecrets

# Default — reads from os.environ. Same behaviour as pre-M10.
agent = Agent("...", model="claude-opus-4-7", secrets=EnvSecrets())

# In-memory for tests or vault-fetched-once-at-startup:
agent = Agent(
    "...",
    model="claude-opus-4-7",
    secrets=DictSecrets({
        "ANTHROPIC_API_KEY": api_key_from_vault,
        "OPENAI_API_KEY": openai_key_from_vault,
    }),
)
```

Resolution order inside model adapters:

1. Explicit `api_key=` argument on the model adapter
2. `secrets.lookup_sync(<ENV_VAR_NAME>)` if a `Secrets` backend is wired
3. `os.environ[<ENV_VAR_NAME>]` as the bare fallback

Production callers running on AWS / GCP / Vault should write a small
custom adapter:

```python
class VaultSecrets:
    """Pulls from HashiCorp Vault, caches into a local dict for
    the constructor-time ``lookup_sync`` path."""

    def __init__(self, vault_client, path: str) -> None:
        self._client = vault_client
        self._cache: dict[str, str] = {}

    async def resolve(self, ref: str) -> str:
        if ref in self._cache:
            return self._cache[ref]
        secret = await self._client.read(f"secret/data/{ref}")
        value = secret["data"]["value"]
        self._cache[ref] = value
        return value

    async def store(self, ref: str, value: str) -> None:
        await self._client.write(f"secret/data/{ref}", value=value)
        self._cache[ref] = value

    def redact(self, text: str) -> str:
        # Use the default regex set or extend with your own:
        from loomflow.security.secrets import _apply_redaction
        return _apply_redaction(text)

    def lookup_sync(self, ref: str) -> str | None:
        # Constructor-time path — return whatever's already in the
        # cache; if nothing's there, return None and let the caller
        # fall back to env-vars.
        return self._cache.get(ref)


# Pre-warm the cache at startup so lookup_sync hits during Agent()
# construction:
async def boot():
    vault = await connect_vault(...)
    secrets = VaultSecrets(vault, path="prod/jeeves")
    await secrets.resolve("ANTHROPIC_API_KEY")
    await secrets.resolve("OPENAI_API_KEY")

    agent = Agent("...", model="claude-opus-4-7", secrets=secrets)
```

`secrets.redact(text)` masks common API-key shapes (OpenAI,
Anthropic, AWS access keys, GitHub PATs) — useful inside
`@agent.before_tool` hooks that log tool args, or before payload
strings hit the audit log.

## Audit log with per-user attribution

`AuditEntry` carries a top-level `user_id` field. Every
`run_started` / `tool_call` / `tool_result` / `run_completed` entry
the framework writes is attributed to the active user:

```python
from loomflow import Agent
from loomflow.security import FileAuditLog

agent = Agent(
    "...",
    audit_log=FileAuditLog("./audit.jsonl", secret="prod-secret"),
)

await agent.run("export my data", user_id="alice", session_id="s1")

# Compliance query — filter the log by user:
alice_entries = await agent._audit_log.query(user_id="alice")
# Or by session:
session_entries = await agent._audit_log.query(session_id="s1")
# Or both:
both = await agent._audit_log.query(user_id="alice", action="run_completed")
```

The HMAC signature on each entry includes `user_id`, so tampering
that swaps a different user's id into an entry breaks the signature.
`verify_signature(entries, secret)` validates the chain on read.

## Verifying isolation under load

`bench/multi_tenant.py` simulates N concurrent users × M turns
through one shared Agent and reports p50 / p99 latency, RSS growth,
isolation violations (cross-user data leakage), and budget
accounting mismatches:

```bash
python bench/multi_tenant.py --users 100 --turns 3
python bench/multi_tenant.py --users 500 --turns 5  # stress
```

Output for the 500 × 5 stress run on a developer laptop:

```
============================================================
  Multi-tenant load bench
============================================================
  users          : 500
  turns / user   : 5
  total runs     : 2500

  p50 turn latency : 1008.04 ms
  p99 turn latency : 1057.01 ms
  RSS growth         : 10179 KB
  per-user growth    : 20.36 KB/user

  isolation violations : 0
  budget mismatches    : 0

  PASS: isolation + budget accounting hold under load
```

A smoke-test variant runs as part of the regular pytest suite
(`tests/test_multi_tenant_load.py`) at lower scale (10 users × 2
turns) so a regression to the isolation contract gets caught in CI.

## Migrating from 0.9.x to 0.10

The 0.10 release is **additive** — every change is opt-in. Three
things to be aware of when you upgrade:

1. **Postgres `memory_blocks` schema migrated.** The empty-string
   anonymous bucket is replaced with a reserved sentinel
   (`__jeeves_anon_user__`). The framework's `init_schema()` runs
   the rewriting `UPDATE` automatically; if you bypass it and
   manage migrations yourself, run:
   ```sql
   UPDATE memory_blocks SET user_id = '__jeeves_anon_user__'
     WHERE user_id = '';
   ALTER TABLE memory_blocks ALTER COLUMN user_id
     SET DEFAULT '__jeeves_anon_user__';
   ```
   Attempting to use the sentinel as a real `user_id` now raises
   `ValueError` — defense against impersonating the anonymous
   bucket.
2. **Custom `Memory` / `Permissions` / `HookHost` / `AuditLog` impls
   without a `user_id=` kwarg emit `LoomDeprecationWarning`.** The
   shim layer still calls the legacy shape, so nothing breaks today.
   The deprecation will turn into a hard removal in 1.0. To silence:
   ```python
   class MyMemory:
       async def remember(
           self, episode: Episode, *, user_id: str | None = None,
       ) -> str:
           ...
   ```
3. **`StandardBudget._by_user` and `InMemoryMemory._blocks` are now
   bounded by default.** A workflow that legitimately pins >100k
   active users in process needs to opt out or raise the cap:
   ```python
   StandardBudget(BudgetConfig(), max_users=None, user_idle_ttl_seconds=None)
   InMemoryMemory(max_users=None, user_idle_ttl_seconds=None)
   ```

That's the whole upgrade. No public API got renamed; no constructor
arg got removed.

## What you turn on, listed

For a checklist-driven shipment, these are the production flips:

| Concern | Opt-in |
|---|---|
| Per-user quota | `BudgetConfig(per_user_max_*)` |
| Tenant-specific permissions | `PerUserPermissions(policies=, default=)` |
| Human-in-the-loop for destructive tools | `Agent(approval_handler=...)` |
| Bounded in-process state | Defaults active; tune `max_users` / `user_idle_ttl_seconds` |
| Vault-backed API keys | `Agent(secrets=VaultSecrets(...))` |
| Auto-extract metrics | `Agent(telemetry=OTelTelemetry(...))` |
| Per-user audit | `FileAuditLog(...)` (attribution is automatic) |
| Load-test isolation | `bench/multi_tenant.py` |

Pair this with the `Production checklist` in [recipes](recipes.md#8-production-checklist)
for the broader operational concerns (durable runtime, persistent
memory, sandbox, etc.).
