"""26_devops_diagnostic — Read-only on-call assistant.

Real workflow: at 3am, an alert fires. The on-call engineer wants
an LLM agent that can READ logs, READ metrics, READ pod status —
correlate them — and propose a fix. **Crucially, it must NOT be
able to make changes** (no kubectl apply, no service restart, no
config rollback) without explicit human approval.

What this example shows
-----------------------

* **Hard permission boundaries** — every write/destructive tool is
  flagged ``destructive=True``. Permissions in DEFAULT mode + a
  policy hook that **denies** any destructive call outright (rather
  than asking — at 3am, "ask" without a human approver is the same
  as "no"). Audit log captures every denial for postmortem.
* **Synthetic but realistic data** — three "services" with logs,
  metrics, and pod-status responses pre-baked. The agent
  correlates the slow ``checkout`` API ↔ the spiking
  ``checkout-db`` connection-pool exhaustion ↔ a recent deploy
  rollback to the cache layer.
* **Reflexion** wraps the underlying ReAct so if the first
  diagnosis misses, the evaluator + reflector loop teaches it on
  retry. Optional but realistic.
* **Streaming** so a UI can show the diagnosis unfolding live.

Run::

    pip install -e '.[dev,openai]'
    # add OPENAI_API_KEY=sk-... to .env at repo root
    python examples/26_devops_diagnostic.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

if not os.environ.get("OPENAI_API_KEY"):
    print(
        "\n  ⊘ OPENAI_API_KEY not set — skipping this example.\n"
        "    Add OPENAI_API_KEY=sk-... to .env at repo root to run.\n"
    )
    sys.exit(0)

from jeevesagent import (  # noqa: E402
    Agent,
    FileAuditLog,
    HookRegistry,
    Mode,
    StandardPermissions,
    tool,
)
from jeevesagent.core.types import PermissionDecision, ToolCall, ToolResult  # noqa: E402
from jeevesagent.governance.budget import BudgetConfig, StandardBudget  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic incident state — the puzzle the agent must solve.
#
# Story: checkout API has been timing out since 14:55. The cause is
# DB connection-pool exhaustion on checkout-db (visible in metrics).
# Root cause: a 14:50 redis-cache rollback caused cache misses,
# pushing all reads to the DB. The agent should correlate logs +
# metrics + pod state and propose: roll the cache forward.
# ---------------------------------------------------------------------------

LOGS: dict[str, str] = {
    "checkout-api": (
        "2026-05-07T14:50:12Z INFO  checkout: ok latency=89ms\n"
        "2026-05-07T14:55:02Z WARN  checkout: latency=1240ms\n"
        "2026-05-07T14:55:30Z ERROR checkout: db_query_timeout after 5000ms (conn_pool=exhausted)\n"
        "2026-05-07T14:56:01Z ERROR checkout: db_query_timeout after 5000ms (conn_pool=exhausted)\n"
        "2026-05-07T14:56:48Z ERROR checkout: db_query_timeout after 5000ms (conn_pool=exhausted)\n"
        "2026-05-07T14:57:22Z ERROR checkout: HTTP 503 returned to caller (upstream timeout)\n"
    ),
    "checkout-db": (
        "2026-05-07T14:50:00Z INFO  pg: connections=12/100 idle=80\n"
        "2026-05-07T14:54:30Z INFO  pg: connections=42/100 idle=51\n"
        "2026-05-07T14:55:10Z WARN  pg: connections=98/100 idle=2 — approaching pool limit\n"
        "2026-05-07T14:55:35Z WARN  pg: connection wait_time_ms p99=4200\n"
        "2026-05-07T14:56:00Z ERROR pg: connection_pool_exhausted, queue_depth=47\n"
    ),
    "cache-layer": (
        "2026-05-07T14:48:55Z INFO  redis: hit_rate=92.4% qps=11420\n"
        "2026-05-07T14:50:01Z WARN  redis: ROLLBACK to v3.2.1 (operator: alex@oncall, reason='regression in v3.3.0 hot-path')\n"
        "2026-05-07T14:50:12Z WARN  redis: hit_rate=8.1% qps=421 — KEYS not yet warmed\n"
        "2026-05-07T14:53:00Z WARN  redis: hit_rate=11.0% qps=998\n"
        "2026-05-07T14:55:00Z WARN  redis: hit_rate=15.3% qps=1840\n"
    ),
}

METRICS: dict[str, dict[str, list[tuple[str, float]]]] = {
    "checkout-api": {
        "latency_p99_ms": [
            ("14:50", 95),
            ("14:53", 110),
            ("14:55", 1340),
            ("14:57", 5120),
        ],
        "error_rate_pct": [
            ("14:50", 0.01),
            ("14:55", 4.2),
            ("14:57", 38.4),
        ],
    },
    "checkout-db": {
        "connection_pool_used_pct": [
            ("14:50", 12),
            ("14:53", 45),
            ("14:55", 99),
            ("14:57", 100),
        ],
    },
    "cache-layer": {
        "hit_rate_pct": [
            ("14:48", 92.4),
            ("14:50", 8.1),
            ("14:53", 11.0),
            ("14:55", 15.3),
        ],
    },
}

POD_STATUS: dict[str, dict[str, object]] = {
    "checkout-api-7b9c": {
        "service": "checkout-api",
        "status": "Running",
        "restarts": 0,
        "ready": "8/8",
    },
    "checkout-db-primary": {
        "service": "checkout-db",
        "status": "Running",
        "restarts": 0,
        "ready": "1/1",
    },
    "cache-layer-redis-0": {
        "service": "cache-layer",
        "status": "Running",
        "restarts": 1,
        "ready": "1/1",
        "image": "redis:v3.2.1",
        "note": "rolled back from v3.3.0 at 14:50",
    },
}


# ---------------------------------------------------------------------------
# Read-only diagnostic tools.
# ---------------------------------------------------------------------------


@tool
async def read_logs(service: str, since: str = "14:48") -> str:
    """Read recent logs for a service. Read-only."""
    log = LOGS.get(service)
    if log is None:
        return f"ERROR: unknown service {service!r}. Known: {', '.join(LOGS)}"
    # Crude time filter for the demo.
    lines = [
        line for line in log.splitlines()
        if since in line or any(t in line for t in ("14:55", "14:56", "14:57"))
        or service != "cache-layer"
    ]
    return f"$ logs {service}\n" + "\n".join(lines)


@tool
async def query_metrics(service: str, metric: str | None = None) -> str:
    """Query metrics for a service. Pass ``metric`` to filter to a
    single timeseries; omit for all available metrics. Read-only."""
    metrics = METRICS.get(service)
    if metrics is None:
        return f"ERROR: unknown service {service!r}"
    if metric:
        series = metrics.get(metric)
        if series is None:
            return (
                f"ERROR: metric {metric!r} not found for {service}. "
                f"Available: {', '.join(metrics)}"
            )
        rows = "\n".join(
            f"  {ts}  {val}" for ts, val in series
        )
        return f"{service}.{metric}:\n{rows}"
    parts = []
    for m, series in metrics.items():
        rows = "\n".join(
            f"    {ts}  {val}" for ts, val in series
        )
        parts.append(f"  {m}:\n{rows}")
    return f"{service} metrics:\n" + "\n".join(parts)


@tool
async def get_pod_status(service: str | None = None) -> str:
    """Pod status across the cluster. Optionally filter by service.
    Read-only."""
    rows = []
    for pod, info in POD_STATUS.items():
        if service and info.get("service") != service:
            continue
        line = (
            f"  {pod:<24} service={info.get('service'):<14} "
            f"status={info.get('status')} restarts={info.get('restarts')} "
            f"ready={info.get('ready')}"
        )
        if "note" in info:
            line += f"   note: {info['note']}"
        rows.append(line)
    if not rows:
        return f"No pods found{f' for service {service!r}' if service else ''}"
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# WRITE tools — all destructive. The policy hook will deny these.
# Marked here only so the agent SEES them in its tool list and tries
# them; the demo's whole point is that the permissions block them.
# ---------------------------------------------------------------------------


@tool
async def restart_pod(pod: str) -> str:
    """Restart a pod. DESTRUCTIVE — requires approval."""
    return f"RESTARTED {pod}"  # Never actually called in the demo.


@tool
async def rollback_deployment(service: str, target_version: str) -> str:
    """Roll a service to a previous version. DESTRUCTIVE — requires approval."""
    return f"ROLLED BACK {service} to {target_version}"


@tool
async def kill_connections(service: str) -> str:
    """Kill all connections for a service (force pool reset). DESTRUCTIVE."""
    return f"KILLED all connections for {service}"


for _t in (restart_pod, rollback_deployment, kill_connections):
    _t.destructive = True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Permission hook — read-only-at-3am policy.
# ---------------------------------------------------------------------------


def _build_readonly_hooks() -> HookRegistry:
    hooks = HookRegistry()

    DESTRUCTIVE = {"restart_pod", "rollback_deployment", "kill_connections"}

    @hooks.register_pre_tool
    async def readonly_policy(
        call: ToolCall,
    ) -> PermissionDecision | None:
        if call.tool in DESTRUCTIVE:
            msg = (
                f"{call.tool} is a write/destructive action. The "
                f"on-call diagnostic policy hard-denies these — "
                f"a human operator must explicitly approve "
                f"recovery actions. Suggest the action in your "
                f"response instead of attempting to execute it."
            )
            print(f"    [POLICY] DENY  {call.tool}({call.args}) — read-only mode")
            return PermissionDecision.deny_(msg)
        return None

    @hooks.register_pre_tool
    async def trace_call(call: ToolCall) -> PermissionDecision | None:
        preview = ", ".join(
            f"{k}={str(v)[:40]!r}" for k, v in call.args.items()
        )
        print(f"    [tool]   {call.tool}({preview})")
        return None

    @hooks.register_post_tool
    async def trace_result(
        call: ToolCall, result: ToolResult
    ) -> None:
        if result.denied or result.error:
            return  # already shown by the deny policy
        preview = (result.output or "").split("\n")[0][:80]
        print(f"    [ok]     {call.tool} → {preview}")

    return hooks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    workdir = Path(  # noqa: ASYNC240 — demo startup
        tempfile.mkdtemp(prefix="jeeves_oncall_")
    ).resolve()
    audit_path = workdir / "incident_audit.jsonl"

    print("=" * 70)
    print("On-call diagnostic — read-only incident assistant")
    print("=" * 70)
    print("Incident: checkout API timing out since 14:55 UTC")
    print(f"Audit log: {audit_path}\n")

    permissions = StandardPermissions(mode=Mode.DEFAULT)
    audit = FileAuditLog(audit_path, secret="oncall-demo")
    budget = StandardBudget(
        BudgetConfig(max_tokens=80_000, max_cost_usd=1.0)
    )

    agent = Agent(
        instructions=(
            "You are an on-call diagnostic assistant. You can READ "
            "logs, metrics, and pod status. You CANNOT execute any "
            "fixes — the policy denies all write actions. Your job:\n"
            "1. Investigate the reported incident by reading logs "
            "+ metrics + pod state across the relevant services.\n"
            "2. Correlate the timing of events across services.\n"
            "3. Identify the root cause.\n"
            "4. Propose a specific remediation as a recommendation "
            "(do NOT call write tools — they are denied; describe "
            "what a human should do instead).\n\n"
            "Be specific: name the services, the timestamps, the "
            "metrics that confirm your diagnosis. Cite log lines "
            "verbatim where relevant."
        ),
        model="gpt-4.1-mini",
        tools=[
            read_logs,
            query_metrics,
            get_pod_status,
            restart_pod,
            rollback_deployment,
            kill_connections,
        ],
        permissions=permissions,
        audit_log=audit,
        budget=budget,
        hooks=_build_readonly_hooks(),
    )

    incident = (
        "Alert: checkout API p99 latency >5s since 14:55 UTC, "
        "503s spiking. Investigate and recommend remediation. "
        "Services involved may include checkout-api, checkout-db, "
        "cache-layer."
    )
    print(f"Incident: {incident}\n")
    print("─" * 70)

    async for ev in agent.stream(incident):
        kind = ev.kind.value
        if kind == "model_chunk":
            chunk = ev.payload.get("chunk", {})
            if chunk.get("kind") == "text" and chunk.get("text"):
                print(chunk["text"], end="", flush=True)
        elif kind == "completed":
            result = ev.payload.get("result") or {}
            print("\n" + "─" * 70)
            print("\n══ DIAGNOSIS ══")
            print(result.get("output", "(no output)"))
            print(
                f"\nTurns: {result.get('turns')}  "
                f"Tokens: in={result.get('tokens_in')} "
                f"out={result.get('tokens_out')}  "
                f"Cost: ${float(result.get('cost_usd', 0) or 0):.4f}"
            )

    # Postmortem trail — every read AND every denied write logged.
    print(f"\n{'═' * 70}")
    print("POSTMORTEM AUDIT TRAIL")
    print(f"{'═' * 70}")
    log = FileAuditLog(audit_path, secret="oncall-demo")
    entries = await log.query()
    deny_count = sum(1 for e in entries if "deny" in e.action.lower())
    print(f"  Total entries:    {len(entries)}")
    print(f"  Denied attempts:  {deny_count}")
    print(f"  Audit file:       {audit_path}\n")
    for e in entries[:15]:
        print(
            f"  seq={e.seq:3d} actor={e.actor:8s} action={e.action}"
        )
    if len(entries) > 15:
        print(f"  ... ({len(entries) - 15} more)")
    print(f"\n(Workdir kept at {workdir} for inspection.)")


if __name__ == "__main__":
    asyncio.run(main())
