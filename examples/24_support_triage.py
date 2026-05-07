"""24_support_triage — Customer support with role-scoped tools + permissions.

A real triage system: a customer's message arrives, a Router
classifies it once, and dispatches to the right specialist agent
— each specialist has only the tools its role needs, with
permission boundaries enforced on the high-risk actions.

The three specialists
---------------------

* **refund_agent** — has ``lookup_order``, ``process_refund``.
  Refunds **over $100 require approval** (policy enforced via a
  permission hook). The agent doesn't need to know about the
  policy; the hook intercepts and either auto-approves under-$100
  refunds or denies the over-$100 ones with a clear reason the
  agent can read in its tool result.
* **technical_agent** — has ``run_diagnostic`` (read-only) and
  ``restart_service`` (always requires explicit approval — wired
  to deny by default; a real deployment would prompt a human).
* **faq_agent** — has only ``search_kb`` over the help-center
  docs (built on top of our ``InMemoryVectorStore``). No write
  capabilities at all.

What's production-shaped here
-----------------------------

* **Router architecture** — one classifier call + one specialist
  run. Cheaper than letting one big agent hold all the tools.
* **Per-tool permission decisions** logged to a file audit. You
  can tail the audit log live during a run to see every decision.
* **Hard budget cap** so a runaway turn loop dies quickly.
* **Streaming** — each specialist's output streams up through the
  Router's stream, so a UI can show "the refund agent is typing".

Run::

    pip install -e '.[dev,openai]'
    # add OPENAI_API_KEY=sk-... to .env at repo root
    python examples/24_support_triage.py
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
    HashEmbedder,
    HookRegistry,
    InMemoryVectorStore,
    Mode,
    RouterRoute,
    StandardPermissions,
    Team,
    tool,
)
from jeevesagent.core.types import PermissionDecision, ToolCall, ToolResult  # noqa: E402
from jeevesagent.governance.budget import BudgetConfig, StandardBudget  # noqa: E402
from jeevesagent.loader.base import Chunk  # noqa: E402

# ---------------------------------------------------------------------------
# Fake "back-office" data — orders + KB docs.
# ---------------------------------------------------------------------------

ORDERS: dict[str, dict[str, object]] = {
    "ORD-1001": {
        "customer": "alice@example.com",
        "amount": 49.99,
        "status": "delivered",
        "item": "Bluetooth headphones",
    },
    "ORD-1002": {
        "customer": "bob@example.com",
        "amount": 249.00,
        "status": "delivered",
        "item": "Smart watch",
    },
    "ORD-1003": {
        "customer": "carla@example.com",
        "amount": 12.50,
        "status": "shipped",
        "item": "Phone case",
    },
}

KB_DOCS: dict[str, str] = {
    "shipping.md": (
        "Shipping policy: standard shipping is 3-5 business days. "
        "Express is next-day for orders placed before 2pm ET. We "
        "ship to the US, Canada, EU, and UK. Tracking links are "
        "emailed when the order ships."
    ),
    "returns.md": (
        "Return policy: 30 days from delivery, item must be in "
        "original packaging. Refunds process within 5 business "
        "days to the original payment method. Damaged or defective "
        "items can be returned at any time for a full refund."
    ),
    "warranty.md": (
        "Warranty: all electronics carry a 1-year manufacturer "
        "warranty against defects. Battery wear is not covered. "
        "Contact warranty@example.com with the order ID for "
        "warranty claims."
    ),
}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
async def lookup_order(order_id: str) -> str:
    """Look up an order by its ID. Read-only."""
    order = ORDERS.get(order_id)
    if order is None:
        return f"ERROR: order {order_id!r} not found"
    return (
        f"Order {order_id}: customer={order['customer']}, "
        f"amount=${order['amount']}, status={order['status']}, "
        f"item={order['item']}"
    )


@tool
async def process_refund(
    order_id: str, amount: float, reason: str
) -> str:
    """Issue a refund against an order. POLICY: refunds over $100
    require approval — the permission hook will deny those unless
    explicitly approved by an authorised hook.
    """
    order = ORDERS.get(order_id)
    if order is None:
        return f"ERROR: order {order_id!r} not found"
    if amount > float(order["amount"]):  # type: ignore[arg-type]
        return (
            f"ERROR: refund amount ${amount} exceeds order total "
            f"${order['amount']}"
        )
    return (
        f"REFUND PROCESSED: order={order_id}, amount=${amount}, "
        f"reason={reason!r} — credited to customer "
        f"{order['customer']}"
    )


# Mark process_refund as destructive so StandardPermissions in
# DEFAULT mode treats it as needing approval. Our budget-aware
# hook below intercepts that and auto-approves under-$100 refunds
# while leaving over-$100 ones to be denied.
process_refund.destructive = True  # type: ignore[attr-defined]


@tool
async def run_diagnostic(service: str) -> str:
    """Run a read-only diagnostic on a service. Returns synthetic
    metrics for demo purposes."""
    return (
        f"Diagnostic for {service}: status=ok, p99_latency=120ms, "
        f"error_rate=0.02%, last_restart=3d ago"
    )


@tool
async def restart_service(service: str, reason: str) -> str:
    """Restart a service. ALWAYS requires explicit approval —
    permissions hard-deny this in the demo."""
    return f"RESTARTED {service} (reason: {reason!r})"


restart_service.destructive = True  # type: ignore[attr-defined]


# Module-global KB; populated in main() before any agent runs.
_KB = InMemoryVectorStore(embedder=HashEmbedder(dimensions=128))


@tool
async def search_kb(query: str) -> str:
    """Semantic search over the help-center knowledge base."""
    results = await _KB.search(query, k=2)
    if not results:
        return "(no matches in the knowledge base)"
    return "\n---\n".join(
        f"[{r.chunk.metadata.get('source', '?')}] {r.chunk.content}"
        for r in results
    )


# ---------------------------------------------------------------------------
# Permission hook — enforces the refund-amount and restart policies
# ---------------------------------------------------------------------------


def _build_policy_hooks() -> HookRegistry:
    hooks = HookRegistry()

    @hooks.register_pre_tool
    async def policy(call: ToolCall) -> PermissionDecision | None:
        """Hard-coded business policy ahead of the StandardPermissions
        check. First deny wins."""
        if call.tool == "process_refund":
            amount = float(call.args.get("amount", 0))
            if amount > 100.0:
                msg = (
                    f"Refund of ${amount:.2f} exceeds the $100 "
                    f"auto-approval threshold. A human supervisor "
                    f"must approve refunds over $100."
                )
                print(f"    [POLICY] DENY  process_refund: {msg}")
                return PermissionDecision.deny_(msg)
            print(
                f"    [POLICY] ALLOW process_refund: "
                f"${amount:.2f} ≤ $100 auto-approval threshold"
            )
            # Explicit allow so StandardPermissions doesn't ask
            # again for the destructive flag.
            return PermissionDecision.allow_()

        if call.tool == "restart_service":
            msg = (
                f"Service restart of {call.args.get('service', '?')!r} "
                f"requires on-call approval. Demo policy denies all "
                f"restart requests automatically."
            )
            print(f"    [POLICY] DENY  restart_service: {msg}")
            return PermissionDecision.deny_(msg)
        return None

    @hooks.register_pre_tool
    async def trace(call: ToolCall) -> PermissionDecision | None:
        # Visibility — print every tool call attempt.
        preview = ", ".join(
            f"{k}={str(v)[:40]!r}" for k, v in call.args.items()
        )
        print(f"    [tool]   {call.tool}({preview})")
        return None

    @hooks.register_post_tool
    async def trace_result(
        call: ToolCall, result: ToolResult
    ) -> None:
        if result.denied:
            return  # already printed by the policy hook
        if result.error:
            print(f"    [error]  {call.tool}: {result.error}")
        else:
            preview = (result.output or "")[:90].replace("\n", " ")
            print(f"    [ok]     {call.tool} → {preview}...")

    return hooks


# ---------------------------------------------------------------------------
# Build the three specialists.
# ---------------------------------------------------------------------------


def _build_specialists() -> dict[str, Agent]:
    refund_agent = Agent(
        instructions=(
            "You handle refund and order-status questions. "
            "Workflow: 1) call `lookup_order(order_id)` to verify "
            "the order, 2) explain the refund decision to the "
            "customer, 3) call `process_refund(order_id, amount, "
            "reason)` if a refund is warranted. NOTE: refunds "
            "over $100 require human approval and the tool will "
            "deny those — explain to the customer that approval "
            "is needed and surface the request to a supervisor. "
            "Be empathetic; cite the order ID in your reply."
        ),
        model="gpt-4.1-mini",
        tools=[lookup_order, process_refund],
    )

    technical_agent = Agent(
        instructions=(
            "You handle technical issues and service problems. "
            "Workflow: 1) call `run_diagnostic(service)` to check "
            "current health, 2) explain what you observed, 3) if "
            "a restart is needed, call `restart_service(service, "
            "reason)` — but be aware restarts ALWAYS require human "
            "approval and the call will be denied. In that case, "
            "tell the customer you've escalated and given the "
            "diagnostic to the on-call engineer."
        ),
        model="gpt-4.1-mini",
        tools=[run_diagnostic, restart_service],
    )

    faq_agent = Agent(
        instructions=(
            "You answer general policy questions (shipping, "
            "returns, warranty). Workflow: 1) call `search_kb(query)` "
            "with a focused query, 2) quote the relevant policy "
            "verbatim, 3) cite the source filename. If the KB "
            "doesn't cover the question, say so and recommend "
            "contacting support directly."
        ),
        model="gpt-4.1-mini",
        tools=[search_kb],
    )

    return {
        "refund": refund_agent,
        "technical": technical_agent,
        "faq": faq_agent,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    workdir = Path(  # noqa: ASYNC240 — demo startup
        tempfile.mkdtemp(prefix="jeeves_support_")
    ).resolve()
    audit_path = workdir / "audit.jsonl"

    print("=" * 70)
    print("Customer support triage — Router + role-scoped specialists")
    print("=" * 70)
    print(f"Audit log: {audit_path}\n")

    # Index the FAQ KB synchronously so tests/searches are ready
    # before the first agent turn.
    chunks = [
        Chunk(content=content, metadata={"source": name})
        for name, content in KB_DOCS.items()
    ]
    await _KB.add(chunks)

    specialists = _build_specialists()
    permissions = StandardPermissions(mode=Mode.DEFAULT)
    audit = FileAuditLog(audit_path, secret="support-demo")
    budget = StandardBudget(
        BudgetConfig(max_tokens=80_000, max_cost_usd=1.0)
    )
    hooks = _build_policy_hooks()

    # Team.router builds the same Agent as the explicit nested form
    # but reads like familiar router builders from other frameworks.
    triage = Team.router(
        routes=[
            RouterRoute(
                name="refund",
                description=(
                    "Refunds, returns, order issues, money back."
                ),
                agent=specialists["refund"],
            ),
            RouterRoute(
                name="technical",
                description=(
                    "Service down, errors, login issues, "
                    "broken features, anything that needs "
                    "diagnostics or a restart."
                ),
                agent=specialists["technical"],
            ),
            RouterRoute(
                name="faq",
                description=(
                    "General policy questions about shipping, "
                    "returns policy, warranty terms."
                ),
                agent=specialists["faq"],
            ),
        ],
        instructions=(
            "You are a customer support triage agent. Read the "
            "customer's message and route it to the correct "
            "specialist. Don't try to answer yourself — your job "
            "is just classification."
        ),
        model="gpt-4.1-mini",
        permissions=permissions,
        audit_log=audit,
        budget=budget,
        hooks=hooks,
    )

    # Three concrete customer scenarios that exercise different
    # permission paths: under-$100 refund (auto-approved),
    # over-$100 refund (denied + escalated), service restart
    # (denied + escalated), policy question (no permissions
    # needed).
    scenarios = [
        (
            "Under-$100 refund — should auto-approve",
            "I want to return ORD-1001. The headphones don't fit "
            "well. Can I get my $49.99 back?",
        ),
        (
            "Over-$100 refund — should be denied + escalated",
            "Order ORD-1002 — I changed my mind on the smart watch. "
            "I'd like a full $249 refund please.",
        ),
        (
            "Service restart — should be denied",
            "The login service has been timing out for 30 minutes. "
            "Please restart it.",
        ),
        (
            "FAQ — no permissions needed",
            "How long does standard shipping take and where do you ship to?",
        ),
    ]

    for i, (title, message) in enumerate(scenarios, 1):
        print(f"\n{'═' * 70}")
        print(f"SCENARIO {i}: {title}")
        print(f"{'═' * 70}")
        print(f'Customer: "{message}"\n')

        async for ev in triage.stream(message):
            kind = ev.kind.value
            if kind == "model_chunk":
                chunk = ev.payload.get("chunk", {})
                if chunk.get("kind") == "text" and chunk.get("text"):
                    print(chunk["text"], end="", flush=True)
            elif kind == "architecture_event":
                name = ev.payload.get("name", "")
                if name == "router.dispatched":
                    print(
                        f"\n  → routed to: {ev.payload.get('route', '?')}"
                    )
            elif kind == "completed":
                result = ev.payload.get("result") or {}
                print(
                    f"\n\n  [tokens in={result.get('tokens_in')} "
                    f"out={result.get('tokens_out')} "
                    f"cost=${float(result.get('cost_usd', 0) or 0):.4f}]"
                )

    # Show what landed in the audit log so users can see the trail.
    print(f"\n{'═' * 70}")
    print("AUDIT LOG (first 10 entries)")
    print(f"{'═' * 70}")
    log = FileAuditLog(audit_path, secret="support-demo")
    entries = await log.query()
    for e in entries[:10]:
        print(
            f"  seq={e.seq:3d} actor={e.actor:8s} action={e.action}"
        )
    print(
        f"\n(Total {len(entries)} audit entries written to "
        f"{audit_path})"
    )


if __name__ == "__main__":
    asyncio.run(main())
