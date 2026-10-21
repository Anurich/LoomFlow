"""34_studio.py — record agent runs and browse them in Loomflow Studio.

Studio records what your agents do, with one switch and no change to
the agents themselves::

    import loomflow.studio
    loomflow.studio.enable()          # or: export LOOMFLOW_STUDIO=1

Every Agent and Workflow run in the process is then saved to a local
SQLite file: its turns, the messages sent to the model and the reply,
tool calls with their arguments, results and permission decisions,
guardrail hits, sub-agent and workflow steps, tokens, cost and time.
Each agent is catalogued by its definition, and a new version is
recorded whenever its instructions, model or tools change. Eval runs
become experiments you can compare case by case.

This script records a small support bot (a billing agent in two
versions, a tech-support agent, a routing workflow, an escalations
team, two eval runs) using scripted models, so it runs OFFLINE with no API key, then tells
you how to browse it::

    python examples/34_studio.py
    loom studio --db .loomflow/studio-demo.db      # needs loomflow[studio]

Open http://127.0.0.1:7600 and look at:

* **Traces**: every run; filter to "Problems only" for the error and
  the run that hit its turn limit.
* A **refund trace**: the approval before ``refund_charge`` ran, and
  the PII guard removing a card number from the reply.
* **Agents**: each agent's and workflow's architecture (the routing
  workflow, the escalations team and its specialists), billing-agent
  v1 and v2, and what changed between them.
* An **escalations trace**: its architecture panel highlights the
  coordinator and the specialist it delegated to.
* **Experiments**: compare the two eval runs; v2 fixed three cases
  and broke one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anyio

import loomflow.studio
from loomflow import Agent, Mode, StandardPermissions, Workflow, tool
from loomflow.architecture import Supervisor
from loomflow.core.types import ToolCall, Usage
from loomflow.eval import Contains, EvalHarness
from loomflow.eval.dataset import Case, Dataset
from loomflow.guardrails import PIIGuard
from loomflow.model.scripted import ScriptedModel, ScriptedTurn

PROJECT = "studio-demo"


class DemoModel(ScriptedModel):
    """A scripted model that reports a real model's name, for display."""

    def __init__(self, name: str, turns: list[ScriptedTurn]) -> None:
        super().__init__(turns)
        self.name = name


class OverloadedModel(DemoModel):
    async def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("upstream model overloaded (529)")


def usage(tokens_in: int, tokens_out: int, per_million_in: float = 0.10, per_million_out: float = 0.50) -> Usage:
    cost = tokens_in * per_million_in / 1e6 + tokens_out * per_million_out / 1e6
    return Usage(input_tokens=tokens_in, output_tokens=tokens_out, cost_usd=cost)


def turn(text: str = "", *, calls: list[ToolCall] | None = None, tin: int = 600, tout: int = 60) -> ScriptedTurn:
    return ScriptedTurn(text=text, tool_calls=calls or [], usage=usage(tin, tout))


# ---- tools -------------------------------------------------------------


@tool
def lookup_charges(order_id: str) -> list[dict[str, object]]:
    """Look up the charges for an order."""
    return [
        {"id": "ch_8812", "amount": 49.00, "date": "2026-10-20"},
        {"id": "ch_8813", "amount": 49.00, "date": "2026-10-20"},
    ]


@tool(destructive=True)
def refund_charge(charge_id: str) -> dict[str, str]:
    """Refund one charge. Asks for approval first."""
    return {"refund_id": "re_311", "charge_id": charge_id, "status": "succeeded"}


@tool
def search_kb(query: str) -> str:
    """Search the help-center articles."""
    return "Restart the router, then re-pair the device. If the VPN drops, update the client to 6.2."


async def approve(call: ToolCall, user_id: str | None) -> bool:
    return True  # a real app would ask a person


# ---- agents ------------------------------------------------------------

BILLING_V1 = "You are the billing agent for Acme. Look up charges before you answer."
BILLING_V2 = (
    "You are the billing agent for Acme. Look up charges before you act.\n"
    "Only refund a charge that is a confirmed duplicate, and refund it once.\n"
    "Never repeat a card number back to the customer."
)


def v1_priced(turns: list[ScriptedTurn]) -> list[ScriptedTurn]:
    """Claude Haiku 4.5 prices ($1 / $5 per million tokens) for v1."""
    for t in turns:
        t.usage = usage(t.usage.input_tokens, t.usage.output_tokens, 1.0, 5.0)
    return turns


def billing_v1(turns: list[ScriptedTurn]) -> Agent:
    return Agent(BILLING_V1, name="billing-agent", model=DemoModel("claude-haiku-4-5", v1_priced(turns)),
                 tools=[lookup_charges], auto_extract=False)


def billing_v2(turns: list[ScriptedTurn]) -> Agent:
    return Agent(
        BILLING_V2,
        name="billing-agent",
        model=DemoModel("claude-haiku-5-5", turns),
        tools=[lookup_charges, refund_charge],
        permissions=StandardPermissions(mode=Mode.DEFAULT),
        approval_handler=approve,
        guardrails=[PIIGuard()],
        auto_extract=False,
    )


def tech_support(turns: list[ScriptedTurn], *, model: DemoModel | None = None) -> Agent:
    return Agent(
        "You help customers fix network and device problems. Search the help center first.",
        name="tech-support",
        model=model or DemoModel("claude-sonnet-5-5", turns),
        tools=[search_kb],
        max_turns=6,
        auto_extract=False,
    )


def lookup(order: str) -> ToolCall:
    return ToolCall(tool="lookup_charges", args={"order_id": order})


async def record_runs() -> None:
    # billing-agent v1: answers, but can't refund.
    r = await billing_v1([turn(calls=[lookup("4471")], tin=520, tout=40),
                          turn("I see two charges of $49.00 on Oct 20. I can't issue refunds; "
                               "please contact support.", tin=640, tout=55)]
                         ).run("My card was charged twice for order #4471", user_id="alice", session_id="conv_42")
    loomflow.studio.score(r, "resolved", False, source="online judge")
    await billing_v1([turn("Invoices are issued in USD. EUR invoices need the Pro plan.", tin=480, tout=30)]
                     ).run("Can I get my invoice in EUR?", user_id="gus")

    # billing-agent v2: refunds a confirmed duplicate, with approval, and
    # the PII guard keeps the card number out of the reply.
    r = await billing_v2([
        turn(calls=[lookup("4471")], tin=520, tout=48),
        turn(calls=[ToolCall(tool="refund_charge", args={"charge_id": "ch_8813"}, destructive=True)],
             tin=590, tout=41),
        turn("I found two charges of $49.00 on Oct 20 and refunded the duplicate to your card "
             "4242 4242 4242 4242. It should show up in 3 to 5 business days.", tin=640, tout=73),
    ]).run("Please refund the duplicate charge on order #4471", user_id="alice", session_id="conv_42")
    loomflow.studio.score(r, "resolved", True, source="online judge")
    loomflow.studio.score(r, "user_rating", "thumbs up", source="end user")
    await billing_v2([turn("Thanks! Anything else I can help with?", tin=700, tout=12)]
                     ).run("Great, thank you", user_id="alice", session_id="conv_42")
    r = await billing_v2([turn(calls=[lookup("5120")], tin=500, tout=40),
                          turn("Your invoice is higher because the annual plan renewed on Oct 1.", tin=610, tout=44)]
                         ).run("Why is my invoice higher this month?", user_id="dana")
    loomflow.studio.score(r, "resolved", True, source="online judge")

    # tech-support: one good run, one that hits its turn limit, one where
    # the model is down.
    await tech_support([turn(calls=[ToolCall(tool="search_kb", args={"query": "vpn drops"})], tin=900, tout=30),
                        turn("Update your VPN client to 6.2; that fixes the ten-minute drops.", tin=1100, tout=40)]
                       ).run("VPN drops every ten minutes on my laptop", user_id="chen")
    looping = [turn(calls=[ToolCall(tool="search_kb", args={"query": f"printer offline {i}"})], tin=900, tout=25)
               for i in range(8)]
    await tech_support(looping).run("Printer says it's offline but it's plugged in", user_id="eli")
    try:
        await tech_support([], model=OverloadedModel("claude-sonnet-5-5", [])).run(
            "What's the guest Wi-Fi password?", user_id="hana")
    except RuntimeError:
        pass  # recorded as an error trace

    # A routing workflow in front of both agents.
    async def classify(text: str) -> str:
        return "billing" if "charge" in text.lower() or "invoice" in text.lower() else "tech"

    def router() -> Workflow:
        return Workflow.route(
            classify,
            {
                "billing": billing_v2([turn("Your next invoice is on Nov 1.", tin=450, tout=20)]),
                "tech": tech_support([turn("Restart the router, then re-pair the device.", tin=800, tout=25)]),
            },
            name="support-router",
        )

    await router().run("When is my next invoice charged?", user_id="fatima")
    await router().run("My phone won't connect to Wi-Fi", user_id="bob")

    # An escalations team: a coordinator delegating to specialists.
    @tool
    def card_activity(order_id: str) -> list[dict[str, str]]:
        """Recent card activity for an order."""
        return [{"country": "PT", "minutes_ago": "7"}, {"country": "BR", "minutes_ago": "2"}]

    fraud = Agent("You assess whether card activity is fraud.", name="fraud-analyst",
                  model=DemoModel("claude-sonnet-5-5", [turn("Two charges from different countries five "
                                                             "minutes apart: likely card fraud.", tin=700, tout=30)]),
                  tools=[card_activity], auto_extract=False)
    refunds = Agent("You issue refunds that policy allows.", name="refund-specialist",
                    model=DemoModel("claude-haiku-5-5", []), tools=[refund_charge], auto_extract=False)
    coordinator = DemoModel("claude-sonnet-5-5", [
        turn(calls=[ToolCall(tool="delegate", args={"worker": "fraud", "instructions": "Check order 9921."})],
             tin=900, tout=40),
        turn("This looks like fraud. I've frozen the card and opened case FR-118.", tin=1100, tout=35),
    ])
    escalations = Agent("You handle escalated cases. Delegate to the right specialist.", name="escalations",
                        model=coordinator, auto_extract=False,
                        architecture=Supervisor(workers={"fraud": fraud, "refunds": refunds}))
    await escalations.run("Two charges on order 9921 I don't recognise", user_id="ivy")


# ---- experiments -------------------------------------------------------

CASES = [
    Case(id="refund-dup", input="Refund the duplicate charge on order 4471", expected="refunded"),
    Case(id="why-twice", input="Why was I charged twice this month?", expected="duplicate"),
    Case(id="eur", input="Can I get my invoice in EUR?", expected="Pro plan"),
    Case(id="refund-again", input="Refund order 4471 again", expected="already refunded"),
    Case(id="pro-price", input="What does the Pro plan cost per year?", expected="$468"),
    Case(id="refund-late", input="I was refunded but I don't see it", expected="3 to 5 business days"),
]
V1_ANSWERS = [
    "I can't issue refunds; please contact support.",
    "It looks like a duplicate charge; support can reverse it.",
    "EUR invoices need the Pro plan.",
    "This order was already refunded.",
    "Plans start at $39 per month.",
    "Refunds can take a while.",
]
V2_ANSWERS = [
    "I refunded the duplicate charge ch_8813.",
    "You were charged twice: the second charge is a duplicate and I can refund it.",
    "EUR invoices need the Pro plan.",
    "Done: I refunded order 4471 again.",
    "The Pro plan is $468 per year.",
    "Refunds show up in 3 to 5 business days.",
]


async def record_experiments() -> None:
    dataset = Dataset(CASES, name="billing-faq")
    metrics = [Contains()]
    v1 = billing_v1([turn(a, tin=400, tout=30) for a in V1_ANSWERS])
    await EvalHarness(v1, metrics, concurrency=1).run(dataset, experiment="billing v1")
    v2 = billing_v2([turn(a, tin=420, tout=34) for a in V2_ANSWERS])
    await EvalHarness(v2, metrics, concurrency=1).run(dataset, experiment="billing v2")


async def main(db: Path) -> None:
    # redact="pii": keep card numbers, emails and phone numbers out of the
    # file. (The agent's PIIGuard changes what the customer receives;
    # Studio records what the model actually produced.)
    loomflow.studio.enable(db, project=PROJECT, redact="pii")
    await record_runs()
    await record_experiments()
    loomflow.studio.flush()
    print(f"Recorded the demo into {db}")
    print(f"Browse it:  loom studio --db {db}   (then open http://127.0.0.1:7600)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=Path(".loomflow") / "studio-demo.db")
    args = parser.parse_args()
    anyio.run(main, args.db)
