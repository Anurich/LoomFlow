"""Example 7 — Route to specialist Agents.

Pattern: classify the user's question, then dispatch to a
specialist Agent. The classifier is a tiny model (cheap +
deterministic routing); the specialists are full models (the part
where reasoning matters).

The DEVELOPER controls the routing — the classifier just produces
a label, and a Python dict says which Agent handles each label.
The LLM only does the work each step asks of it; it doesn't
decide control flow.

This is the most common production shape: "trustworthy skeleton
with smart leaves."

Run::

    OPENAI_API_KEY=sk-... python examples/07_workflow_route.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

if not os.environ.get("OPENAI_API_KEY"):
    print(
        "\n  ⊘ Skipping: OPENAI_API_KEY is not set. "
        "Export it (or add it to .env) to run this example.\n"
    )
    sys.exit(0)


from jeevesagent import Agent, InMemoryMemory, Workflow  # noqa: E402

MODEL = "gpt-4.1-mini"


async def classify_topic(text: str) -> str:
    """Tag the question as one of: billing | tech | general.

    The classifier IS an Agent — but it's used as a deterministic
    routing primitive. The real branching logic lives in the
    workflow's ``Workflow.route`` map below, not in the model.
    """
    classifier = Agent(
        "Classify the customer question into exactly one of: "
        "billing, tech, general. Reply with only the single word.",
        model=MODEL,
        memory=InMemoryMemory(),
    )
    result = await classifier.run(text)
    label = result.output.strip().lower()
    return label if label in {"billing", "tech", "general"} else "general"


def make_specialist(system: str) -> Agent:
    return Agent(system, model=MODEL, memory=InMemoryMemory())


async def main() -> None:
    print("\n  Example 7 — Route to specialist Agents\n")

    billing = make_specialist(
        "You handle billing questions. Be concise, cite invoice numbers "
        "if helpful, tell the user how to escalate if you can't help."
    )
    tech = make_specialist(
        "You handle technical questions about Acme's product. Keep "
        "answers short, suggest specific commands or settings."
    )
    general = make_specialist(
        "You handle generic customer questions. Friendly tone, "
        "redirect to a specialist if the topic is really billing or tech."
    )

    # The dict IS the routing logic. The workflow runs the
    # classifier, looks up the result in the dict, runs the
    # matching Agent. No LLM decides between billing/tech/general
    # — the classifier just returns a label.
    wf = Workflow.route(
        classifier=classify_topic,
        routes={"billing": billing, "tech": tech, "general": general},
    )

    questions = [
        "I think I was double-charged last month — what should I do?",
        "How do I rotate my API key?",
        "Hi, just wanted to say thanks for the great support.",
    ]

    for q in questions:
        print("─" * 72)
        print(f"Q: {q}")
        result = await wf.run(q, user_id="alice", session_id="route-demo")
        # ``visited`` shows which branch the workflow actually
        # picked — useful for debugging classifier behaviour.
        branch = next(n for n in result.visited if n.startswith("route_"))
        print(f"   → routed to {branch}")
        print(f"A: {result.output}")
    print("─" * 72)


if __name__ == "__main__":
    asyncio.run(main())
