"""Example 10 — Agent with a non-default Architecture inside a Workflow.

Same outer shape as Example 7 (workflow wraps an agent) — but now
the agent uses ``architecture="self-refine"`` instead of the
default ReAct.

What this means concretely:

* Default ReAct Agent — one model call (or a tool-loop), the
  output is whatever the model said in that pass.
* SelfRefine Agent — internally drafts an answer, critiques it,
  refines, possibly multiple times. The OUTPUT is the
  post-refinement answer.

From the Workflow's perspective neither is visible — both shapes
return one string. The architecture is encapsulated in the
agent. **Workflow controls the pipeline; Architecture controls
the agent's internal reasoning. They're orthogonal axes.**

The audit log shows both: the workflow's ``step_started`` /
``step_completed`` AND the agent's ``run_started`` /
``run_completed`` — all in one trace, all attributed to the same
``user_id``.

Other architectures swap in identically — try
``architecture="reflexion"``, ``"tree_of_thoughts"``,
``"plan_and_execute"``, etc. The workflow doesn't care.

Run::

    OPENAI_API_KEY=sk-... python examples/10_workflow_architecture.py
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


from jeevesagent import Agent, InMemoryAuditLog, InMemoryMemory, Workflow  # noqa: E402

MODEL = "gpt-4.1-mini"


async def main() -> None:
    print("\n  Example 10 — Agent with non-default Architecture in Workflow\n")

    audit = InMemoryAuditLog()

    # The specialist uses SelfRefine — drafts → critiques → refines
    # internally. Same construction as a default Agent, just one
    # extra ``architecture=`` kwarg.
    specialist = Agent(
        "You are a senior technical writer for Acme. The user gives "
        "you a question; produce a clear, accurate, well-formatted "
        "answer (3-6 sentences max). Cite specific UI labels or CLI "
        "flags where they help.",
        model=MODEL,
        memory=InMemoryMemory(),
        audit_log=audit,
        architecture="self-refine",  # ← non-default; default is "react"
    )

    async def annotate_request(question: str) -> str:
        return f"[customer asked]\n{question}"

    async def package_response(answer: str) -> dict[str, str]:
        return {"answer": answer, "delivered_at": "2026-05-09T00:00:00Z"}

    wf = Workflow.chain(
        [annotate_request, specialist, package_response],
        name="self-refine-pipeline",
        audit_log=audit,
    )

    question = "How do I configure log retention in AcmeTrace?"
    print(f"  question: {question}\n")

    result = await wf.run(question, user_id="alice", session_id="arch-demo")

    answer = result.output["answer"]  # type: ignore[index]
    print(f"  visited : {' → '.join(result.visited)}")
    print(f"  answer  : {answer}\n")

    # Unified attribution — workflow events AND agent events under
    # one user_id, one session_id, one audit log.
    entries = await audit.query(user_id="alice")
    workflow_actions = [e.action for e in entries if e.actor == "workflow"]
    agent_actions = [e.action for e in entries if e.actor != "workflow"]
    print(f"  audit (workflow): {len(workflow_actions)} entries")
    print(f"  audit (agent)   : {len(agent_actions)} entries")


if __name__ == "__main__":
    asyncio.run(main())
