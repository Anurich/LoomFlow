"""12_supervisor — Software-dev team coordinated by a supervisor.

What it shows:
* Supervisor wraps a base architecture (default ReAct) with a
  ``delegate(worker, instructions)`` tool injected into the loop.
  Multiple delegations in one supervisor turn run in parallel.
* Real-world use: any decomposable task with specialist roles —
  research → code → review pipelines, multi-domain queries.
* Streaming + tool events from BOTH the supervisor AND its workers
  flow through to ``agent.stream(...)`` consumers.

Run:
    pip install -e '.[dev,openai]'
    # add OPENAI_API_KEY=sk-... to .env at repo root
    python examples/12_supervisor.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

if not os.environ.get("OPENAI_API_KEY"):
    print(
        "\n  ⊘ OPENAI_API_KEY not set — skipping this example.\n"
        "    Add OPENAI_API_KEY=sk-... to .env at repo root to run.\n"
    )
    sys.exit(0)

from jeevesagent import Agent, Team, tool  # noqa: E402


@tool
def search_best_practices(topic: str) -> str:
    """Look up canonical best practices for a programming topic."""
    fake_kb = {
        "email validation": (
            "Best practice: do NOT roll your own regex. Use the "
            "rules from RFC 5321/5322 (or simply send a confirmation "
            "email). For a quick check, ensure exactly one '@', a "
            "non-empty local part, and a dotted domain. Recommended "
            "library in Python: ``email-validator``."
        ),
        "password hashing": (
            "Use Argon2id (preferred) or bcrypt. Never MD5/SHA-1. "
            "Always use a per-user salt. Library: ``argon2-cffi``."
        ),
    }
    for k, v in fake_kb.items():
        if k in topic.lower():
            return v
    return f"no entry for {topic!r}"


@tool
def lint_python(code: str) -> str:
    """Static-check a Python snippet for common issues. Returns a
    list of findings (or 'clean' if none)."""
    findings = []
    if "import re" not in code and "re.match" in code:
        findings.append("uses re without importing")
    if "except:" in code or "except Exception:" in code:
        findings.append(
            "broad except — catch specific exceptions instead"
        )
    if "print(" in code:
        findings.append(
            "uses print for non-debug output — consider logging"
        )
    if not findings:
        return "clean: no issues found"
    return "issues: " + "; ".join(findings)


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


researcher = Agent(
    "You are a senior engineer who researches best practices. "
    "Use search_best_practices to find canonical guidance. "
    "Summarize findings tightly.",
    model="gpt-4.1-mini",
    tools=[search_best_practices],
)

coder = Agent(
    "You are a Python developer. Write idiomatic, well-typed code "
    "that follows the best practices the researcher provided. "
    "Output ONLY the code — no markdown fences, no commentary.",
    model="gpt-4.1-mini",
)

reviewer = Agent(
    "You review Python code for issues. Use lint_python on the "
    "code, then add any human-eye issues you spot. Be concise.",
    model="gpt-4.1-mini",
    tools=[lint_python],
)


async def main() -> None:
    agent = Team.supervisor(
        workers={
            "researcher": researcher,
            "coder": coder,
            "reviewer": reviewer,
        },
        instructions=(
            "You manage a small dev team. Delegate research first, "
            "then coding, then review. Combine the outputs into a "
            "final answer with the reviewed code + a short summary "
            "of best practices."
        ),
        model="gpt-4.1-mini",
    )

    prompt = (
        "Build a Python function `is_valid_email(email: str) -> bool` "
        "that follows current best practices."
    )

    print("=" * 70)
    print("Supervisor — software dev team")
    print("=" * 70)
    print(f"Task: {prompt}\n")

    async for ev in agent.stream(prompt):
        kind = ev.kind.value
        if kind == "model_chunk":
            chunk = ev.payload.get("chunk", {})
            if chunk.get("kind") == "text" and chunk.get("text"):
                print(chunk["text"], end="", flush=True)
        elif kind == "tool_call":
            call = ev.payload.get("call", {})
            tool_name = call.get("tool", "")
            args = call.get("args", {})
            if tool_name == "delegate":
                worker = args.get("worker", "?")
                preview = args.get("instructions", "")[:70]
                print(f"\n\n  [supervisor → {worker}] {preview}...")
            else:
                print(f"\n  [tool] {tool_name}({args})")
        elif kind == "tool_result":
            result = ev.payload.get("result", {})
            call_id = result.get("call_id", "")
            # Only print delegate results (worker outputs); inner
            # tool results already printed by their workers.
            if call_id.startswith("call_"):
                output = (result.get("output") or "")[:100]
                print(f"  [→] {output}")


if __name__ == "__main__":
    asyncio.run(main())
