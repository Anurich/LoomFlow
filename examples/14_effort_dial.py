"""Example 14 — Reasoning-effort dial across providers.

LLM labs ship "think harder before answering" knobs under different
names: OpenAI's o-series / GPT-5 use ``reasoning_effort``, Anthropic
Opus 4.7 uses adaptive thinking + ``output_config.effort``, older
Claude Sonnets use a ``thinking.budget_tokens`` integer, LiteLLM
normalises everyone to OpenAI's shape, Gemini has its own integer
budget — and every adapter accepts a slightly different argument.

Loom unifies all of this behind one enum::

    effort = "minimal" | "low" | "medium" | "high" | "xhigh" | "max"

Pass it once, the framework picks the right provider-native shape
for the model you're talking to. Models that don't support reasoning
effort emit a one-time warning per ``(model, effort)`` pair and
drop the kwarg — opt into hard-fail with ``strict_effort=True``.

Two equivalent ways to wire it on an Agent, pick whichever reads
better at your call site:

  * **Dict form** — everything model-related in one place::

        Agent(
            "...",
            model={
                "name": "claude-opus-4-7",
                "effort": "high",
                "strict_effort": True,
            },
        )

  * **Explicit kwargs** — good for short inline construction::

        Agent("...", model="claude-opus-4-7", effort="high")

This example shows both forms, plus per-call override and the
strict-fail path. Run with::

    ANTHROPIC_API_KEY=sk-... python examples/14_effort_dial.py
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

if not os.environ.get("ANTHROPIC_API_KEY"):
    print(
        "\n  ⊘ Skipping: ANTHROPIC_API_KEY is not set. "
        "Export it (or add it to .env) to run this example.\n"
    )
    sys.exit(0)


from loomflow import Agent  # noqa: E402
from loomflow.model._effort import EffortNotSupportedError  # noqa: E402

# A question that benefits from more thinking — multi-step
# arithmetic + a small constraint check.
QUESTION = (
    "I have a 3-litre jug and a 5-litre jug, both unmarked, and an "
    "unlimited supply of water. Walk me through every step required "
    "to measure out EXACTLY 4 litres. Be careful — the wrong "
    "sequence wastes water but still ends at 4L; I want the "
    "minimum-steps solution."
)


async def main() -> None:
    print("\n  Example 14 — Reasoning-effort dial across providers\n")

    # ------------------------------------------------------------------
    # 1. Same question at each effort tier.
    # ------------------------------------------------------------------
    #
    # We sweep effort across "low → medium → high → xhigh" on
    # ``claude-opus-4-7``, the only regime that accepts the full
    # enum unclamped. The dict form keeps the model + dial in one
    # place; ``run(..., effort=)`` overrides it per call.

    agent = Agent(
        "You are a careful reasoner. Show your steps.",
        # Dict-form ``model={...}`` — the model spec and any
        # related agent defaults live together. Same shape as
        # ``audit_log={...}``: one parameter, structured config.
        model={
            "name": "claude-opus-4-7",
            "effort": "medium",  # default for runs without their own override
        },
    )

    print("─" * 72)
    print("  Asking the same question at each effort level")
    print(f"  Model: {agent.model.name}  (agent default effort: medium)")
    print("─" * 72)

    for effort in ("low", "medium", "high", "xhigh"):
        # Per-call effort wins over the agent default. Useful when
        # the same Agent serves cheap chit-chat AND occasional deep
        # reasoning — no need for two Agents.
        result = await agent.run(QUESTION, effort=effort)
        preview = result.output.replace("\n", " ")[:140]
        print(
            f"\n  effort={effort:<7} "
            f"tokens={result.tokens_in:>5}+{result.tokens_out:<5}  "
            f"turns={result.turns}"
        )
        print(f"    {preview}{'...' if len(result.output) > 140 else ''}")

    # ------------------------------------------------------------------
    # 2. Equivalent explicit-kwarg form (for callers who prefer it).
    # ------------------------------------------------------------------
    #
    # Dict and kwargs are interchangeable; the framework normalises
    # both to the same internal state. Pick whichever reads better
    # at the call site — config-style for declarative setups, kwargs
    # for short inline construction.

    print("\n" + "─" * 72)
    print("  Equivalent: explicit kwargs instead of dict")
    print("─" * 72)

    explicit_agent = Agent(
        "You are a careful reasoner.",
        model="claude-opus-4-7",
        effort="medium",
    )
    result = await explicit_agent.run(QUESTION)
    print(
        f"\n  Same default (medium) via explicit kwargs: "
        f"tokens={result.tokens_in}+{result.tokens_out}"
    )

    # When both forms are present, the explicit top-level kwarg
    # wins — useful for environment-specific overrides on a shared
    # config dict.
    override_agent = Agent(
        "You are a careful reasoner.",
        model={"name": "claude-opus-4-7", "effort": "low"},
        effort="high",  # ← wins over the "low" in the dict
    )
    result = await override_agent.run(QUESTION)
    print(
        f"  Explicit kwarg overrides dict (now high): "
        f"tokens={result.tokens_in}+{result.tokens_out}"
    )

    # ------------------------------------------------------------------
    # 3. strict_effort=True — fail loudly on an unsupported model.
    # ------------------------------------------------------------------
    #
    # ``claude-haiku-3-5`` doesn't support reasoning effort. With
    # the default warn-and-drop behaviour the run would proceed
    # silently; ``strict_effort=True`` turns the drop into a hard
    # error so wiring mistakes surface immediately during
    # development. Use this in CI / pre-prod for catching typos.
    #
    # Dict form keeps the failure-mode flag next to the model it
    # applies to — easy to read, easy to flip per environment.

    print("\n" + "─" * 72)
    print("  strict_effort=True — fail loudly when the model can't honour it")
    print("─" * 72)

    strict_agent = Agent(
        "Answer briefly.",
        model={
            "name": "claude-haiku-3-5",
            "effort": "high",
            "strict_effort": True,
        },
    )

    try:
        await strict_agent.run("What's 2 + 2?")
    except EffortNotSupportedError as exc:
        print("\n  ✓ Caught EffortNotSupportedError as expected:")
        print(f"    {exc}")

    print("\n  Done.\n")


if __name__ == "__main__":
    asyncio.run(main())
