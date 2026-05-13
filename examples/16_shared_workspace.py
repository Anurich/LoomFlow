"""Example 16 — Shared notebook for multi-agent teams (live).

A research team — four specialists + a synthesizer — collaborates
on one question by writing notes to a shared notebook. Each
specialist runs in parallel; none of them sees the others'
transcripts. The synthesizer then reads everyone's notes via
``list_notes`` / ``read_note`` and writes the final recommendation
into the same notebook.

Why this matters:

  * **No history sharing.** Specialists run hermetically — no
    cross-agent transcript flood. Only their structured findings
    end up in the notebook.
  * **Synthesizer context stays small.** It reads N curated notes
    (each ~100 tokens) instead of N×K full transcripts.
  * **Filesystem-mounted.** ``WORKSPACE.md`` is auto-regenerated;
    you can ``cat`` it during or after the run.
  * **Cross-run persistence.** Use a fixed path instead of temp
    and a second run picks up where the first left off.

What the framework does for you:

  * Auto-attribution — each agent's notes are tagged with its team
    role (``[tech_architect]`` etc.) because the author identity is
    baked into the workspace tools' closure.
  * Auto-index — every ``note(...)`` call regenerates
    ``WORKSPACE.md`` atomically so reads always see a consistent
    table of contents.
  * Slug + frontmatter — filenames are invisible; agents only see
    titles. The disk backend handles ``001-foo.md`` naming, YAML
    frontmatter, and per-author subdirs.

A note on real-world LLM behaviour you'll see in the output:

  Even with strict "ONE note then stop" prompts, smaller models
  (gpt-4.1-mini included) occasionally write multiple notes in a
  single turn — emitting parallel tool calls covering several
  angles. The workspace handles this gracefully: notes are
  attributed per-author, the index stays consistent, and the
  synthesizer reads whatever's there and produces a coherent
  recommendation regardless. This is actually a feature of the
  contributor → aggregator pattern — messy individual contributors
  are fine when the aggregator is the one synthesising.

Run with::

    # .env should contain OPENAI_API_KEY=sk-...
    python examples/16_shared_workspace.py

Uses ``gpt-4.1-mini`` so the demo is cheap (~$0.02 per run).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Prefer the in-repo source over any site-packages install so the
# example exercises the same code as the local tests.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

if not os.environ.get("OPENAI_API_KEY"):
    print(
        "\n  ⊘ Skipping: OPENAI_API_KEY is not set. "
        "Add it to .env (or export it) to run this example.\n"
    )
    sys.exit(0)


import anyio  # noqa: E402

from loomflow import Agent, LocalDiskWorkspace  # noqa: E402
from loomflow.governance.budget import BudgetConfig, StandardBudget  # noqa: E402

# ---------------------------------------------------------------------------
# The question + the team
# ---------------------------------------------------------------------------

QUESTION = (
    "A 10-engineer startup with $2M ARR is considering migrating their "
    "Django monolith to microservices. Should they do it now, wait, or "
    "stay on the monolith? Each teammate, share ONE strong perspective "
    "from your angle."
)

# Specialists are FIRE-AND-FORGET contributors. They don't browse
# the workspace — they each write exactly ONE note in their OWN
# voice and stop. The synthesizer (Phase 2) is the one that reads
# everyone's notes and produces the final view.
#
# Why this division of labour: when specialists run in parallel
# and ALSO read the workspace, they over-eagerly write multiple
# notes "responding" to teammates' findings as those land. The
# clean pattern is contributor → aggregator, not contributors
# observing each other mid-run.

_NOTE_RULE = (
    "STRICT RULES:\n"
    "1. Call note(title=..., content=..., kind='finding') EXACTLY ONCE.\n"
    "2. The note is YOUR voice only. Do NOT write notes from other "
    "roles' perspectives. You are not the architect, PM, or investor "
    "(unless that IS your role) — you are one specific role and you "
    "write one note from that one angle.\n"
    "3. The note body is 5-7 sentences in YOUR voice. Use first "
    "person (\"I think\", \"In my experience...\").\n"
    "4. After the single note call, respond with the word 'done' "
    "and nothing else. Do NOT call any other tools.\n"
    "5. Do NOT call list_notes / read_note / search_notes — "
    "teammates run in parallel; the synthesizer reads everything "
    "at the end."
)

SPECIALISTS: dict[str, str] = {
    "tech_architect": (
        "ROLE: You are the TECH ARCHITECT — a senior software "
        "architect with 15 years scaling SaaS startups.\n\n"
        "ASSIGNMENT: Form ONE opinionated TECHNICAL view on the real "
        "engineering trade-offs of microservices at small scale "
        "(10 engineers, $2M ARR). Things like: service boundary "
        "discipline, data consistency, observability cost, the cost "
        "of a wrong split.\n\n" + _NOTE_RULE
    ),
    "engineer": (
        "ROLE: You are the ENGINEER — a hands-on senior IC who's "
        "done two monolith→services migrations and is currently "
        "on-call.\n\n"
        "ASSIGNMENT: Form ONE PRACTICAL view focused on lived "
        "operational reality at 10-engineer scale: on-call burden, "
        "deploy friction, dev-loop speed, debug pain.\n\n"
        + _NOTE_RULE
    ),
    "product_manager": (
        "ROLE: You are the PRODUCT MANAGER — senior PM responsible "
        "for the roadmap.\n\n"
        "ASSIGNMENT: Form ONE view on BUSINESS / VELOCITY: how the "
        "migration affects feature shipping cadence, customer "
        "risk, hiring velocity.\n\n" + _NOTE_RULE
    ),
    "investor": (
        "ROLE: You are the SEED-STAGE INVESTOR — you've watched 50 "
        "startups make this call.\n\n"
        "ASSIGNMENT: Form ONE view on CAPITAL EFFICIENCY: burn rate "
        "impact, engineering ROI, what Series A VCs will ask about "
        "this decision at the next round.\n\n" + _NOTE_RULE
    ),
}

SYNTHESIZER_INSTRUCTIONS = (
    "You synthesize a research team's findings into a final "
    "recommendation. Your teammates have each written one note in "
    "the shared notebook.\n\n"
    "Steps:\n"
    "1. Call list_notes() to see every note.\n"
    "2. For each note, call read_note(slug) to read the full content.\n"
    "3. Form a synthesis: where does the team converge? Where do "
    "they disagree? What's the strongest case?\n"
    "4. Call note(title='Final recommendation', content=..., "
    "kind='summary') with 10-12 sentences: the recommendation, the "
    "key trade-off, and one concrete next step.\n"
    "5. Output the recommendation as your final response."
)


# ---------------------------------------------------------------------------
# Build + run the team
# ---------------------------------------------------------------------------


async def run_specialist(
    name: str,
    instructions: str,
    workspace: LocalDiskWorkspace,
) -> None:
    """Build a specialist Agent, wire the workspace, run once.

    Each specialist gets its own Agent with ``workspace=ws`` and
    ``workspace_name=<role>``. The five notebook tools are
    auto-installed onto the agent's tool host at construction
    time; the system prompt is auto-augmented with the workspace
    nudges so the model knows to use them.
    """
    teammates = list(SPECIALISTS.keys())
    agent = Agent(
        instructions,
        model="gpt-4.1-mini",
        # One kwarg, three semantics: the shared notebook + this
        # agent's author identity + the team they're collaborating
        # with. ``ws.member(...)`` collapses what used to be three
        # separate Agent kwargs into a single chained call.
        workspace=workspace.member(name, teammates=teammates),
        max_turns=2,  # one note call + final "done" message — hard cap
        budget=StandardBudget(
            BudgetConfig(max_tokens=20_000, max_cost_usd=0.10),
        ),
    )
    print(f"  → {name} thinking...")
    result = await agent.run(QUESTION, user_id="research_team")
    print(f"  ✓ {name} done ({result.turns} turn(s), "
          f"${result.cost_usd:.4f})")


async def main() -> None:
    workspace = LocalDiskWorkspace.temp(prefix="loom-research-", cleanup=False)
    print(f"Workspace: {workspace.root}")
    print(f"Question:  {QUESTION}\n")

    # ---- Phase 1 — specialists in parallel -------------------------------
    print("Phase 1 — four specialists writing their findings in parallel")
    async with anyio.create_task_group() as tg:
        for name, instructions in SPECIALISTS.items():
            tg.start_soon(run_specialist, name, instructions, workspace)

    print()
    print("─" * 60)
    print("After Phase 1, the notebook contains:")
    print("─" * 60)
    print(await workspace.render_index(user_id="research_team"))

    # ---- Phase 2 — synthesizer reads everyone's notes -------------------
    print("─" * 60)
    print("Phase 2 — synthesizer reads the team's findings and writes "
          "the recommendation")
    print("─" * 60)
    synthesizer = Agent(
        SYNTHESIZER_INSTRUCTIONS,
        model="gpt-4.1-mini",
        workspace=workspace.member(
            "synthesizer", teammates=list(SPECIALISTS.keys())
        ),
        budget=StandardBudget(
            BudgetConfig(max_tokens=30_000, max_cost_usd=0.15),
        ),
    )
    result = await synthesizer.run(
        "Read the team's findings and write the final recommendation.",
        user_id="research_team",
    )
    print()
    print(f"Synthesizer: {result.turns} turn(s), "
          f"${result.cost_usd:.4f}")
    print()
    print("─" * 60)
    print("Synthesizer's response:")
    print("─" * 60)
    print(result.output)

    # ---- Final notebook ------------------------------------------------
    print()
    print("═" * 60)
    print("FINAL NOTEBOOK")
    print("═" * 60)
    print(await workspace.render_index(user_id="research_team"))
    print(f"\nFull notebook on disk: {workspace.root}")
    print("Inspect the raw notes:")
    print(f"  ls {workspace.root}/research_team/notes/")
    print(f"  cat {workspace.root}/research_team/WORKSPACE.md")


if __name__ == "__main__":
    asyncio.run(main())
