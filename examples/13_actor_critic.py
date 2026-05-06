"""13_actor_critic — Actor + adversarial critic, asymmetric by design.

What it shows:
* The ``ActorCritic`` architecture orchestrates two separate ``Agent``
  instances: an actor that generates output and a critic that
  reviews it adversarially. Different prompts (and ideally different
  models) catch different blind spots.
* The critic emits structured JSON with ``issues``, ``score``, and
  ``summary``. ``ActorCritic`` parses it (markdown fences and loose
  text are both handled) and either terminates on
  ``score >= approval_threshold`` or feeds the critique back to the
  actor for revision.
* Quality-driven termination — no fixed ``max_turns``; loops until
  the critic approves OR ``max_rounds`` is hit.

Production canonical use: code review. Configure actor with one
model (e.g. Claude Opus) and critic with a different model (e.g.
GPT-4o) plus an adversarial system prompt. The asymmetry — different
priors, different blind spots — is the point.

We use ``ScriptedModel`` so the example runs deterministically. In
production each agent has its own real ``model="claude-opus-4-7"``
or similar.

Run:
    python examples/13_actor_critic.py
"""

from __future__ import annotations

import asyncio

from jeevesagent import (
    ActorCritic,
    Agent,
    ScriptedModel,
    ScriptedTurn,
)


def _scripted_agent(role_instructions: str, replies: list[str]) -> Agent:
    return Agent(
        role_instructions,
        model=ScriptedModel([ScriptedTurn(text=r) for r in replies]),
    )


async def main() -> None:
    # Actor produces a draft, then a polished revision after critique.
    actor = _scripted_agent(
        (
            "You write production Python. Be complete and correct. "
            "When given a critique, address every point."
        ),
        replies=[
            # Round 0: initial draft (a deliberately incomplete one)
            "def divide(a, b):\n    return a / b",
            # Round 1 refine: improved version addressing the critique
            (
                "def divide(a: float, b: float) -> float:\n"
                '    """Divide a by b. Raises ValueError on b == 0."""\n'
                "    if b == 0:\n"
                "        raise ValueError('cannot divide by zero')\n"
                "    return a / b"
            ),
        ],
    )

    # Critic plays an adversarial reviewer with structured JSON output.
    # In production, give the critic a DIFFERENT model than the actor —
    # the asymmetry is what catches actor blind spots.
    critic = _scripted_agent(
        (
            "You review code adversarially. Find every issue: missing "
            "type hints, no docstring, unhandled edge cases, bad names. "
            'Output JSON: {"issues": [...], "score": 0-1, "summary": str}.'
        ),
        replies=[
            (
                "{\n"
                '  "issues": [\n'
                '    "no type hints",\n'
                '    "no docstring",\n'
                '    "unhandled ZeroDivisionError"\n'
                "  ],\n"
                '  "score": 0.4,\n'
                '  "summary": "Functional but unsafe and undocumented."\n'
                "}"
            ),
            (
                "{\n"
                '  "issues": [],\n'
                '  "score": 0.95,\n'
                '  "summary": "Clean, typed, documented, handles edge."\n'
                "}"
            ),
        ],
    )

    agent = Agent(
        "Code-quality coordinator.",
        model=ScriptedModel(
            [ScriptedTurn(text="never reached")]
        ),  # outer model unused — ActorCritic drives sub-agents
        architecture=ActorCritic(
            actor=actor,
            critic=critic,
            max_rounds=3,
            approval_threshold=0.9,
        ),
    )

    print("=== Streaming events ===")
    async for event in agent.stream(
        "Write a Python function that divides two numbers."
    ):
        if event.kind == "architecture_event":
            name = event.payload.get("name", "")
            if name == "actor_critic.actor_started":
                round_num = event.payload["round"]
                phase = event.payload.get("phase", "?")
                print(f"\n[actor, round {round_num}, {phase}]")
            elif name == "actor_critic.critique":
                round_num = event.payload["round"]
                score = event.payload["score"]
                issues = event.payload["issues"]
                print(
                    f"[critic, round {round_num}] score={score:.2f}"
                )
                for issue in issues:
                    print(f"  • {issue}")
            elif name == "actor_critic.approved":
                round_num = event.payload["round"]
                score = event.payload["score"]
                print(
                    f"\n[approved on round {round_num}, "
                    f"score={score:.2f}]"
                )
            elif name == "actor_critic.max_rounds_reached":
                print("\n[max rounds reached]")

    # Re-run with fresh scripted agents to print the final result.
    fresh_actor = _scripted_agent(
        "actor",
        [
            "def divide(a, b):\n    return a / b",
            (
                "def divide(a: float, b: float) -> float:\n"
                '    """Divide a by b. Raises on b == 0."""\n'
                "    if b == 0:\n"
                "        raise ValueError('cannot divide by zero')\n"
                "    return a / b"
            ),
        ],
    )
    fresh_critic = _scripted_agent(
        "critic",
        [
            '{"issues": ["no types", "no docstring"], "score": 0.4, "summary": ""}',
            '{"issues": [], "score": 0.95, "summary": ""}',
        ],
    )
    fresh_agent = Agent(
        "host",
        model=ScriptedModel([ScriptedTurn(text="never reached")]),
        architecture=ActorCritic(
            actor=fresh_actor,
            critic=fresh_critic,
            max_rounds=3,
            approval_threshold=0.9,
        ),
    )
    result = await fresh_agent.run(
        "Write a divide function with proper handling."
    )
    print("\n=== Final code ===")
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
