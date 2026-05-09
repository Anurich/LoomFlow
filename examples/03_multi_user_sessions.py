"""Example 3 — Multi-user namespacing + session continuity.

End-to-end demonstration of the two M1/M2 contracts:

  1. **Namespace partitioning by ``user_id``**
     One ``Agent`` shared across many users. Memory is partitioned
     so user A's history NEVER surfaces in user B's recall — even
     though they share the same ``InMemoryMemory`` instance.

  2. **Conversation continuity by ``session_id``**
     Reusing the same ``session_id`` across calls to ``agent.run``
     continues the conversation — prior user/assistant turns get
     rehydrated as real :class:`Message` history so the model sees
     the thread, not just a recall summary.

Layout::

    one Agent + one InMemoryMemory
        ├── user_id="alice"
        │   └── session_id="conv_a"   ← three turns, conversation continues
        ├── user_id="bob"
        │   └── session_id="conv_b"   ← bob's separate history, never sees alice
        └── tool ``whoami`` reads ``get_run_context().user_id``

What you should see:

  * Alice tells the bot her favourite food in turn 1; in turn 2 she
    asks "what's my favourite food?" and the bot answers correctly
    (rehydration works).
  * Bob, on the same Agent, asks the same question after Alice has
    answered — and the bot says it doesn't know (namespace isolation
    works).
  * The ``whoami`` tool always returns the right user_id because the
    framework installs a contextvar for the duration of every run.

Run::

    OPENAI_API_KEY=sk-... python examples/03_multi_user_sessions.py
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


from loomflow import Agent, get_run_context, tool  # noqa: E402
from loomflow.memory.inmemory import InMemoryMemory  # noqa: E402

# --------------------------------------------------------------------
# A tool that reads scope from the run context.
#
# Notice the signature has NO user_id parameter — the model never
# sees ``user_id`` in the schema, can't pass the wrong one, and the
# tool always gets the framework-managed value.
# --------------------------------------------------------------------


@tool
async def whoami() -> str:
    """Return the user_id of the user currently chatting.

    Used by the assistant only when explicitly asked who the user
    is. Reads the live :class:`RunContext` via ``get_run_context()``.
    """
    ctx = get_run_context()
    return ctx.user_id or "anonymous"


def banner(title: str) -> None:
    print()
    print("─" * 72)
    print(title)
    print("─" * 72)


async def main() -> None:
    print("\n  Example 3 — Multi-user + session continuity\n")

    # ONE agent, ONE memory. The framework partitions by user_id
    # and threads session_id through automatically — no extra
    # plumbing in user code.
    memory = InMemoryMemory()
    agent = Agent(
        instructions=(
            "You are a friendly personal assistant. The user is talking "
            "to you over a continuing chat thread; refer back to things "
            "they have told you earlier in this conversation when "
            "relevant. Keep replies short — one or two sentences."
        ),
        model="gpt-4.1-mini",
        memory=memory,
        tools=[whoami],
    )

    # ----------------------------------------------------------------
    # Alice's conversation — session_id="conv_a"
    # ----------------------------------------------------------------

    banner("Alice — turn 1: tells the bot her favourite food")
    r = await agent.run(
        "Hi! My favourite food is pizza. Just remember that.",
        user_id="alice",
        session_id="conv_a",
    )
    print(f"  bot: {r.output}")

    banner("Alice — turn 2: asks the bot to recall it")
    r = await agent.run(
        "Without using any tools, what is my favourite food?",
        user_id="alice",
        session_id="conv_a",   # ← SAME session_id → conversation continues
    )
    print(f"  bot: {r.output}")
    print("  (the bot has rehydrated its prior turn from session memory)")

    banner("Alice — turn 3: asks the whoami tool")
    r = await agent.run(
        "Use the whoami tool and tell me what user id it returns.",
        user_id="alice",
        session_id="conv_a",
    )
    print(f"  bot: {r.output}")

    # ----------------------------------------------------------------
    # Bob's conversation — different user_id, different session_id,
    # SAME Agent + SAME Memory.
    # ----------------------------------------------------------------

    banner("Bob — turn 1: asks for HIS favourite food (must NOT see Alice's)")
    r = await agent.run(
        "Without using any tools, what is my favourite food?",
        user_id="bob",
        session_id="conv_b",
    )
    print(f"  bot: {r.output}")
    print(
        "  (the bot does NOT know — bob has never said. alice's history is "
        "in a different namespace partition and never reaches bob's recall.)"
    )

    banner("Bob — turn 2: tells the bot his own favourite food")
    r = await agent.run(
        "OK my favourite food is sushi. Remember that.",
        user_id="bob",
        session_id="conv_b",
    )
    print(f"  bot: {r.output}")

    banner("Bob — turn 3: re-asks; the bot should now know")
    r = await agent.run(
        "Without using any tools, what is my favourite food?",
        user_id="bob",
        session_id="conv_b",
    )
    print(f"  bot: {r.output}")

    # ----------------------------------------------------------------
    # Sanity-check the partition by going BACK to Alice and confirming
    # she still sees her own history (and not Bob's).
    # ----------------------------------------------------------------

    banner("Alice — turn 4: still pizza, untouched by Bob's session")
    r = await agent.run(
        "Without using any tools, what is my favourite food?",
        user_id="alice",
        session_id="conv_a",
    )
    print(f"  bot: {r.output}")

    # ----------------------------------------------------------------
    # Inspect the persisted state to make the partition concrete.
    # ----------------------------------------------------------------

    banner("Inspecting persisted memory (Memory.recall scoped per user)")
    alice_eps = await memory.recall("food", user_id="alice", limit=10)
    bob_eps = await memory.recall("food", user_id="bob", limit=10)
    anon_eps = await memory.recall("food", user_id=None, limit=10)
    print(f"  alice has {len(alice_eps)} episode(s) in her partition")
    print(f"  bob   has {len(bob_eps)} episode(s) in his partition")
    print(f"  anonymous bucket has {len(anon_eps)} episode(s)")
    print(
        "  → distinct buckets, no cross-contamination, "
        "all from one InMemoryMemory."
    )


if __name__ == "__main__":
    asyncio.run(main())
