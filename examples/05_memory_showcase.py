"""Example 5 — Memory: every backend, profile, fact extraction.

End-to-end showcase of the framework's memory story:

  1. Picking a backend with the ``memory=`` URL/dict resolver
  2. Using each backend in turn (inmemory / sqlite / chroma / postgres /
     redis) — same Agent code, the only thing that changes is the
     string you pass
  3. Inspecting what the bot knows: ``memory.profile(user_id=...)``
  4. GDPR ops: ``memory.forget(user_id=...)`` and
     ``memory.export(user_id=...)``
  5. Fact extraction via the bundled ``Consolidator`` — the bot
     remembers structured claims, not just raw chat history

Postgres / Redis sections SKIP gracefully when their respective
``DATABASE_URL`` / ``REDIS_URL`` env vars aren't set, so this file
runs out of the box on any laptop with just an ``OPENAI_API_KEY``.

Run::

    OPENAI_API_KEY=sk-...                      \\
    [DATABASE_URL=postgres://...]               \\
    [REDIS_URL=redis://localhost:6379/0]        \\
    python examples/05_memory_showcase.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
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


from loomflow import (  # noqa: E402
    Agent,
    MemoryProfile,
)
from loomflow.memory.embedder import OpenAIEmbedder  # noqa: E402

MODEL = "gpt-4.1-mini"


def banner(title: str) -> None:
    print()
    print("─" * 78)
    print(f"  {title}")
    print("─" * 78)


def show_profile(label: str, profile: MemoryProfile) -> None:
    print(f"  [{label}] profile for user_id={profile.user_id!r}:")
    print(f"     episodes:        {profile.episode_count}")
    print(f"     facts:           {profile.fact_count}")
    print(f"     last_seen:       {profile.last_seen}")
    print(f"     recent_sessions: {profile.recent_sessions[:3]}")
    if profile.sample_facts:
        print("     sample_facts:")
        for f in profile.sample_facts[:3]:
            print(f"       • {f.subject} {f.predicate} {f.object}")


# --------------------------------------------------------------------
# 1. Pick a backend with one parameter — same code for every backend.
# --------------------------------------------------------------------


async def demo_backend(name: str, memory_spec: str | dict | None) -> None:
    """Run the same scenario against any memory backend.

    Two users (alice + bob) chat with the bot across two sessions
    each. We verify the partition (bob never sees alice's data),
    show the profile + recall, then exercise ``forget`` + ``export``.
    """
    banner(f"Backend: {name}")
    print(f"  memory= {memory_spec!r}")

    agent = Agent(
        "You are a brief, helpful assistant. Reply in one sentence.",
        model=MODEL,
        memory=memory_spec,
    )

    # Alice tells the bot about herself.
    await agent.run(
        "Hi, I'm Alice and I love jazz music.",
        user_id="alice",
        session_id="alice_chat_1",
    )
    await agent.run(
        "Without using any tools, what music do I love?",
        user_id="alice",
        session_id="alice_chat_1",
    )

    # Bob — separate user_id. Should NOT see Alice's data.
    bob = await agent.run(
        "Without using any tools, what music do I love?",
        user_id="bob",
        session_id="bob_chat_1",
    )
    print(f"  bob (no memory of alice's jazz): {bob.output[:80]!r}")

    # Inspect what the memory holds for each user.
    alice_profile = await agent.memory.profile(user_id="alice")
    bob_profile = await agent.memory.profile(user_id="bob")
    show_profile("alice", alice_profile)
    show_profile("bob  ", bob_profile)

    # Export — full data dump for portability / DSAR responses.
    export = await agent.memory.export(user_id="alice")
    print(f"  alice export: {len(export.episodes)} episodes, "
          f"{len(export.facts)} facts; serialisable as JSON "
          f"({len(export.model_dump_json())} chars)")

    # Forget — GDPR right-to-erasure.
    deleted = await agent.memory.forget(user_id="alice")
    print(f"  forget(user_id='alice') → {deleted} records erased")
    after = await agent.memory.profile(user_id="alice")
    print(f"  after forget — alice episodes: {after.episode_count}, "
          f"bob episodes: {(await agent.memory.profile(user_id='bob')).episode_count}")


# --------------------------------------------------------------------
# 2. Auto fact extraction (the default UX) — the framework runs the
#    bundled Consolidator after every agent.run, pulling structured
#    (subject, predicate, object) facts out of the conversation and
#    writing them to the bi-temporal fact store automatically.
#
#    No manual Consolidator wiring; no .consolidate() call. This is
#    what "your bot just remembers things" means.
# --------------------------------------------------------------------


async def demo_auto_extract(memory_spec: str | dict | None) -> None:
    banner("Auto fact extraction (default ON for real models)")
    print(f"  memory= {memory_spec!r}")

    # Build an Agent the normal way. ``auto_extract=True`` is the
    # default for OpenAI / Anthropic / LiteLLM; we set it
    # explicitly so the print above is unambiguous.
    agent = Agent(
        "You are a brief, friendly assistant. Acknowledge what the "
        "user tells you in one sentence.",
        model=MODEL,
        memory=memory_spec,
        auto_extract=True,
    )

    # Three normal conversational turns. Every one of them has the
    # framework run the Consolidator behind the scenes — no extra
    # code from us.
    for prompt in [
        "Hi! I work at Acme Corp as a senior platform engineer.",
        "Quick context: I'm based in Lisbon but originally from Brazil.",
        "My favourite programming language is Python; I use Rust for hot paths.",
    ]:
        await agent.run(
            prompt, user_id="alice", session_id="onboarding"
        )

    # Inspect what the agent extracted automatically.
    profile = await agent.memory.profile(user_id="alice")
    print(f"  alice now has {profile.fact_count} auto-extracted facts:")
    for f in profile.sample_facts[:6]:
        print(f"    • {f.subject} {f.predicate} {f.object}")

    # Semantic recall on a question the user never literally asked
    # — answered from extracted facts, not raw chat history.
    recall = await agent.memory.recall_facts(
        "Where does the user live?", user_id="alice"
    )
    if recall:
        print("  recall_facts('Where does the user live?') → ")
        for f in recall[:3]:
            print(f"    • {f.subject} {f.predicate} {f.object}")

    # Power-user escape hatch: the same Consolidator is still
    # accessible if you want to run it on bulk historical data, on
    # a different model (cheaper extraction), or with a custom
    # system prompt. ``auto_extract=False`` turns off the
    # automatic pass.
    print(
        "  (power user: pass auto_extract=False + drive Consolidator "
        "yourself for bulk / off-line extraction.)"
    )


# --------------------------------------------------------------------
# Driver — exercises every backend that's reachable in the
# environment. Postgres / Redis skip gracefully when no DSN is set.
# --------------------------------------------------------------------


async def main() -> None:
    print("\n  Example 5 — Memory: every backend, profile, fact extraction\n")

    # ----------------------------------------------------------------
    # Backends 1–3: zero-config — always available.
    # ----------------------------------------------------------------
    await demo_backend("inmemory (default)", None)

    # SQLite gets its own temp dir so the example doesn't pollute cwd.
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "showcase.db"
        await demo_backend(f"sqlite (persistent file: {db.name})", f"sqlite:{db}")

        # Show that SQLite is genuinely persistent: re-open the same
        # path with a fresh Agent and the data we left behind for
        # bob is still there.
        banner("SQLite persistence sanity-check")
        agent2 = Agent("...", model=MODEL, memory=f"sqlite:{db}")
        prof = await agent2.memory.profile(user_id="bob")
        print(f"  reopened {db.name}: bob still has "
              f"{prof.episode_count} episode(s) — file survived.")

    # Chroma ephemeral — in-process, no infra. (Persistent path used
    # in example 01.)
    await demo_backend("chroma (ephemeral, in-process)", "chroma")

    # ----------------------------------------------------------------
    # Backend 4: Postgres — only runs when DATABASE_URL is set.
    # ----------------------------------------------------------------
    pg_url = os.environ.get("DATABASE_URL")
    if pg_url and pg_url.startswith(("postgres://", "postgresql://")):
        await demo_backend("postgres (DATABASE_URL set)", pg_url)
    else:
        banner("Backend: postgres")
        print("  Skipped — set DATABASE_URL=postgres://... to enable.")
        print("  Code is identical:")
        print("    Agent(..., memory='postgres://user:pw@host/db')")
        print("  The framework lazy-connects on first agent.run; sync")
        print("  Agent constructor stays sync regardless of backend.")

    # ----------------------------------------------------------------
    # Backend 5: Redis — only runs when REDIS_URL is set.
    # ----------------------------------------------------------------
    redis_url = os.environ.get("REDIS_URL")
    if redis_url and redis_url.startswith(("redis://", "rediss://")):
        await demo_backend("redis (REDIS_URL set)", redis_url)
    else:
        banner("Backend: redis")
        print("  Skipped — set REDIS_URL=redis://localhost:6379/0 to enable.")

    # ----------------------------------------------------------------
    # Fact extraction: works against any backend; we use SQLite for
    # the demo so the extracted facts persist somewhere visible.
    # ----------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "facts.db"
        await demo_auto_extract(f"sqlite:{db}")

    # ----------------------------------------------------------------
    # Config-dict form — for the backends where the URL string
    # doesn't carry every option you need (custom embedder,
    # explicit namespace, etc.).
    # ----------------------------------------------------------------
    banner("Config-dict form (Tier 2)")
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "configdict.db"
        agent = Agent(
            "You are helpful.",
            model=MODEL,
            memory={
                "backend": "sqlite",
                "path": str(db),
                "embedder": "openai",
                "with_facts": True,
            },
        )
        await agent.run("Hi!", user_id="alice", session_id="s")
        p = await agent.memory.profile(user_id="alice")
        print(f"  agent memory: {type(agent.memory).__name__}")
        print(f"  embedder:     {type(agent.memory.embedder).__name__}")
        print(f"  facts on:     {agent.memory.facts is not None}")
        print(f"  alice has {p.episode_count} episode(s) after one run")

    # ----------------------------------------------------------------
    # Tier 3 — explicit instance. Today's API; still supported.
    # ----------------------------------------------------------------
    banner("Tier 3: explicit instance")
    from loomflow.memory import SqliteMemory
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "explicit.db"
        memory = SqliteMemory(
            db,
            with_facts=True,
            embedder=OpenAIEmbedder("text-embedding-3-small"),
        )
        agent = Agent(
            "You are helpful.", model=MODEL, memory=memory
        )
        print(f"  agent.memory is the same instance: "
              f"{agent.memory is memory}")
        await agent.run("Hi!", user_id="alice", session_id="s")
        print(f"  ran one turn; memory persisted to {db.name}")

    print()
    print("─" * 78)
    print("  Done. The single ``memory=`` parameter picked the backend")
    print("  in every section above; the rest of the agent code is")
    print("  identical regardless of where memory lives.")
    print("─" * 78)


if __name__ == "__main__":
    asyncio.run(main())
