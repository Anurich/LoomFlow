"""Example 17 — Prompt caching, one boolean flag.

Prompt caching is the single biggest cost lever for an agent with a
big stable system prompt — typically 50-90% off the input bill,
with a measurable latency drop too. Every major lab implements it
differently:

  * **OpenAI** — fully automatic. Set ``prompt_caching=True`` and
    loomflow parses ``cached_tokens`` out of the API response so
    ``RunResult.cached_tokens_in`` reports the hit rate. The
    50% discount applies automatically. The optional ``cache_key``
    routes related requests to the same backend cache for higher hit
    rates.
  * **Anthropic** — opt-in via ``cache_control`` markers. Set
    ``prompt_caching=True`` and loomflow injects ``cache_control``
    on the LAST system block + LAST tool definition (2 of the 4
    available breakpoints). Cached tokens are billed at **0.1x**;
    cache writes cost 1.25x (5min TTL) or 2x (1h TTL).
  * **Gemini** — separate ``CachedContent.create()`` flow; not
    supported in this release.

You don't have to know any of that. ``prompt_caching=True`` does the
right thing per provider.

This example demonstrates:

  1. The simple boolean form on both Anthropic and OpenAI.
  2. The dict form for advanced control (``ttl="1h"``, ``cache_key``).
  3. **Live cost evidence** — two back-to-back runs with the same big
     prompt: the second run shows non-zero ``result.cached_tokens_in``
     and a markedly lower ``result.cost_usd``.

Run with::

    # .env must contain either ANTHROPIC_API_KEY or OPENAI_API_KEY
    python examples/17_prompt_caching.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass


from loomflow import Agent  # noqa: E402

# A LARGE static system prompt that's the same across both runs.
# This is the part the cache will pay off on — repeating it for every
# turn is what makes the 10x discount on Anthropic / 2x on OpenAI
# materially shape the bill. In a real agent this is your domain
# instructions, tool catalog, few-shot examples, framework rules,
# etc — all the stuff that doesn't change between user prompts.
LARGE_SYSTEM_PROMPT = (
    "You are a meticulous research analyst.\n\n"
    "Your job is to read documents carefully, extract structured "
    "insights, and produce concise summaries with citations.\n\n"
    # Inflate well past the minimum cacheable threshold (OpenAI: 1024
    # tokens; Anthropic: 1024-4096 depending on model). ~3K tokens
    # here so we're comfortably above both. In a real agent this is
    # your domain instructions, tool catalog, few-shot examples,
    # taxonomy — all the stable stuff that benefits from caching.
    + (
        "## Guidelines\n\n"
        "- Be concrete. Names, numbers, dates over generalities.\n"
        "- Quote source text when making claims; surround with backticks.\n"
        "- When asked for an opinion, give the strongest case for ONE\n"
        "  position rather than hedging across all of them.\n"
        "- Format all output as markdown. Headings, lists, tables.\n"
        "- Cap each answer at 200 words unless explicitly asked for more.\n"
        "- Never apologise. Never preamble. Get to the point.\n"
        "- When evidence is thin, say so explicitly: 'Not enough to "
        "claim X, but the pattern suggests Y.'\n\n"
        "## Reference taxonomy\n\n"
        "All reports must classify findings against this taxonomy:\n\n"
        "- **Factual** — directly extracted, verbatim or near-verbatim.\n"
        "- **Inferred** — derived from facts via stated reasoning.\n"
        "- **Speculative** — your hypothesis; clearly flagged as such.\n"
        "- **Counterfactual** — what would change the conclusion.\n\n"
        "## Citation format\n\n"
        "Cite as [SOURCE:PAGE:LINE] inline. Example: 'The Q3 figure "
        "(see [10-K:p12:l5]) shows a 14% YoY drop.'\n\n"
        "## Forbidden phrases\n\n"
        "Never use: 'arguably', 'some say', 'as we all know', 'it is "
        "important to note', 'in conclusion', 'when push comes to "
        "shove', 'at the end of the day'. They signal vague thinking.\n\n"
    ) * 6  # repeat to grow the prefix safely past 1024 tokens
)


async def demo_anthropic_caching() -> None:
    """Run the same prompt twice with prompt_caching=True against
    Claude. First call writes to cache, second hits it."""
    print("=" * 60)
    print("Demo 1 — Anthropic (Claude Opus 4.7) with caching ON")
    print("=" * 60)

    agent = Agent(
        LARGE_SYSTEM_PROMPT,
        model="claude-opus-4-7",
        prompt_caching=True,  # ← the magic boolean
        max_turns=2,
    )

    # First run — populates the cache. Expect cache_write_tokens > 0
    # (those tokens are billed at 1.25x).
    print("\nFirst run (cache cold)...")
    r1 = await agent.run(
        "What's the most important quality in a research analyst?",
        user_id="alice",
        session_id="research-001",
    )
    print(f"  output: {r1.output[:120]}...")
    print(
        f"  tokens: {r1.tokens_in} uncached / "
        f"{r1.cached_tokens_in} cached / "
        f"{r1.cache_write_tokens} written to cache / "
        f"{r1.tokens_out} output"
    )
    print(f"  cost:   ${r1.cost_usd:.4f}")

    # Second run — same system prompt → cache HIT.
    # Expect cached_tokens_in > 0 and a noticeably lower cost.
    print("\nSecond run (cache warm)...")
    r2 = await agent.run(
        "Now name three habits good analysts share.",
        user_id="alice",
        session_id="research-001",
    )
    print(f"  output: {r2.output[:120]}...")
    print(
        f"  tokens: {r2.tokens_in} uncached / "
        f"{r2.cached_tokens_in} cached / "
        f"{r2.cache_write_tokens} written to cache / "
        f"{r2.tokens_out} output"
    )
    print(f"  cost:   ${r2.cost_usd:.4f}")

    print()
    if r2.cached_tokens_in > 0:
        savings = (r1.cost_usd - r2.cost_usd) / max(r1.cost_usd, 1e-9)
        print(
            f"✓ Cache HIT — second run was "
            f"{savings * 100:.0f}% cheaper than the first."
        )
    else:
        print(
            "⚠ No cache hit registered. The cache window is 5 minutes "
            "by default — make sure the two runs happened back-to-back."
        )


async def demo_openai_caching() -> None:
    """OpenAI's prompt caching is automatic — ``prompt_caching=True``
    enables the cache-aware accounting + routing hint but doesn't
    change behaviour on the provider side."""
    print("=" * 60)
    print("Demo 2 — OpenAI (gpt-4.1-mini) with caching ON + cache_key")
    print("=" * 60)

    # Dict form: enable + add a cache_key for better routing on
    # shared-prefix requests (OpenAI's prompt_cache_key parameter).
    agent = Agent(
        LARGE_SYSTEM_PROMPT,
        model="gpt-4.1-mini",
        prompt_caching={
            "enabled": True,
            "cache_key": "demo-session",
        },
        max_turns=2,
    )

    # Note: NO session_id — each run is fresh, so the prefix
    # (system prompt) is byte-identical across runs and OpenAI's
    # cache routing has a stable key to hit.
    print("\nFirst run (cache cold)...")
    r1 = await agent.run(
        "What makes a great research analyst stand out?",
    )
    print(
        f"  tokens: {r1.tokens_in} uncached / "
        f"{r1.cached_tokens_in} cached / "
        f"{r1.tokens_out} output"
    )
    print(f"  cost:   ${r1.cost_usd:.4f}")

    print("\nSecond run (cache warm)...")
    r2 = await agent.run(
        "Now describe a typical day in their work.",
    )
    print(
        f"  tokens: {r2.tokens_in} uncached / "
        f"{r2.cached_tokens_in} cached / "
        f"{r2.tokens_out} output"
    )
    print(f"  cost:   ${r2.cost_usd:.4f}")

    print()
    if r2.cached_tokens_in > 0:
        ratio = r2.cached_tokens_in / max(
            r2.tokens_in + r2.cached_tokens_in, 1
        )
        print(
            f"✓ Cache HIT — {ratio * 100:.0f}% of input tokens "
            "came from cache."
        )
    else:
        print(
            "⚠ No cache hit registered. OpenAI requires prompts ≥ 1024 "
            "tokens; check the system prompt length."
        )


async def main() -> None:
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    if not (has_anthropic or has_openai):
        print(
            "\n  ⊘ Skipping: set ANTHROPIC_API_KEY or OPENAI_API_KEY "
            "in .env to run.\n"
        )
        return

    if has_anthropic:
        await demo_anthropic_caching()
        print()

    if has_openai:
        await demo_openai_caching()


if __name__ == "__main__":
    asyncio.run(main())
