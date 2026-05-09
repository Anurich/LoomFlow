"""Live integration tests against the paid OpenAI endpoint.

These exercise the framework against a *real* model — catching the
class of regression that ``ScriptedModel`` can't see (e.g. an SDK
upgrade subtly changes the response shape, the structured-output
prompt directive isn't doing its job in practice, the retry
classifier mishandles an exception type the SDK started raising).

Skipped by default. Opt in with::

    OPENAI_API_KEY=sk-... pytest -m live

The tests are intentionally **fast**, **cheap**, and **few** —
they're a smoke suite, not an exhaustive matrix. Each one runs
exactly one short prompt through ``gpt-4.1-mini`` so the whole
file finishes in well under a minute and costs a fraction of a
cent. If you find yourself adding expensive scenarios here,
they probably belong in a separate evaluation harness instead.
"""

from __future__ import annotations

import gc
import os
from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import anyio.lowlevel
import pytest
from pydantic import BaseModel

from loomflow import Agent, get_run_context, tool
from loomflow.governance import RetryPolicy
from loomflow.memory.inmemory import InMemoryMemory

# Try .env (developer ergonomics). CI sets OPENAI_API_KEY directly.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

pytestmark = [
    pytest.mark.live,
    pytest.mark.anyio,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    ),
]

MODEL = "gpt-4.1-mini"


# Force the asyncio backend; trio isn't supported by openai's SDK and
# without pinning, anyio will try both and emit confusing failures.
@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# AsyncOpenAI ties its httpx transport to whichever event loop
# created it. pytest-anyio gives each test a fresh loop (correct),
# but stale clients from prior tests can still try to close their
# transports against a closed loop during garbage collection. Force
# a GC + an event-loop tick at the end of every test so any pending
# close callbacks drain on THIS test's loop, not the next one.
@pytest.fixture(autouse=True)
async def _drain_async_clients() -> AsyncIterator[None]:
    yield
    gc.collect()
    await anyio.lowlevel.checkpoint()


# ---------------------------------------------------------------------------
# Smoke: model is reachable + agent loop terminates
# ---------------------------------------------------------------------------


async def test_basic_run_returns_a_response() -> None:
    """The simplest case: one round-trip, no tools. Verifies the
    model adapter is alive and the agent loop terminates."""
    agent = Agent("Be brief.", model=MODEL)
    result = await agent.run("Reply with the single word: ready.")
    assert result.output  # non-empty
    assert result.turns >= 1
    assert result.tokens_in > 0
    assert result.tokens_out > 0


# ---------------------------------------------------------------------------
# Tool calling: parallel dispatch
# ---------------------------------------------------------------------------


async def test_tool_call_round_trip() -> None:
    """End-to-end tool dispatch — model decides to call the tool,
    framework runs it, model integrates the result."""
    calls: list[str] = []

    @tool
    async def lookup_password_policy() -> str:
        """Return the company password policy.

        Use this tool whenever the user asks about password rules.
        """
        calls.append("lookup_password_policy")
        return (
            "Passwords must be at least 14 characters long, contain "
            "letters, numbers, and a symbol."
        )

    agent = Agent(
        "When asked about company policy, ALWAYS use the available "
        "tools to look it up. Then answer in one sentence.",
        model=MODEL,
        tools=[lookup_password_policy],
    )
    result = await agent.run(
        "What's our minimum password length?"
    )
    assert "14" in result.output
    assert calls == ["lookup_password_policy"]


# ---------------------------------------------------------------------------
# Multi-tenant memory: one Agent, two users, no leakage
# ---------------------------------------------------------------------------


async def test_user_id_partitions_memory() -> None:
    """Alice tells the bot a fact in turn 1; Bob — same Agent, same
    Memory — must NOT see Alice's fact in turn 1 of his own
    session."""
    memory = InMemoryMemory()
    agent = Agent(
        "Reply in one sentence. Refer to information the current user "
        "told you earlier in THIS conversation when relevant. Do not "
        "invent facts.",
        model=MODEL,
        memory=memory,
    )
    await agent.run(
        "My name is Alice. Just remember.",
        user_id="alice",
        session_id="convA",
    )
    bob_first = await agent.run(
        "Without using any tools, what is my name?",
        user_id="bob",
        session_id="convB",
    )
    # Bob should not see "alice".
    assert "alice" not in bob_first.output.lower()


async def test_session_id_continues_conversation() -> None:
    """Same session_id reused → conversation continues, the model
    sees prior turns as real chat history."""
    memory = InMemoryMemory()
    agent = Agent(
        "Reply in one sentence. Refer back to what the user told you "
        "earlier in this conversation.",
        model=MODEL,
        memory=memory,
    )
    await agent.run(
        "My favourite color is teal. Just remember.",
        user_id="alice",
        session_id="conv1",
    )
    follow_up = await agent.run(
        "Without using any tools, what is my favourite color?",
        user_id="alice",
        session_id="conv1",  # SAME session
    )
    assert "teal" in follow_up.output.lower()


# ---------------------------------------------------------------------------
# Tool sees framework-managed user_id via contextvar
# ---------------------------------------------------------------------------


async def test_tool_reads_user_id_from_run_context() -> None:
    """A tool calls ``get_run_context()`` mid-dispatch and sees the
    same user_id the run was scoped to. Real-model sanity-check
    of the contextvar plumbing."""
    seen: list[str | None] = []

    @tool
    async def whoami() -> str:
        """Return the user_id of the user currently chatting.

        Use this whenever the user asks who they are.
        """
        ctx = get_run_context()
        seen.append(ctx.user_id)
        return ctx.user_id or "anonymous"

    agent = Agent(
        "When the user asks who they are, use the whoami tool and "
        "report exactly what it returns.",
        model=MODEL,
        tools=[whoami],
    )
    await agent.run(
        "Use whoami and tell me the user id verbatim.",
        user_id="alice",
    )
    assert seen == ["alice"]


# ---------------------------------------------------------------------------
# Structured outputs: end-to-end validation against a real model
# ---------------------------------------------------------------------------


class CompanyInfo(BaseModel):
    name: str
    founded_year: int
    headquarters: str


async def test_structured_output_returns_validated_instance() -> None:
    """Real model + output_schema. The schema directive in the
    system prompt should be enough to get clean JSON back; if it
    isn't, the validation-retry path covers the gap. Either way
    ``result.parsed`` arrives as a typed instance."""
    agent = Agent(
        "Extract structured company info from the user's prompt. "
        "Be faithful to what's in the text; do not invent fields.",
        model=MODEL,
    )
    result = await agent.run(
        "Acme Corp was founded in 2008 and is headquartered in Berlin.",
        output_schema=CompanyInfo,
    )
    assert isinstance(result.parsed, CompanyInfo)
    assert result.parsed.founded_year == 2008
    assert result.parsed.headquarters.lower() == "berlin"


async def test_structured_output_retry_on_validation_failure() -> None:
    """A schema with a tightly-typed enum forces the model to
    self-correct. The retry-with-feedback path means we get a
    valid instance back even when the first emission misses."""

    from typing import Literal

    class Mood(BaseModel):
        sentiment: Literal["positive", "neutral", "negative"]
        confidence: float

    agent = Agent(
        "Classify the sentiment of the input. Use ONLY the values "
        "positive, neutral, or negative.",
        model=MODEL,
    )
    result = await agent.run(
        "I am absolutely thrilled about this!",
        output_schema=Mood,
        output_validation_retries=1,
    )
    assert isinstance(result.parsed, Mood)
    assert result.parsed.sentiment == "positive"


# ---------------------------------------------------------------------------
# Retry classification on real auth error
# ---------------------------------------------------------------------------


async def test_invalid_api_key_classifies_as_authentication_error() -> None:
    """A bad API key should surface as ``AuthenticationError``
    (permanent, no retries) — not buried inside a raw SDK
    exception. Verifies the classifier handles a real OpenAI
    auth response."""
    from loomflow.core import AuthenticationError
    from loomflow.model.openai import OpenAIModel

    model = OpenAIModel(MODEL, api_key="sk-deliberately-invalid")
    agent = Agent(
        "...",
        model=model,
        # Default policy would retry transient errors; auth errors
        # are permanent so this run should fail fast on the first
        # attempt regardless of policy.
        retry_policy=RetryPolicy(
            max_attempts=2, initial_delay_s=0.0, jitter=0.0
        ),
    )
    with pytest.raises(AuthenticationError):
        await agent.run("hi")


async def test_auto_extract_populates_facts_after_one_run() -> None:
    """The headline M8 UX: a single ``agent.run`` against a real
    model auto-extracts structured facts into memory, partitioned
    by user_id, with no manual ``Consolidator`` call from the
    user. ``auto_extract=True`` is the default for in-tree network
    adapters."""
    from loomflow.memory.inmemory import InMemoryMemory

    memory = InMemoryMemory()
    agent = Agent(
        "You are helpful. Acknowledge what the user says briefly.",
        model=MODEL,
        memory=memory,
        # Default is True for OpenAIModel; pin for clarity.
        auto_extract=True,
    )
    await agent.run(
        "Hi! I'm Alice and my favourite programming language is Python.",
        user_id="alice",
    )

    # The Consolidator should have run and persisted at least one
    # fact to alice's partition.
    facts = await memory.facts.query(user_id="alice", limit=10)
    assert facts, "expected auto-extracted facts after one run"
    blob = " ".join(f"{f.subject} {f.predicate} {f.object}" for f in facts)
    assert "Python" in blob or "python" in blob


async def test_streaming_yields_token_events() -> None:
    """``agent.stream`` produces model-chunk events end-to-end
    against a real model — verifies the stream wiring + retry
    wrapper's pre-first-chunk gate behave on the wire."""
    agent = Agent("Be brief.", model=MODEL)
    saw_text = False
    saw_completed = False
    async for event in agent.stream("Reply with one word: hi."):
        kind = event.kind.value
        if kind == "model_chunk":
            saw_text = True
        elif kind == "completed":
            saw_completed = True
    assert saw_text
    assert saw_completed
