"""End-to-end test for ``Router(conversation_scope='shared')``.

The unit tests in ``test_router.py`` cover that shared mode passes
the parent session_id through. This file proves the WHOLE chain
works — that the routed agent actually rehydrates prior turns from
``Memory.session_messages`` keyed by that shared session_id, so a
follow-up question across routes sees the earlier conversation.

The bug this guards against (loom-code, observed 2026-05): turn 1
"what is this code about?" → router picks COMPLEX → supervisor
answers. Turn 2 "can you check what is this code about?" → router
picks SIMPLE → simple coder runs with empty history, asks the user
to clarify what code. With ``conversation_scope='shared'`` both
turns share a session_id; ReAct's prompt builder rehydrates the
prior turn via ``session_messages`` and turn 2 has continuity.

We deliberately do NOT include an inverse "per_route isolates"
test. The unit test in ``test_router.py`` covers session_id
derivation directly. Asserting that per_route routes can't see
each other's *content* via the model's input would also pick up
``Memory.recall()`` (semantic cross-session search) — which is
intentional and not the route-isolation primitive.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loomflow import Agent, ScriptedModel, ScriptedTurn
from loomflow.architecture import Router, RouterRoute
from loomflow.memory.sqlite import SqliteMemory

pytestmark = pytest.mark.anyio


class _RecordingScriptedModel:
    """Wraps ScriptedModel — captures the message list each
    ``stream()`` call receives so the test can assert what the
    model saw."""

    def __init__(self, replies: list[str]) -> None:
        self._inner = ScriptedModel(
            [ScriptedTurn(text=r) for r in replies]
        )
        self.name = "recording"
        self.seen: list[list[str]] = []

    async def stream(self, messages, *, tools=None, **kwargs):  # type: ignore[no-untyped-def]
        self.seen.append([str(m.content) for m in messages])
        async for chunk in self._inner.stream(
            messages, tools=tools, **kwargs
        ):
            yield chunk


async def test_shared_router_rehydrates_across_routes(
    tmp_path: Path,
) -> None:
    """Turn 1 → route A, turn 2 → route B, same parent session_id.
    With shared scope, route B's agent sees turn 1's USER +
    ASSISTANT messages via session_messages."""
    db = str(tmp_path / "m.db")
    mem_classifier = SqliteMemory(db)
    mem_route_a = SqliteMemory(db)
    mem_route_b = SqliteMemory(db)

    route_a_model = _RecordingScriptedModel(
        ["The code is about a TUI editor with embedded AI."]
    )
    route_b_model = _RecordingScriptedModel(
        ["Yes — same as I described before, a TUI editor with AI."]
    )
    classifier_model = ScriptedModel(
        [
            # Turn 1 → route_a; turn 2 → route_b.
            ScriptedTurn(text="route: a\nconfidence: 0.99"),
            ScriptedTurn(text="route: b\nconfidence: 0.99"),
        ]
    )

    route_a_agent = Agent(
        instructions="route a specialist",
        model=route_a_model,  # type: ignore[arg-type]
        memory=mem_route_a,
    )
    route_b_agent = Agent(
        instructions="route b specialist",
        model=route_b_model,  # type: ignore[arg-type]
        memory=mem_route_b,
    )

    coordinator = Agent(
        instructions="router test",
        model=classifier_model,
        memory=mem_classifier,
        architecture=Router(
            routes=[
                RouterRoute(name="a", agent=route_a_agent),
                RouterRoute(name="b", agent=route_b_agent),
            ],
            conversation_scope="shared",
        ),
    )

    PARENT = "repl-session-1"
    USER = "alice"

    # Turn 1.
    r1 = await coordinator.run(
        "what is this code about?",
        user_id=USER,
        session_id=PARENT,
    )
    assert r1.session_id == PARENT

    # Turn 2.
    r2 = await coordinator.run(
        "can you check what is this code about?",
        user_id=USER,
        session_id=PARENT,
    )
    assert r2.session_id == PARENT

    # The smoking gun: route B's specialist must have been called
    # exactly once, and the message list it saw on that call must
    # contain a stringified form of turn 1's USER prompt OR turn
    # 1's ASSISTANT reply. Both come from session_messages
    # rehydration of the shared parent session.
    assert len(route_b_model.seen) == 1, (
        f"route B's specialist was called "
        f"{len(route_b_model.seen)} times, expected once"
    )
    seen_blob = "\n".join(route_b_model.seen[0]).lower()
    assert (
        "what is this code about" in seen_blob
        or "tui editor" in seen_blob
    ), (
        "Route B did NOT see turn 1's USER prompt or ASSISTANT reply "
        "in its rehydrated message list — shared rehydration is "
        "broken end-to-end.\nMessages seen by route B:\n"
        + "\n---\n".join(route_b_model.seen[0])
    )
