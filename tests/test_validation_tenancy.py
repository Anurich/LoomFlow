"""Nothing one user's run produced reaches another user's run.

Journals and checkpoints were keyed by session id alone, so a second
user reusing a session id replayed the first user's results or resumed
from their transcript, and a persistent worker pinned to one user let
an anonymous run replay that user's journaled tool results.
"""

from __future__ import annotations

from typing import Any

import pytest

from loomflow import Agent, ScriptedModel, ScriptedTurn, tool
from loomflow.agent.api import Tuning
from loomflow.core.context import get_run_context
from loomflow.core.types import ToolCall
from loomflow.runtime import (
    InMemoryJournalStore,
    JournaledRuntime,
    SqliteRuntime,
    journal_scope,
)
from loomflow.team import Team

pytestmark = pytest.mark.anyio


class _Spy(ScriptedModel):
    def __init__(self, turns: list[ScriptedTurn]) -> None:
        super().__init__(turns)
        self.prompts: list[str] = []

    def _record(self, messages: Any) -> None:
        self.prompts.append(" | ".join(str(m.content) for m in messages))

    async def complete(self, messages: Any, **kwargs: Any) -> Any:
        self._record(messages)
        return await super().complete(messages, **kwargs)

    async def stream(self, messages: Any, **kwargs: Any) -> Any:
        self._record(messages)
        async for chunk in super().stream(messages, **kwargs):
            yield chunk


def test_journal_scope_is_injective() -> None:
    pairs = [
        ("s", None),
        ("s", "a"),
        ("s", "b"),
        ("user:1:a:s", None),
        ("anon:user:1:a:s", None),
        ("anon:s", None),
        ("a:s", "x"),
        ("s", "x:a"),
        ("", ""),
        ("", None),
    ]
    keys = [journal_scope(s, u) for s, u in pairs]
    assert len(set(keys)) == len(keys)
    # Runs without a user keep the layout earlier versions wrote.
    assert journal_scope("conv_42", None) == "conv_42"


async def test_journal_never_replays_another_users_results() -> None:
    rt = JournaledRuntime(InMemoryJournalStore())
    first = Agent("x", model=ScriptedModel([ScriptedTurn(text="alice's answer")]), runtime=rt)
    await first.run("hi", session_id="s9", user_id="alice")

    bob = Agent("x", model=ScriptedModel([ScriptedTurn(text="fresh for bob")]), runtime=rt)
    assert (await bob.run("hi", session_id="s9", user_id="bob")).output == "fresh for bob"

    anon = Agent("x", model=ScriptedModel([ScriptedTurn(text="fresh for anon")]), runtime=rt)
    assert (await anon.run("hi", session_id="s9")).output == "fresh for anon"

    # The same user still replays their own journal (crash-resume).
    again = Agent("x", model=ScriptedModel([ScriptedTurn(text="re-run")]), runtime=rt)
    assert (await again.run("hi", session_id="s9", user_id="alice")).output == "alice's answer"


async def test_checkpoints_resume_only_for_their_own_user(tmp_path: Any) -> None:
    rt = SqliteRuntime(tmp_path / "t.db")
    writer = Agent(
        "x",
        model=ScriptedModel([ScriptedTurn(text="noted")]),
        runtime=rt,
        tuning=Tuning(checkpoint=True),
    )
    await writer.run("my PIN is 4321", session_id="conv_42", user_id="alice")
    assert len(await writer.list_checkpoints("conv_42", user_id="alice")) == 1
    assert await writer.list_checkpoints("conv_42", user_id="mallory") == []
    assert await writer.list_checkpoints("conv_42") == []

    mallory_model = _Spy([ScriptedTurn(text="ok")])
    mallory = Agent("x", model=mallory_model, runtime=rt, tuning=Tuning(checkpoint=True))
    await mallory.resume("what did I say?", session_id="conv_42", user_id="mallory")
    assert not any("4321" in p for p in mallory_model.prompts)

    alice_model = _Spy([ScriptedTurn(text="ok")])
    alice = Agent("x", model=alice_model, runtime=rt, tuning=Tuning(checkpoint=True))
    await alice.resume("what did I say?", session_id="conv_42", user_id="alice")
    assert any("4321" in p for p in alice_model.prompts)


def _delegate_turns() -> list[ScriptedTurn]:
    delegate = ScriptedTurn(
        tool_calls=[
            ToolCall(
                id="c1",
                tool="delegate",
                args={"worker": "acct", "instructions": "balance?"},
            )
        ]
    )
    return [delegate, ScriptedTurn(text="ok")] * 3


async def test_persistent_worker_never_replays_another_users_tool_results() -> None:
    rt = JournaledRuntime(InMemoryJournalStore())
    executions: list[str | None] = []

    @tool
    async def get_balance() -> str:
        """Return the caller's balance."""
        uid = get_run_context().user_id
        executions.append(uid)
        return f"balance for {uid}"

    worker = Agent(
        "acct",
        model=ScriptedModel(
            [
                ScriptedTurn(tool_calls=[ToolCall(id="g1", tool="get_balance", args={})]),
                ScriptedTurn(text="WORKER DONE"),
            ]
            * 3
        ),
        tools=[get_balance],
        runtime=rt,
    )
    team = Team.supervisor({"acct": worker}, model=ScriptedModel(_delegate_turns()))
    outputs: dict[str | None, list[str]] = {}
    for user in ("alice", None, "bob"):
        async for ev in team.stream("balance?", user_id=user):
            if ev.kind == "tool_result":
                outputs.setdefault(user, []).append(str(ev.payload["result"]["output"]))
    # Every user's run executed the tool itself, as that user.
    assert executions == ["alice", None, "bob"]
    assert not any("alice" in out for out in outputs[None] + outputs["bob"])
