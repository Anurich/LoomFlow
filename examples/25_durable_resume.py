"""25_durable_resume.py — durable checkpoint / resume / fork.

Long agent runs die: the process gets OOM-killed, the deploy rolls,
the laptop lid closes. With ``Tuning(checkpoint=True)`` on a
checkpoint-capable runtime (:class:`SqliteRuntime` here — also
``JournaledRuntime`` / ``PostgresRuntime``), the agent loop snapshots
the full transcript + turn count + cumulative usage after EVERY
architecture pass. A later process picks the work up exactly where it
stopped::

    agent = Agent(..., runtime=SqliteRuntime("runtime.db"),
                  tuning=Tuning(checkpoint=True))
    await agent.run("do the task", session_id="job-1")   # ...crash...

    # new process, same DB file:
    metas  = await agent.list_checkpoints("job-1")       # newest first
    result = await agent.resume("pick it up", session_id="job-1")

The contract worth paying for:

* prior turns are restored, **never re-executed or re-billed** —
  usage rolls up restored + new;
* memory re-seeding is skipped on resume (the transcript IS the state);
* ``resume(prompt=None)`` injects an internal continuation nudge;
* ``from_checkpoint=<older id>`` **forks**: a fresh session continues
  from the old snapshot and the original session stays untouched
  (time travel for "try a different approach from turn 1").

Runs OFFLINE with :class:`ScriptedModel` (no API key) — a scripted
"crash" model raises mid-run to simulate the process dying.

Run with::

    python examples/25_durable_resume.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import anyio

from loomflow import (
    Agent,
    ScriptedModel,
    ScriptedTurn,
    StopHookResult,
    Tuning,
    Usage,
)
from loomflow.runtime import SqliteRuntime


class CrashAfterScript(ScriptedModel):
    """Replays its script, then RAISES — simulating the process dying
    on the (N+1)th model call."""

    name = "crash_after"

    async def complete(self, messages: Any, **kwargs: Any) -> Any:
        if self.remaining == 0:
            raise RuntimeError("simulated crash (process killed)")
        return await super().complete(messages, **kwargs)


class ContinueOnce:
    """Stop hook that forces one extra architecture pass — so the run
    has somewhere to crash AFTER the first checkpoint landed."""

    name = "continue_once"

    def __init__(self) -> None:
        self.fired = False

    async def __call__(
        self, session: Any, deps: Any, *, iteration: int
    ) -> StopHookResult | None:
        if self.fired:
            return None
        self.fired = True
        return StopHookResult(inject_message="keep going", reason="demo")


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "runtime.db"
        sid = "report-q3"

        # ---- 1. A run that crashes after turn 1 ----------------------
        print("=" * 64)
        print("Part 1 — run crashes after the first checkpoint")
        print("=" * 64)

        agent = Agent(
            "You write quarterly reports.",
            model=CrashAfterScript(
                [
                    ScriptedTurn(
                        text="Drafted the outline and section 1.",
                        usage=Usage(input_tokens=100, output_tokens=40, cost_usd=0.02),
                    ),
                ]
            ),
            runtime=SqliteRuntime(db),
            tuning=Tuning(checkpoint=True, stop_hooks=[ContinueOnce()]),
        )
        try:
            await agent.run("Write the Q3 report.", session_id=sid)
        except RuntimeError as exc:
            print(f"  run died: {exc}")

        metas = await agent.list_checkpoints(sid)
        print(f"  checkpoints survived on disk: {len(metas)}")
        for m in metas:
            print(f"    turn {m.turn}  id={m.checkpoint_id}")

        # ---- 2. "New process": reopen the DB, resume the session -----
        print()
        print("=" * 64)
        print("Part 2 — new process resumes from the latest checkpoint")
        print("=" * 64)

        agent2 = Agent(
            "You write quarterly reports.",
            model=ScriptedModel(
                [
                    ScriptedTurn(
                        text="Finished sections 2-4. Report complete.",
                        usage=Usage(input_tokens=50, output_tokens=20, cost_usd=0.01),
                    ),
                ]
            ),
            runtime=SqliteRuntime(db),  # same file — a fresh "process"
            tuning=Tuning(checkpoint=True),
        )
        result = await agent2.resume("Pick it up where you left off.", session_id=sid)
        print(f"  output:     {result.output!r}")
        print(f"  turns:      {result.turns}  (1 restored + 1 new — turn 1 not re-run)")
        print(f"  tokens in:  {result.tokens_in}  (100 restored + 50 new)")
        print(f"  cost:       ${result.cost_usd:.2f}  (rolled up, not re-billed)")

        # ---- 3. Fork from the OLDER checkpoint (time travel) ----------
        print()
        print("=" * 64)
        print("Part 3 — fork from an older checkpoint")
        print("=" * 64)

        metas = await agent2.list_checkpoints(sid)  # newest first
        older = metas[-1]  # the turn-1 snapshot from the crashed run
        print(f"  session {sid!r} now has {len(metas)} checkpoints; "
              f"forking from turn {older.turn}")

        agent3 = Agent(
            "You write quarterly reports.",
            model=ScriptedModel(
                [ScriptedTurn(text="Alternate take: wrote an exec summary instead.")]
            ),
            runtime=SqliteRuntime(db),
            tuning=Tuning(checkpoint=True),
        )
        fork = await agent3.resume(
            "Try a different angle from where the draft stood.",
            session_id=sid,
            from_checkpoint=older.checkpoint_id,
        )
        print(f"  fork output:      {fork.output!r}")
        print(f"  fork session_id:  {fork.session_id!r}  (fresh — not {sid!r})")
        original = await agent3.list_checkpoints(sid)
        print(f"  original session: {len(original)} checkpoints, untouched")
        forked = await agent3.list_checkpoints(fork.session_id)
        print(f"  forked session:   {len(forked)} checkpoint(s) of its own")
        print("  → Resume by LATEST id continues in place; an OLDER id forks.")


if __name__ == "__main__":
    anyio.run(main)
