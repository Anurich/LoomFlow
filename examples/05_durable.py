"""05_durable — SqliteRuntime journal + cross-instance replay.

What it shows:
* ``SqliteRuntime`` records every step (model call + tool call) by
  ``(session_id, step_name)`` in a sqlite file.
* A *fresh* ``SqliteRuntime`` against the same DB, opening the same
  session, returns cached values without re-executing anything.
* This is the foundation for crash-recovery: the journal survives a
  process exit, so the next start picks up where the last one left off.

Run:
    python examples/05_durable.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from jeevesagent import SqliteRuntime


async def main() -> None:
    # An "expensive" function we want to cache:
    call_count = {"runs": 0}

    async def expensive() -> str:
        call_count["runs"] += 1
        return f"v{call_count['runs']}"

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "journal.db"

        # First instance: runs the function and journals the result.
        rt1 = SqliteRuntime(db)
        async with rt1.session("fixed-id"):
            v1 = await rt1.step("computed_value", expensive)
        print(f"first run:  {v1!r}  (calls so far: {call_count['runs']})")

        # Simulate a process restart with a brand-new runtime instance
        # against the same DB.
        rt2 = SqliteRuntime(db)
        async with rt2.session("fixed-id"):
            v2 = await rt2.step("computed_value", expensive)
        print(f"replay:     {v2!r}  (calls so far: {call_count['runs']})")

        # Different session ⇒ executes again.
        async with rt2.session("different-session"):
            v3 = await rt2.step("computed_value", expensive)
        print(f"new sess:   {v3!r}  (calls so far: {call_count['runs']})")

        assert v1 == v2  # same session, same step ⇒ cached
        assert v3 != v1  # different session ⇒ fresh execution


if __name__ == "__main__":
    asyncio.run(main())
