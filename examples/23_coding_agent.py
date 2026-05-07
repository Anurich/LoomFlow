"""23_coding_agent — Test-driven autonomous coding loop.

A real coding agent: given a buggy Python source file plus a
failing pytest suite, the agent reads the tests to understand the
spec, edits the source to make the tests pass, runs pytest, reads
the failures, and iterates until everything is green.

What's production-shaped here
-----------------------------

* **Sandboxed bash via a focused tool.** The agent does NOT get a
  free-form shell — only a ``run_tests(target)`` tool that runs
  ``pytest`` inside the workdir. No ``rm -rf``, no ``pip install``,
  no shell escapes.
* **Filesystem ops scoped to a tempdir** via the
  built-in ``read_tool`` / ``write_tool`` / ``edit_tool`` factories
  (they share a per-process tempdir by default).
* **Audit log on disk.** Every tool call + permission decision lands
  in ``${WORKDIR}/audit.jsonl`` for postmortem.
* **Budget cap.** Hard ceiling on total tokens/cost so the agent
  can't loop indefinitely on a hard bug.
* **Streaming events** — you watch the agent reason, edit, test,
  fail, edit again, in real time.

The pre-seeded scenario
-----------------------

We drop two files into the workdir:

* ``mathlib.py`` — a deliberately buggy ``factorial(n)`` and
  ``fibonacci(n)``. The factorial returns ``n``-not-``n!`` (off by
  the recursive multiplication). Fibonacci has a base-case bug.
* ``test_mathlib.py`` — a small pytest suite that exercises both
  functions on known values.

Run it::

    pip install -e '.[dev,openai]'
    # add OPENAI_API_KEY=sk-... to .env at repo root
    python examples/23_coding_agent.py

You should see the agent read the failing tests, deduce the
expected behaviour, edit ``mathlib.py``, and re-run pytest until
both functions pass.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit(
        "\n  ✗ OPENAI_API_KEY required. "
        "Add OPENAI_API_KEY=sk-... to .env at repo root.\n"
    )

from collections.abc import Callable  # noqa: E402

from jeevesagent import (  # noqa: E402
    Agent,
    FileAuditLog,
    HookRegistry,
    Mode,
    StandardPermissions,
    Tool,
    edit_tool,
    read_tool,
    tool,
    write_tool,
)
from jeevesagent.core.types import PermissionDecision, ToolCall, ToolResult  # noqa: E402
from jeevesagent.governance.budget import BudgetConfig, StandardBudget  # noqa: E402

# ---------------------------------------------------------------------------
# The buggy code we'll ask the agent to fix.
# ---------------------------------------------------------------------------

BUGGY_MATHLIB = '''\
"""mathlib — small numeric helpers (DELIBERATELY BUGGY)."""


def factorial(n: int) -> int:
    """Return n! for n >= 0."""
    if n < 0:
        raise ValueError("factorial is undefined for negatives")
    if n <= 1:
        return 1
    # BUG: returns n + factorial(n-1) instead of n * factorial(n-1)
    return n + factorial(n - 1)


def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number (0-indexed)."""
    if n < 0:
        raise ValueError("fibonacci is undefined for negatives")
    # BUG: should return n when n < 2; returns 1 always.
    if n < 2:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)
'''

TEST_MATHLIB = '''\
"""Tests for mathlib. The agent's job is to make these pass."""

import pytest
from mathlib import factorial, fibonacci


def test_factorial_zero() -> None:
    assert factorial(0) == 1


def test_factorial_one() -> None:
    assert factorial(1) == 1


def test_factorial_five() -> None:
    assert factorial(5) == 120


def test_factorial_negative_raises() -> None:
    with pytest.raises(ValueError):
        factorial(-1)


def test_fibonacci_zero() -> None:
    assert fibonacci(0) == 0


def test_fibonacci_one() -> None:
    assert fibonacci(1) == 1


def test_fibonacci_seven() -> None:
    assert fibonacci(7) == 13


def test_fibonacci_negative_raises() -> None:
    with pytest.raises(ValueError):
        fibonacci(-1)
'''


# ---------------------------------------------------------------------------
# A focused, narrow tool: run pytest in the workdir. The agent does
# NOT get bash; this is the only "execute" capability it has.
# ---------------------------------------------------------------------------


def _materialize_project(workdir: Path) -> None:
    (workdir / "mathlib.py").write_text(BUGGY_MATHLIB)
    (workdir / "test_mathlib.py").write_text(TEST_MATHLIB)


def _make_test_runner(workdir: Path) -> Tool:
    @tool
    async def run_tests(target: str = "test_mathlib.py") -> str:
        """Run pytest against a test file in the working directory.

        Returns the combined stdout+stderr of pytest. Pass the
        bare filename (e.g. ``test_mathlib.py``); the path is
        resolved relative to the workdir. Output truncates after
        4000 characters so it stays manageable for the model.
        """
        # Validate the target stays inside the workdir — no escapes.
        full = (workdir / target).resolve()
        # Sync filesystem stat is fine here — this is a one-shot
        # validation, the subprocess below dominates the call's time.
        if not str(full).startswith(str(workdir.resolve())):  # noqa: ASYNC240
            return f"ERROR: target {target!r} escapes workdir"
        if not full.exists():  # noqa: ASYNC240
            return f"ERROR: {target} does not exist"
        # ``pytest --tb=short -q`` produces compact output the model
        # can quickly diagnose.
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "pytest",
            str(full),
            "--tb=short",
            "-q",
            cwd=str(workdir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        text = stdout.decode("utf-8", errors="replace")
        if len(text) > 4000:
            text = text[:4000] + "\n... [output truncated]"
        return f"$ pytest {target}\n(exit code: {proc.returncode})\n\n{text}"

    return run_tests


# ---------------------------------------------------------------------------
# Hooks — visible permission decisions + audit
# ---------------------------------------------------------------------------


def _build_hooks() -> HookRegistry:
    hooks = HookRegistry()

    @hooks.register_pre_tool
    async def show_decision(call: ToolCall) -> PermissionDecision | None:
        # Print every tool call before it runs so users can watch
        # the agent operate. Returning ``None`` lets the configured
        # StandardPermissions drive the actual decision.
        preview = ", ".join(
            f"{k}={str(v)[:40]!r}" for k, v in call.args.items()
        )
        print(f"    [pre-tool] {call.tool}({preview})")
        return None

    @hooks.register_post_tool
    async def show_result(call: ToolCall, result: ToolResult) -> None:
        if result.denied:
            print(f"    [DENIED]   {call.tool}: {result.reason}")
        elif result.error:
            print(f"    [error]    {call.tool}: {result.error}")
        else:
            preview = (result.output or "")[:80].replace("\n", " ")
            print(f"    [ok]       {call.tool} → {preview}...")

    return hooks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    workdir = Path(  # noqa: ASYNC240 — demo startup
        tempfile.mkdtemp(prefix="jeeves_coding_")
    ).resolve()
    audit_path = workdir / "audit.jsonl"

    print("=" * 70)
    print("Coding agent — fix the buggy mathlib")
    print("=" * 70)
    print(f"Workdir: {workdir}")
    print(f"Audit:   {audit_path}\n")

    _materialize_project(workdir)
    print("Pre-seeded:")
    print(f"  - mathlib.py        ({(workdir / 'mathlib.py').stat().st_size} bytes, buggy)")
    print(f"  - test_mathlib.py   ({(workdir / 'test_mathlib.py').stat().st_size} bytes, 8 tests)\n")

    # Production-grade infra wired up:
    # * StandardPermissions in DEFAULT mode — destructive calls would
    #   need explicit approval (none of our tools are flagged
    #   destructive=True so the agent runs unattended for this demo,
    #   but the policy is in place if you flag your own tools).
    # * FileAuditLog appends every call/result to disk as JSONL.
    # * Budget caps at 50K input tokens / 5K output tokens / $0.50 USD —
    #   plenty for this puzzle, low enough that a runaway loop dies.
    permissions = StandardPermissions(mode=Mode.DEFAULT)
    audit = FileAuditLog(audit_path, secret="coding-agent-demo")
    budget = StandardBudget(
        BudgetConfig(max_tokens=200_000, max_cost_usd=0.50)
    )

    # Filesystem tools are pinned to the workdir so the agent can't
    # touch anything outside (the framework's bash_tool default is a
    # shared tempdir; we override here for clarity).
    # ``list[Tool | Callable[..., object]]`` — the broad type Agent
    # accepts (it tolerates raw callables alongside Tool instances).
    tools: list[Tool | Callable[..., object]] = [
        read_tool(workdir=workdir),
        write_tool(workdir=workdir),
        edit_tool(workdir=workdir),
        _make_test_runner(workdir),
    ]

    agent = Agent(
        instructions=(
            "You are an autonomous Python coding agent. Your task: "
            "make every test in the working directory pass.\n\n"
            "Process:\n"
            "1. Call `read('test_mathlib.py')` to understand what the "
            "code is supposed to do.\n"
            "2. Call `read('mathlib.py')` to see the current "
            "(buggy) implementation.\n"
            "3. Call `run_tests('test_mathlib.py')` to see which "
            "tests are failing and why.\n"
            "4. Use `edit(path, old_string, new_string)` to fix "
            "bugs. ``old_string`` must match the file EXACTLY "
            "(whitespace included) and must be unique. Read the "
            "file again afterwards if you're unsure.\n"
            "5. Re-run tests after each edit.\n"
            "6. Loop until ALL tests pass. Then summarize what was "
            "broken and what you changed.\n\n"
            "Be concise in commentary; let your tool calls do the "
            "talking. Don't ask clarifying questions — just fix it."
        ),
        model="gpt-4.1-mini",
        tools=tools,
        permissions=permissions,
        audit_log=audit,
        budget=budget,
        hooks=_build_hooks(),
    )

    prompt = "Make all tests in test_mathlib.py pass."
    print(f"Goal: {prompt}\n")
    print("─" * 70)

    async for ev in agent.stream(prompt):
        kind = ev.kind.value
        if kind == "model_chunk":
            chunk = ev.payload.get("chunk", {})
            if chunk.get("kind") == "text" and chunk.get("text"):
                print(chunk["text"], end="", flush=True)
        elif kind == "completed":
            result = ev.payload.get("result") or {}
            print("\n" + "─" * 70)
            print("\nFINAL ANSWER:")
            print(result.get("output", "(no output)"))
            print(
                f"\nTurns: {result.get('turns')}  "
                f"Tokens: in={result.get('tokens_in')} "
                f"out={result.get('tokens_out')}  "
                f"Cost: ${float(result.get('cost_usd', 0) or 0):.4f}"
            )

    # Show the final state of mathlib.py so users can see the fix.
    print("\n" + "=" * 70)
    print("FINAL mathlib.py")
    print("=" * 70)
    print((workdir / "mathlib.py").read_text())

    # Sanity check — run pytest one more time outside the agent to
    # confirm everything passes.
    print("\n" + "=" * 70)
    print("Verification (running pytest ourselves)")
    print("=" * 70)
    result = subprocess.run(  # noqa: S603, ASYNC221 — final sync check
        [sys.executable, "-m", "pytest", "test_mathlib.py", "-v"],
        cwd=str(workdir),
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode == 0:
        print("✓ All tests pass — agent succeeded.\n")
    else:
        print("✗ Some tests still fail. Inspect the workdir.\n")
    print(f"(Workdir kept at {workdir} for inspection.)")


if __name__ == "__main__":
    asyncio.run(main())
