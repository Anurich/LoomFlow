"""Subprocess sandbox: runs each tool call in a child Python process.

What you get:

* **Process isolation** — a tool that crashes (segfault, OOM, etc.)
  takes down its own subprocess, not the agent.
* **Hard timeout** — the parent process kills the child if it
  exceeds ``timeout_seconds``; the call returns
  ``ToolResult.error_(...)`` with a clear timeout message.
* **Memory boundary** — the child's heap is independent; large
  intermediate values get GC'd by process exit even if the tool
  leaks them.

What you don't get (yet):

* Filesystem isolation, network restrictions, or syscall sandboxing.
  For real OS-level isolation, layer this with a Bubblewrap /
  Seatbelt / Docker / gVisor wrapper as Phase 6 follow-up.

Constraints:

* The wrapped tool host must be an :class:`InProcessToolHost` because
  we need access to the registered ``Tool.fn`` callable to ship it
  to the child process. MCP / external hosts can't be sandboxed
  this way (they're already a process boundary themselves —
  re-process-isolating them adds nothing).
* The tool function and its arguments must be **picklable**. That
  means module-level functions (top-level ``def`` in a module);
  closures and locally-defined functions can't cross the process
  boundary. The ``@tool``-decorated functions in your application
  modules are usually fine.

Cost:

* Spawning a Python subprocess takes ~100-300ms on most platforms
  (macOS uses ``spawn`` start method which is slower than fork).
  Don't use this for fast tools — the spawn dwarfs the work. It
  pays off for tools that take seconds, can crash, or use a lot of
  memory.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
from collections.abc import AsyncIterator, Mapping
from typing import Any

import anyio

from ...core.errors import ConfigError
from ...core.protocols import ToolHost
from ...core.types import ToolDef, ToolEvent, ToolResult
from ...tools.registry import InProcessToolHost, Tool


class SubprocessSandbox:
    """Run each tool call in a fresh child Python process."""

    def __init__(
        self,
        inner: ToolHost,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not isinstance(inner, InProcessToolHost):
            raise ConfigError(
                "SubprocessSandbox wraps InProcessToolHost only "
                "(it needs access to the registered Tool.fn callable). "
                f"Got {type(inner).__name__}."
            )
        self._inner = inner
        self._timeout = timeout_seconds

    @property
    def inner(self) -> ToolHost:
        return self._inner

    @property
    def timeout_seconds(self) -> float:
        return self._timeout

    # ---- ToolHost protocol ----------------------------------------------

    async def list_tools(self, *, query: str | None = None) -> list[ToolDef]:
        return await self._inner.list_tools(query=query)

    async def call(
        self,
        tool: str,
        args: Mapping[str, Any],
        *,
        call_id: str = "",
    ) -> ToolResult:
        # We narrowed the host type at __init__; cast for mypy.
        host = self._inner
        assert isinstance(host, InProcessToolHost)
        registered: Tool | None = host.get(tool)
        if registered is None:
            return ToolResult.error_(call_id=call_id, message=f"unknown tool: {tool}")

        try:
            output = await _run_in_subprocess(
                registered.fn,
                dict(args),
                timeout=self._timeout,
            )
        except _SubprocessTimeoutError as exc:
            return ToolResult.error_(
                call_id=call_id, message=str(exc)
            )
        except _SubprocessExecutionError as exc:
            return ToolResult.error_(
                call_id=call_id, message=str(exc)
            )
        except (pickle.PickleError, TypeError) as exc:
            return ToolResult.error_(
                call_id=call_id,
                message=(
                    f"SubprocessSandbox: tool {tool!r} or its arguments "
                    f"are not picklable ({exc}). Use a module-level "
                    "function and primitive args."
                ),
            )

        return ToolResult.success(call_id=call_id, output=output)

    async def watch(self) -> AsyncIterator[ToolEvent]:
        async for event in self._inner.watch():
            yield event


# ---------------------------------------------------------------------------
# Subprocess machinery
# ---------------------------------------------------------------------------


class _SubprocessTimeoutError(Exception):
    """Raised when a subprocess tool call exceeds the configured timeout."""


class _SubprocessExecutionError(Exception):
    """Raised when a subprocess tool call ran but raised an exception."""


def _worker(
    fn_pickled: bytes,
    args: dict[str, Any],
    queue: multiprocessing.Queue[tuple[str, Any]],
) -> None:
    """Child-process entry point. Module-level so it pickles."""
    try:
        fn = pickle.loads(fn_pickled)
        import asyncio
        import inspect

        if inspect.iscoroutinefunction(fn):
            result = asyncio.run(fn(**args))
        else:
            result = fn(**args)
        queue.put(("ok", result))
    except BaseException as exc:  # noqa: BLE001 — relay all errors to parent
        queue.put(("err", f"{type(exc).__name__}: {exc}"))


async def _run_in_subprocess(
    fn: Any,
    args: dict[str, Any],
    *,
    timeout: float,  # noqa: ASYNC109 — anyio.fail_after doesn't help here; we kill the subprocess directly
) -> Any:
    """Spawn a subprocess, run ``fn(**args)`` there, return the result.

    Raises:
        :class:`_SubprocessTimeoutError`: child didn't finish in time.
        :class:`_SubprocessExecutionError`: child raised; message wraps
            the original exception's repr.
        ``pickle.PickleError``/``TypeError``: ``fn`` or ``args`` weren't
            picklable (caught at the parent before spawn).
    """
    fn_pickled = pickle.dumps(fn)
    ctx = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue[tuple[str, Any]] = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(fn_pickled, args, queue))
    proc.start()

    def _wait_blocking() -> tuple[str, Any] | None:
        proc.join(timeout=timeout)
        if proc.is_alive():
            return None  # timed out
        try:
            return queue.get_nowait()
        except Exception:  # noqa: BLE001 — empty queue / closed
            return None

    outcome = await anyio.to_thread.run_sync(_wait_blocking)

    if outcome is None and proc.is_alive():
        # Timeout: kill the child cooperatively, then forcefully.
        proc.terminate()
        await anyio.to_thread.run_sync(lambda: proc.join(1.0))
        if proc.is_alive():
            os.kill(proc.pid, 9)  # type: ignore[arg-type]
            await anyio.to_thread.run_sync(lambda: proc.join(1.0))
        raise _SubprocessTimeoutError(
            f"SubprocessSandbox: tool exceeded {timeout}s"
        )

    if outcome is None:
        raise _SubprocessExecutionError(
            "SubprocessSandbox: subprocess exited without producing a result"
        )

    status, payload = outcome
    if status == "ok":
        return payload
    raise _SubprocessExecutionError(
        f"SubprocessSandbox: tool raised {payload}"
    )
