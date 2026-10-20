"""Module-level tool functions used by ``test_subprocess_sandbox.py``.

The subprocess sandbox pickles the function before sending it to the
child process. Pickling only works for functions defined at module
scope — closures and ``def``s inside test functions can't cross the
process boundary. So the test tools live here and the test file
imports them.
"""

from __future__ import annotations

import os
import time

from loomflow import tool


def double(x: int) -> int:
    """Return ``x * 2``."""
    return x * 2


def get_pid() -> int:
    """Return the current process's PID. Used to verify subprocess isolation."""
    return os.getpid()


def slow_tool(delay_seconds: float) -> str:
    """Sleep for ``delay_seconds`` then return ``"done"``. Used for
    timeout tests."""
    time.sleep(delay_seconds)
    return "done"


def crashy_tool(message: str) -> str:
    """Always raises a ValueError. Used to verify error propagation."""
    raise ValueError(message)


async def async_double(x: int) -> int:
    """Async variant of :func:`double` so we exercise the
    ``asyncio.run`` branch in the worker."""
    return x * 2


def write_under(path: str, content: str) -> str:
    """Write ``content`` to ``path`` — used by OSSandbox tests to check
    write containment. Returns ``"wrote"`` on success; raises on a
    kernel-denied write so the sandbox surfaces it as a tool error."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return "wrote"


def reach_network() -> str:
    """Attempt an outbound TCP connect — used by OSSandbox tests to check
    network policy. Returns ``"connected"`` if the socket opens; raises
    otherwise (a kernel-denied connection surfaces as a tool error)."""
    import socket

    s = socket.create_connection(("1.1.1.1", 53), timeout=3)
    s.close()
    return "connected"


@tool
def tool_triple(x: int) -> int:
    """Return ``x * 3``. ``@tool`` rebinds this module name to the Tool
    wrapper, so the sandbox has to send it by name, not by pickle."""
    return x * 3


def read_text(path: str) -> str:
    """Read ``path`` — used by OSSandbox tests to check read containment."""
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def get_env(name: str) -> str | None:
    """Return environment variable ``name`` as the tool sees it."""
    return os.environ.get(name)


def make_set() -> set[int]:
    """A result that isn't JSON-serialisable."""
    return {1, 2}
