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
