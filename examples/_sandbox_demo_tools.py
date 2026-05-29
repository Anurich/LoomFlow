"""Module-level tool functions for ``21_sandbox_os.py``.

These live in their own module on purpose. ``OSSandbox`` (like
``SubprocessSandbox``) ships the tool callable to a CHILD process and
unpickles it there — and pickle can only resolve functions that live at
module scope with a stable import path. A function defined in a script
run as ``__main__`` (or a closure / local def) cannot be unpickled in
the child, so the sandbox would reject it. Keeping the demo tools here
makes them importable as ``examples._sandbox_demo_tools.write_file``.
"""

from __future__ import annotations

from pathlib import Path


def write_file(path: str, content: str) -> str:
    """Write ``content`` to ``path``. Succeeds inside the sandbox's
    allowed root; kernel-denied (raises) outside it."""
    Path(path).write_text(content, encoding="utf-8")
    return f"wrote {len(content)} bytes to {path}"


def fetch_url() -> str:
    """Attempt an outbound TCP connection. Blocked by default
    (``allow_network=False``) — raises inside the sandbox."""
    import socket

    sock = socket.create_connection(("1.1.1.1", 53), timeout=3)
    sock.close()
    return "connected"
