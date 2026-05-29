"""Example 21: OS-level sandbox — kernel-enforced tool isolation.

Run (as a module, so the demo tools are importable in the sandbox child):
    python -m examples.21_sandbox_os

``OSSandbox`` is the strongest sandbox tier: it runs each tool call in a
child process wrapped by the platform's kernel isolation facility, so a
tool genuinely *cannot* write outside its declared roots or reach the
network even if its code tries — enforced by the OS, not by Python
argument inspection.

Backends are auto-selected:
  * macOS  -> sandbox-exec (Seatbelt)
  * Linux  -> bwrap (Bubblewrap)
  * else   -> graceful degrade to FilesystemSandbox(SubprocessSandbox)

The active backend is reported via ``OSSandbox.mode`` so you always know
whether you got kernel isolation or the degraded fallback.

Key API:
    from loomflow import OSSandbox
    sb = OSSandbox(inproc_host, roots=[allowed_dir], allow_network=False)
    sb.mode            # "seatbelt" | "bubblewrap" | "degraded"
    await sb.call(...)  # runs the tool in the kernel sandbox

No API key needed — this exercises the sandbox directly, not a model.

NOTE on WHY this runs as ``python -m``: the sandbox ships the tool
callable to a child process and unpickles it there, so tools must live
at module scope with a stable import path (a ``__main__`` script's
functions can't be unpickled in the child). The demo tools therefore
live in ``examples/_sandbox_demo_tools.py`` and this file imports them.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import anyio

from loomflow import OSSandbox
from loomflow.tools.registry import InProcessToolHost, Tool

# The demo tools must be importable by a STABLE module path so the
# sandbox child can unpickle them. Import the sibling helper whether the
# example is run as ``python -m examples.21_sandbox_os`` (package import)
# or as ``python examples/21_sandbox_os.py`` (script — add the dir so the
# child resolves ``_sandbox_demo_tools`` the same way).
try:
    from examples import _sandbox_demo_tools as tools
except ImportError:  # run as a plain script
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _sandbox_demo_tools as tools  # type: ignore[no-redef]


def _host() -> InProcessToolHost:
    host = InProcessToolHost()
    host.register(
        Tool(
            name="write_file",
            description="write content to a path",
            fn=tools.write_file,
            input_schema={"type": "object"},
        )
    )
    host.register(
        Tool(
            name="fetch_url",
            description="attempt an outbound network connection",
            fn=tools.fetch_url,
            input_schema={"type": "object"},
        )
    )
    return host


async def main() -> None:
    # The sandbox may write under this root; anything else is denied.
    allowed_root = Path(tempfile.mkdtemp(prefix="loomflow-sandbox-demo-"))
    sandbox = OSSandbox(_host(), roots=[allowed_root], allow_network=False)

    print(f"Sandbox backend in force: {sandbox.mode}")
    if sandbox.mode == "degraded":
        print(
            "  (no kernel backend on this host — install bwrap on Linux, "
            "or run on macOS, for true kernel isolation. The calls below "
            "still run, path-contained + process-isolated.)"
        )
    print()

    # 1. A write INSIDE the allowed root succeeds.
    inside = allowed_root / "notes.txt"
    res = await sandbox.call(
        "write_file",
        {"path": str(inside), "content": "hello from the sandbox"},
        call_id="1",
    )
    print(f"write inside root  -> ok={res.ok}  {res.output or res.error}")

    # 2. A write OUTSIDE the root (here: $HOME) is blocked by the kernel.
    outside = Path.home() / ".loomflow-sandbox-demo-escape.txt"
    if outside.exists():
        outside.unlink()
    res = await sandbox.call(
        "write_file",
        {"path": str(outside), "content": "i should not exist"},
        call_id="2",
    )
    blocked = (not res.ok) and (not outside.exists())
    print(
        f"write outside root -> ok={res.ok}  "
        f"blocked={blocked}  ({res.error or res.output})"
    )
    if outside.exists():  # only reachable in degraded mode w/o fs guard
        outside.unlink()

    # 3. Network is denied by default.
    res = await sandbox.call("fetch_url", {}, call_id="3")
    print(
        f"network (default)  -> ok={res.ok}  "
        f"({res.error or res.output})"
    )

    print()
    print(
        "Kernel-enforced result: writes stay inside the root, the network "
        "is off — even though the tool code tried otherwise."
    )


if __name__ == "__main__":
    anyio.run(main)
