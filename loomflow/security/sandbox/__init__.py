"""Sandbox layer.

Sandboxes wrap a :class:`~loomflow.core.protocols.ToolHost` and
mediate every call. The wrapper *is* itself a ``ToolHost`` (it
re-exports ``list_tools`` / ``call`` / ``watch``) so it slots straight
into ``Agent(tools=sandbox)`` with zero changes to the agent core.

What's here today:

* :class:`NoSandbox` — pass-through. Useful as a layer placeholder and
  to demonstrate the wrapping pattern.
* :class:`FilesystemSandbox` — validates path-typed arguments don't
  escape one or more declared roots; symlinks are resolved before the
  containment check. Auto-detects path arguments by name (``path``,
  ``file``, ``directory``, ...) or by the value containing ``/``;
  callers can also pass an explicit ``path_args=`` allowlist.

* :class:`OSSandbox` — kernel-enforced isolation: Seatbelt
  (``sandbox-exec``) on macOS, Bubblewrap (``bwrap``) on Linux, and a
  graceful degrade to ``FilesystemSandbox(SubprocessSandbox(...))`` on
  platforms with no backend (Windows, or Linux without ``bwrap``). The
  active backend is reported via ``OSSandbox.mode``.
"""

from .base import NoSandbox
from .filesystem import FilesystemSandbox
from .os_sandbox import OSSandbox, SandboxMode
from .subprocess_ import SubprocessSandbox

__all__ = [
    "FilesystemSandbox",
    "NoSandbox",
    "OSSandbox",
    "SandboxMode",
    "SubprocessSandbox",
]
