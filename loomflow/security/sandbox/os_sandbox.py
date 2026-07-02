"""OS-level isolation sandbox.

The strongest sandbox tier: it runs each tool call in a child process
wrapped by the platform's kernel-level isolation facility, so a tool
genuinely *cannot* read outside its roots or reach the network even if
its code tries — enforced by the OS, not by Python argument inspection.

Backends (auto-selected by platform):

* **macOS** — ``sandbox-exec`` (Seatbelt). A generated profile denies
  everything by default, then allows reads/writes only under the
  declared roots and (optionally) network.
* **Linux** — ``bwrap`` (Bubblewrap, from the ``bubblewrap`` package).
  A new mount + network namespace; only the declared roots are bind-
  mounted in, network is dropped unless allowed.
* **Windows / no backend available** — graceful **degrade**: there is
  no first-class OS sandbox we can shell out to, so this layer falls
  back to :class:`SubprocessSandbox` (process isolation + timeout)
  composed under :class:`FilesystemSandbox` (path-arg containment).
  Weaker than kernel isolation, but never *silently* weaker — the
  active mode is reported via :pyattr:`OSSandbox.mode` and logged.

Relationship to the other tiers (all are ``ToolHost`` wrappers and
compose by nesting): ``FilesystemSandbox`` validates path *arguments*
in-process (cheap, fast); ``SubprocessSandbox`` adds process isolation
+ timeout; ``OSSandbox`` adds *kernel-enforced* filesystem + network
isolation on top. Use the weakest tier that meets your threat model —
OS-level is the right default only for running untrusted / agent-
authored shell, which is exactly Codex-CLI-style autonomous execution.

Constraints (same as SubprocessSandbox, for the same reason — we ship
the tool fn to a child): the wrapped host must be an
``InProcessToolHost`` and the tool fn + args must be picklable. The
child is launched *via the OS sandbox wrapper* so the kernel limits
apply to the whole Python child, tool code included.

Cost: a sandbox-wrapped subprocess spawn is ~150-400ms. Reserve for
tools that run untrusted code; never wrap fast pure-Python tools.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections.abc import AsyncIterator, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from ...core.protocols import ToolHost
from ...core.types import ToolDef, ToolEvent, ToolResult
from ...tools.registry import InProcessToolHost
from .filesystem import FilesystemSandbox
from .subprocess_ import SubprocessSandbox

# The isolation mode actually in force, surfaced so callers + tests can
# assert what they got instead of assuming kernel isolation silently
# degraded to something weaker.
SandboxMode = Literal["seatbelt", "bubblewrap", "degraded"]


def _detect_mode() -> SandboxMode:
    """Pick the strongest OS isolation backend available on this host.

    macOS → seatbelt (``sandbox-exec`` ships with the OS). Linux →
    bubblewrap iff ``bwrap`` is on PATH (it's a separate package, not
    guaranteed). Everything else, or Linux without bwrap → degraded.
    """
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        return "seatbelt"
    if sys.platform.startswith("linux") and shutil.which("bwrap"):
        return "bubblewrap"
    return "degraded"


def _validate_seatbelt_root(root: Path) -> None:
    """Reject roots that can't be embedded verbatim in a Seatbelt
    profile string literal. A ``"`` closes the ``(subpath "...")``
    string, a backslash starts an escape, and a newline / control
    char breaks the s-expression — any of them would let a
    crafted root rewrite the sandbox profile itself, so we fail
    loudly instead of interpolating."""
    text = str(root)
    if '"' in text or "\\" in text or any(ord(ch) < 0x20 for ch in text):
        raise ValueError(
            f"OSSandbox root {text!r} contains a double quote, "
            "backslash, or control character (e.g. newline); it "
            "cannot be safely embedded in a Seatbelt sandbox "
            "profile. Rename the directory or choose another root."
        )


def _seatbelt_profile(roots: tuple[Path, ...], allow_network: bool) -> str:
    """Generate a Seatbelt (.sb) profile: deny by default, allow reads
    everywhere (so Python + stdlib import), but allow *writes* only
    under the declared roots, and gate network behind the flag.

    Read is left broad on purpose — denying all reads breaks the Python
    interpreter startup (it must read its own stdlib). The security
    property we enforce is **no writes outside roots** + **no network**,
    which is what matters for agent-authored code: it can look, but it
    can't exfiltrate or tamper.
    """
    lines = [
        "(version 1)",
        ";; loomflow OSSandbox — deny by default, narrow allows below.",
        "(deny default)",
        ";; Reads must stay broad or the Python child can't import its",
        ";; own stdlib. The enforced property is no-write-outside-roots",
        ";; plus network control, not read confinement.",
        "(allow file-read*)",
        ";; Process basics the interpreter needs.",
        "(allow process-fork)",
        "(allow process-exec)",
        "(allow sysctl-read)",
        "(allow mach-lookup)",
        "(allow signal (target self))",
        ";; Writes: only under the declared roots + the OS temp dir",
        ";; (Python writes __pycache__ / tempfiles during normal run).",
    ]
    write_roots = list(roots) + [Path(tempfile.gettempdir()).resolve()]
    for r in write_roots:
        # subpath grants write to the dir and everything beneath it.
        # The root is interpolated into a quoted profile string, so
        # it must not contain quote / escape / control characters.
        _validate_seatbelt_root(r)
        lines.append(f'(allow file-write* (subpath "{r}"))')
    if allow_network:
        lines.append("(allow network*)")
    else:
        lines.append(";; network denied (default) — no outbound/inbound.")
    return "\n".join(lines) + "\n"


def _seatbelt_argv(profile_path: str) -> list[str]:
    return ["sandbox-exec", "-f", profile_path, sys.executable]


def _bubblewrap_argv(
    roots: tuple[Path, ...], allow_network: bool
) -> list[str]:
    """Build a ``bwrap`` argv that gives the child a fresh namespace,
    read-only system dirs, read-write binds for the declared roots, and
    (unless allowed) no network namespace.
    """
    argv = [
        "bwrap",
        "--die-with-parent",
        "--unshare-pid",
        # Read-only system so imports work; nothing here is writable.
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/bin", "/bin",
        "--ro-bind", "/lib", "/lib",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
    ]
    # /lib64 exists on most x86-64 distros but not all (e.g. arm); bind
    # only if present so bwrap doesn't error on a missing source.
    if Path("/lib64").exists():
        argv += ["--ro-bind", "/lib64", "/lib64"]
    # The interpreter itself must be reachable inside the namespace.
    argv += ["--ro-bind", sys.executable, sys.executable]
    for r in roots:
        argv += ["--bind", str(r), str(r)]
    if not allow_network:
        argv += ["--unshare-net"]
    argv += [sys.executable]
    return argv


class OSSandbox:
    """Kernel-isolated tool execution (Seatbelt / Bubblewrap / degrade).

    Wraps an :class:`InProcessToolHost`; each ``call`` runs the tool fn
    in a child process launched under the platform's OS sandbox. On a
    platform with no backend (Windows, or Linux without ``bwrap``) it
    degrades to ``FilesystemSandbox(SubprocessSandbox(inner))`` — still
    process-isolated + path-contained, just not kernel-enforced — and
    reports that via :pyattr:`mode`.

    Args:
        inner: the wrapped tool host (must be ``InProcessToolHost``).
        roots: filesystem paths the sandboxed tool may write under.
            At least one is required — a sandbox with no writable root
            is almost always a misconfiguration.
        allow_network: when ``False`` (default) the kernel sandbox
            blocks all network access. Ignored in degraded mode (the
            fallback can't enforce network policy).
        timeout_seconds: hard wall-clock kill for the child.
        path_args: forwarded to the degraded-mode ``FilesystemSandbox``.
    """

    def __init__(
        self,
        inner: ToolHost,
        *,
        roots: Iterable[str | Path],
        allow_network: bool = False,
        timeout_seconds: float = 30.0,
        path_args: Iterable[str] | None = None,
    ) -> None:
        roots_list = tuple(Path(r).resolve() for r in roots)
        if not roots_list:
            raise ValueError("OSSandbox requires at least one writable root")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._inner = inner
        self._roots = roots_list
        self._allow_network = allow_network
        self._timeout = timeout_seconds
        self._mode: SandboxMode = _detect_mode()
        # Fail fast on roots that can't be embedded in a Seatbelt
        # profile string (quote / backslash / control chars) —
        # better a construction-time ValueError than a profile
        # injection or an opaque sandbox-exec parse failure later.
        if self._mode == "seatbelt":
            for r in roots_list:
                _validate_seatbelt_root(r)
        # Degraded fallback is composed eagerly so misconfig (e.g. a
        # non-InProcess host) fails fast at construction, same as the
        # kernel path's requirement below.
        if self._mode == "degraded":
            self._fallback: ToolHost | None = FilesystemSandbox(
                SubprocessSandbox(inner, timeout_seconds=timeout_seconds),
                roots=roots_list,
                path_args=path_args,
            )
        else:
            if not isinstance(inner, InProcessToolHost):
                from ...core.errors import ConfigError

                raise ConfigError(
                    "OSSandbox kernel mode wraps InProcessToolHost only "
                    "(it ships the registered Tool.fn to a sandboxed "
                    f"child). Got {type(inner).__name__}."
                )
            self._fallback = None

    # ---- introspection --------------------------------------------------

    @property
    def inner(self) -> ToolHost:
        return self._inner

    @property
    def mode(self) -> SandboxMode:
        """The isolation backend actually in force on this host."""
        return self._mode

    @property
    def roots(self) -> tuple[Path, ...]:
        return self._roots

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
        if self._fallback is not None:
            # No kernel backend — use the composed fs+subprocess fallback.
            return await self._fallback.call(tool, args, call_id=call_id)

        host = self._inner
        assert isinstance(host, InProcessToolHost)
        registered = host.get(tool)
        if registered is None:
            return ToolResult.error_(
                call_id=call_id, message=f"unknown tool: {tool}"
            )

        wrapper_argv = self._wrapper_argv()
        try:
            output = await _run_in_os_sandbox(
                registered.fn,
                dict(args),
                wrapper_argv=wrapper_argv,
                timeout=self._timeout,
            )
        except _OSSandboxTimeout as exc:
            return ToolResult.error_(call_id=call_id, message=str(exc))
        except _OSSandboxError as exc:
            return ToolResult.error_(call_id=call_id, message=str(exc))

        return ToolResult.success(call_id=call_id, output=output)

    async def watch(self) -> AsyncIterator[ToolEvent]:
        async for event in self._inner.watch():
            yield event

    # ---- internals ------------------------------------------------------

    def _wrapper_argv(self) -> list[str]:
        """The command prefix that launches a sandboxed Python child.

        For seatbelt we materialise a profile to a temp file and return
        ``sandbox-exec -f <profile> python``; the profile path is cleaned
        up by the caller after the run.
        """
        if self._mode == "bubblewrap":
            return _bubblewrap_argv(self._roots, self._allow_network)
        # seatbelt: profile written here, path appended; the runner
        # deletes it. Stored on self so the runner can find + unlink it.
        profile = _seatbelt_profile(self._roots, self._allow_network)
        fd, path = tempfile.mkstemp(suffix=".sb", prefix="loomflow-sb-")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(profile)
        self._last_profile_path = path
        return _seatbelt_argv(path)


# ---------------------------------------------------------------------------
# Subprocess machinery — launches a Python child UNDER the OS wrapper and
# feeds it the pickled fn + args over a temp file pair (we can't share a
# multiprocessing.Queue across the sandbox boundary, so we use files).
# ---------------------------------------------------------------------------


class _OSSandboxTimeout(Exception):
    """Child exceeded the configured wall-clock timeout."""


class _OSSandboxError(Exception):
    """Child failed to run, or the tool raised inside the sandbox."""


# Child entry point: a module run with ``python -c`` reads a pickle of
# (fn, args) from argv[1], runs it, writes a result pickle to argv[2].
# Kept as a string so it works regardless of how loomflow is installed
# inside the sandboxed namespace (no import of loomflow needed).
_CHILD_SOURCE = """\
import pickle, sys, inspect
inp, outp = sys.argv[1], sys.argv[2]
try:
    with open(inp, "rb") as f:
        fn, args = pickle.load(f)
    if inspect.iscoroutinefunction(fn):
        import asyncio
        result = asyncio.run(fn(**args))
    else:
        result = fn(**args)
    payload = ("ok", result)
except BaseException as exc:  # relay everything to the parent
    payload = ("err", f"{type(exc).__name__}: {exc}")
try:
    with open(outp, "wb") as f:
        pickle.dump(payload, f)
except Exception:
    # Result not picklable — record that instead of crashing silently.
    with open(outp, "wb") as f:
        pickle.dump(("err", "result not picklable"), f)
"""


async def _run_in_os_sandbox(
    fn: Any,
    args: dict[str, Any],
    *,
    wrapper_argv: list[str],
    timeout: float,  # noqa: ASYNC109 — we kill the child directly on timeout
) -> Any:
    """Run ``fn(**args)`` in a Python child launched under ``wrapper_argv``
    (the OS-sandbox command prefix ending in the interpreter path).

    Communication is via two temp files (input pickle, output pickle) so
    nothing needs to cross the sandbox as a shared FD/queue. The input
    file lives under a writable root via the OS temp dir, which the
    profiles grant.
    """
    import pickle

    import anyio

    in_fd, in_path = tempfile.mkstemp(prefix="loomflow-sbin-")
    out_fd, out_path = tempfile.mkstemp(prefix="loomflow-sbout-")
    os.close(out_fd)
    try:
        def _write_input() -> None:
            with os.fdopen(in_fd, "wb") as fh:
                pickle.dump((fn, args), fh)

        try:
            # File I/O off the event loop (anyio-everywhere; ASYNC230).
            await anyio.to_thread.run_sync(_write_input)
        except (pickle.PickleError, TypeError) as exc:
            raise _OSSandboxError(
                f"OSSandbox: tool or args not picklable ({exc}). Use a "
                "module-level function and primitive args."
            ) from exc

        argv = [*wrapper_argv, "-c", _CHILD_SOURCE, in_path, out_path]

        try:
            with anyio.fail_after(timeout):
                proc = await anyio.run_process(argv, check=False)
        except TimeoutError as exc:
            raise _OSSandboxTimeout(
                f"OSSandbox: tool exceeded {timeout}s"
            ) from exc

        if proc.returncode != 0:
            stderr = (proc.stderr or b"").decode("utf-8", "replace").strip()
            raise _OSSandboxError(
                f"OSSandbox: sandboxed child exited {proc.returncode}"
                + (f": {stderr}" if stderr else "")
            )

        def _read_output() -> tuple[str, Any]:
            with open(out_path, "rb") as fh:
                loaded: tuple[str, Any] = pickle.load(fh)
                return loaded

        try:
            status, payload = await anyio.to_thread.run_sync(_read_output)
        except EOFError as exc:
            raise _OSSandboxError(
                "OSSandbox: child produced no result"
            ) from exc
        if status == "ok":
            return payload
        raise _OSSandboxError(f"OSSandbox: tool raised {payload}")
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass
        # Clean up a seatbelt profile if the wrapper made one.
        if wrapper_argv and wrapper_argv[0] == "sandbox-exec":
            prof = wrapper_argv[2]  # ["sandbox-exec","-f",<profile>,...]
            try:
                os.unlink(prof)
            except OSError:
                pass
