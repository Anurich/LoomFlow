"""OS-level isolation sandbox.

The strongest sandbox tier: it runs each tool call in a child process
wrapped by the platform's kernel-level isolation facility, so a tool
genuinely *cannot* read outside its roots or reach the network even if
its code tries — enforced by the OS, not by Python argument inspection.

Backends (auto-selected by platform):

* **macOS** — ``sandbox-exec`` (Seatbelt). A generated profile denies
  everything by default, then allows writes only under the declared
  roots, reads only under the roots, the Python installation, the
  tool's own top-level package (and loomflow) and the system
  libraries, and (optionally) network. The rest of your project and
  your home directory can be ``stat``-ed but not read or listed. The
  child starts in the first root.
* **Linux** — ``bwrap`` (Bubblewrap, from the ``bubblewrap`` package).
  A new mount + PID (+ network) namespace with a private ``/tmp``:
  the system directories, the Python installation and import paths
  are mounted read-only, the declared roots read-write, and nothing
  else exists. Network is dropped unless allowed.

Either way the child starts with a minimal environment (no API keys
or other secrets from the parent's), plus whatever you pass as
``env=``, and its result comes back as JSON, never as a pickle, so a
compromised tool can't run code in the parent.
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
``InProcessToolHost``, the tool must be importable by module path
(``@tool`` functions are fine) and its args picklable, and its return
value should be JSON-serialisable (anything else comes back as its
``str()``). The child is launched *via the OS sandbox wrapper* so the
kernel limits apply to the whole Python child, tool code included.

Cost: a sandbox-wrapped subprocess spawn is ~150-400ms. Reserve for
tools that run untrusted code; never wrap fast pure-Python tools.
"""

from __future__ import annotations

import inspect
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
from .base import register_inside
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


#: System locations a Python child needs to read on macOS: dyld's
#: shared cache, system libraries, devices, time zone data.
_SEATBELT_SYSTEM_READS = (
    "/usr/lib",
    "/usr/share",
    "/System",
    "/Library/Apple",
    "/private/var/db/dyld",
    "/private/var/db/timezone",
    "/private/etc/localtime",
    "/dev",
)
#: Extra reads needed to use the network (DNS, TLS roots).
_SEATBELT_NETWORK_READS = (
    "/private/etc/hosts",
    "/private/etc/resolv.conf",
    "/private/etc/ssl",
    "/private/var/run/resolv.conf",
)


def _python_paths() -> list[Path]:
    """The Python installation the child runs: interpreter, standard
    library and site-packages (the virtualenv's and the base install's),
    outermost first."""
    import sysconfig

    candidates = [
        sys.prefix,
        sys.base_prefix,
        sys.exec_prefix,
        sys.base_exec_prefix,
        os.path.dirname(os.path.realpath(sys.executable)),
        *(sysconfig.get_paths().get(k) for k in ("stdlib", "platstdlib", "purelib", "platlib")),
    ]
    return _outermost(_existing(candidates))


def _import_paths(fn: Any) -> list[Path]:
    """What the child must read to import ``fn`` (and loomflow, which
    ``@tool`` modules import) when they live outside the installation,
    e.g. in your project or an editable install: just their top-level
    packages, plus any ``sandbox_paths`` the callable declares (a
    skill script's directory). Nothing else in the project is
    visible."""
    import importlib.util

    from .subprocess_ import _file_module

    owner = fn if inspect.isfunction(fn) or inspect.ismethod(fn) else type(fn)
    modules = {getattr(owner, "__module__", "") or "", "loomflow"}
    found: list[str | None] = [str(p) for p in getattr(fn, "sandbox_paths", ())]
    from_file = _file_module(fn)
    if from_file is not None:
        # Shipped as its file (a skill's tools.py): the child reads
        # that file's directory.
        found.append(os.path.dirname(from_file[1]))
    for name in modules:
        top = name.split(".")[0]
        if not top or top == "__main__":
            continue
        try:
            spec = importlib.util.find_spec(top)
        except (ImportError, ValueError):
            continue
        if spec is None:
            continue
        found.extend(spec.submodule_search_locations or [spec.origin])
    install = _python_paths()
    return [
        p
        for p in _outermost(_existing(found))
        if not any(p == i or p.is_relative_to(i) for i in install)
    ]


def _existing(candidates: Iterable[str | None]) -> list[Path]:
    found = []
    for c in candidates:
        if not c:
            continue
        path = Path(c).resolve()
        if path.exists():
            found.append(path)
    return found


def _outermost(paths: Iterable[Path]) -> list[Path]:
    """``paths`` without duplicates or paths inside another one."""
    unique = sorted(set(paths), key=lambda p: len(p.parts))
    kept: list[Path] = []
    for path in unique:
        if not any(path == k or path.is_relative_to(k) for k in kept):
            kept.append(path)
    return kept


def _seatbelt_profile(
    roots: tuple[Path, ...],
    allow_network: bool,
    *,
    read_paths: Iterable[Path] = (),
    io_dir: Path | None = None,
) -> str:
    """Generate a Seatbelt (.sb) profile: deny by default; allow reads
    of the system libraries, ``read_paths`` (the Python installation
    and the tool's packages), the roots and ``io_dir``; writes only
    under the roots and ``io_dir``; network only when allowed. ``stat``
    works anywhere (path resolution needs it) and the directories that
    hold ``read_paths`` can be listed (so imports find them), but file
    contents and other listings outside those paths are denied."""
    read_paths = list(read_paths)
    writable = [*roots, *([io_dir] if io_dir is not None else [])]
    readable = [
        *(Path(p) for p in _SEATBELT_SYSTEM_READS),
        *(Path(p) for p in (_SEATBELT_NETWORK_READS if allow_network else ())),
        *read_paths,
        *writable,
    ]
    listable = sorted({p.parent for p in read_paths} - set(readable))
    for path in [*readable, *listable]:
        # Interpolated into a quoted profile string: must not contain
        # quote / escape / control characters.
        _validate_seatbelt_root(path)

    def rule(action: str, paths: Iterable[Path]) -> str:
        filters = " ".join(f'(subpath "{p}")' for p in paths)
        return f"(allow {action} {filters})"

    lines = [
        "(version 1)",
        ";; loomflow OSSandbox — deny by default, narrow allows below.",
        "(deny default)",
        "(allow process-fork)",
        "(allow process-exec)",
        "(allow sysctl-read)",
        "(allow mach-lookup)",
        "(allow signal (target self))",
        ";; stat() anywhere (realpath, import machinery); no contents.",
        "(allow file-read-metadata)",
        ";; dyld reads the root directory entry itself at startup.",
        '(allow file-read* (literal "/"))',
        rule("file-read*", readable),
        *(
            ['(allow file-read-data ' + " ".join(f'(literal "{p}")' for p in listable) + ")"]
            if listable
            else []
        ),
        rule("file-write*", writable),
        '(allow file-write-data (literal "/dev/null"))',
    ]
    if allow_network:
        lines.append("(allow network*)")
    else:
        lines.append(";; network denied (default) — no outbound/inbound.")
    return "\n".join(lines) + "\n"


def _seatbelt_argv(profile_path: str) -> list[str]:
    return ["sandbox-exec", "-f", profile_path, sys.executable, "-E", "-s"]


#: Top-level system directories bound read-only (or recreated as the
#: symlinks they are on merged-/usr distros).
_BWRAP_SYSTEM_DIRS = ("/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32")
#: The dynamic loader's config, without which a virtualenv's python
#: can't find ``libpython`` in /usr/local/lib.
_BWRAP_LOADER_FILES = ("/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d")
#: DNS and TLS configuration, mounted only when network is allowed.
_BWRAP_NETWORK_FILES = (
    "/etc/resolv.conf",
    "/etc/hosts",
    "/etc/nsswitch.conf",
    "/etc/ssl",
    "/etc/ca-certificates",
    "/etc/pki",
)


def _bubblewrap_argv(
    roots: tuple[Path, ...],
    allow_network: bool,
    *,
    read_paths: Iterable[Path] = (),
    io_dir: Path | None = None,
) -> list[str]:
    """Build a ``bwrap`` argv: a fresh mount + PID (+ network)
    namespace with read-only system directories, the Python
    installation and import paths (``read_paths``) read-only, a private
    ``/tmp``, and the roots and ``io_dir`` read-write."""
    argv = [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--ro-bind", "/usr", "/usr",
    ]
    for d in _BWRAP_SYSTEM_DIRS:
        if os.path.islink(d):
            argv += ["--symlink", os.readlink(d), d]
        elif os.path.isdir(d):
            argv += ["--ro-bind", d, d]
    extra_files = _BWRAP_LOADER_FILES + (
        _BWRAP_NETWORK_FILES if allow_network else ()
    )
    for f in extra_files:
        argv += ["--ro-bind-try", f, f]
    argv += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    # After the tmpfs, so paths under /tmp are mounted on top of it.
    # Paths already visible through /usr aren't bound again (binding
    # the interpreter's symlink over /usr made bwrap fail).
    for path in _outermost(read_paths):
        if path == Path("/usr") or path.is_relative_to("/usr"):
            continue
        argv += ["--ro-bind", str(path), str(path)]
    writable = [*roots, *([io_dir] if io_dir is not None else [])]
    for path in writable:
        argv += ["--bind", str(path), str(path)]
    if not allow_network:
        argv += ["--unshare-net"]
    if roots:
        argv += ["--chdir", str(roots[0])]
    argv += [sys.executable, "-E", "-s"]
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
        env: environment variables to give the child on top of the
            minimal default (``PATH``, ``HOME`` = the first root,
            ``TMPDIR``, locale). The parent's environment is not
            passed through, so secrets in it stay out of the sandbox.
    """

    def __init__(
        self,
        inner: ToolHost,
        *,
        roots: Iterable[str | Path],
        allow_network: bool = False,
        timeout_seconds: float = 30.0,
        path_args: Iterable[str] | None = None,
        env: Mapping[str, str] | None = None,
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
        self._env = dict(env or {})
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

    def register(self, tool: Any) -> None:
        """Add ``tool`` under this sandbox (see :func:`register_inside`)."""
        register_inside(self, tool)

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

        try:
            output = await _run_in_os_sandbox(
                registered.fn,
                dict(args),
                mode=self._mode,
                roots=self._roots,
                allow_network=self._allow_network,
                timeout=self._timeout,
                env=self._env,
            )
        except _OSSandboxTimeout as exc:
            return ToolResult.error_(call_id=call_id, message=str(exc))
        except _OSSandboxError as exc:
            return ToolResult.error_(call_id=call_id, message=str(exc))

        return ToolResult.success(call_id=call_id, output=output)

    async def watch(self) -> AsyncIterator[ToolEvent]:
        async for event in self._inner.watch():
            yield event

# ---------------------------------------------------------------------------
# Subprocess machinery — launches a Python child UNDER the OS wrapper and
# exchanges input/output through files in a private per-call directory
# (we can't share a multiprocessing.Queue across the sandbox boundary).
# ---------------------------------------------------------------------------


class _OSSandboxTimeout(Exception):
    """Child exceeded the configured wall-clock timeout."""


class _OSSandboxError(Exception):
    """Child failed to run, or the tool raised inside the sandbox."""


# Child entry point, run with ``python -c``. It reads the parent's
# sys.path, the tool reference and the args from argv[1] (pickles the
# parent wrote) and writes its outcome to argv[2] as JSON: the parent
# never unpickles anything the sandboxed code could have written.
# Doesn't import loomflow, so it runs however loomflow is installed.
_CHILD_SOURCE = """\
import sys
inp, outp = sys.argv[1], sys.argv[2]
import asyncio, importlib, importlib.util, inspect, json, pickle

def load():
    with open(inp, "rb") as f:
        sys.path[:] = pickle.load(f)
        ref = pickle.load(f)
        args = pickle.load(f)
    if ref[0] == "fn":
        return ref[1], args
    if ref[0] == "file":
        spec = importlib.util.spec_from_file_location(ref[1], ref[2])
        fn = importlib.util.module_from_spec(spec)
        sys.modules[ref[1]] = fn
        spec.loader.exec_module(fn)
        qualname = ref[3]
    else:
        fn = importlib.import_module(ref[1])
        qualname = ref[2]
    for part in qualname.split("."):
        fn = getattr(fn, part)
    if type(fn).__name__ == "Tool" and callable(getattr(fn, "fn", None)):
        fn = fn.fn
    return fn, args

async def wait(awaitable):
    return await awaitable

try:
    fn, args = load()
    result = fn(**args)
    if inspect.isawaitable(result):
        result = asyncio.run(wait(result))
    try:
        payload = json.dumps({"ok": True, "result": result})
    except (TypeError, ValueError):
        payload = json.dumps({"ok": True, "result": str(result)})
except BaseException as exc:  # relay everything to the parent
    payload = json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
with open(outp, "w", encoding="utf-8") as f:
    f.write(payload)
"""


def _child_env(io_dir: Path, roots: tuple[Path, ...], extra: Mapping[str, str]) -> dict[str, str]:
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(roots[0]) if roots else str(io_dir),
        "TMPDIR": str(io_dir),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if sys.platform == "darwin" and "LC_ALL" in os.environ:
        env["LC_ALL"] = os.environ["LC_ALL"]
    env.update(extra)
    return env


async def _run_in_os_sandbox(
    fn: Any,
    args: dict[str, Any],
    *,
    mode: SandboxMode,
    roots: tuple[Path, ...],
    allow_network: bool,
    timeout: float,  # noqa: ASYNC109 — we kill the child directly on timeout
    env: Mapping[str, str],
) -> Any:
    """Run ``fn(**args)`` in a Python child under the OS sandbox and
    return its JSON-decoded result.

    Input and output files (and the Seatbelt profile) live in a fresh
    private directory, made visible inside the sandbox read-write.
    """
    import json
    import pickle

    import anyio

    from .subprocess_ import _dump_function

    io_dir = Path(
        await anyio.to_thread.run_sync(
            lambda: os.path.realpath(tempfile.mkdtemp(prefix="loomflow-sbio-"))
        )
    )
    in_path, out_path = io_dir / "in.pkl", io_dir / "out.json"
    try:
        try:
            ref = pickle.loads(_dump_function(fn))
            blob = (
                pickle.dumps(list(sys.path))
                + pickle.dumps(ref)
                + pickle.dumps(args)
            )
        except (pickle.PickleError, TypeError, AttributeError) as exc:
            raise _OSSandboxError(
                f"OSSandbox: tool or args not picklable ({exc}). Use a "
                "module-level function and primitive args."
            ) from exc
        await anyio.Path(in_path).write_bytes(blob)

        read_paths = await anyio.to_thread.run_sync(
            lambda: [*_python_paths(), *_import_paths(fn)]
        )
        if mode == "bubblewrap":
            wrapper = _bubblewrap_argv(
                roots, allow_network, read_paths=read_paths, io_dir=io_dir
            )
        else:
            profile = io_dir / "profile.sb"
            await anyio.Path(profile).write_text(
                _seatbelt_profile(
                    roots, allow_network, read_paths=read_paths, io_dir=io_dir
                ),
                encoding="utf-8",
            )
            wrapper = _seatbelt_argv(str(profile))
        argv = [*wrapper, "-c", _CHILD_SOURCE, str(in_path), str(out_path)]

        try:
            with anyio.fail_after(timeout):
                # The child starts in the first root, as under
                # Bubblewrap's ``--chdir``: the parent's directory may
                # be unreadable under the profile, and Python's import
                # system fails on an unreadable working directory.
                proc = await anyio.run_process(
                    argv,
                    check=False,
                    env=_child_env(io_dir, roots, env),
                    cwd=roots[0],
                )
        except TimeoutError as exc:
            raise _OSSandboxTimeout(
                f"OSSandbox: tool exceeded {timeout}s"
            ) from exc

        try:
            outcome = json.loads(await anyio.Path(out_path).read_text("utf-8"))
        except (OSError, ValueError) as exc:
            stderr = (proc.stderr or b"").decode("utf-8", "replace").strip()
            raise _OSSandboxError(
                f"OSSandbox: sandboxed child exited {proc.returncode} "
                "without a result" + (f": {stderr[-2000:]}" if stderr else "")
            ) from exc
        if not isinstance(outcome, dict) or "ok" not in outcome:
            raise _OSSandboxError("OSSandbox: malformed result from the child")
        if outcome["ok"]:
            return outcome.get("result")
        raise _OSSandboxError(f"OSSandbox: tool raised {outcome.get('error')}")
    finally:
        shutil.rmtree(io_dir, ignore_errors=True)
