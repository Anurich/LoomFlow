"""Code-execution seam: the :class:`CodeExecutor` protocol and a
safe-default :class:`SubprocessExecutor`.

This module defines the isolation boundary for running model-authored
code. The protocol is deliberately tiny — one async ``run()`` method —
so remote/backends slot in without touching call sites:

* :class:`SubprocessExecutor` (shipped here) — a fresh local Python /
  shell subprocess per call.
* E2B / Modal / Daytona / Docker adapters (NOT shipped) — implement
  the same ``run()`` signature against a remote or containerized
  runtime and pass the instance anywhere a ``CodeExecutor`` is
  accepted (``bash_tool(executor=...)``,
  ``make_code_mode_tools(..., executor=...)``).

Isolation tiers — be honest about what you get
----------------------------------------------

``SubprocessExecutor`` provides **process isolation + hard timeout +
environment hygiene only**:

* the code runs in its own process (a crash/OOM doesn't take the
  agent down),
* it is killed after ``timeout_s`` seconds,
* it sees a minimal allowlisted environment (host API keys / tokens
  are NOT forwarded), and Python runs in ``-I`` isolated mode (no
  user site-packages, no ``PYTHON*`` env influence, cwd not on
  ``sys.path``).

It does **NOT** block network access, filesystem reads outside the
scratch dir, or dangerous syscalls. For kernel-enforced isolation,
run the agent under :class:`~loomflow.security.sandbox.OSSandbox`
(Seatbelt / Bubblewrap / Landlock) — or implement this protocol
against a remote sandbox (E2B, Modal, Docker, gVisor), which is the
protocol's whole purpose.

File I/O contract
-----------------

``files`` are materialised (relative paths only) into a fresh scratch
directory that becomes the subprocess's cwd. Code that wants to
return file outputs should write them under ``./out/`` — everything
new or changed under that directory is collected into
:attr:`ExecResult.artifacts` after the run, keyed by path relative to
``out/``. The scratch directory is deleted afterwards (unless a fixed
``cwd=`` was supplied at construction).
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import anyio
from pydantic import BaseModel, Field

from .builtin import _DEFAULT_ENV_ALLOWLIST

#: Environment variables forwarded to executor subprocesses by
#: default. Shared with ``bash_tool``'s allowlist — enough for
#: command lookup, temp files, and locale-correct output, but NOT
#: the parent process's API keys / tokens / cloud credentials.
MINIMAL_ENV_ALLOWLIST: tuple[str, ...] = _DEFAULT_ENV_ALLOWLIST

#: Directory (relative to the scratch cwd) the executed code should
#: write outputs to; its contents come back as ``ExecResult.artifacts``.
OUTPUT_DIR_NAME = "out"

#: Per-stream output cap (characters). Beyond this the stream is
#: truncated with an explicit marker so the model knows data is missing.
DEFAULT_MAX_OUTPUT_CHARS = 100_000

#: Total artifact budget (bytes). Collection stops once exceeded so a
#: runaway loop writing gigabytes under ``out/`` can't OOM the agent.
DEFAULT_MAX_ARTIFACT_BYTES = 16_000_000


class ExecResult(BaseModel):
    """Outcome of one :meth:`CodeExecutor.run` invocation."""

    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False
    #: Files the code wrote under the designated output dir
    #: (``./out/``), keyed by path relative to that dir.
    artifacts: dict[str, bytes] = Field(default_factory=dict)


@runtime_checkable
class CodeExecutor(Protocol):
    """Anything that can run a string of code and report the outcome.

    Implementations decide *where* the code runs (local subprocess,
    Docker container, remote microVM) — callers only see this
    signature. ``language`` is a hint (``"python"`` and ``"bash"``
    are the two loomflow itself uses); implementations may support
    more or raise :class:`ValueError` for ones they don't.
    """

    async def run(
        self,
        code: str,
        *,
        language: str = "python",
        timeout_s: float = 30.0,
        files: Mapping[str, bytes] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ExecResult: ...


class SubprocessExecutor:
    """Run code in a fresh local subprocess with a scratch cwd.

    Safe *default* tier — see the module docstring for what this does
    and does not isolate. Construction knobs:

    * ``python`` — interpreter used for ``language="python"``
      (default: the current interpreter). Always invoked with ``-I``
      (isolated mode).
    * ``cwd`` — fixed working directory. Default ``None`` creates a
      fresh temporary directory per call and deletes it afterwards;
      pass a path to run in-place (e.g. to share ``bash_tool``'s
      workdir) — nothing is deleted then.
    * ``env_allowlist`` — parent-env variable names forwarded to the
      subprocess (default: the same minimal list ``bash_tool`` uses).
      The ``env`` mapping passed to :meth:`run` is merged on top.
    * ``max_output_chars`` — per-stream truncation cap.
    * ``max_artifact_bytes`` — total artifact collection budget.
    """

    def __init__(
        self,
        python: str = sys.executable,
        *,
        cwd: Path | None = None,
        env_allowlist: Sequence[str] = MINIMAL_ENV_ALLOWLIST,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    ) -> None:
        self._python = python
        self._cwd = cwd.resolve() if cwd is not None else None
        self._env_allowlist = tuple(env_allowlist)
        self._max_output_chars = max_output_chars
        self._max_artifact_bytes = max_artifact_bytes

    async def run(
        self,
        code: str,
        *,
        language: str = "python",
        timeout_s: float = 30.0,
        files: Mapping[str, bytes] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ExecResult:
        """Execute ``code`` and return the collected outcome.

        On timeout the subprocess is killed and the result carries
        ``timed_out=True`` with ``returncode=-1``; any artifacts the
        code managed to write under ``out/`` before the kill are
        still collected.
        """
        if language not in ("python", "bash", "sh", "shell"):
            raise ValueError(
                f"SubprocessExecutor supports language='python' or 'bash'; got {language!r}"
            )

        tmp: tempfile.TemporaryDirectory[str] | None = None
        if self._cwd is not None:
            workdir = self._cwd
        else:
            tmp = tempfile.TemporaryDirectory(prefix="loom_exec_")
            workdir = Path(tmp.name)
        try:
            seeded = await anyio.to_thread.run_sync(
                self._setup, workdir, dict(files or {})
            )

            if language == "python":
                script = workdir / "_loom_code_mode_main.py"
                await anyio.to_thread.run_sync(script.write_text, code)
                argv = [self._python, "-I", str(script)]
            elif sys.platform == "win32":
                argv = ["cmd.exe", "/c", code]
            else:
                argv = ["/bin/sh", "-c", code]

            run_env = {
                key: val
                for key in self._env_allowlist
                if (val := os.environ.get(key)) is not None
            }
            if env:
                run_env.update(env)

            timed_out = False
            stdout = stderr = ""
            returncode = -1
            try:
                # anyio kills the child on cancellation (Process.aclose),
                # mirroring bash_tool's fail_after + run_process discipline.
                with anyio.fail_after(timeout_s):
                    proc = await anyio.run_process(
                        argv,
                        cwd=str(workdir),
                        env=run_env,
                        check=False,
                    )
            except TimeoutError:
                timed_out = True
                stderr = f"ERROR: process killed after {timeout_s:.1f}s timeout"
            else:
                stdout = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
                stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
                returncode = proc.returncode if proc.returncode is not None else -1

            artifacts = await anyio.to_thread.run_sync(
                self._collect_artifacts, workdir, seeded
            )
            return ExecResult(
                stdout=self._truncate(stdout),
                stderr=self._truncate(stderr),
                returncode=returncode,
                timed_out=timed_out,
                artifacts=artifacts,
            )
        finally:
            if tmp is not None:
                await anyio.to_thread.run_sync(tmp.cleanup)

    # ---- internals --------------------------------------------------------

    def _truncate(self, text: str) -> str:
        cap = self._max_output_chars
        if len(text) <= cap:
            return text
        return text[:cap] + f"\n... (truncated, {len(text) - cap} more chars)"

    def _setup(self, workdir: Path, files: dict[str, bytes]) -> dict[str, bytes]:
        """Materialise input files; return the seeded out/ contents.

        Seeded contents of ``out/`` are remembered so only *new or
        changed* files come back as artifacts.
        """
        workdir.mkdir(parents=True, exist_ok=True)
        out_dir = workdir / OUTPUT_DIR_NAME
        out_dir.mkdir(parents=True, exist_ok=True)
        seeded: dict[str, bytes] = {}
        workdir_resolved = workdir.resolve()
        for rel, data in files.items():
            target = (workdir / rel).resolve()
            try:
                target.relative_to(workdir_resolved)
            except ValueError:
                raise ValueError(
                    f"files key {rel!r} escapes the scratch directory "
                    "(absolute paths and '..' are not allowed)"
                ) from None
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            try:
                seeded[target.relative_to(out_dir.resolve()).as_posix()] = data
            except ValueError:
                pass  # not under out/ — never an artifact candidate
        return seeded

    def _collect_artifacts(
        self, workdir: Path, seeded: dict[str, bytes]
    ) -> dict[str, bytes]:
        """Gather new/changed files under ``out/``, within budget."""
        out_dir = workdir / OUTPUT_DIR_NAME
        if not out_dir.is_dir():
            return {}
        artifacts: dict[str, bytes] = {}
        budget = self._max_artifact_bytes
        for dirpath, dirnames, filenames in os.walk(out_dir):
            dirnames.sort()
            for fname in sorted(filenames):
                fp = Path(dirpath) / fname
                rel = fp.relative_to(out_dir).as_posix()
                try:
                    data = fp.read_bytes()
                except OSError:
                    continue
                if seeded.get(rel) == data:
                    continue  # unchanged input echo — not an artifact
                if len(data) > budget:
                    return artifacts  # budget exhausted; stop collecting
                artifacts[rel] = data
                budget -= len(data)
        return artifacts


__all__ = [
    "DEFAULT_MAX_ARTIFACT_BYTES",
    "DEFAULT_MAX_OUTPUT_CHARS",
    "MINIMAL_ENV_ALLOWLIST",
    "OUTPUT_DIR_NAME",
    "CodeExecutor",
    "ExecResult",
    "SubprocessExecutor",
]
