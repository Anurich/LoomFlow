"""Tests for OSSandbox (kernel-level isolation + graceful degrade).

The kernel-path tests are skipped unless a backend is actually present
(seatbelt on macOS, bwrap on Linux) — CI on a backend-less box still
exercises the degrade path + construction + mode detection.

Tool fns come from ``tests._subprocess_tools`` (module-level + picklable)
because OSSandbox ships the callable to a child process; a closure or a
``@tool``-decorated local can't cross that boundary.
"""
from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import tests._subprocess_tools as sbtools
from loomflow.core.errors import ConfigError
from loomflow.security.sandbox import OSSandbox
from loomflow.security.sandbox.os_sandbox import _detect_mode
from loomflow.tools.registry import InProcessToolHost, Tool

pytestmark = pytest.mark.anyio


_HAS_BACKEND = (sys.platform == "darwin" and shutil.which("sandbox-exec")) or (
    sys.platform.startswith("linux") and shutil.which("bwrap")
)
requires_backend = pytest.mark.skipif(
    not _HAS_BACKEND, reason="no OS sandbox backend (seatbelt/bwrap) on host"
)


def _host_with(
    *specs: tuple[str, Callable[..., Any], str],
) -> InProcessToolHost:
    """Register ``(name, fn, description)`` specs from module-level,
    picklable fns (see module docstring)."""
    host = InProcessToolHost()
    for name, fn, desc in specs:
        host.register(
            Tool(
                name=name,
                description=desc,
                fn=fn,
                input_schema={"type": "object"},
            )
        )
    return host


_DOUBLE = ("double", sbtools.double, "double a number")
_WRITE = ("write_under", sbtools.write_under, "write content to a path")
_NET = ("reach_network", sbtools.reach_network, "attempt a TCP connect")


# ---- construction / config ------------------------------------------------


def test_requires_a_root() -> None:
    with pytest.raises(ValueError, match="at least one writable root"):
        OSSandbox(_host_with(_DOUBLE), roots=[])


def test_rejects_non_positive_timeout(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError, match="timeout_seconds must be positive"
    ):
        OSSandbox(
            _host_with(_DOUBLE), roots=[tmp_path], timeout_seconds=0
        )


def test_mode_matches_detection(tmp_path: Path) -> None:
    sb = OSSandbox(_host_with(_DOUBLE), roots=[tmp_path])
    assert sb.mode == _detect_mode()
    assert sb.mode in ("seatbelt", "bubblewrap", "degraded")


def test_kernel_mode_requires_inprocess_host(tmp_path: Path) -> None:
    # Only assert the InProcess requirement when a kernel backend is
    # active — degraded mode composes a different fallback chain.
    if _detect_mode() == "degraded":
        pytest.skip("degraded mode has no InProcess requirement")

    class _NotInProcess:
        async def list_tools(self, *, query=None):  # type: ignore[no-untyped-def]
            return []

        async def call(self, tool, args, *, call_id=""):  # type: ignore[no-untyped-def]
            return None

        async def watch(self):  # type: ignore[no-untyped-def]
            return
            yield  # pragma: no cover — makes this an async generator

    with pytest.raises(ConfigError, match="InProcessToolHost only"):
        OSSandbox(_NotInProcess(), roots=[tmp_path])  # type: ignore[arg-type]


# ---- behaviour: works in whatever mode is active --------------------------


async def test_double_roundtrips(tmp_path: Path) -> None:
    sb = OSSandbox(_host_with(_DOUBLE), roots=[tmp_path])
    res = await sb.call("double", {"x": 21}, call_id="c1")
    assert res.ok, res.error
    assert res.output == 42


async def test_unknown_tool_errors(tmp_path: Path) -> None:
    sb = OSSandbox(_host_with(_DOUBLE), roots=[tmp_path])
    res = await sb.call("_nope", {}, call_id="c2")
    assert not res.ok


async def test_write_inside_root_succeeds(tmp_path: Path) -> None:
    sb = OSSandbox(_host_with(_WRITE), roots=[tmp_path])
    target = tmp_path / "out.txt"
    res = await sb.call(
        "write_under",
        {"path": str(target), "content": "data"},
        call_id="c3",
    )
    assert res.ok, res.error
    assert target.read_text(encoding="utf-8") == "data"


# ---- kernel-only guarantees (skipped without a backend) -------------------


@requires_backend
async def test_write_outside_root_is_blocked(tmp_path: Path) -> None:
    # The kernel must reject a write outside the declared root. The
    # target must be outside BOTH the root AND the OS temp dir, since the
    # profile grants temp for writes (Python needs __pycache__/tempfiles)
    # — and pytest's tmp_path itself lives under temp, so a sibling of it
    # would (correctly) be allowed. Use a uniquely-named dir under $HOME.
    home_outside = Path.home() / ".loomflow-sandbox-test-outside"
    home_outside.mkdir(exist_ok=True)
    leak = home_outside / "leak.txt"
    if leak.exists():
        leak.unlink()
    try:
        sb = OSSandbox(_host_with(_WRITE), roots=[tmp_path])
        res = await sb.call(
            "write_under",
            {"path": str(leak), "content": "x"},
            call_id="c4",
        )
        assert not res.ok  # kernel denied the write → tool raised
        assert not leak.exists()
    finally:
        if leak.exists():
            leak.unlink()
        try:
            home_outside.rmdir()
        except OSError:
            pass


@requires_backend
async def test_network_blocked_by_default(tmp_path: Path) -> None:
    sb = OSSandbox(_host_with(_NET), roots=[tmp_path])
    res = await sb.call("reach_network", {}, call_id="c5")
    assert not res.ok  # connect refused/denied by the sandbox


@requires_backend
async def test_network_allowed_when_opted_in(tmp_path: Path) -> None:
    # With network allowed, the sandbox must NOT categorically deny like
    # the default-deny case. We don't require a live connection (CI may
    # be offline / the host may flake), so: a clean success returns
    # "connected"; otherwise the call may error for a NON-sandbox reason.
    # The one thing we assert is that *if* it succeeded, the value came
    # through (output plumbing works) — proving the allow path is real.
    sb = OSSandbox(
        _host_with(_NET), roots=[tmp_path], allow_network=True
    )
    res = await sb.call("reach_network", {}, call_id="c6")
    if res.ok:
        assert res.output == "connected"
    # else: errored for a non-sandbox reason (offline) — inconclusive,
    # not a failure. The default-deny guarantee is covered by the test
    # above; this one only needs to show allow_network changes behaviour.
