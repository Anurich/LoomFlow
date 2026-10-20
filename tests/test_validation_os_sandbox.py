"""OSSandbox confines reads and the environment, not only writes.

Seatbelt used to allow reads anywhere (a secret outside the roots was
read, the home directory listed) and writes to the system temp dir;
Bubblewrap couldn't launch at all; and the child inherited the
parent's environment (API keys included). The parent also unpickled
the child's output, which the sandboxed code controls.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

import tests._subprocess_tools as sbtools
from loomflow.security.sandbox import OSSandbox
from loomflow.security.sandbox.os_sandbox import (
    _bubblewrap_argv,
    _import_paths,
    _python_paths,
    _seatbelt_profile,
)
from loomflow.tools.registry import InProcessToolHost, Tool

pytestmark = pytest.mark.anyio

_HAS_BACKEND = (sys.platform == "darwin" and shutil.which("sandbox-exec")) or (
    sys.platform.startswith("linux") and shutil.which("bwrap")
)
requires_backend = pytest.mark.skipif(
    not _HAS_BACKEND, reason="no OS sandbox backend (seatbelt/bwrap) on host"
)


def _sandbox(root: Path, *fns: object, **kwargs: object) -> OSSandbox:
    host = InProcessToolHost(
        [Tool(name=f.__name__, description="t", fn=f) for f in fns]  # type: ignore[attr-defined]
    )
    return OSSandbox(host, roots=[root], **kwargs)  # type: ignore[arg-type]


def test_import_paths_expose_only_the_tools_package() -> None:
    # The tool lives in tests/, which is outside the installation: only
    # that package (and loomflow) is readable, not the project root.
    paths = _import_paths(sbtools.read_text)
    project = Path(__file__).resolve().parent.parent
    assert Path(sbtools.__file__).resolve().parent in paths
    assert project not in paths
    install = _python_paths()
    assert all(not any(p.is_relative_to(i) for i in install) for p in paths)


def test_policies_never_mount_or_allow_the_whole_project(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    reads = [*_python_paths(), *_import_paths(sbtools.read_text)]
    project = str(Path(__file__).resolve().parent.parent)
    profile = _seatbelt_profile((root,), False, read_paths=reads, io_dir=root / "io")
    assert f'(subpath "{project}")' not in profile
    assert "(allow file-read*)" not in profile  # no blanket read
    argv = _bubblewrap_argv((root,), False, read_paths=reads, io_dir=root / "io")
    binds = {argv[i + 1] for i, a in enumerate(argv) if a in ("--ro-bind", "--bind")}
    assert project not in binds
    # /tmp is a private tmpfs, and the I/O dir is mounted after it.
    assert argv.index("/tmp") < argv.index(str(root / "io"))
    assert "--unshare-net" in argv


@requires_backend
async def test_reads_outside_the_roots_are_blocked(tmp_path: Path) -> None:
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("SECRET", encoding="utf-8")
    (root / "mine.txt").write_text("MINE", encoding="utf-8")
    sb = _sandbox(root, sbtools.read_text)
    denied = await sb.call("read_text", {"path": str(outside / "secret.txt")}, call_id="c")
    assert not denied.ok and "SECRET" not in str(denied.output)
    allowed = await sb.call("read_text", {"path": str(root / "mine.txt")}, call_id="c")
    assert allowed.ok and allowed.output == "MINE"


@requires_backend
async def test_writes_to_the_system_temp_dir_are_blocked(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    escape = tmp_path / "escape.txt"  # under the system temp dir, outside root
    sb = _sandbox(root, sbtools.write_under)
    res = await sb.call("write_under", {"path": str(escape), "content": "x"}, call_id="c")
    # Seatbelt denies the write; under bwrap /tmp is a private tmpfs,
    # so the write lands there and never reaches the host.
    if sb.mode == "seatbelt":
        assert not res.ok
    assert not escape.exists()


@requires_backend
async def test_child_does_not_inherit_the_parents_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOOMFLOW_TEST_SECRET", "sk-parent")
    sb = _sandbox(tmp_path, sbtools.get_env)
    res = await sb.call("get_env", {"name": "LOOMFLOW_TEST_SECRET"}, call_id="c")
    assert res.ok and res.output is None
    sb = _sandbox(tmp_path, sbtools.get_env, env={"LOOMFLOW_TEST_SECRET": "given"})
    res = await sb.call("get_env", {"name": "LOOMFLOW_TEST_SECRET"}, call_id="c")
    assert res.ok and res.output == "given"


@requires_backend
async def test_tool_functions_async_tools_and_odd_results(tmp_path: Path) -> None:
    host = InProcessToolHost([sbtools.tool_triple])
    host.register(Tool(name="async_double", description="d", fn=sbtools.async_double))
    host.register(Tool(name="make_set", description="s", fn=sbtools.make_set))
    sb = OSSandbox(host, roots=[tmp_path])
    assert (await sb.call("tool_triple", {"x": 3}, call_id="c")).output == 9
    assert (await sb.call("async_double", {"x": 4}, call_id="c")).output == 8
    # Results come back as JSON; anything else as its str().
    assert (await sb.call("make_set", {}, call_id="c")).output == "{1, 2}"
