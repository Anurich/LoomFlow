"""Regression tests for the tools-layer review fixes (WSF2 batch).

Covers:

* bash_tool no longer inherits the full parent environment — host
  secrets are withheld by default; ``env_allowlist`` /
  ``inherit_env`` / ``extra_env`` opt back in.
* grep_tool / find_tool prune noise dirs during the walk and stop
  early at ``max_results`` (behavioural assertions: caps + noise
  exclusion + deterministic output + path-glob support).
* Arg coercion no longer silently truncates non-integral floats
  ("8.5" for an int param passes through unchanged).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from loomflow.tools.builtin import bash_tool, find_tool, grep_tool
from loomflow.tools.registry import _coerce_to_json_type, tool

pytestmark = pytest.mark.anyio


def _echo_env_cmd(var: str) -> str:
    if sys.platform == "win32":
        return f"echo %{var}%"
    return f"echo ${var}"


# ---------------------------------------------------------------------------
# bash_tool environment isolation
# ---------------------------------------------------------------------------


async def test_bash_default_env_hides_host_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WSF2_SECRET_TOKEN", "sekret-value-123")
    t = bash_tool(tmp_path, timeout=10.0)
    out = await t.execute(
        {"command": _echo_env_cmd("WSF2_SECRET_TOKEN")}
    )
    assert "sekret-value-123" not in out


async def test_bash_default_env_keeps_path(tmp_path: Path) -> None:
    t = bash_tool(tmp_path, timeout=10.0)
    out = await t.execute({"command": _echo_env_cmd("PATH")})
    # PATH is on the allowlist — command lookup must still work and
    # the variable must be non-empty in the child.
    assert "[exit=0]" in out
    assert "--- stdout ---" in out


async def test_bash_inherit_env_opts_back_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WSF2_SECRET_TOKEN", "sekret-value-123")
    t = bash_tool(tmp_path, timeout=10.0, inherit_env=True)
    out = await t.execute(
        {"command": _echo_env_cmd("WSF2_SECRET_TOKEN")}
    )
    assert "sekret-value-123" in out


async def test_bash_env_allowlist_param(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WSF2_ALLOWED_VAR", "allowed-value-xyz")
    monkeypatch.setenv("WSF2_OTHER_VAR", "other-value-abc")
    t = bash_tool(
        tmp_path,
        timeout=10.0,
        env_allowlist=["PATH", "WSF2_ALLOWED_VAR"],
    )
    out_allowed = await t.execute(
        {"command": _echo_env_cmd("WSF2_ALLOWED_VAR")}
    )
    out_other = await t.execute(
        {"command": _echo_env_cmd("WSF2_OTHER_VAR")}
    )
    assert "allowed-value-xyz" in out_allowed
    assert "other-value-abc" not in out_other


async def test_bash_extra_env_still_merged(tmp_path: Path) -> None:
    t = bash_tool(
        tmp_path, timeout=10.0, extra_env={"WSF2_EXTRA": "extra-val-42"}
    )
    out = await t.execute({"command": _echo_env_cmd("WSF2_EXTRA")})
    assert "extra-val-42" in out


# ---------------------------------------------------------------------------
# grep_tool — pruned walk + early exit
# ---------------------------------------------------------------------------


async def test_grep_caps_at_max_results(tmp_path: Path) -> None:
    for i in range(10):
        (tmp_path / f"f{i:02d}.txt").write_text("NEEDLE here\n")
    grep = grep_tool(tmp_path, max_results=3)
    out = await grep.execute({"pattern": "NEEDLE"})
    assert "capped at 3" in out
    # Exactly 3 match lines (plus the cap notice).
    match_lines = [line for line in out.splitlines() if "NEEDLE" in line]
    assert len(match_lines) == 3


async def test_grep_still_skips_noise_dirs(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("NEEDLE\n")
    nm = tmp_path / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("NEEDLE\n")
    grep = grep_tool(tmp_path)
    out = await grep.execute({"pattern": "NEEDLE"})
    assert "src" in out
    assert "node_modules" not in out


async def test_grep_output_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("hit\n")
    (tmp_path / "a.txt").write_text("hit\n")
    grep = grep_tool(tmp_path)
    out1 = await grep.execute({"pattern": "hit"})
    out2 = await grep.execute({"pattern": "hit"})
    assert out1 == out2
    lines = out1.splitlines()
    assert lines[0].startswith("a.txt")
    assert lines[1].startswith("b.txt")


async def test_grep_single_file_path_still_works(tmp_path: Path) -> None:
    (tmp_path / "only.py").write_text("x = 1\nNEEDLE = 2\n")
    grep = grep_tool(tmp_path)
    out = await grep.execute({"pattern": "NEEDLE", "path": "only.py"})
    assert "only.py:2" in out


# ---------------------------------------------------------------------------
# find_tool — pruned walk + early exit + path globs
# ---------------------------------------------------------------------------


async def test_find_caps_at_max_results(tmp_path: Path) -> None:
    for i in range(10):
        (tmp_path / f"m{i:02d}.py").write_text("")
    find = find_tool(tmp_path, max_results=4)
    out = await find.execute({"glob": "*.py"})
    assert "capped at 4" in out


async def test_find_still_skips_noise_dirs(tmp_path: Path) -> None:
    (tmp_path / "keep.py").write_text("")
    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "hidden.py").write_text("")
    find = find_tool(tmp_path)
    out = await find.execute({"glob": "*.py"})
    assert "keep.py" in out
    assert ".venv" not in out


async def test_find_path_glob_with_doublestar(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "config.json").write_text("")
    find = find_tool(tmp_path)
    out = await find.execute({"glob": "**/config.*"})
    assert "config.yaml" in out
    assert str(Path("sub") / "config.json") in out


async def test_find_output_sorted(tmp_path: Path) -> None:
    (tmp_path / "zz.py").write_text("")
    (tmp_path / "aa.py").write_text("")
    find = find_tool(tmp_path)
    out = await find.execute({"glob": "*.py"})
    assert out.splitlines() == ["aa.py", "zz.py"]


# ---------------------------------------------------------------------------
# Arg coercion — no silent truncation
# ---------------------------------------------------------------------------


def test_non_integral_float_string_not_truncated_to_int() -> None:
    # "8.5" for an int param used to become int(float("8.5")) == 8 —
    # a silently WRONG value. Now the original string passes through
    # so the function / validation surfaces a real error.
    assert _coerce_to_json_type("8.5", "integer") == "8.5"


def test_integral_float_string_still_coerces() -> None:
    assert _coerce_to_json_type("8.0", "integer") == 8
    assert _coerce_to_json_type("8", "integer") == 8
    assert _coerce_to_json_type("-3", "integer") == -3
    assert _coerce_to_json_type(" 12.0 ", "integer") == 12


async def test_int_tool_gets_clear_error_not_wrong_answer() -> None:
    @tool
    def paginate(offset: int) -> str:
        "Paginate."
        return f"[{offset}:{offset + 10}]"

    # Integral strings still work end-to-end...
    assert await paginate.execute({"offset": "5"}) == "[5:15]"
    # ...but "8.5" must NOT silently become 8; the original value
    # reaches the function and raises rather than corrupting.
    with pytest.raises(TypeError):
        await paginate.execute({"offset": "8.5"})
