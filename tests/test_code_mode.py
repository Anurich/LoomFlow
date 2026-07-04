"""G2 — code-execution seam: SubprocessExecutor, code-mode two-tool
pattern (search_api / run_code), and the bash_tool executor seam.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import anyio
import pytest

from loomflow.tools import (
    CodeExecutor,
    ExecResult,
    InProcessToolHost,
    SubprocessExecutor,
    bash_tool,
    make_code_mode_tools,
    tool,
)

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# SubprocessExecutor
# ---------------------------------------------------------------------------


async def test_subprocess_executor_happy_path() -> None:
    ex = SubprocessExecutor()
    res = await ex.run("print('hello from child')")
    assert res.returncode == 0
    assert not res.timed_out
    assert "hello from child" in res.stdout
    assert res.stderr == ""


async def test_subprocess_executor_nonzero_exit_and_stderr() -> None:
    ex = SubprocessExecutor()
    res = await ex.run("import sys; sys.stderr.write('boom'); sys.exit(3)")
    assert res.returncode == 3
    assert "boom" in res.stderr


async def test_subprocess_executor_timeout_kills() -> None:
    ex = SubprocessExecutor()
    res = await ex.run("import time; time.sleep(60)", timeout_s=0.5)
    assert res.timed_out is True
    assert res.returncode == -1
    assert "timeout" in res.stderr.lower()


async def test_subprocess_executor_files_in_artifacts_out() -> None:
    ex = SubprocessExecutor()
    code = (
        "from pathlib import Path\n"
        "data = Path('data.txt').read_text()\n"
        "Path('out/result.txt').write_text(data.upper())\n"
    )
    res = await ex.run(code, files={"data.txt": b"hello"})
    assert res.returncode == 0
    assert res.artifacts == {"result.txt": b"HELLO"}


async def test_subprocess_executor_seeded_out_files_not_echoed() -> None:
    """Unchanged files seeded under out/ don't come back as artifacts."""
    ex = SubprocessExecutor()
    res = await ex.run(
        "from pathlib import Path\nPath('out/new.txt').write_text('n')\n",
        files={"out/seed.txt": b"same"},
    )
    assert "seed.txt" not in res.artifacts
    assert res.artifacts == {"new.txt": b"n"}


async def test_subprocess_executor_rejects_escaping_file_paths() -> None:
    ex = SubprocessExecutor()
    with pytest.raises(ValueError, match="escapes"):
        await ex.run("pass", files={"../evil.txt": b"x"})


async def test_subprocess_executor_env_is_minimal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOOM_TEST_SECRET_G2", "s3cr3t-value")
    ex = SubprocessExecutor()
    res = await ex.run(
        "import os; print(os.environ.get('LOOM_TEST_SECRET_G2', 'ABSENT'))"
    )
    assert res.returncode == 0
    assert "ABSENT" in res.stdout
    assert "s3cr3t-value" not in res.stdout


async def test_subprocess_executor_env_param_merges_on_top() -> None:
    ex = SubprocessExecutor()
    res = await ex.run(
        "import os; print(os.environ['EXTRA_VAR'])",
        env={"EXTRA_VAR": "forwarded"},
    )
    assert "forwarded" in res.stdout


async def test_subprocess_executor_output_caps() -> None:
    ex = SubprocessExecutor(max_output_chars=100)
    res = await ex.run("print('x' * 5000)")
    assert len(res.stdout) < 300
    assert "truncated" in res.stdout


async def test_subprocess_executor_bash_language(tmp_path: Path) -> None:
    ex = SubprocessExecutor(cwd=tmp_path)
    res = await ex.run("echo shell-works", language="bash")
    assert res.returncode == 0
    assert "shell-works" in res.stdout


async def test_subprocess_executor_rejects_unknown_language() -> None:
    ex = SubprocessExecutor()
    with pytest.raises(ValueError, match="language"):
        await ex.run("puts 'hi'", language="ruby")


async def test_subprocess_executor_satisfies_protocol() -> None:
    assert isinstance(SubprocessExecutor(), CodeExecutor)


# ---------------------------------------------------------------------------
# Code mode — search_api
# ---------------------------------------------------------------------------


@tool
async def get_weather(city: str, units: str = "metric") -> str:
    """Fetch the current weather for a city."""
    return f"{city}: 21C sunny"


@tool
async def big_data() -> str:
    """Return a huge blob of raw data."""
    return "x" * 100_000


def _code_tools(**kwargs: Any) -> tuple[Any, Any]:
    tools = make_code_mode_tools([get_weather, big_data], **kwargs)
    assert [t.name for t in tools] == ["search_api", "run_code"]
    return tools[0], tools[1]


async def test_search_api_finds_tool_by_keyword() -> None:
    search_api, _ = _code_tools()
    out = await search_api.execute({"query": "weather"})
    assert "async def get_weather(city: str, units: str = ...)" in out
    assert "Fetch the current weather" in out
    assert "big_data" not in out


async def test_search_api_empty_query_lists_all() -> None:
    search_api, _ = _code_tools()
    out = await search_api.execute({"query": ""})
    assert "get_weather" in out
    assert "big_data" in out


async def test_search_api_no_match_names_available_functions() -> None:
    search_api, _ = _code_tools()
    out = await search_api.execute({"query": "zzzznomatch"})
    assert "No API functions match" in out
    assert "get_weather" in out  # available names listed


# ---------------------------------------------------------------------------
# Code mode — run_code (in-process tool-binding mode)
# ---------------------------------------------------------------------------


async def test_run_code_calls_tool_and_filters_huge_output() -> None:
    _, run_code = _code_tools()
    out = await run_code.execute(
        {"code": "data = await big_data()\nresult = len(data)"}
    )
    assert out == "100000"
    assert "xxx" not in out  # raw tool output never reaches the model


async def test_run_code_tool_available_via_module_namespace() -> None:
    _, run_code = _code_tools(module_name="api")
    out = await run_code.execute(
        {"code": "result = await api.get_weather(city='Paris')"}
    )
    assert "Paris" in out


async def test_run_code_result_variable_contract() -> None:
    _, run_code = _code_tools()
    out = await run_code.execute({"code": "result = {'a': 1}"})
    assert out == "{'a': 1}"


async def test_run_code_falls_back_to_print_output() -> None:
    _, run_code = _code_tools()
    out = await run_code.execute({"code": "print('printed', 42)"})
    assert out == "printed 42"


async def test_run_code_no_output_hint() -> None:
    _, run_code = _code_tools()
    out = await run_code.execute({"code": "x = 1"})
    assert "assign to `result`" in out


async def test_run_code_restricted_imports() -> None:
    _, run_code = _code_tools()
    out = await run_code.execute({"code": "import os\nresult = os.getcwd()"})
    assert "ERROR" in out
    assert "not allowed" in out
    assert "json, math" in out or "json" in out  # names the allowed modules


async def test_run_code_allowed_modules_work() -> None:
    _, run_code = _code_tools()
    out = await run_code.execute(
        {"code": "import json\nresult = json.dumps({'a': 1})"}
    )
    assert out == '{"a": 1}'


async def test_run_code_dangerous_builtins_absent() -> None:
    _, run_code = _code_tools()
    for snippet in ("open('/etc/hosts')", "eval('1+1')", "exec('x=1')"):
        out = await run_code.execute({"code": f"result = {snippet}"})
        assert "ERROR: NameError" in out


async def test_run_code_timeout() -> None:
    @tool
    async def slow() -> str:
        """Sleep for a long time."""
        await anyio.sleep(30)
        return "done"

    _, run_code = make_code_mode_tools([slow], timeout_s=0.3)
    out = await run_code.execute({"code": "result = await slow()"})
    assert "ERROR" in out
    assert "timed out" in out


async def test_run_code_tool_error_surfaces_as_text() -> None:
    @tool
    async def broken() -> str:
        """Always fails."""
        raise ValueError("kaput")

    _, run_code = make_code_mode_tools([broken])
    out = await run_code.execute({"code": "result = await broken()"})
    assert "ERROR" in out
    assert "kaput" in out


async def test_run_code_syntax_error_is_reported() -> None:
    _, run_code = _code_tools()
    out = await run_code.execute({"code": "def broken(:"})
    assert out.startswith("ERROR: SyntaxError")


async def test_run_code_output_capped_at_20k() -> None:
    _, run_code = _code_tools()
    out = await run_code.execute({"code": "result = 'y' * 50000"})
    assert len(out) < 21_000
    assert "truncated" in out


async def test_run_code_accepts_existing_host() -> None:
    host = InProcessToolHost([get_weather])
    _, run_code = make_code_mode_tools(host)
    out = await run_code.execute(
        {"code": "result = await get_weather(city='Oslo')"}
    )
    assert "Oslo" in out


async def test_run_code_is_destructive_search_api_is_not() -> None:
    search_api, run_code = _code_tools()
    assert run_code.destructive is True
    assert search_api.destructive is False


# ---------------------------------------------------------------------------
# Code mode — run_code (executor / data-transform mode)
# ---------------------------------------------------------------------------


async def test_run_code_executor_mode_no_tool_bindings() -> None:
    _, run_code = _code_tools(executor=SubprocessExecutor())
    out = await run_code.execute({"code": "print(sum(range(10)))"})
    assert "45" in out
    # tools are NOT bound out-of-process
    out2 = await run_code.execute({"code": "print(get_weather)"})
    assert "NameError" in out2 or "[exit=1]" in out2


async def test_run_code_executor_mode_timeout() -> None:
    _, run_code = _code_tools(executor=SubprocessExecutor(), timeout_s=0.5)
    out = await run_code.execute({"code": "import time; time.sleep(60)"})
    assert "ERROR" in out
    assert "timed out" in out


async def test_run_code_executor_mode_reports_artifacts() -> None:
    _, run_code = _code_tools(executor=SubprocessExecutor())
    out = await run_code.execute(
        {
            "code": (
                "from pathlib import Path\n"
                "Path('out/report.csv').write_text('a,b')\n"
                "print('wrote it')"
            )
        }
    )
    assert "wrote it" in out
    assert "report.csv" in out


# ---------------------------------------------------------------------------
# bash_tool executor seam
# ---------------------------------------------------------------------------


async def test_bash_tool_with_subprocess_executor(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("seam-content")
    t = bash_tool(tmp_path, executor=SubprocessExecutor(cwd=tmp_path))
    out = await t.execute({"command": "cat f.txt"})
    assert "seam-content" in out
    assert "[exit=0]" in out


async def test_bash_tool_executor_deny_patterns_still_apply(
    tmp_path: Path,
) -> None:
    t = bash_tool(tmp_path, executor=SubprocessExecutor(cwd=tmp_path))
    out = await t.execute({"command": "sudo ls"})
    assert out.startswith("ERROR")
    assert "denylist" in out


async def test_bash_tool_executor_timeout(tmp_path: Path) -> None:
    t = bash_tool(tmp_path, executor=SubprocessExecutor(cwd=tmp_path))
    out = await t.execute({"command": "sleep 30", "timeout_sec": 0.3})
    assert out.startswith("ERROR")
    assert "timed out" in out


async def test_bash_tool_executor_env_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOOM_TEST_SECRET_G2B", "hidden")
    t = bash_tool(tmp_path, executor=SubprocessExecutor(cwd=tmp_path))
    out = await t.execute(
        {"command": "echo ${LOOM_TEST_SECRET_G2B:-ABSENT}"}
    )
    assert "ABSENT" in out
    assert "hidden" not in out


async def test_bash_tool_executor_extra_env_forwarded(tmp_path: Path) -> None:
    t = bash_tool(
        tmp_path,
        executor=SubprocessExecutor(cwd=tmp_path),
        extra_env={"MY_FLAG": "on"},
    )
    out = await t.execute({"command": "echo $MY_FLAG"})
    assert "on" in out


async def test_bash_tool_with_fake_executor_result_formatting(
    tmp_path: Path,
) -> None:
    """Any CodeExecutor implementation slots into the seam."""

    class FakeExecutor:
        async def run(
            self,
            code: str,
            *,
            language: str = "python",
            timeout_s: float = 30.0,
            files: Mapping[str, bytes] | None = None,
            env: Mapping[str, str] | None = None,
        ) -> ExecResult:
            assert language == "bash"
            return ExecResult(stdout="fake-out", stderr="fake-err", returncode=7)

    t = bash_tool(tmp_path, executor=FakeExecutor())
    out = await t.execute({"command": "anything"})
    assert "[exit=7]" in out
    assert "fake-out" in out
    assert "fake-err" in out


async def test_bash_tool_default_path_unchanged(tmp_path: Path) -> None:
    """executor=None keeps the existing host-subprocess behavior."""
    t = bash_tool(tmp_path)
    out = await t.execute({"command": "echo direct"})
    assert "direct" in out
    assert "[exit=0]" in out
