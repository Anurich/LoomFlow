"""Tests for the ``loom`` CLI (loomflow/cli.py, G12).

All offline: EchoModel agents, tmp-path datasets, monkeypatched
imports. ``main()`` is called directly — no subprocess, no sockets.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from loomflow import Agent, __version__
from loomflow.cli import main

# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


def test_version_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    assert __version__ in capsys.readouterr().out


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_run_echo(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "hi", "--model", "echo"]) == 0
    assert "Echo: hi" in capsys.readouterr().out


def test_run_stream_prints_event_lines(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "hi", "--model", "echo", "--stream"]) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    kinds = [line.split("\t", 1)[0] for line in lines]
    assert "started" in kinds
    assert "completed" in kinds
    # Each line's payload is the Event's JSON serialization.
    for line in lines:
        kind, payload = line.split("\t", 1)
        assert json.loads(payload)["kind"] == kind


def test_run_failure_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    # An unknown provider prefix cannot be resolved to a model.
    rc = main(["run", "hi", "--model", "definitely-not-a-provider:nope"])
    assert rc == 1
    assert "error" in capsys.readouterr().err.lower()


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


@pytest.fixture
def echo_agent_module(monkeypatch: pytest.MonkeyPatch) -> str:
    name = "_loom_cli_test_mod"
    module = types.ModuleType(name)
    module.agent = Agent("t", model="echo")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, module)
    return name


def _write_dataset(path: str, cases: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case) + "\n")


def test_eval_passing_threshold_exits_0(
    tmp_path: Path,
    echo_agent_module: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = f"{tmp_path}/ds.jsonl"
    _write_dataset(
        dataset,
        [
            {"input": "hi", "expected": "Echo: hi"},
            {"input": "yo", "expected": "Echo: yo"},
        ],
    )
    rc = main(
        [
            "eval",
            dataset,
            "--agent",
            f"{echo_agent_module}:agent",
            "--metric",
            "exact",
            "--threshold",
            "exact_match=1.0",
        ]
    )
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["passed"] is True
    assert summary["metrics"]["exact_match"]["mean"] == 1.0


def test_eval_failing_threshold_exits_1(
    tmp_path: Path,
    echo_agent_module: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = f"{tmp_path}/ds.jsonl"
    _write_dataset(
        dataset,
        [
            {"input": "hi", "expected": "Echo: hi"},
            {"input": "yo", "expected": "something else entirely"},
        ],
    )
    rc = main(
        [
            "eval",
            dataset,
            "--agent",
            f"{echo_agent_module}:agent",
            "--metric",
            "exact",
            "--threshold",
            "exact_match=1.0",
        ]
    )
    assert rc == 1
    assert "exact_match" in capsys.readouterr().err


def test_eval_bad_agent_spec_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = f"{tmp_path}/ds.jsonl"
    _write_dataset(dataset, [{"input": "hi", "expected": "Echo: hi"}])
    assert main(["eval", dataset, "--agent", "no-colon-here"]) == 1
    assert "module:attribute" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def test_serve_missing_uvicorn_exits_1_with_hint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # ``None`` in sys.modules makes ``import uvicorn`` raise ImportError,
    # regardless of whether uvicorn is installed in the test env.
    monkeypatch.setitem(sys.modules, "uvicorn", None)  # type: ignore[arg-type]
    rc = main(["serve", "some.module:agent"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "loomflow[serve]" in err


def test_serve_bad_target_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Fake uvicorn so the target-loading error path is reached without
    # a real server dependency.
    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda *a, **kw: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    rc = main(["serve", "not-a-valid-spec"])
    assert rc == 1
    assert "module:attribute" in capsys.readouterr().err
