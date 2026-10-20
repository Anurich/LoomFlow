"""Regressions in 0.17.0-0.17.2, found while auditing the docs.

* Giving sandboxes ``register()`` put the workspace and living-plan
  tools inside a process sandbox, which can't ship their closures to
  its child, so every call failed.
* A skill's ``tools.py`` (loaded under a name the child can't import)
  failed under a process sandbox.
* A Seatbelt child inherited the parent's working directory and failed
  to start when the profile didn't let it read that directory.
* ``agent.memory`` is the backend, not the auto-extract wrapper, so
  ``agent.memory.forget()`` let a running extraction write the erased
  facts back.
* An interrupted Agent branch of ``Workflow.parallel`` reached
  ``merge`` as a private object, and the run wasn't interrupted.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

import tests._subprocess_tools as sbtools
from loomflow import Agent, InMemoryMemory, ScriptedModel, ScriptedTurn, Workflow, tool
from loomflow.core.types import ToolCall
from loomflow.security.sandbox import OSSandbox, SubprocessSandbox
from loomflow.tools.registry import InProcessToolHost, Tool

pytestmark = pytest.mark.anyio

_HAS_SEATBELT = sys.platform == "darwin" and shutil.which("sandbox-exec")
_HAS_BACKEND = _HAS_SEATBELT or (sys.platform.startswith("linux") and shutil.which("bwrap"))
requires_backend = pytest.mark.skipif(
    not _HAS_BACKEND, reason="no OS sandbox backend (seatbelt/bwrap) on host"
)


def _model() -> ScriptedModel:
    return ScriptedModel([ScriptedTurn(text="ok")])


# ---------------------------------------------------------------------------
# Framework tools stay outside process sandboxes
# ---------------------------------------------------------------------------


async def test_workspace_and_plan_tools_work_under_a_process_sandbox() -> None:
    from loomflow import InMemoryWorkspace

    sandbox = SubprocessSandbox(InProcessToolHost([]))
    agent = Agent(
        "x", model=_model(), tools=sandbox, workspace=InMemoryWorkspace(), living_plan=True
    )
    listed = await agent.tool_host.call("list_notes", {})
    assert listed.ok, listed.error
    plan = await agent.tool_host.call("plan_read", {})
    assert plan.ok, plan.error


# ---------------------------------------------------------------------------
# A skill's tools.py runs inside the sandbox
# ---------------------------------------------------------------------------


def _python_skill(base: Path) -> Path:
    skill = base / "greeter"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: greeter\ndescription: Greets people.\n---\nUse greeter__say_hi.\n",
        encoding="utf-8",
    )
    (skill / "tools.py").write_text(
        "import os\n"
        "from loomflow import tool\n\n"
        "@tool\n"
        "async def say_hi(name: str) -> str:\n"
        '    """Say hi."""\n'
        '    return f"Hi {name} from {os.getpid()}"\n',
        encoding="utf-8",
    )
    return skill


async def _say_hi(agent: Agent) -> Any:
    await agent.tool_host.call("load_skill", {"name": "greeter"})
    return await agent.tool_host.call("greeter__say_hi", {"name": "A"})


async def test_skill_python_tools_run_in_a_subprocess_sandbox(tmp_path: Path) -> None:
    import os

    skill = _python_skill(tmp_path)
    sandbox = SubprocessSandbox(InProcessToolHost([]))
    result = await _say_hi(Agent("x", model=_model(), tools=sandbox, skills=[skill]))
    assert result.ok, result.error
    assert str(result.output).startswith("Hi A from ")
    assert str(result.output) != f"Hi A from {os.getpid()}"  # ran in the child


async def test_a_closure_tool_is_an_error_result_not_a_crash() -> None:
    # What a skill's build_tools(ctx) returns: closures can't reach a
    # child process. On Python 3.12 / 3.13 pickle raised AttributeError,
    # which escaped SubprocessSandbox.call.
    def make() -> Tool:
        def greet(name: str) -> str:
            return f"hello {name}"

        return Tool(name="greet", description="g", fn=greet)

    sandbox = SubprocessSandbox(InProcessToolHost([make()]))
    result = await sandbox.call("greet", {"name": "A"}, call_id="c")
    assert not result.ok and "not picklable" in (result.error or "")


@requires_backend
async def test_skill_python_tools_run_in_the_os_sandbox(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    skill = _python_skill(tmp_path)
    sandbox = OSSandbox(InProcessToolHost([]), roots=[root])
    result = await _say_hi(Agent("x", model=_model(), tools=sandbox, skills=[skill]))
    assert result.ok, result.error


# ---------------------------------------------------------------------------
# Seatbelt child's working directory
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_SEATBELT, reason="Seatbelt only")
async def test_seatbelt_child_starts_in_its_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, elsewhere = tmp_path / "root", tmp_path / "elsewhere"
    root.mkdir()
    elsewhere.mkdir()
    (root / "mine.txt").write_text("MINE", encoding="utf-8")
    # The parent runs from a directory the profile doesn't let the
    # child read.
    monkeypatch.chdir(elsewhere)
    host = InProcessToolHost([Tool(name="read_text", description="t", fn=sbtools.read_text)])
    sandbox = OSSandbox(host, roots=[root])
    result = await sandbox.call("read_text", {"path": "mine.txt"}, call_id="c")
    assert result.ok, result.error
    assert result.output == "MINE"  # relative paths resolve in the root


# ---------------------------------------------------------------------------
# agent.memory.forget() stops background extraction
# ---------------------------------------------------------------------------


_FACT = '[{"subject": "alice", "predicate": "ssn_is", "object": "123-45-6789", "confidence": 0.9}]'


class _SlowExtraction(ScriptedModel):
    async def stream(self, messages: Any, **kwargs: Any) -> Any:
        if self._idx == 1:  # the extraction call
            await asyncio.sleep(0.3)
        async for chunk in super().stream(messages, **kwargs):
            yield chunk


async def test_forget_on_the_backend_stops_background_extraction() -> None:
    backend = InMemoryMemory()
    agent = Agent(
        "x",
        model=_SlowExtraction([ScriptedTurn(text="ok"), ScriptedTurn(text=_FACT)]),
        memory=backend,
        auto_extract=True,
    )
    await agent.run("my ssn is 123-45-6789", user_id="alice")
    assert agent.memory is backend
    await agent.memory.forget(user_id="alice")
    await agent.aclose()
    assert await backend.facts.query(user_id="alice") == []


async def test_forget_for_another_user_leaves_extraction_running() -> None:
    backend = InMemoryMemory()
    agent = Agent(
        "x",
        model=_SlowExtraction([ScriptedTurn(text="ok"), ScriptedTurn(text=_FACT)]),
        memory=backend,
        auto_extract=True,
    )
    await agent.run("my ssn is 123-45-6789", user_id="alice")
    await agent.memory.forget(user_id="bob")
    await agent.aclose()
    assert [f.object for f in await backend.facts.query(user_id="alice")] == ["123-45-6789"]


# ---------------------------------------------------------------------------
# Workflow.parallel and interrupted branches
# ---------------------------------------------------------------------------


@tool
def _ping(n: int) -> str:
    """Ping."""
    return "pong"


def _runs_out_of_turns() -> Agent:
    calls = [
        ScriptedTurn(tool_calls=[ToolCall(id=f"c{i}", tool="_ping", args={"n": i})])
        for i in range(5)
    ]
    return Agent("x", model=ScriptedModel(calls), tools=[_ping], max_turns=2)


@pytest.mark.parametrize("return_exceptions", [False, True])
async def test_parallel_stops_at_an_interrupted_branch(return_exceptions: bool) -> None:
    merged: list[Any] = []

    def merge(results: list[Any]) -> Any:
        merged.append(results)
        return results

    fine = Agent("y", model=ScriptedModel([ScriptedTurn(text="fine")]))
    flow = Workflow.parallel(
        [_runs_out_of_turns(), fine], merge=merge, return_exceptions=return_exceptions
    )
    result = await flow.run("go")
    assert result.interrupted
    assert result.interruption_reason == "fan_out:branch_0:max_turns_exceeded"
    assert merged == []


async def test_parallel_without_interruptions_merges_as_before() -> None:
    a = Agent("a", model=ScriptedModel([ScriptedTurn(text="A")]))
    b = Agent("b", model=ScriptedModel([ScriptedTurn(text="B")]))
    result = await Workflow.parallel([a, b]).run("go")
    assert not result.interrupted and result.output == ["A", "B"]


# ---------------------------------------------------------------------------
# ReWOO step references
# ---------------------------------------------------------------------------


def test_rewoo_default_prompt_teaches_the_reference_it_parses() -> None:
    # The prose said ``{{En}}`` in source, which str.format renders as
    # ``{En}``: the model was taught a form the executor didn't read.
    from loomflow.architecture import rewoo
    from loomflow.architecture.helpers import fill_template

    rendered = fill_template(rewoo.DEFAULT_PLANNER_PROMPT, tool_descriptions="- t")
    assert "using `{{En}}`" in rendered and "`{En}`" not in rendered


async def test_rewoo_reads_single_brace_references() -> None:
    import json

    from loomflow.architecture import ReWOO

    seen: list[str] = []

    @tool
    async def fetch(url: str) -> str:
        """Fetch a URL."""
        seen.append(url)
        return f"page<{url}>"

    plan = [
        {"id": "E1", "tool": "fetch", "args": {"url": "a"}},
        {"id": "E2", "tool": "fetch", "args": {"url": "{E1}"}},
    ]
    model = ScriptedModel(
        [ScriptedTurn(text=json.dumps(plan)), ScriptedTurn(text="ANSWER")]
    )
    await Agent("x", model=model, tools=[fetch], architecture=ReWOO()).run("task")
    assert seen == ["a", "page<a>"]


# ---------------------------------------------------------------------------
# Role-model spec strings and the Agent's secrets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "key"), [("gpt-4.1-mini", "OPENAI_API_KEY"), ("deepseek-chat", "DEEPSEEK_API_KEY")]
)
def test_role_model_key_can_live_only_in_the_agents_secrets(
    spec: str, key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An OpenAI client needs its key when it's built, so these specs
    # failed in the architecture's constructor before the Agent could
    # hand them its secrets.
    pytest.importorskip("openai")
    from loomflow import Tuning
    from loomflow.architecture import Reflexion
    from loomflow.security import DictSecrets

    for name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    agent = Agent(
        "x",
        model=_model(),
        architecture=Reflexion(evaluator_model=spec),
        tuning=Tuning(secrets=DictSecrets({key: "sk-from-secrets"})),
    )
    evaluator = agent._architecture._evaluator_model  # type: ignore[attr-defined]
    assert evaluator._client.api_key == "sk-from-secrets"


def test_a_bad_role_model_spec_still_fails_at_construction() -> None:
    from loomflow.architecture import Reflexion
    from loomflow.core.errors import ConfigError

    with pytest.raises(ConfigError):
        Reflexion(evaluator_model="no-such-model-family")
