"""Loomflow Studio: recording runs, the catalog, experiments, the web app."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pytest

import loomflow.studio as studio
from loomflow import Agent, Mode, StandardPermissions, Workflow, tool
from loomflow.architecture import Supervisor
from loomflow.core.types import ToolCall, Usage
from loomflow.eval import Contains, EvalHarness
from loomflow.eval.dataset import Case, Dataset
from loomflow.model.scripted import ScriptedModel, ScriptedTurn
from loomflow.observability import InMemoryTelemetry
from loomflow.studio import _state

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def _clean_studio(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv(studio.ENV_VAR, raising=False)
    monkeypatch.delenv(studio.PROJECT_ENV_VAR, raising=False)
    monkeypatch.delenv(studio.KEEP_DAYS_ENV_VAR, raising=False)
    studio.disable()
    yield
    studio.disable()
    for recorder in _state.all_recorders():
        recorder.close()
    _state._recorders.clear()


@tool
def lookup(order_id: str) -> dict[str, Any]:
    """Look up an order."""
    return {"order": order_id, "charges": 2}


def _agent(turns: list[ScriptedTurn], **kwargs: Any) -> Agent:
    kwargs.setdefault("name", "billing")
    kwargs.setdefault("auto_extract", False)
    return Agent(kwargs.pop("instructions", "You handle billing."), model=ScriptedModel(turns), **kwargs)


def _tool_turns(text: str = "Refunded.") -> list[ScriptedTurn]:
    return [
        ScriptedTurn(tool_calls=[ToolCall(id="c1", tool="lookup", args={"order_id": "4471"})],
                     usage=Usage(input_tokens=100, output_tokens=10, cost_usd=0.001)),
        ScriptedTurn(text=text, usage=Usage(input_tokens=150, output_tokens=20, cost_usd=0.002)),
    ]


def _db(path: Path) -> sqlite3.Connection:
    assert studio.flush()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _traces(path: Path) -> list[dict[str, Any]]:
    return [dict(r) for r in _db(path).execute("SELECT * FROM traces ORDER BY started_at")]


def _spans(path: Path, trace_id: str) -> list[dict[str, Any]]:
    rows = _db(path).execute("SELECT * FROM spans WHERE trace_id = ? ORDER BY started_at, rowid", (trace_id,))
    return [{**dict(r), "content": json.loads(r["content"]), "attributes": json.loads(r["attributes"])}
            for r in rows]


# ---------------------------------------------------------------------------
# Recording an agent run
# ---------------------------------------------------------------------------


async def test_records_an_agent_run_with_model_and_tool_content(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    studio.enable(db, project="p")
    result = await _agent(_tool_turns(), tools=[lookup]).run("charged twice", user_id="alice", session_id="conv")
    assert result.run_id
    [trace] = _traces(db)
    assert trace["project"] == "p" and trace["kind"] == "agent" and trace["name"] == "billing"
    assert trace["status"] == "ok" and trace["run_id"] == result.run_id
    assert (trace["user_id"], trace["session_id"]) == ("alice", "conv")
    assert (trace["input"], trace["output"]) == ("charged twice", "Refunded.")
    assert (trace["tokens_in"], trace["tokens_out"]) == (250, 30)
    assert trace["cost_usd"] == pytest.approx(0.003)
    assert trace["agent_key"] == "billing" and trace["agent_version"] == 1
    spans = _spans(db, trace["trace_id"])
    kinds = [s["kind"] for s in spans]
    assert kinds.count("model") == 2 and kinds.count("tool") == 1 and kinds.count("turn") == 2
    first_model = next(s for s in spans if s["kind"] == "model")
    assert [m["role"] for m in first_model["content"]["messages"]] == ["system", "user"]
    assert first_model["content"]["tool_calls"] == [{"tool": "lookup", "args": {"order_id": "4471"}}]
    tool_span = next(s for s in spans if s["kind"] == "tool")
    assert tool_span["content"]["args"] == {"order_id": "4471"}
    assert tool_span["content"]["result"] == {"order": "4471", "charges": 2}
    # Parent links form one tree under the agent run.
    by_id = {s["span_id"]: s for s in spans}
    root = next(s for s in spans if s["parent_id"] is None)
    assert root["kind"] == "agent" and all(s["parent_id"] in by_id for s in spans if s is not root)


async def test_recording_doesnt_change_how_a_run_executes(tmp_path: Path) -> None:
    # A run() nobody streams must still use complete(), recorded or not.
    calls: list[str] = []

    class Spy(ScriptedModel):
        async def complete(self, messages: Any, **kwargs: Any) -> Any:
            calls.append("complete")
            return await super().complete(messages, **kwargs)

        async def stream(self, messages: Any, **kwargs: Any) -> Any:
            calls.append("stream")
            async for chunk in super().stream(messages, **kwargs):
                yield chunk

    studio.enable(tmp_path / "s.db")
    agent = Agent("x", model=Spy([ScriptedTurn(text="ok")]), auto_extract=False)
    assert (await agent.run("hi")).output == "ok"
    assert calls == ["complete"]


async def test_a_failed_run_raises_and_is_recorded_as_an_error(tmp_path: Path) -> None:
    class Boom(ScriptedModel):
        async def complete(self, messages: Any, **kwargs: Any) -> Any:
            raise RuntimeError("upstream overloaded")

    db = tmp_path / "s.db"
    studio.enable(db)
    with pytest.raises(RuntimeError, match="overloaded"):
        await Agent("x", model=Boom([]), auto_extract=False).run("hi")
    [trace] = _traces(db)
    assert trace["status"] == "error" and "upstream overloaded" in trace["error"]


async def test_score_attaches_to_the_run(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    studio.enable(db)
    result = await _agent([ScriptedTurn(text="ok")]).run("hi")
    studio.score(result, "resolved", True, comment="looks right", source="user")
    studio.score(result.run_id or "", "rating", 4)
    studio.score(result, "label", "polite")
    rows = [dict(r) for r in _db(db).execute("SELECT key, score, value, source, trace_id FROM scores")]
    [trace] = _traces(db)
    assert {r["trace_id"] for r in rows} == {trace["trace_id"]}
    assert [(r["key"], r["score"], r["value"]) for r in rows] == [
        ("resolved", 1.0, "yes"), ("rating", 4.0, None), ("label", None, "polite"),
    ]


async def test_score_without_studio_raises() -> None:
    with pytest.raises(RuntimeError, match="isn't recording"):
        studio.score("run_x", "k", 1)


async def test_your_own_telemetry_still_gets_spans_but_no_content(tmp_path: Path) -> None:
    telemetry = InMemoryTelemetry()
    studio.enable(tmp_path / "s.db")
    await _agent(_tool_turns(), tools=[lookup], telemetry=telemetry).run("secret prompt")
    names = [s.name for s in telemetry.spans()]
    assert {"loom.run", "loom.turn", "loom.model.complete", "loom.tool"} <= set(names)
    dumped = json.dumps([dict(s.attributes) for s in telemetry.spans()], default=str)
    assert "secret prompt" not in dumped and "messages" not in dumped


async def test_tool_spans_carry_the_permission_decision(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    telemetry = InMemoryTelemetry()

    @tool(destructive=True)
    def refund(charge_id: str) -> str:
        """Refund a charge."""
        return "done"

    async def approve(call: ToolCall, user_id: str | None) -> bool:
        return True

    studio.enable(db)
    agent = _agent(
        [ScriptedTurn(tool_calls=[ToolCall(tool="refund", args={"charge_id": "ch_1"}, destructive=True)]),
         ScriptedTurn(text="refunded")],
        tools=[refund], permissions=StandardPermissions(mode=Mode.DEFAULT), approval_handler=approve,
        telemetry=telemetry,
    )
    await agent.run("refund it")
    [trace] = _traces(db)
    tool_span = next(s for s in _spans(db, trace["trace_id"]) if s["kind"] == "tool")
    assert tool_span["attributes"]["loom.permission"] == "ask"
    assert tool_span["attributes"]["loom.approval"] == "approved"
    # The same plain attributes reach your own telemetry.
    [otel_tool] = [s for s in telemetry.spans() if s.name == "loom.tool"]
    assert otel_tool.attributes["loom.approval"] == "approved"


# ---------------------------------------------------------------------------
# Switching recording on and off
# ---------------------------------------------------------------------------


async def test_off_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert not studio.is_enabled()
    await _agent([ScriptedTurn(text="ok")]).run("hi")
    assert not (tmp_path / ".loomflow").exists()


async def test_environment_variable_turns_it_on_and_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "env.db"
    monkeypatch.setenv(studio.ENV_VAR, str(db))
    monkeypatch.setenv(studio.PROJECT_ENV_VAR, "from-env")
    await _agent([ScriptedTurn(text="ok")]).run("hi")
    assert [t["project"] for t in _traces(db)] == ["from-env"]
    monkeypatch.setenv(studio.ENV_VAR, "off")
    assert not studio.is_enabled()
    await _agent([ScriptedTurn(text="ok")]).run("hi")
    assert len(_traces(db)) == 1


async def test_default_path_and_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(studio.ENV_VAR, "1")
    await _agent([ScriptedTurn(text="ok")]).run("hi")
    db = tmp_path / ".loomflow" / "studio.db"
    assert [t["project"] for t in _traces(db)] == [tmp_path.name]


async def test_session_overrides_enable(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    studio.enable(db, project="base")
    with studio.session(enabled=False):
        await _agent([ScriptedTurn(text="ok")]).run("skipped")
    with studio.session(project="scoped", tags=["nightly"], metadata={"build": 7}):
        with studio.session(tags=["inner"]):
            await _agent([ScriptedTurn(text="ok")]).run("kept")
    [trace] = _traces(db)
    assert trace["input"] == "kept" and trace["project"] == "scoped"
    assert json.loads(trace["tags"]) == ["nightly", "inner"]
    assert json.loads(trace["metadata"]) == {"build": 7}


async def test_session_can_record_when_studio_is_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with studio.session(enabled=True):
        await _agent([ScriptedTurn(text="ok")]).run("hi")
    assert len(_traces(tmp_path / ".loomflow" / "studio.db")) == 1


async def test_sample_rate_zero_records_nothing_not_even_nested_runs(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    studio.enable(db, sample_rate=0.0)
    inner = _agent([ScriptedTurn(text="inner")], name="inner")

    @tool
    async def ask_inner(question: str) -> str:
        """Ask the inner agent."""
        return (await inner.run(question)).output

    outer = _agent([ScriptedTurn(tool_calls=[ToolCall(tool="ask_inner", args={"question": "q"})]),
                    ScriptedTurn(text="done")], name="outer", tools=[ask_inner])
    await outer.run("go")
    assert _traces(db) == []


def test_enable_rejects_a_bad_sample_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        studio.enable(sample_rate=1.5)


# ---------------------------------------------------------------------------
# Nesting
# ---------------------------------------------------------------------------


async def test_a_sub_agent_run_joins_its_callers_trace(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    studio.enable(db)
    inner = _agent([ScriptedTurn(text="inner answer")], name="researcher")

    @tool
    async def research(question: str) -> str:
        """Ask the researcher."""
        return (await inner.run(question)).output

    outer = _agent([ScriptedTurn(tool_calls=[ToolCall(tool="research", args={"question": "why"})]),
                    ScriptedTurn(text="final")], name="lead", tools=[research])
    await outer.run("go")
    [trace] = _traces(db)
    spans = _spans(db, trace["trace_id"])
    agents = [s for s in spans if s["kind"] == "agent"]
    assert [a["name"] for a in agents] == ["lead", "researcher"]
    tool_span = next(s for s in spans if s["kind"] == "tool")
    assert agents[1]["parent_id"] == tool_span["span_id"]
    keys = {r["agent_key"] for r in _db(db).execute("SELECT agent_key FROM agent_versions")}
    assert keys == {"lead", "researcher"}


async def test_a_workflow_is_one_trace_with_its_steps(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    studio.enable(db)
    classify = _agent([ScriptedTurn(text="billing")], name="classifier")
    answer = _agent([ScriptedTurn(text="answered")], name="answerer")
    result = await Workflow.chain([classify, answer], name="support").run("hello", user_id="bob")
    assert result.visited == ["classifier", "answerer"]  # agents' names name the nodes
    [trace] = _traces(db)
    assert (trace["kind"], trace["name"], trace["user_id"]) == ("workflow", "support", "bob")
    assert (trace["input"], trace["output"]) == ("hello", "answered")
    spans = _spans(db, trace["trace_id"])
    steps = [s for s in spans if s["kind"] == "step"]
    assert [(s["name"], s["content"]["output"]) for s in steps] == [("classifier", "billing"), ("answerer", "answered")]
    nested = [s for s in spans if s["kind"] == "agent"]
    assert [n["parent_id"] for n in nested] == [s["span_id"] for s in steps]


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------


async def test_a_changed_agent_is_a_new_version(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    studio.enable(db)
    await _agent([ScriptedTurn(text="a")]).run("1")
    await _agent([ScriptedTurn(text="b")]).run("2")
    await _agent([ScriptedTurn(text="c")], instructions="You handle billing. Refund once.", tools=[lookup]).run("3")
    assert [t["agent_version"] for t in _traces(db)] == [1, 1, 2]
    rows = [dict(r) for r in _db(db).execute("SELECT version, changes FROM agent_versions ORDER BY version")]
    assert rows[0]["changes"] == "First seen."
    assert "Tool added: lookup." in rows[1]["changes"] and "Instructions edited." in rows[1]["changes"]


async def test_unnamed_agents_are_keyed_by_their_first_line(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    studio.enable(db)
    await Agent("You are the triage agent.\nBe brief.", model=ScriptedModel([ScriptedTurn(text="x")]),
                auto_extract=False).run("hi")
    [trace] = _traces(db)
    assert trace["agent_key"].startswith("agent:") and trace["name"] == "You are the triage agent."


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


async def test_each_version_records_its_team_structure(tmp_path: Path) -> None:
    from loomflow.studio._store import StudioStore

    db = tmp_path / "s.db"
    studio.enable(db, project="arch")

    def team(*, refunds: bool, fraud_tools: tuple[Any, ...] = ()) -> Agent:
        workers = {"fraud": _agent([], name="fraud-analyst", tools=list(fraud_tools))}
        if refunds:
            workers["refunds"] = _agent([], name="refund-specialist")
        return _agent([ScriptedTurn(text="done")], name="escalations", architecture=Supervisor(workers=workers))

    await team(refunds=False).run("1")
    await team(refunds=True).run("2")
    await team(refunds=True, fraud_tools=(lookup,)).run("3")
    rows = [dict(r) for r in _db(db).execute(
        "SELECT version, changes FROM agent_versions WHERE agent_key = 'escalations' ORDER BY version")]
    assert [r["changes"] for r in rows] == ["First seen.", "Team or routing changed.", "A team member's tools changed."]
    structure = StudioStore(db, readonly=True).version_definition("arch", "escalations", 2)["structure"]  # type: ignore[index]
    nodes = {n["label"]: n for n in structure["nodes"]}
    assert nodes["escalations"]["detail"] == "supervisor"
    assert nodes["refund-specialist"]["keys"] == ["refund-specialist"]
    assert {e["label"] for e in structure["edges"]} >= {"delegate: fraud", "delegate: refunds"}


def _structure() -> dict[str, Any]:
    return {
        "nodes": [{"id": "a", "label": "lead <b>", "kind": "agent", "keys": ["lead"]},
                  {"id": "w", "label": "worker", "kind": "subagent", "keys": ["worker"]},
                  {"id": "t", "label": "lookup", "kind": "tool", "keys": ["lookup"]}],
        "edges": [{"source": "a", "target": "w", "label": "delegate: worker"},
                  {"source": "a", "target": "w", "label": "forward: worker", "style": "dashed"},
                  {"source": "w", "target": "t", "label": "uses"},
                  {"source": "w", "target": "a", "label": "critique"}],
        "subgraphs": [{"id": "g", "label": "workers", "members": ["w"]}],
    }


def test_the_diagram_folds_tools_merges_edges_and_draws_loops() -> None:
    from loomflow.studio._diagram import render

    out = render(_structure(), uid="x", title="lead")
    assert out.startswith('<div class="diagram-box"><svg') and "<script" not in out
    assert "lead &lt;b&gt;" in out and "<b>" not in out
    assert "tools: lookup" in out and out.count('<g class="dg-node') == 2  # the tool sits inside its agent
    assert ">delegate, forward<" in out  # two edges, one line; the target already names the worker
    assert 'class="dg-edge dashed"' in out  # the edge back up the graph
    assert "dg-on" not in out and "dg-off" not in out
    assert "No architecture" in render(None, uid="x", title="t")


def test_the_diagram_highlights_what_a_run_used() -> None:
    from loomflow.studio._diagram import render

    out = render(_structure(), uid="x", title="lead", active={"lead"})
    assert '<g class="dg-node dg-agent dg-on"><title>lead &lt;b&gt;</title>' in out
    assert '<g class="dg-node dg-subagent dg-off"><title>worker</title>' in out
    assert "lead &lt;b&gt; (used in this run)" in out


def test_a_swarm_becomes_a_note_on_its_group() -> None:
    from loomflow.studio._diagram import render

    peers = [{"id": p, "label": p, "kind": "subagent"} for p in "abc"]
    handoffs = [{"source": s, "target": t, "label": "handoff", "style": "dashed"}
                for s in "abc" for t in "abc" if s != t]
    out = render({"nodes": peers, "edges": handoffs, "subgraphs": [{"id": "g", "label": "swarm", "members": list("abc")}]},
                 uid="s", title="swarm")
    assert "swarm (any can hand off to any other)" in out and "dg-edge" not in out


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------


async def test_capture_content_off_keeps_only_structure(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    studio.enable(db, capture_content=False)
    await _agent(_tool_turns("the answer"), tools=[lookup]).run("my secret question")
    [trace] = _traces(db)
    assert trace["input"] is None and trace["output"] is None and trace["tokens_in"] == 250
    # Content and attributes only: span ids are random hex and may
    # contain "4471" by chance.
    dumped = json.dumps([(s["content"], s["attributes"]) for s in _spans(db, trace["trace_id"])])
    assert "secret question" not in dumped and "the answer" not in dumped and '"4471"' not in dumped


async def test_redact_scrubs_text_before_it_is_written(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    studio.enable(db, redact="pii")
    await _agent([ScriptedTurn(text="Mail me at jo@example.com")]).run("card 4242 4242 4242 4242")
    [trace] = _traces(db)
    wal = db.with_name(db.name + "-wal")
    raw = db.read_bytes() + (wal.read_bytes() if wal.exists() else b"")
    assert "4242 4242" not in trace["input"] and "jo@example.com" not in trace["output"]
    dumped = json.dumps(_spans(db, trace["trace_id"]))
    assert "jo@example.com" not in dumped and "4242 4242 4242 4242" not in dumped
    assert b"jo@example.com" not in raw


async def test_redact_takes_a_function(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    studio.enable(db, redact=lambda text: text.replace("hunter2", "***"))
    await _agent([ScriptedTurn(text="ok")]).run("password hunter2")
    assert _traces(db)[0]["input"] == "password ***"


def test_enable_rejects_an_unknown_redactor() -> None:
    with pytest.raises(ValueError, match="redact"):
        studio.enable(redact="everything")


# ---------------------------------------------------------------------------
# Experiments and scores
# ---------------------------------------------------------------------------


def _cases() -> list[Case]:
    return [Case(id="a", input="2+2", expected="4"), Case(id="b", input="3+3", expected="6")]


async def test_an_eval_run_is_an_experiment(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    studio.enable(db)
    agent = _agent([ScriptedTurn(text="4"), ScriptedTurn(text="7")], name="calc")
    report = await EvalHarness(agent, [Contains()], concurrency=1).run(Dataset(_cases(), name="math"),
                                                                    experiment="calc v1")
    conn = _db(db)
    [exp] = [dict(r) for r in conn.execute("SELECT * FROM experiments")]
    assert (exp["name"], exp["dataset"], exp["agent_key"], exp["agent_version"]) == ("calc v1", "math", "calc", 1)
    assert json.loads(exp["summary"])["metrics"]["contains"]["mean"] == report.mean("contains")
    cases = [dict(r) for r in conn.execute("SELECT * FROM experiment_cases ORDER BY position")]
    assert [(c["case_key"], json.loads(c["scores"])) for c in cases] == [("a", {"contains": 1.0}), ("b", {"contains": 0.0})]
    assert all(c["trace_id"] for c in cases)
    traces = _traces(db)
    assert {t["experiment_id"] for t in traces} == {exp["id"]} and {t["case_key"] for t in traces} == {"a", "b"}
    assert conn.execute("SELECT COUNT(*) FROM scores WHERE source = 'eval'").fetchone()[0] == 2


def test_cases_without_ids_are_matched_by_content() -> None:
    from loomflow.studio._hooks import case_key

    first, second = Case(input="q", expected="a"), Case(input="q", expected="a")
    assert first.id != second.id and case_key(first) == case_key(second)
    assert case_key(Case(id="mine", input="q")) == "mine"


async def test_the_online_scorer_records_scores(tmp_path: Path) -> None:
    from loomflow.eval.harness import OnlineScorer

    db = tmp_path / "s.db"
    studio.enable(db)

    class Length:
        name = "length"

        async def score(self, case: Any, result: Any, events: Any) -> float:
            return float(len(result.output))

    result = await _agent([ScriptedTurn(text="four")]).run("hi")
    await OnlineScorer([Length()], sample_rate=1.0).maybe_score("hi", result, [])
    rows = [dict(r) for r in _db(db).execute("SELECT key, score, source, trace_id FROM scores")]
    assert rows[0]["key"] == "length" and rows[0]["score"] == 4.0 and rows[0]["source"] == "online"
    assert rows[0]["trace_id"] == _traces(db)[0]["trace_id"]


async def test_aclose_waits_for_studio_writes(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    studio.enable(db)
    agent = _agent([ScriptedTurn(text="ok")])
    await agent.run("hi")
    await agent.aclose()
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM traces WHERE status = 'ok'").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# The web app
# ---------------------------------------------------------------------------


async def _get(
    app: Any,
    path: str,
    method: str = "GET",
    *,
    form: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
) -> tuple[int, dict[str, str], str]:
    raw_path, _, query = path.partition("?")
    sent_headers = dict(headers or {})
    if form is not None:
        body = urlencode(form, doseq=True).encode()
        sent_headers.setdefault("content-type", "application/x-www-form-urlencoded")
    scope = {"type": "http", "method": method, "path": raw_path, "query_string": query.encode(),
             "headers": [(k.encode(), v.encode()) for k, v in sent_headers.items()]}
    sent: list[dict[str, Any]] = []
    messages = [{"type": "http.request", "body": body or b"", "more_body": False}]

    async def receive() -> dict[str, Any]:
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(scope, receive, send)
    start = sent[0]
    headers = {k.decode(): v.decode() for k, v in start["headers"]}
    body = b"".join(m.get("body", b"") for m in sent[1:]).decode()
    return start["status"], headers, body


async def _recorded_db(tmp_path: Path) -> tuple[Path, str]:
    db = tmp_path / "s.db"
    studio.enable(db, project="web")
    result = await _agent(_tool_turns(), tools=[lookup]).run("<script>alert(1)</script> charged twice", user_id="alice")
    studio.score(result, "resolved", True)
    agent = _agent([ScriptedTurn(text="4"), ScriptedTurn(text="7")], name="calc")
    await EvalHarness(agent, [Contains()], concurrency=1).run(Dataset(_cases(), name="math"), experiment="one")
    agent = _agent([ScriptedTurn(text="5"), ScriptedTurn(text="6")], name="calc")
    await EvalHarness(agent, [Contains()], concurrency=1).run(Dataset(_cases(), name="math"), experiment="two")
    studio.flush()
    return db, _traces(db)[0]["trace_id"]


async def test_the_web_app_pages(tmp_path: Path) -> None:
    from loomflow.studio.web import create_app

    db, trace_id = await _recorded_db(tmp_path)
    app = create_app(db)
    status, headers, body = await _get(app, "/traces")
    assert status == 200 and "billing" in body and "resolved: yes" in body
    assert "script-src" not in headers["content-security-policy"] and "default-src 'none'" in headers["content-security-policy"]
    assert "<script>alert(1)</script>" not in body and "&lt;script&gt;" in body
    status, _, body = await _get(app, f"/traces/{trace_id}")
    assert status == 200 and "Timeline" in body and "lookup" in body and "&lt;script&gt;" in body
    spans = _spans(db, trace_id)
    tool_id = next(s["span_id"] for s in spans if s["kind"] == "tool")
    status, _, body = await _get(app, f"/traces/{trace_id}?span={tool_id}")
    assert status == 200 and "Arguments" in body and "4471" in body
    status, headers, body = await _get(app, f"/traces/{trace_id}.json")
    assert status == 200 and headers["content-type"] == "application/json" and json.loads(body)["trace_id"] == trace_id
    for path in ("/agents", "/agents/billing?project=web", "/experiments", "/traces?problems=1&q=twice&range=24h",
                 "/health", "/traces?agent=billing&version=1"):
        status, _, _ = await _get(app, path)
        assert status == 200, path
    exps = [dict(r) for r in _db(db).execute("SELECT id, name FROM experiments ORDER BY started_at")]
    status, _, body = await _get(app, f"/experiments/compare?a={exps[0]['id']}&b={exps[1]['id']}&show=regressed")
    assert status == 200 and "1 improved, 1 regressed" in body
    status, _, body = await _get(app, f"/experiments/{exps[1]['id']}")
    assert status == 200 and "two" in body


async def test_the_pages_draw_the_architecture(tmp_path: Path) -> None:
    from loomflow.studio.web import create_app

    db = tmp_path / "s.db"
    studio.enable(db, project="web")
    await _agent(_tool_turns(), tools=[lookup]).run("charged twice")
    await _agent([ScriptedTurn(text="ok")], instructions="You handle billing. Be brief.").run("hi")

    async def classify(text: str) -> str:
        return "billing"

    router = Workflow.route(classify, {"billing": _agent([ScriptedTurn(text="paid")]),
                                       "tech": _agent([], name="tech")}, name="router")
    await router.run("invoice?")
    app = create_app(db)
    status, _, body = await _get(app, "/agents/billing?project=web")
    assert status == 200 and "Architecture of billing v3" in body and "<svg" in body and "tools: lookup" not in body
    status, _, body = await _get(app, "/agents/billing?project=web&version=1")
    assert "Architecture of billing v1" in body and "tools: lookup" in body
    first, *_, last = _traces(db)
    status, _, body = await _get(app, f"/traces/{first['trace_id']}")
    assert status == 200 and "Highlighted: the agents" in body and "dg-on" in body
    status, _, body = await _get(app, f"/traces/{last['trace_id']}")
    assert '<g class="dg-node dg-subagent dg-on"><title>billing</title>' in body
    assert '<g class="dg-node dg-subagent dg-off"><title>tech</title>' in body
    status, _, body = await _get(app, "/agents/workflow%3Arouter?project=web")
    assert status == 200 and "Architecture of router v1" in body and ">start<" in body


async def test_the_web_app_without_a_database(tmp_path: Path) -> None:
    from loomflow.studio.web import create_app

    app = create_app(tmp_path / "missing.db")
    status, _, body = await _get(app, "/traces")
    assert status == 200 and "No runs recorded yet" in body
    assert not (tmp_path / "missing.db").exists()


async def test_the_web_app_is_read_only_and_404s(tmp_path: Path) -> None:
    from loomflow.studio.web import create_app

    db, trace_id = await _recorded_db(tmp_path)
    app = create_app(db, read_only=True)
    assert (await _get(app, "/traces", method="POST"))[0] == 405
    status, _, body = await _get(app, f"/traces/{trace_id}")
    assert status == 200 and 'name="token"' not in body and "Add to a dataset" not in body
    assert (await _get(app, "/traces", method="DELETE"))[0] == 405
    assert (await _get(create_app(db), "/traces", method="PUT"))[0] == 405
    assert (await _get(app, "/traces/nope"))[0] == 404
    assert (await _get(app, "/nowhere"))[0] == 404
    assert (await _get(app, "/"))[0] == 302


# ---------------------------------------------------------------------------
# Search and cleanup
# ---------------------------------------------------------------------------


def _store(db: Path) -> Any:
    from loomflow.studio._store import StudioStore

    assert studio.flush()
    return StudioStore(db, readonly=True)


async def test_search_uses_the_full_text_index(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    studio.enable(db)
    first = await _agent(_tool_turns("Refunded the duplicate."), tools=[lookup]).run("Charged twice?", user_id="ann")
    await _agent([ScriptedTurn(text="Restart the router.")], name="tech").run("Wi-Fi is down")
    store = _store(db)
    assert _db(db).execute("SELECT value FROM meta WHERE key = 'search'").fetchone()[0] == "fts5"

    def found(text: str) -> list[str]:
        return [t["name"] for t in store.list_traces(search=text)["traces"]]

    assert found("4471") == ["billing"]  # a tool argument
    assert found("refund") == ["billing"]  # the start of a word
    assert found("charges 2") == ["billing"]  # words anywhere in the trace, here a tool result
    assert found("router") == ["tech"]
    assert found(first.run_id or "") == ["billing"]
    assert found("nothing like this") == []
    assert found("Wi-Fi") == ["tech"]


async def test_search_scans_when_there_is_no_index(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    studio.enable(db)
    await _agent(_tool_turns(), tools=[lookup]).run("Charged twice?")
    store = _store(db)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE meta SET value = 'off' WHERE key = 'search'")
    assert [t["name"] for t in store.list_traces(search="447")["traces"]] == ["billing"]  # any part of a word


async def test_a_file_without_an_index_gets_one(tmp_path: Path) -> None:
    from loomflow.studio._store import StudioStore

    db = tmp_path / "s.db"
    studio.enable(db)
    await _agent(_tool_turns(), tools=[lookup]).run("Charged twice?")
    assert studio.flush()
    for recorder in _state.all_recorders():
        recorder.close()
    _state._recorders.clear()
    with sqlite3.connect(db) as conn:  # as a file 0.18.0 wrote
        conn.execute("DROP TABLE search")
        conn.execute("DROP TABLE search_docs")
        conn.execute("DELETE FROM meta WHERE key = 'search'")
    conn = StudioStore(db).writer()
    assert StudioStore.prepare_search(conn)
    conn.close()
    assert [t["name"] for t in _store(db).list_traces(search="4471")["traces"]] == ["billing"]


async def test_old_runs_are_deleted_and_eval_runs_kept(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    studio.enable(db, project="p")
    old = await _agent(_tool_turns(), tools=[lookup]).run("old run")
    studio.score(old, "resolved", True)
    await _agent([ScriptedTurn(text="new")]).run("new run")
    await EvalHarness(_agent([ScriptedTurn(text="4"), ScriptedTurn(text="7")], name="calc"), [Contains()],
                      concurrency=1).run(Dataset(_cases(), name="math"))
    assert studio.flush()
    week_ago = time.time() - 7 * 86400
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE traces SET started_at = ? WHERE input = 'old run' OR experiment_id IS NOT NULL",
                     (week_ago,))
    assert studio.prune(3, path=db) == 1
    rows = _db(db).execute("SELECT input, experiment_id FROM traces").fetchall()
    assert sorted(r["input"] for r in rows if r["experiment_id"] is None) == ["new run"]
    assert sum(r["experiment_id"] is not None for r in rows) == 2  # eval runs stay
    assert _db(db).execute("SELECT COUNT(*) FROM scores WHERE key = 'resolved'").fetchone()[0] == 0
    assert _db(db).execute(
        "SELECT COUNT(*) FROM spans WHERE trace_id NOT IN (SELECT trace_id FROM traces)").fetchone()[0] == 0
    assert _store(db).list_traces(search="old")["total"] == 0
    assert studio.prune(3, path=tmp_path / "missing.db") == 0


async def test_keep_days_deletes_old_runs_as_it_records(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    studio.enable(db)
    await _agent([ScriptedTurn(text="a")]).run("old run")
    assert studio.flush()
    for recorder in _state.all_recorders():
        recorder.close()
    _state._recorders.clear()
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE traces SET started_at = ?", (time.time() - 40 * 86400,))
    studio.enable(db, keep_days=30)
    await _agent([ScriptedTurn(text="b")]).run("new run")
    assert [t["input"] for t in _traces(db)] == ["new run"]
    assert _db(db).execute("PRAGMA auto_vacuum").fetchone()[0] == 2  # incremental: space goes back


def test_keep_days_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for bad in (0, -1, True, "7"):
        with pytest.raises(ValueError, match="keep_days"):
            studio.enable(tmp_path / "s.db", keep_days=bad)  # type: ignore[arg-type]
    monkeypatch.setenv(studio.KEEP_DAYS_ENV_VAR, "14")
    studio.enable(tmp_path / "s.db")
    assert _state.get_global().keep_days == 14.0  # type: ignore[union-attr]
    monkeypatch.setenv(studio.ENV_VAR, "1")
    assert _state.settings_from_env().keep_days == 14.0  # type: ignore[union-attr]
    monkeypatch.setenv(studio.KEEP_DAYS_ENV_VAR, "soon")
    with pytest.raises(ValueError, match=studio.KEEP_DAYS_ENV_VAR):
        _state.settings_from_env()


# ---------------------------------------------------------------------------
# Scoring and datasets from the web app
# ---------------------------------------------------------------------------


def _token(body: str) -> str:
    match = re.search(r'name="token" value="([^"]+)"', body)
    assert match, "no form on the page"
    return match[1]


async def test_scoring_a_run_from_its_page(tmp_path: Path) -> None:
    from loomflow.studio.web import create_app

    db, trace_id = await _recorded_db(tmp_path)
    app = create_app(db)
    _, _, body = await _get(app, f"/traces/{trace_id}")
    token = _token(body)
    status, headers, _ = await _get(app, f"/traces/{trace_id}/scores", "POST",
                                    form={"token": token, "key": "review", "verdict": "bad", "comment": "too long"})
    assert status == 303 and headers["location"] == f"/traces/{trace_id}?saved=score#scores"
    await _get(app, f"/traces/{trace_id}/scores", "POST", form={"token": token, "key": "stars", "value": "4"})
    await _get(app, f"/traces/{trace_id}/scores", "POST", form={"token": token, "key": "tone", "value": "curt"})
    rows = {r["key"]: dict(r) for r in _db(db).execute("SELECT * FROM scores WHERE source = 'ui'")}
    assert (rows["review"]["score"], rows["review"]["value"], rows["review"]["comment"]) == (0.0, "no", "too long")
    assert (rows["stars"]["score"], rows["tone"]["value"]) == (4.0, "curt")
    status, _, body = await _get(app, f"/traces/{trace_id}?saved=score")
    assert "Score saved." in body and "too long" in body and "Remove" in body
    assert '<option value="resolved">' in body  # keys already in use are suggested
    status, headers, _ = await _get(app, f"/scores/{rows['review']['id']}/delete", "POST", form={"token": token})
    assert status == 303 and headers["location"] == f"/traces/{trace_id}#scores"
    assert _db(db).execute("SELECT COUNT(*) FROM scores WHERE key = 'review'").fetchone()[0] == 0
    status, _, body = await _get(app, f"/traces/{trace_id}/scores", "POST", form={"token": token, "key": "x"})
    assert status == 400 and "Enter a value" in body
    status, _, body = await _get(app, "/traces?unscored=1")
    assert status == 200 and "charged twice" not in body  # it has scores


async def test_forms_from_elsewhere_are_refused(tmp_path: Path) -> None:
    from loomflow.studio.web import create_app

    db, trace_id = await _recorded_db(tmp_path)
    app = create_app(db)
    _, _, body = await _get(app, f"/traces/{trace_id}")
    form = {"token": _token(body), "key": "review", "verdict": "good"}
    path = f"/traces/{trace_id}/scores"
    assert (await _get(app, path, "POST", form={**form, "token": "guess"}))[0] == 403
    assert (await _get(create_app(db), path, "POST", form=form))[0] == 403  # another app's token
    assert (await _get(app, path, "POST", form=form,
                       headers={"host": "127.0.0.1:7600", "origin": "http://evil.example"}))[0] == 403
    assert (await _get(app, path, "POST", form=form, headers={"sec-fetch-site": "cross-site"}))[0] == 403
    assert (await _get(app, path, "POST", body=json.dumps(form).encode(),
                       headers={"content-type": "application/json"}))[0] == 415
    assert (await _get(app, path, "POST", form={**form, "comment": "x" * 1_100_000}))[0] == 413
    # A name pointed at 127.0.0.1 by another site can't read it either.
    assert (await _get(app, "/traces", headers={"host": "evil.example:7600"}))[0] == 421
    for host in ("127.0.0.1:7600", "localhost:7600", "[::1]:7600", "studio.localhost"):
        assert (await _get(app, "/traces", headers={"host": host}))[0] == 200, host
    assert (await _get(create_app(db, allowed_hosts=["studio.internal"]), "/traces",
                       headers={"host": "studio.internal"}))[0] == 200
    assert (await _get(create_app(db, allowed_hosts=["*"]), "/traces", headers={"host": "anything"}))[0] == 200
    ok = await _get(app, path, "POST", form=form, headers={"host": "127.0.0.1:7600", "origin": "http://127.0.0.1:7600",
                                                            "sec-fetch-site": "same-origin"})
    assert ok[0] == 303
    assert _db(db).execute("SELECT COUNT(*) FROM scores WHERE source = 'ui'").fetchone()[0] == 1


async def test_building_a_dataset_from_runs(tmp_path: Path) -> None:
    from loomflow.studio.web import create_app

    db, trace_id = await _recorded_db(tmp_path)
    app = create_app(db)
    _, _, body = await _get(app, f"/traces/{trace_id}")
    token = _token(body)
    assert "Add to a dataset" in body and "charged twice</textarea>" in body  # the input, ready to edit
    status, headers, _ = await _get(app, f"/traces/{trace_id}/datasets", "POST", form={
        "token": token, "dataset": "billing regressions", "input": "Was I charged twice?",
        "expected": "Yes; refunded"})
    assert status == 303
    dataset_id = headers["location"].split("added=")[1].split("#")[0]
    _, _, body = await _get(app, headers["location"].split("#")[0])
    assert "Added to" in body and "billing regressions" in body and "1 case." in body
    status, _, body = await _get(app, "/datasets")
    assert status == 200 and "billing regressions" in body
    status, _, body = await _get(app, f"/datasets/{dataset_id}")
    assert status == 200 and "Was I charged twice?" in body and 'loomflow.studio.dataset(&quot;billing regressions&quot;)' in body
    status, headers, body = await _get(app, f"/datasets/{dataset_id}.jsonl")
    assert status == 200 and 'filename="billing_regressions.jsonl"' in headers["content-disposition"]
    (tmp_path / "cases.jsonl").write_text(body)
    from_file = Dataset.from_jsonl(tmp_path / "cases.jsonl")
    loaded = studio.dataset("billing regressions", path=db)
    assert [c.id for c in loaded] == [c.id for c in from_file] == [studio.dataset("billing regressions", path=db)[0].id]
    assert (loaded[0].input, loaded[0].expected, loaded.name) == ("Was I charged twice?", "Yes; refunded",
                                                                  "billing regressions")
    assert loaded[0].metadata == {"studio_trace": trace_id}
    await EvalHarness(_agent([ScriptedTurn(text="Yes; refunded")]), [Contains()]).run(loaded, experiment="check")
    assert studio.flush()
    exp = _db(db).execute("SELECT dataset FROM experiments WHERE name = 'check'").fetchone()
    assert exp["dataset"] == "billing regressions"
    case_id = loaded[0].id
    status, headers, _ = await _get(app, f"/datasets/{dataset_id}/cases/{case_id}/delete", "POST",
                                    form={"token": token})
    assert status == 303 and len(studio.dataset("billing regressions", path=db)) == 0
    with pytest.raises(LookupError, match="no Studio dataset"):
        studio.dataset("nope", path=db)
    status, _, body = await _get(app, f"/traces/{trace_id}/datasets", "POST", form={"token": token, "dataset": ""})
    assert status == 400


async def test_pages_work_on_a_file_from_0_18_0(tmp_path: Path) -> None:
    from loomflow.studio.web import create_app

    db, trace_id = await _recorded_db(tmp_path)
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE dataset_cases")
        conn.execute("DROP TABLE datasets")
    app = create_app(db, read_only=True)
    for path in ("/datasets", f"/traces/{trace_id}", "/traces?q=twice"):
        assert (await _get(app, path))[0] == 200, path
    status, _, body = await _get(create_app(db), f"/traces/{trace_id}")
    token = _token(body)
    status, _, _ = await _get(create_app(db), f"/traces/{trace_id}/datasets", "POST",
                              form={"token": token, "dataset": "d", "input": "x"})
    assert status == 403  # a different app instance's token
    app = create_app(db)
    _, _, body = await _get(app, f"/traces/{trace_id}")
    status, _, _ = await _get(app, f"/traces/{trace_id}/datasets", "POST",
                              form={"token": _token(body), "dataset": "d", "input": "x"})
    assert status == 303  # the first write adds the new tables


# ---------------------------------------------------------------------------
# Agent names, the CLI
# ---------------------------------------------------------------------------


def test_agent_names() -> None:
    assert Agent("x", model="echo", name="helper").name == "helper"
    assert Agent("x", model="echo").name is None
    with pytest.raises(ValueError, match="name"):
        Agent("x", model="echo", name="  ")


def test_agent_name_from_config() -> None:
    assert Agent.from_dict({"instructions": "x", "model": "echo", "name": "cfg-agent"}).name == "cfg-agent"


def test_cli_studio_and_eval_flags() -> None:
    from loomflow.cli import _build_parser, _studio_hosts

    args = _build_parser().parse_args(["studio", "--db", "x.db", "--port", "7700"])
    assert (args.db, args.port, args.host, args.read_only, args.allow_host) == ("x.db", 7700, "127.0.0.1", False, [])
    args = _build_parser().parse_args(["studio", "--read-only", "--allow-host", "studio.internal"])
    assert args.read_only and args.allow_host == ["studio.internal"]
    assert _studio_hosts("127.0.0.1", ["studio.internal"]) == ["localhost", "127.0.0.1", "::1", "studio.internal"]
    assert _studio_hosts("10.0.0.5", []) == ["localhost", "127.0.0.1", "::1", "10.0.0.5"]
    assert _studio_hosts("0.0.0.0", []) == ["*"]
    args = _build_parser().parse_args(["eval", "cases.jsonl", "--agent", "m:a", "--studio"])
    assert args.studio == ""
    args = _build_parser().parse_args(["eval", "cases.jsonl", "--agent", "m:a"])
    assert args.studio is None


async def test_a_case_that_raises_still_links_to_its_trace(tmp_path: Path) -> None:
    class Boom(ScriptedModel):
        # The harness streams (it reads the events), so stream() fails too.
        async def complete(self, messages: Any, **kwargs: Any) -> Any:
            raise RuntimeError("model down")

        async def stream(self, messages: Any, **kwargs: Any) -> Any:
            raise RuntimeError("model down")
            yield  # pragma: no cover

    db = tmp_path / "s.db"
    studio.enable(db)
    agent = Agent("x", name="flaky", model=Boom([]), auto_extract=False)
    report = await EvalHarness(agent, [Contains()]).run(Dataset(_cases()[:1], name="math"))
    assert report.results[0].error
    [case] = [dict(r) for r in _db(db).execute("SELECT * FROM experiment_cases")]
    assert "model down" in case["error"]
    assert case["trace_id"] == _traces(db)[0]["trace_id"]


def test_several_processes_can_record_to_one_new_file(tmp_path: Path) -> None:
    import subprocess
    import sys

    db = tmp_path / "shared.db"
    script = (
        "import sys, anyio, loomflow.studio as studio\n"
        "from loomflow import Agent\n"
        "from loomflow.model.scripted import ScriptedModel, ScriptedTurn\n"
        "async def main():\n"
        f"    studio.enable({str(db)!r})\n"
        "    for i in range(8):\n"
        "        agent = Agent('x', name=sys.argv[1], auto_extract=False,\n"
        "                      model=ScriptedModel([ScriptedTurn(text='ok')]))\n"
        "        await agent.run(str(i))\n"
        "anyio.run(main)\n"
    )
    env = {k: v for k, v in __import__("os").environ.items() if k != studio.ENV_VAR}
    procs = [subprocess.Popen([sys.executable, "-c", script, f"p{i}"], env=env, stderr=subprocess.PIPE)
             for i in range(4)]
    errors = [p.communicate(timeout=60)[1].decode() for p in procs]
    assert all(p.returncode == 0 for p in procs), errors
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*), COUNT(DISTINCT name) FROM traces WHERE status = 'ok'").fetchone() == (32, 4)
