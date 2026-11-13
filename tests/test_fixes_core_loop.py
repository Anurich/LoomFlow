"""Regression tests for the core-loop review fixes (0.20.x).

Each test pins one behaviour that was wrong before:

1.  Truncated / malformed tool-call JSON no longer becomes ``{}`` —
    the tool is NOT run and the model gets a tool error.
2.  Tool-call arguments that are JSON but not an object are a tool
    error, not a crash of ``run()``.
3.  Tool arguments are validated against the annotations after the
    lenient coercion; a bad value never reaches the function.
4.  ``resume()`` after ``max_turns_exceeded`` gets a fresh allowance.
5.  The output-schema validation retry goes through the budget gate,
    the rate limiter and budget accounting.
6.  Auto-compact / tool-result summariser usage is counted.
7.  A failing tool dispatch raises the underlying exception (no bare
    ``ExceptionGroup``); legacy-signature probes inspect instead of
    calling twice.
8.  ``RunResult.finish_reason`` exposes truncation / content filters.
9.  Stream-only models work under ``RetryingModel`` / ``FallbackModel``.
10. Tool timeout is configurable; delegation is exempt from it, keeps
    a worker's spend when it fails, and is depth-limited.
11. A timed-out sync tool's error says it is still running.
12. Destructive tool calls run sequentially; ``parallel_tool_calls``.
13. ``Agent(timeout=)`` only converts its own deadline to RunTimeout.
14. Tool results render as JSON; errors as ``ERROR: <Type>: <msg>``.
15. Duplicate tool names raise ``ConfigError``.
16. A ``max_turns`` exit returns the last assistant text and does not
    persist an empty episode.
17. A ``@tool`` function is still directly callable.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from types import SimpleNamespace as NS
from typing import Any

import anyio
import pytest
from pydantic import BaseModel

from loomflow import Agent, InMemoryMemory, ScriptedModel, ScriptedTurn, tool
from loomflow.agent.api import Tuning
from loomflow.architecture import Supervisor
from loomflow.architecture.helpers import (
    SubagentInvocation,
    accepts_kwarg,
    delegation_depth,
    render_tool_output,
)
from loomflow.core.errors import ConfigError, OutputValidationError, RunTimeout
from loomflow.core.types import (
    BudgetStatus,
    Message,
    ModelChunk,
    PermissionDecision,
    Role,
    ToolCall,
    Usage,
)
from loomflow.governance.budget import BudgetConfig, StandardBudget
from loomflow.governance.retry import RetryPolicy
from loomflow.model._toolargs import parse_tool_args
from loomflow.model.anthropic import AnthropicModel
from loomflow.model.fallback import FallbackModel
from loomflow.model.openai import OpenAIModel
from loomflow.model.retrying import RetryingModel
from loomflow.security.audit import InMemoryAuditLog
from loomflow.tools.registry import InProcessToolHost, ToolArgumentError

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeOpenAI:
    """``chat.completions.create`` → canned non-streaming responses."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.seen: list[dict[str, Any]] = []
        self.chat = NS(completions=NS(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        self.seen.append(kwargs)
        return self.responses.pop(0)


def _oai(
    text: str | None = None,
    calls: tuple[tuple[str, str], ...] = (),
    finish: str = "stop",
) -> Any:
    tool_calls = [
        NS(id=f"call_{i}", function=NS(name=name, arguments=raw))
        for i, (name, raw) in enumerate(calls)
    ]
    message = NS(content=text, tool_calls=tool_calls or None)
    return NS(
        choices=[NS(message=message, finish_reason=finish)],
        usage=NS(prompt_tokens=10, completion_tokens=5, prompt_tokens_details=None),
    )


def _tool_messages(client: _FakeOpenAI, request: int = 1) -> list[str]:
    return [
        m["content"] for m in client.seen[request]["messages"] if m["role"] == "tool"
    ]


class _SpyScripted(ScriptedModel):
    """ScriptedModel that records the messages + kwargs of every call."""

    def __init__(self, turns: list[ScriptedTurn]) -> None:
        super().__init__(turns)
        self.calls: list[tuple[list[Message], dict[str, Any]]] = []

    async def complete(self, messages: list[Message], **kwargs: Any) -> Any:
        self.calls.append((list(messages), kwargs))
        return await super().complete(messages, **kwargs)

    def stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[ModelChunk]:
        self.calls.append((list(messages), kwargs))
        return super().stream(messages, **kwargs)

    def tool_messages(self, call: int = 1) -> list[str]:
        return [m.content for m in self.calls[call][0] if m.role == Role.TOOL]


class _StreamOnly:
    """A protocol-conformant model with no ``complete``."""

    name = "stream-only"

    def __init__(self, text: str = "hi") -> None:
        self._text = text

    async def stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[ModelChunk]:
        yield ModelChunk(kind="text", text=self._text)
        yield ModelChunk(
            kind="finish", finish_reason="stop", usage=Usage(input_tokens=3, output_tokens=1)
        )


@dataclass
class _RecordingBudget:
    """A current-protocol budget that records what it is charged."""

    cost: float = 0.0
    tokens: int = 0
    consume_calls: int = 0

    async def allows_step(self, *, user_id: str | None = None) -> BudgetStatus:
        return BudgetStatus.ok_()

    async def consume(
        self,
        *,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
        user_id: str | None = None,
    ) -> None:
        self.consume_calls += 1
        self.tokens += tokens_in + tokens_out
        self.cost += cost_usd


# ---------------------------------------------------------------------------
# 1. Malformed / truncated tool-call JSON
# ---------------------------------------------------------------------------


async def test_truncated_tool_call_json_does_not_run_the_tool() -> None:
    ran: list[tuple[str, bool]] = []

    @tool(destructive=True)
    def wipe(path: str = "/", recursive: bool = True) -> str:
        "Delete a path."
        ran.append((path, recursive))
        return f"wiped {path}"

    client = _FakeOpenAI(
        [
            _oai(calls=(("wipe", '{"path": "/tmp/scratch/only-this", "recur'),), finish="length"),
            _oai(text="done"),
        ]
    )
    agent = Agent("x", model=OpenAIModel("gpt-4o", client=client), tools=[wipe])
    result = await agent.run("clean up the scratch dir")

    assert ran == []  # NOT run with its defaults (path="/")
    (msg,) = _tool_messages(client)
    assert msg.startswith("ERROR:")
    assert "not valid JSON" in msg
    assert "cut off" in msg and "length" in msg
    assert "NOT run" in msg
    assert result.output == "done"


async def test_malformed_tool_call_json_without_truncation_has_no_cutoff_note() -> None:
    ran: list[int] = []

    @tool
    def ping(n: int = 1) -> str:
        "Ping."
        ran.append(n)
        return "pong"

    client = _FakeOpenAI([_oai(calls=(("ping", "{n: 1}"),), finish="tool_calls"), _oai(text="ok")])
    await Agent("x", model=OpenAIModel("gpt-4o", client=client), tools=[ping]).run("q")
    assert ran == []
    (msg,) = _tool_messages(client)
    assert "not valid JSON" in msg and "cut off" not in msg


async def test_args_error_call_is_not_run_on_the_streaming_path() -> None:
    ran: list[str] = []

    @tool
    def act(x: str = "default") -> str:
        "Act."
        ran.append(x)
        return x

    model = _SpyScripted(
        [
            ScriptedTurn(tool_calls=[ToolCall(tool="act", args_error="arguments were not valid JSON")]),
            ScriptedTurn(text="done"),
        ]
    )
    events = [ev async for ev in Agent("x", model=model, tools=[act]).stream("q")]
    assert ran == []
    results = [e.payload["result"] for e in events if e.kind.value == "tool_result"]
    assert results[0]["ok"] is False and "not valid JSON" in results[0]["error"]


async def test_anthropic_stream_flags_truncated_tool_input() -> None:
    events = [
        NS(type="message_start", message=NS(usage=NS(input_tokens=5, output_tokens=0))),
        NS(type="content_block_start", index=0, content_block=NS(type="tool_use", id="tu_1", name="wipe")),
        NS(type="content_block_delta", index=0, delta=NS(type="input_json_delta", partial_json='{"path": "/tm')),
        NS(type="content_block_stop", index=0),
        NS(type="message_delta", delta=NS(stop_reason="max_tokens"), usage=NS(output_tokens=9)),
    ]

    class _Stream:
        async def __aenter__(self) -> _Stream:
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        async def __aiter__(self) -> AsyncIterator[Any]:
            for ev in events:
                yield ev

    client = NS(messages=NS(stream=lambda **kw: _Stream()))
    model = AnthropicModel("claude-opus-4-7", client=client)
    chunks = [c async for c in model.stream([Message(role=Role.USER, content="hi")])]
    (call,) = [c.tool_call for c in chunks if c.kind == "tool_call"]
    assert call is not None
    assert call.args == {} and call.args_error and "not valid JSON" in call.args_error


async def test_anthropic_complete_flags_tool_use_cut_off_by_max_tokens() -> None:
    async def create(**kwargs: Any) -> Any:
        return NS(
            content=[NS(type="tool_use", id="tu_1", name="wipe", input={"path": "/tm"})],
            stop_reason="max_tokens",
            usage=NS(input_tokens=5, output_tokens=9),
        )

    model = AnthropicModel("claude-opus-4-7", client=NS(messages=NS(create=create)))
    _, calls, _, finish = await model.complete([Message(role=Role.USER, content="hi")])
    assert finish == "max_tokens"
    assert calls[0].args == {} and calls[0].args_error and "cut off" in calls[0].args_error


def test_parse_tool_args_contract() -> None:
    assert parse_tool_args("") == ({}, None)
    assert parse_tool_args(None) == ({}, None)
    assert parse_tool_args('{"a": 1}') == ({"a": 1}, None)
    for raw, needle in (('{"a": ', "not valid JSON"), ("null", "JSON object"), ("[1]", "JSON object")):
        args, error = parse_tool_args(raw)
        assert args == {} and error is not None and needle in error


# ---------------------------------------------------------------------------
# 2. Valid JSON that is not an object
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["null", "[1,2]", '"abc"', "7"])
async def test_non_object_tool_args_are_a_tool_error_not_a_crash(raw: str) -> None:
    ran: list[str] = []

    @tool
    def echo(x: str = "d") -> str:
        "echo"
        ran.append(x)
        return x

    client = _FakeOpenAI([_oai(calls=(("echo", raw),)), _oai(text="done")])
    result = await Agent("x", model=OpenAIModel("gpt-4o", client=client), tools=[echo]).run("hi")
    assert result.output == "done"
    assert ran == []
    (msg,) = _tool_messages(client)
    assert msg.startswith("ERROR:") and "JSON object" in msg


# ---------------------------------------------------------------------------
# 3. Argument validation
# ---------------------------------------------------------------------------


class _Order(BaseModel):
    sku: str
    qty: int


async def test_primitive_argument_that_does_not_validate_is_rejected() -> None:
    called: list[Any] = []

    @tool
    def repeat(n: int, flag: bool = False) -> str:
        "Repeat."
        called.append(n)
        return "x" * n

    with pytest.raises(ToolArgumentError) as err:
        await repeat.execute({"n": "abc"})
    assert called == []
    assert "parameter 'n'" in str(err.value) and "'abc'" in str(err.value)
    assert "repeat" in str(err.value)

    result = await InProcessToolHost([repeat]).call("repeat", {"n": "abc"}, call_id="c1")
    assert not result.ok and "parameter 'n'" in (result.error or "")
    with pytest.raises(ToolArgumentError, match="parameter 'flag'"):
        await repeat.execute({"n": 1, "flag": "maybe"})


async def test_model_argument_that_does_not_validate_is_rejected() -> None:
    received: list[Any] = []

    @tool
    def place(order: _Order) -> str:
        "Place an order."
        received.append(order)
        return order.sku

    with pytest.raises(ToolArgumentError) as err:
        await place.execute({"order": {"sku": "a", "qty": "many"}})
    assert received == []  # the raw dict never reached the function
    assert "order.qty" in str(err.value)
    with pytest.raises(ToolArgumentError, match="parameter 'order'"):
        await place.execute({"order": "not json at all"})


async def test_lenient_coercions_still_work() -> None:
    @tool
    def calc(rate: int, ratio: float = 1.0, on: bool = False, label: str = "") -> str:
        "Calc."
        return f"{rate!r} {ratio!r} {on!r} {label!r}"

    @tool
    def place(order: _Order, tags: list[int] | None = None, limit: int = 10) -> str:
        "Place."
        assert isinstance(order, _Order)
        return f"{order.sku}:{order.qty}:{tags}:{limit}"

    assert await calc.execute({"rate": "8", "ratio": "0.5", "on": "true"}) == "8 0.5 True ''"
    assert await calc.execute({"rate": "8.0"}) == "8 1.0 False ''"
    # JSON-string → object, element coercion, and null for an optional
    # parameter falls back to its default.
    out = await place.execute({"order": '{"sku": "a", "qty": "2"}', "tags": ["1"], "limit": None})
    assert out == "a:2:[1]:10"


async def test_unannotated_parameters_are_not_validated() -> None:
    @tool
    def loose(thing) -> str:  # type: ignore[no-untyped-def]
        "Loose."
        return repr(thing)

    assert await loose.execute({"thing": {"k": 1}}) == "{'k': 1}"


async def test_invalid_argument_becomes_a_tool_error_the_model_can_retry() -> None:
    seen: list[int] = []

    @tool
    def double(n: int) -> int:
        "Double."
        seen.append(n)
        return n * 2

    model = _SpyScripted(
        [
            ScriptedTurn(tool_calls=[ToolCall(tool="double", args={"n": "abc"})]),
            ScriptedTurn(tool_calls=[ToolCall(tool="double", args={"n": "4"})]),
            ScriptedTurn(text="8"),
        ]
    )
    await Agent("x", model=model, tools=[double]).run("q")
    assert seen == [4]
    assert "parameter 'n'" in model.tool_messages(1)[0]
    assert model.tool_messages(2)[-1] == "8"


# ---------------------------------------------------------------------------
# 4. resume() after max_turns_exceeded
# ---------------------------------------------------------------------------


async def test_resume_after_max_turns_gets_a_fresh_turn_allowance() -> None:
    @tool
    def t() -> str:
        "t"
        return "data"

    model = ScriptedModel(
        [ScriptedTurn(tool_calls=[ToolCall(tool="t")])] * 3 + [ScriptedTurn(text="FINAL ANSWER")]
    )
    agent = Agent("x", model=model, tools=[t], max_turns=3, tuning=Tuning(checkpoint=True))
    first = await agent.run("q", session_id="s")
    assert first.interruption_reason == "max_turns_exceeded" and first.turns == 3

    resumed = await agent.resume(session_id="s")
    assert resumed.output == "FINAL ANSWER"
    assert not resumed.interrupted
    assert resumed.turns == 4  # turns keep counting across the resume
    assert model.remaining == 0


async def test_resume_allowance_is_per_call_not_unbounded() -> None:
    @tool
    def t() -> str:
        "t"
        return "data"

    model = ScriptedModel([ScriptedTurn(tool_calls=[ToolCall(tool="t")])] * 10)
    agent = Agent("x", model=model, tools=[t], max_turns=2, tuning=Tuning(checkpoint=True))
    await agent.run("q", session_id="s")
    resumed = await agent.resume(session_id="s")
    assert resumed.interruption_reason == "max_turns_exceeded"
    assert resumed.turns == 4  # 2 + a fresh allowance of 2, no more


# ---------------------------------------------------------------------------
# 5. Output-schema validation retry: gate + accounting
# ---------------------------------------------------------------------------


class _Out(BaseModel):
    n: int


async def test_validation_retry_stops_when_budget_is_exhausted() -> None:
    spend = Usage(input_tokens=1000, output_tokens=1000, cost_usd=1.0)
    model = ScriptedModel([ScriptedTurn(text="not json", usage=spend)] * 6)
    budget = StandardBudget(BudgetConfig(max_tokens=1500))
    agent = Agent("x", model=model, budget=budget)

    with pytest.raises(OutputValidationError, match="budget ran out"):
        await agent.run("q", output_schema=_Out, output_validation_retries=5)
    assert 6 - model.remaining == 1  # no retry call after the cap was hit


async def test_validation_retry_is_charged_and_gated() -> None:
    spend = Usage(input_tokens=100, output_tokens=10, cost_usd=0.5)
    model = _SpyScripted(
        [ScriptedTurn(text="nope", usage=spend), ScriptedTurn(text='{"n": 3}', usage=spend)]
    )
    budget = _RecordingBudget()
    acquired: list[str | None] = []

    class _Limiter:
        async def acquire(self, *, user_id: str | None) -> None:
            acquired.append(user_id)

    agent = Agent(
        "x",
        model=model,
        budget=budget,
        rate_limiter=_Limiter(),  # type: ignore[arg-type]
        prompt_caching=True,
    )
    result = await agent.run("q", output_schema=_Out)

    assert result.parsed == _Out(n=3)
    assert budget.cost == pytest.approx(1.0)  # main call + the retry
    assert budget.tokens == 220
    assert result.cost_usd == pytest.approx(1.0) and result.turns == 2
    assert len(acquired) == 2  # the retry passed the rate limiter too
    retry_kwargs = model.calls[1][1]
    assert retry_kwargs["output_schema"] is _Out
    assert retry_kwargs["prompt_caching"] is not None and retry_kwargs["prompt_caching"].enabled


# ---------------------------------------------------------------------------
# 6. Summariser usage is counted
# ---------------------------------------------------------------------------


async def test_auto_compact_summariser_usage_is_counted() -> None:
    @tool
    def big() -> str:
        "big"
        return "lorem ipsum " * 400

    turn = Usage(input_tokens=100, output_tokens=10, cost_usd=0.01)
    summary = Usage(input_tokens=5000, output_tokens=500, cost_usd=0.50)
    model = ScriptedModel(
        [ScriptedTurn(tool_calls=[ToolCall(tool="big")], usage=turn)] * 6
        + [ScriptedTurn(text="done", usage=turn)]
    )
    summariser = ScriptedModel([ScriptedTurn(text="summary", usage=summary)] * 50)
    budget = _RecordingBudget()
    agent = Agent(
        "x",
        model=model,
        tools=[big],
        budget=budget,
        max_turns=10,
        auto_compact_at_tokens=2000,
        tuning=Tuning(auto_compact_summariser=summariser, auto_compact_keep_recent_turns=1),
    )
    result = await agent.run("q")
    calls = 50 - summariser.remaining
    assert calls >= 1
    expected = 7 * 0.01 + calls * 0.50
    assert result.cost_usd == pytest.approx(expected)
    assert budget.cost == pytest.approx(expected)
    assert result.tokens_in == 7 * 100 + calls * 5000


async def test_tool_result_summariser_usage_is_counted() -> None:
    @tool
    def big() -> str:
        "big"
        return "row\n" * 500

    turn = Usage(input_tokens=10, output_tokens=1, cost_usd=0.01)
    summariser = ScriptedModel(
        [ScriptedTurn(text="short", usage=Usage(input_tokens=700, output_tokens=7, cost_usd=0.25))]
    )
    model = _SpyScripted(
        [ScriptedTurn(tool_calls=[ToolCall(tool="big")], usage=turn), ScriptedTurn(text="ok", usage=turn)]
    )
    budget = _RecordingBudget()
    agent = Agent("x", model=model, tools=[big], budget=budget, tool_result_summarizer=summariser)
    result = await agent.run("q")
    assert model.tool_messages() == ["short"]
    assert result.cost_usd == pytest.approx(0.27)
    assert budget.cost == pytest.approx(0.27)
    assert result.tokens_in == 720


# ---------------------------------------------------------------------------
# 7. ExceptionGroup unwrapping + signature probes
# ---------------------------------------------------------------------------


def _one_tool_agent(**kwargs: Any) -> Agent:
    @tool
    def t() -> str:
        "t"
        return "ok"

    @tool
    def u() -> str:
        "u"
        return "ok"

    model = ScriptedModel(
        [ScriptedTurn(tool_calls=[ToolCall(tool="t"), ToolCall(tool="u")]), ScriptedTurn(text="done")]
    )
    return Agent("x", model=model, tools=[t, u], **kwargs)


class _FailingAudit(InMemoryAuditLog):
    async def append(self, **kwargs: Any) -> Any:
        if kwargs["action"] == "tool_call":
            raise ConnectionError("audit db down")
        return await super().append(**kwargs)


async def test_dispatch_failure_is_raised_as_itself_from_run() -> None:
    with pytest.raises(ConnectionError, match="audit db down"):
        await _one_tool_agent(audit_log=_FailingAudit()).run("q")


async def test_dispatch_failure_is_raised_as_itself_from_stream() -> None:
    with pytest.raises(ConnectionError, match="audit db down"):
        async for _ in _one_tool_agent(audit_log=_FailingAudit()).stream("q"):
            pass


async def test_type_error_inside_permissions_is_not_retried_as_legacy() -> None:
    calls: list[int] = []

    class _Buggy:
        async def check(self, call: ToolCall, *, context: Any, user_id: str | None = None) -> Any:
            calls.append(1)
            raise TypeError("bug inside my policy: NoneType + int")

    with pytest.raises(TypeError, match="bug inside my policy"):
        await _one_tool_agent(
            permissions=_Buggy(), tuning=Tuning(parallel_tool_calls=False)
        ).run("q")
    assert len(calls) == 1  # was 2: the "legacy signature" fallback re-called it


async def test_legacy_permissions_signature_still_supported() -> None:
    calls: list[str] = []

    class _Legacy:
        async def check(self, call: ToolCall, *, context: Any) -> PermissionDecision:
            calls.append(call.tool)
            return PermissionDecision.allow_()

    with pytest.warns(DeprecationWarning):
        result = await _one_tool_agent(permissions=_Legacy()).run("q")
    assert result.output == "done" and sorted(calls) == ["t", "u"]


async def test_type_error_inside_budget_consume_is_not_charged_twice() -> None:
    class _BuggyBudget(_RecordingBudget):
        async def consume(self, **kwargs: Any) -> None:
            self.consume_calls += 1
            raise TypeError("unsupported operand")

    budget = _BuggyBudget()
    agent = Agent("x", model=ScriptedModel([ScriptedTurn(text="hi")]), budget=budget)
    with pytest.raises(TypeError, match="unsupported operand"):
        await agent.run("q")
    assert budget.consume_calls == 1


def test_accepts_kwarg() -> None:
    class _C:
        async def new(self, call: Any, *, user_id: str | None = None) -> None: ...
        async def old(self, call: Any) -> None: ...
        async def catch_all(self, *args: Any, **kwargs: Any) -> None: ...

    c = _C()
    assert accepts_kwarg(c.new, "user_id")
    assert not accepts_kwarg(c.old, "user_id")
    assert accepts_kwarg(c.catch_all, "user_id")


# ---------------------------------------------------------------------------
# 8. finish_reason
# ---------------------------------------------------------------------------


async def test_truncated_final_answer_is_visible_on_the_result() -> None:
    client = _FakeOpenAI([_oai(text="The answer is forty-", finish="length")])
    result = await Agent("x", model=OpenAIModel("gpt-4o", client=client)).run("q")
    assert result.output == "The answer is forty-"
    assert result.finish_reason == "length"
    assert result.truncated is True and result.content_filtered is False


async def test_finish_reason_for_normal_and_filtered_stops() -> None:
    ok = await Agent("x", model=ScriptedModel([ScriptedTurn(text="hi")])).run("q")
    assert ok.finish_reason == "stop" and not ok.truncated and not ok.content_filtered

    client = _FakeOpenAI([_oai(text="", finish="content_filter")])
    filtered = await Agent("x", model=OpenAIModel("gpt-4o", client=client)).run("q")
    assert filtered.finish_reason == "content_filter" and filtered.content_filtered

    streamed = [ev async for ev in Agent("x", model=_StreamOnly()).stream("q")]
    assert streamed[-1].payload["result"]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# 9. Stream-only models under the wrappers
# ---------------------------------------------------------------------------


async def test_stream_only_model_under_retry_and_fallback() -> None:
    retrying = RetryingModel(_StreamOnly("a"), RetryPolicy())
    text, calls, usage, finish = await retrying.complete([Message(role=Role.USER, content="q")])
    assert (text, calls, usage.input_tokens, finish) == ("a", [], 3, "stop")

    fallback = FallbackModel([_StreamOnly("b"), _StreamOnly("c")])
    text, _, _, _ = await fallback.complete([Message(role=Role.USER, content="q")])
    assert text == "b" and fallback.last_served == "stream-only"

    with_retry = Agent("x", model=_StreamOnly(), tuning=Tuning(retry_policy=RetryPolicy()))
    assert (await with_retry.run("q")).output == "hi"
    assert (await Agent("x", model=FallbackModel([_StreamOnly(), _StreamOnly()])).run("q")).output == "hi"


# ---------------------------------------------------------------------------
# 10 + 11. Tool timeout, delegation
# ---------------------------------------------------------------------------


async def _run_one_call(agent_tool: Any, **agent_kwargs: Any) -> tuple[str, _SpyScripted]:
    model = _SpyScripted(
        [ScriptedTurn(tool_calls=[ToolCall(tool=agent_tool.name)]), ScriptedTurn(text="done")]
    )
    await Agent("x", model=model, tools=[agent_tool], **agent_kwargs).run("q")
    return model.tool_messages()[0], model


async def test_tool_timeout_is_configurable_per_agent() -> None:
    @tool
    async def slow() -> str:
        "slow"
        await anyio.sleep(5)
        return "finished"

    started = time.monotonic()
    msg, _ = await _run_one_call(slow, tuning=Tuning(tool_timeout=0.05))
    assert time.monotonic() - started < 2
    assert msg.startswith("ERROR: tool timed out after 0.05s")
    assert "cancelled" in msg


async def test_tool_decorator_timeout_overrides_agent_default() -> None:
    @tool(timeout=0.05)
    async def slow() -> str:
        "slow"
        await anyio.sleep(5)
        return "finished"

    @tool(timeout=float("inf"))
    async def patient() -> str:
        "patient"
        await anyio.sleep(0.15)
        return "finished"

    assert slow.to_def().timeout == 0.05
    msg, _ = await _run_one_call(slow)  # agent default is 120s
    assert msg.startswith("ERROR: tool timed out after 0.05s")
    msg, _ = await _run_one_call(patient, tuning=Tuning(tool_timeout=0.05))
    assert msg == "finished"
    with pytest.raises(ConfigError):
        tool(timeout=0)(lambda: "x")
    with pytest.raises(ValueError):
        Agent("x", model=ScriptedModel([]), tuning=Tuning(tool_timeout=0))


async def test_timed_out_sync_tool_error_says_it_is_still_running() -> None:
    log: list[str] = []

    @tool
    def charge_card(amount: int = 5) -> str:
        "charge"
        time.sleep(0.3)
        log.append(f"charged {amount}")
        return "ok"

    msg, _ = await _run_one_call(charge_card, tuning=Tuning(tool_timeout=0.05))
    assert log == []  # the model is told before the side effect lands
    assert msg.startswith("ERROR: tool timed out after 0.05s")
    assert "still running in the background" in msg
    assert "do not assume it didn't happen" in msg
    await asyncio.sleep(0.4)
    assert log == ["charged 5"]


async def test_tools_own_timeout_error_is_not_reported_as_the_framework_timeout() -> None:
    @tool
    async def flaky() -> str:
        "flaky"
        raise TimeoutError("upstream db timed out")

    msg, _ = await _run_one_call(flaky)
    assert msg == "ERROR: TimeoutError: upstream db timed out"


class _SlowWorkerModel:
    """Spends $5 on a first call, then hangs (or raises) on the second."""

    name = "slow-worker"

    def __init__(self, *, fail: BaseException | None = None) -> None:
        self._calls = 0
        self._fail = fail

    async def complete(self, messages: list[Message], **kwargs: Any) -> Any:
        self._calls += 1
        if self._calls == 1:
            usage = Usage(input_tokens=1000, output_tokens=1000, cost_usd=5.0)
            return "", [ToolCall(id="w1", tool="nope", args={})], usage, "tool_use"
        if self._fail is not None:
            raise self._fail
        await anyio.sleep(30)
        return "worker done", [], Usage(cost_usd=1.0), "stop"

    async def stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[ModelChunk]:
        text, calls, usage, finish = await self.complete(messages, **kwargs)
        if text:
            yield ModelChunk(kind="text", text=text)
        for call in calls:
            yield ModelChunk(kind="tool_call", tool_call=call)
        yield ModelChunk(kind="finish", finish_reason=finish, usage=usage)


def _supervisor(worker: Agent, budget: Any = None, **sup_kwargs: Any) -> tuple[Agent, _SpyScripted]:
    model = _SpyScripted(
        [
            ScriptedTurn(
                tool_calls=[ToolCall(id="d1", tool="delegate", args={"worker": "r", "instructions": "go"})],
                usage=Usage(cost_usd=0.01),
            ),
            ScriptedTurn(text="final", usage=Usage(cost_usd=0.01)),
        ]
    )
    agent = Agent(
        "coordinate",
        model=model,
        budget=budget,
        architecture=Supervisor(workers={"r": worker}, **sup_kwargs),
        tuning=Tuning(tool_timeout=0.05),
    )
    return agent, model


async def test_delegation_is_not_subject_to_the_ordinary_tool_timeout() -> None:
    class _Worker(_StreamOnly):
        async def stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[ModelChunk]:
            await anyio.sleep(0.25)  # 5x the supervisor's tool timeout
            async for chunk in super().stream(messages, **kwargs):
                yield chunk

    sup, model = _supervisor(Agent("researcher", model=_Worker("worker answer")))
    result = await sup.run("task")
    assert result.output == "final"
    assert "worker answer" in model.tool_messages()[0]


@pytest.mark.parametrize("streaming", [False, True])
async def test_timed_out_worker_spend_rolls_up_into_parent(streaming: bool) -> None:
    budget = _RecordingBudget()
    worker = Agent("slow researcher", model=_SlowWorkerModel())
    sup, model = _supervisor(worker, budget=budget, delegate_timeout=0.2)

    if streaming:
        events = [ev async for ev in sup.stream("task")]
        cost = events[-1].payload["result"]["cost_usd"]
    else:
        cost = (await sup.run("task")).cost_usd

    assert "timed out after 0.2s" in model.tool_messages()[0]
    assert cost == pytest.approx(5.02)  # the worker had already spent $5
    assert budget.cost == pytest.approx(5.02)


async def test_raising_worker_spend_rolls_up_into_parent() -> None:
    budget = _RecordingBudget()
    worker = Agent("flaky researcher", model=_SlowWorkerModel(fail=RuntimeError("provider exploded")))
    sup, model = _supervisor(worker, budget=budget)
    result = await sup.run("task")
    assert "RuntimeError: provider exploded" in model.tool_messages()[0]
    assert result.cost_usd == pytest.approx(5.02)
    assert budget.cost == pytest.approx(5.02)


async def test_subagent_invocation_reports_partial_usage_on_failure() -> None:
    from loomflow.architecture import AgentSession

    session = AgentSession(id="parent", instructions="")
    worker = Agent("w", model=_SlowWorkerModel(fail=RuntimeError("boom")))
    invocation = SubagentInvocation(worker, "go", rollup_into=session)
    with pytest.raises(RuntimeError, match="boom"):
        async for _ in invocation.events():
            pass
    assert invocation.result == {}
    assert invocation.usage.cost_usd == pytest.approx(5.0)
    assert session.cumulative_usage.cost_usd == pytest.approx(5.0)


async def test_delegation_depth_limit() -> None:
    depths: list[int] = []

    @tool
    def probe() -> str:
        "probe"
        depths.append(delegation_depth())
        return "ok"

    def delegate_turns() -> list[ScriptedTurn]:
        return [
            ScriptedTurn(tool_calls=[ToolCall(tool="delegate", args={"worker": "w", "instructions": "go"})]),
            ScriptedTurn(text="done"),
        ]

    leaf = Agent("leaf", model=ScriptedModel([ScriptedTurn(tool_calls=[ToolCall(tool="probe")]), ScriptedTurn(text="leaf done")]), tools=[probe])
    mid_model = _SpyScripted(delegate_turns())
    mid = Agent("mid", model=mid_model, architecture=Supervisor(workers={"w": leaf}, max_delegation_depth=1))
    top = Agent("top", model=ScriptedModel(delegate_turns()), architecture=Supervisor(workers={"w": mid}))

    await top.run("task")
    # ``mid`` runs at depth 1 == its limit: its own delegate is refused.
    assert "delegation depth limit reached" in mid_model.tool_messages()[0]
    assert depths == []

    # With room to spare the chain goes through, one level deeper each hop.
    leaf2 = Agent("leaf", model=ScriptedModel([ScriptedTurn(tool_calls=[ToolCall(tool="probe")]), ScriptedTurn(text="leaf done")]), tools=[probe])
    mid2 = Agent("mid", model=ScriptedModel(delegate_turns()), architecture=Supervisor(workers={"w": leaf2}))
    top2 = Agent("top", model=ScriptedModel(delegate_turns()), architecture=Supervisor(workers={"w": mid2}))
    await top2.run("task")
    assert depths == [2]
    with pytest.raises(ValueError):
        Supervisor(workers={"w": leaf}, max_delegation_depth=0)


# ---------------------------------------------------------------------------
# 12. Sequential / destructive dispatch
# ---------------------------------------------------------------------------


def _ordering_tools(log: list[str], *, destructive: bool) -> list[Any]:
    @tool(destructive=destructive)
    async def write(v: str) -> str:
        "w"
        log.append("write-start")
        await anyio.sleep(0.05)
        log.append("write-done")
        return "ok"

    @tool
    async def read() -> str:
        "r"
        log.append("read-start")
        return "x"

    return [write, read]


def _write_then_read() -> ScriptedModel:
    return ScriptedModel(
        [
            ScriptedTurn(tool_calls=[ToolCall(tool="write", args={"v": "1"}), ToolCall(tool="read")]),
            ScriptedTurn(text="done"),
        ]
    )


@pytest.mark.parametrize("streaming", [False, True])
async def test_destructive_tool_calls_run_alone_in_emitted_order(streaming: bool) -> None:
    log: list[str] = []
    agent = Agent("x", model=_write_then_read(), tools=_ordering_tools(log, destructive=True))
    if streaming:
        async for _ in agent.stream("q"):
            pass
    else:
        await agent.run("q")
    assert log == ["write-start", "write-done", "read-start"]


async def test_non_destructive_calls_stay_parallel_by_default() -> None:
    log: list[str] = []
    await Agent("x", model=_write_then_read(), tools=_ordering_tools(log, destructive=False)).run("q")
    assert log == ["write-start", "read-start", "write-done"]


@pytest.mark.parametrize("streaming", [False, True])
async def test_parallel_tool_calls_false_runs_the_turn_sequentially(streaming: bool) -> None:
    log: list[str] = []
    agent = Agent(
        "x",
        model=_write_then_read(),
        tools=_ordering_tools(log, destructive=False),
        tuning=Tuning(parallel_tool_calls=False),
    )
    if streaming:
        async for _ in agent.stream("q"):
            pass
    else:
        await agent.run("q")
    assert log == ["write-start", "write-done", "read-start"]


# ---------------------------------------------------------------------------
# 13. Agent(timeout=)
# ---------------------------------------------------------------------------


async def test_inner_timeout_error_is_not_relabelled_as_run_timeout() -> None:
    class _Model:
        name = "m"

        async def stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[ModelChunk]:
            raise TimeoutError("socket read timed out")
            yield  # pragma: no cover

    with pytest.raises(TimeoutError, match="socket read timed out") as err:
        await Agent("x", model=_Model(), timeout=3600).run("q")
    assert not isinstance(err.value, RunTimeout)


async def test_own_deadline_still_raises_run_timeout() -> None:
    class _Hang:
        name = "m"

        async def stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[ModelChunk]:
            await anyio.sleep(30)
            yield ModelChunk(kind="finish", finish_reason="stop")

    with pytest.raises(RunTimeout) as err:
        await Agent("x", model=_Hang(), timeout=0.05).run("q")
    assert err.value.seconds == 0.05


# ---------------------------------------------------------------------------
# 14. Rendering
# ---------------------------------------------------------------------------


async def test_tool_results_and_errors_render_for_the_model() -> None:
    @tool
    def d() -> dict[str, Any]:
        "d"
        return {"ok": True, "items": None, "name": "it's"}

    @tool
    def m() -> _Order:
        "m"
        return _Order(sku="a", qty=2)

    @tool
    def k() -> str:
        "k"
        empty: dict[str, str] = {}
        return empty["user_id"]

    @tool
    def e() -> str:
        "e"
        raise AssertionError()

    model = _SpyScripted(
        [ScriptedTurn(tool_calls=[ToolCall(tool=n) for n in "dmke"]), ScriptedTurn(text="done")]
    )
    await Agent("x", model=model, tools=[d, m, k, e]).run("q")
    as_dict, as_model, key_error, assertion = model.tool_messages()
    assert json.loads(as_dict) == {"ok": True, "items": None, "name": "it's"}
    assert json.loads(as_model) == {"sku": "a", "qty": 2}
    assert key_error == "ERROR: KeyError: 'user_id'"
    assert assertion == "ERROR: AssertionError"


def test_render_tool_output() -> None:
    class _Opaque:
        pass

    assert render_tool_output("plain") == "plain"
    assert render_tool_output(42) == "42"
    assert render_tool_output(None) == "None"
    assert render_tool_output([1, {"a": (2, 3)}]) == '[1, {"a": [2, 3]}]'
    assert render_tool_output({"é": "ü"}) == '{"é": "ü"}'
    opaque = render_tool_output(_Opaque())
    assert "_Opaque" in opaque and "0x" not in opaque
    nested = render_tool_output({"thing": _Opaque()})
    assert "_Opaque" in nested and "0x" not in nested


# ---------------------------------------------------------------------------
# 15. Duplicate tool names
# ---------------------------------------------------------------------------


async def test_duplicate_tool_names_raise_config_error() -> None:
    @tool(name="search")
    def web(q: str) -> str:
        "web"
        return "web"

    @tool(name="search")
    def db(q: str) -> str:
        "db"
        return "db"

    with pytest.raises(ConfigError, match="duplicate tool name 'search'"):
        Agent("x", model=ScriptedModel([]), tools=[web, db])

    host = InProcessToolHost([web])
    host.register(web)  # the same tool again is a no-op
    with pytest.raises(ConfigError, match="duplicate tool name 'search'"):
        host.register(db)
    host.register(db, replace=True)
    assert (await host.call("search", {"q": "x"})).output == "db"


# ---------------------------------------------------------------------------
# 16. max_turns exit
# ---------------------------------------------------------------------------


async def test_max_turns_exit_keeps_last_assistant_text() -> None:
    @tool
    def t() -> str:
        "t"
        return "data"

    model = ScriptedModel(
        [
            ScriptedTurn(text="Found two candidates so far.", tool_calls=[ToolCall(tool="t")]),
            ScriptedTurn(text="", tool_calls=[ToolCall(tool="t")]),
            ScriptedTurn(text="", tool_calls=[ToolCall(tool="t")]),
        ]
    )
    result = await Agent("x", model=model, tools=[t], max_turns=3).run("q")
    assert result.interrupted and result.interruption_reason == "max_turns_exceeded"
    assert result.output == "Found two candidates so far."


async def test_max_turns_exit_without_text_does_not_persist_an_empty_episode() -> None:
    @tool
    def t() -> str:
        "t"
        return "data"

    memory = InMemoryMemory()
    model = ScriptedModel([ScriptedTurn(tool_calls=[ToolCall(tool="t")])] * 3)
    agent = Agent("x", model=model, tools=[t], max_turns=2, memory=memory)
    result = await agent.run("q", session_id="s1")
    assert result.interrupted and result.output == ""
    assert await memory.session_messages("s1", user_id=None, limit=20) == []

    # A normal run on the same memory is still persisted.
    ok = Agent("x", model=ScriptedModel([ScriptedTurn(text="hello")]), memory=memory)
    await ok.run("q", session_id="s2")
    history = await memory.session_messages("s2", user_id=None, limit=20)
    assert [m.content for m in history] == ["q", "hello"]


# ---------------------------------------------------------------------------
# 17. Tool.__call__
# ---------------------------------------------------------------------------


async def test_decorated_tool_is_directly_callable() -> None:
    @tool
    def add(a: int, b: int = 1) -> int:
        "Add."
        return a + b

    @tool(name="fetch")
    async def fetch(url: str) -> str:
        "Fetch."
        return f"GET {url}"

    assert add(2, b=3) == 5
    assert add(2) == 3
    assert await fetch("http://x") == "GET http://x"
    assert add.name == "add" and (await add.execute({"a": "2"})) == 3
