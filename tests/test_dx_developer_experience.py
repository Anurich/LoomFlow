"""Developer-experience regressions: clear errors for bad input, a
blocking ``run_sync``, a ``ScriptedModel`` tests can assert against,
and a synchronous ``Agent.tools``.

Everything here is offline — no API keys, no network.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from loomflow import (
    Agent,
    ConfigError,
    OutputValidationError,
    Role,
    ScriptedModel,
    ScriptedTurn,
    Tool,
    ToolCall,
    Tuning,
    tool,
)
from loomflow.architecture import RouterRoute
from loomflow.model import ScriptedCall, ScriptExhaustedError
from loomflow.security import DictSecrets
from loomflow.team import Team

pytestmark = pytest.mark.anyio


@tool
def get_weather(city: str) -> str:
    """Look up the weather."""
    return f"sunny in {city}"


# ---------------------------------------------------------------------------
# ScriptedModel: records calls, fails loudly when the script runs out
# ---------------------------------------------------------------------------


async def test_scripted_model_records_messages_and_tools() -> None:
    model = ScriptedModel(
        [
            ScriptedTurn(tool_calls=[ToolCall(tool="get_weather", args={"city": "Tokyo"})]),
            ScriptedTurn(text="It is sunny."),
        ]
    )
    agent = Agent("You are a travel assistant.", model=model, tools=[get_weather])

    result = await agent.run("Weather in Tokyo?")

    assert result.output == "It is sunny."
    assert len(model.calls) == 2
    first, second = model.calls
    assert isinstance(first, ScriptedCall)
    assert first.messages[0].role == Role.SYSTEM
    assert "travel assistant" in first.messages[0].content
    assert first.messages[-1].content == "Weather in Tokyo?"
    assert first.tools is not None
    assert [t.name for t in first.tools] == ["get_weather"]
    # The second call saw the tool result; the first call's snapshot
    # did not grow when the agent appended to its message list.
    assert any("sunny in Tokyo" in m.content for m in second.messages)
    assert len(first.messages) < len(second.messages)


async def test_scripted_model_records_stream_calls() -> None:
    model = ScriptedModel([ScriptedTurn(text="hello there")])
    agent = Agent("x", model=model)
    async for _ in agent.stream("hi"):
        pass
    assert len(model.calls) == 1
    assert model.calls[0].messages[-1].content == "hi"


async def test_scripted_model_raises_when_the_script_runs_out() -> None:
    model = ScriptedModel([ScriptedTurn(text="only one")])
    agent = Agent("x", model=model)
    await agent.run("first")

    with pytest.raises(ScriptExhaustedError) as excinfo:
        await agent.run("second")

    message = str(excinfo.value)
    assert "1 turn(s) were scripted" in message
    assert "model call #2" in message
    assert "on_exhausted='empty'" in message


async def test_scripted_model_empty_answers_are_opt_in() -> None:
    model = ScriptedModel([], on_exhausted="empty")
    result = await Agent("x", model=model).run("hi")
    assert result.output == ""
    assert len(model.calls) == 1


def test_scripted_model_rejects_an_unknown_on_exhausted() -> None:
    with pytest.raises(ValueError, match="on_exhausted"):
        ScriptedModel([], on_exhausted="ignore")  # type: ignore[arg-type]


async def test_scripted_subclass_with_its_own_calls_counter_still_works() -> None:
    class Counting(ScriptedModel):
        def __init__(self, turns: list[ScriptedTurn]) -> None:
            super().__init__(turns)
            self.calls = 0  # type: ignore[assignment]

    model = Counting([ScriptedTurn(text="ok")])
    assert (await Agent("x", model=model).run("hi")).output == "ok"


# ---------------------------------------------------------------------------
# Bad input raises a framework error that says what was passed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [123, 4.5, ["gpt-4o"], object()])
def test_model_of_the_wrong_type_is_a_config_error(bad: object) -> None:
    with pytest.raises(ConfigError) as excinfo:
        Agent("x", model=bad)  # type: ignore[arg-type]
    message = str(excinfo.value)
    assert type(bad).__name__ in message
    assert "Model instance" in message


async def test_non_string_prompt_is_a_type_error() -> None:
    agent = Agent("x", model="echo")
    with pytest.raises(TypeError, match="prompt must be a string, not list"):
        await agent.run(["hi"])  # type: ignore[arg-type]


async def test_non_string_prompt_fails_a_stream_with_the_same_message() -> None:
    agent = Agent("x", model="echo")
    with pytest.raises(TypeError, match="prompt must be a string, not list"):
        async for _ in agent.stream(["hi"]):  # type: ignore[arg-type]
            pass


async def test_output_schema_that_is_not_a_model_is_a_config_error() -> None:
    model = ScriptedModel([ScriptedTurn(text="{}")])
    agent = Agent("x", model=model)
    with pytest.raises(ConfigError) as excinfo:
        await agent.run("hi", output_schema=dict)
    message = str(excinfo.value)
    assert "BaseModel" in message
    assert "<class 'dict'>" in message
    assert model.calls == [], "the model must not be called for an unusable schema"


def test_unknown_agent_kwarg_suggests_the_close_match() -> None:
    with pytest.raises(TypeError) as excinfo:
        Agent("x", model="echo", tool=[get_weather])  # type: ignore[call-arg]
    message = str(excinfo.value)
    assert "unexpected keyword argument(s): tool" in message
    assert "Did you mean 'tools'?" in message


def test_unknown_agent_kwarg_suggests_tuning_fields_too() -> None:
    with pytest.raises(TypeError, match="Did you mean 'retry_policy'"):
        Agent("x", model="echo", retry_polcy=None)  # type: ignore[call-arg]


def test_several_unknown_agent_kwargs_each_get_a_hint() -> None:
    with pytest.raises(TypeError) as excinfo:
        Agent("x", model="echo", tool=[], max_turn=3)  # type: ignore[call-arg]
    message = str(excinfo.value)
    assert "'tools' (for 'tool')" in message
    assert "'max_turns' (for 'max_turn')" in message


def test_unknown_agent_kwarg_without_a_close_match_has_no_hint() -> None:
    with pytest.raises(TypeError) as excinfo:
        Agent("x", model="echo", zzzzqqqq=1)  # type: ignore[call-arg]
    assert "Did you mean" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# Missing API key for a model-name string
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "env_var"),
    [
        ("gpt-4.1-mini", "OPENAI_API_KEY"),
        ("o3", "OPENAI_API_KEY"),
        ("claude-opus-4-7", "ANTHROPIC_API_KEY"),
    ],
)
def test_missing_api_key_is_a_config_error_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch, spec: str, env_var: str
) -> None:
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ConfigError) as excinfo:
        Agent("x", model=spec)
    message = str(excinfo.value)
    assert env_var in message
    assert spec in message


def test_api_key_from_the_environment_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    assert Agent("x", model="gpt-4.1-mini").model.name == "gpt-4.1-mini"


def test_api_key_from_a_secrets_backend_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("anthropic")
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    secrets = DictSecrets({"ANTHROPIC_API_KEY": "sk-ant-test-not-real"})
    agent = Agent("x", model="claude-opus-4-7", tuning=Tuning(secrets=secrets))
    assert agent.model.name == "claude-opus-4-7"


def test_anthropic_auth_token_counts_as_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token-not-real")
    assert Agent("x", model="claude-opus-4-7").model.name == "claude-opus-4-7"


# ---------------------------------------------------------------------------
# run_sync
# ---------------------------------------------------------------------------


def test_run_sync_returns_the_same_result_as_run() -> None:
    agent = Agent("x", model="echo")
    result = agent.run_sync("hello", user_id="alice", session_id="s1")
    assert result.output == "Echo: hello"
    assert result.session_id == "s1"
    # A second call on the same agent starts a fresh loop and still works.
    assert agent.run_sync("again", user_id="alice").output == "Echo: again"


def test_run_sync_passes_output_schema_through() -> None:
    class Answer(BaseModel):
        value: int

    agent = Agent("x", model=ScriptedModel([ScriptedTurn(text='{"value": 4}')]))
    result = agent.run_sync("2 + 2?", output_schema=Answer)
    assert result.parsed == Answer(value=4)


async def test_run_sync_inside_an_event_loop_says_to_await_run() -> None:
    agent = Agent("x", model="echo")
    with pytest.raises(RuntimeError, match=r"await agent\.run\("):
        agent.run_sync("hello")


# ---------------------------------------------------------------------------
# OutputValidationError names what was wrong
# ---------------------------------------------------------------------------


async def test_output_validation_error_message_includes_the_pydantic_cause() -> None:
    class Company(BaseModel):
        name: str
        founded_year: int

    model = ScriptedModel([ScriptedTurn(text='{"name": "Acme", "founded_year": "long ago"}')])
    agent = Agent("x", model=model)

    with pytest.raises(OutputValidationError) as excinfo:
        await agent.run("go", output_schema=Company, output_validation_retries=0)

    message = str(excinfo.value)
    assert "did not validate against Company" in message
    assert "founded_year" in message
    assert "valid integer" in message
    assert excinfo.value.cause is not None


def test_output_validation_error_summary_is_capped() -> None:
    class Wide(BaseModel):
        a: int
        b: int
        c: int
        d: int
        e: int

    with pytest.raises(Exception) as caught:  # noqa: PT011 — we want the ValidationError itself
        Wide.model_validate({})
    error = OutputValidationError("Did not validate.", raw="{}", schema=Wide, cause=caught.value)
    message = str(error)
    assert message.startswith("Did not validate: a: Field required; b: Field required; c: ")
    assert message.endswith("and 2 more")


def test_output_validation_error_without_a_cause_keeps_its_message() -> None:
    error = OutputValidationError("Plain message.", raw="", schema=dict, cause=None)
    assert str(error) == "Plain message."


# ---------------------------------------------------------------------------
# Agent.tools — the Tool objects, synchronously
# ---------------------------------------------------------------------------


async def test_tools_property_returns_tool_objects_without_await() -> None:
    agent = Agent("x", model="echo", tools=[get_weather])

    tools = agent.tools

    assert [t.name for t in tools] == ["get_weather"]
    assert all(isinstance(t, Tool) for t in tools)
    assert tools[0] is get_weather
    assert [t.name for t in tools] == await agent.tools_list()


async def test_tools_property_follows_add_and_remove() -> None:
    agent = Agent("x", model="echo")
    assert agent.tools == []

    @agent.with_tool
    def ping() -> str:
        """Ping."""
        return "pong"

    assert [t.name for t in agent.tools] == ["ping"]
    agent.remove_tool("ping")
    assert agent.tools == []


async def test_tools_property_includes_framework_tools() -> None:
    agent = Agent("x", model="echo", tools=[get_weather], living_plan=True)
    names = [t.name for t in agent.tools]
    assert names[0] == "get_weather"
    assert {"plan_write", "plan_read"} <= set(names)
    assert sorted(names) == sorted(await agent.tools_list())


# ---------------------------------------------------------------------------
# Team.router: the documented classifier format
# ---------------------------------------------------------------------------


async def test_team_router_dispatches_on_the_documented_route_line() -> None:
    billing = Agent("billing", model=ScriptedModel([ScriptedTurn(text="refund issued")]))
    tech = Agent("tech", model=ScriptedModel([ScriptedTurn(text="rebooted")]))
    router = Team.router(
        routes=[
            RouterRoute(name="billing", agent=billing, description="payments"),
            RouterRoute(name="tech", agent=tech, description="outages"),
        ],
        model=ScriptedModel([ScriptedTurn(text="route: billing")]),
    )

    result = await router.run("I was charged twice")

    assert result.output == "refund issued"
