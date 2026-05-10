"""Structured-output (output_schema=) tests.

Covers the M4 contract end-to-end:

* Happy path: a well-behaved model emits valid JSON, ``result.parsed``
  is populated as a typed Pydantic instance.
* Markdown-fence tolerance: models that wrap their JSON in ``` ```
  fences (a common bad-but-real habit) are handled cleanly.
* Validation retry: a malformed first response triggers a single
  retry turn that injects the validation error as a USER message;
  if the retry succeeds the run completes normally.
* Validation exhaust: when retries are configured to 0 and the
  model's first response is bad, ``OutputValidationError`` is
  raised with the right cause / raw / schema attached.
* Schema directive: the appended system-prompt directive includes
  the JSON-schema text so the model has it in front of it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from loomflow import Agent, OutputValidationError, Role
from loomflow.model.scripted import ScriptedModel, ScriptedTurn

pytestmark = pytest.mark.anyio


class CompanyInfo(BaseModel):
    name: str
    founded_year: int
    headquarters: str


# ---------------------------------------------------------------------------
# Capturing scripted model — lets us see what the framework feeds the
# model so we can assert on the system-prompt augmentation behaviour.
# ---------------------------------------------------------------------------


class _CapturingScripted(ScriptedModel):
    """Same as ScriptedModel but records the messages it was called
    with so tests can verify the system-prompt directive shape."""

    def __init__(self, turns: list[ScriptedTurn]) -> None:
        super().__init__(turns)
        self.captured: list[list[Any]] = []

    async def complete(self, messages: Any, **kwargs: Any) -> Any:
        self.captured.append(list(messages))
        return await super().complete(messages, **kwargs)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_output_schema_populates_parsed_with_typed_instance() -> None:
    payload = {
        "name": "Acme",
        "founded_year": 2008,
        "headquarters": "Berlin",
    }
    model = ScriptedModel([ScriptedTurn(text=json.dumps(payload))])
    agent = Agent("extract company info", model=model)

    result = await agent.run("...", output_schema=CompanyInfo)

    assert isinstance(result.parsed, CompanyInfo)
    assert result.parsed.name == "Acme"
    assert result.parsed.founded_year == 2008
    assert result.parsed.headquarters == "Berlin"
    # ``output`` keeps the raw text; cleaned of any whitespace.
    assert json.loads(result.output) == payload


async def test_output_schema_strips_markdown_code_fences() -> None:
    """Models often wrap their JSON in ``` ``` despite being told not
    to. The parser tolerates ``` and ```json fences."""
    payload = {
        "name": "Acme",
        "founded_year": 2008,
        "headquarters": "Berlin",
    }
    fenced = "```json\n" + json.dumps(payload) + "\n```"
    model = ScriptedModel([ScriptedTurn(text=fenced)])
    agent = Agent("extract company info", model=model)

    result = await agent.run("...", output_schema=CompanyInfo)

    assert isinstance(result.parsed, CompanyInfo)
    assert result.parsed.name == "Acme"
    # Persisted output is the cleaned JSON, not the fenced version.
    assert "```" not in result.output


async def test_run_without_output_schema_leaves_parsed_none() -> None:
    """Default behaviour — no output_schema means parsed stays None
    and output is whatever the model returned."""
    model = ScriptedModel([ScriptedTurn(text="some free-form answer")])
    agent = Agent("...", model=model)

    result = await agent.run("...")

    assert result.parsed is None
    assert result.output == "some free-form answer"


# ---------------------------------------------------------------------------
# Schema directive in the system prompt
# ---------------------------------------------------------------------------


async def test_output_schema_appends_directive_to_system_prompt() -> None:
    payload = {"name": "X", "founded_year": 2020, "headquarters": "NYC"}
    model = _CapturingScripted([ScriptedTurn(text=json.dumps(payload))])
    agent = Agent("You are an extractor.", model=model)

    await agent.run("go", output_schema=CompanyInfo)

    # First message of the first model call is the system prompt;
    # it should contain BOTH the original instructions and the
    # JSON-schema directive.
    sys_prompt = model.captured[0][0]
    assert sys_prompt.role.value == "system"
    assert "You are an extractor." in sys_prompt.content
    assert "STRUCTURED OUTPUT REQUIRED" in sys_prompt.content
    # The schema itself is embedded as JSON.
    assert "founded_year" in sys_prompt.content
    assert "headquarters" in sys_prompt.content


async def test_run_persists_unaugmented_static_instructions() -> None:
    """The schema directive is per-run; ``agent._instructions`` is
    not mutated, so a follow-up run with no schema sees the
    original prompt only."""
    payload = {"name": "X", "founded_year": 2020, "headquarters": "NYC"}
    model = _CapturingScripted([
        ScriptedTurn(text=json.dumps(payload)),
        ScriptedTurn(text="hello"),
    ])
    agent = Agent("You are an extractor.", model=model)

    await agent.run("first", output_schema=CompanyInfo)
    await agent.run("second")  # no schema

    second_sys = model.captured[1][0]
    assert "STRUCTURED OUTPUT REQUIRED" not in second_sys.content
    assert second_sys.content == "You are an extractor."


# ---------------------------------------------------------------------------
# Validation-retry behaviour
# ---------------------------------------------------------------------------


async def test_validation_retry_recovers_from_initial_bad_output() -> None:
    """First model response fails validation; retry turn fixes it.
    The successful retry should populate ``parsed`` and let the
    run complete normally."""
    bad = json.dumps({"name": "X"})  # missing required fields
    good = json.dumps(
        {"name": "X", "founded_year": 2020, "headquarters": "NYC"}
    )
    model = _CapturingScripted([
        ScriptedTurn(text=bad),
        ScriptedTurn(text=good),
    ])
    agent = Agent("...", model=model)

    result = await agent.run(
        "...", output_schema=CompanyInfo, output_validation_retries=1
    )

    assert isinstance(result.parsed, CompanyInfo)
    assert result.parsed.founded_year == 2020
    # The retry consumed a second model call.
    assert len(model.captured) == 2
    # Retry's messages include the validation-error follow-up.
    retry_messages = model.captured[1]
    user_messages = [
        m.content for m in retry_messages if m.role.value == "user"
    ]
    assert any(
        "failed schema validation" in m.lower() for m in user_messages
    )


async def test_validation_failure_with_zero_retries_raises_immediately() -> None:
    bad = json.dumps({"name": "X"})  # missing required fields
    model = ScriptedModel([ScriptedTurn(text=bad)])
    agent = Agent("...", model=model)

    with pytest.raises(OutputValidationError) as excinfo:
        await agent.run(
            "...", output_schema=CompanyInfo, output_validation_retries=0
        )

    err = excinfo.value
    assert err.schema is CompanyInfo
    assert "name" in err.raw  # raw has the model's bad output
    assert err.cause is not None  # underlying ValidationError attached


async def test_validation_failure_after_exhausted_retries_raises() -> None:
    """When every retry also fails, the framework gives up and
    raises with the latest validation error attached."""
    bad1 = json.dumps({"wrong_field": 1})
    bad2 = json.dumps({"also_wrong": 2})
    model = ScriptedModel([
        ScriptedTurn(text=bad1),
        ScriptedTurn(text=bad2),
    ])
    agent = Agent("...", model=model)

    with pytest.raises(OutputValidationError):
        await agent.run(
            "...", output_schema=CompanyInfo, output_validation_retries=1
        )


# ---------------------------------------------------------------------------
# Native structured output skips the in-prompt schema directive
# ---------------------------------------------------------------------------


async def test_native_structured_output_skips_in_prompt_schema_directive() -> None:
    """When the model adapter declares
    ``supports_native_structured_output = True``, the agent loop
    must NOT also paste the JSON Schema into the system prompt —
    that's pure dead tokens (the API-level constraint already
    forces valid JSON). This test guards against accidentally
    re-introducing the double-injection that bloated structured-
    output token usage by ~3× in the v0.9 benchmark vs LangGraph
    and Pydantic AI.
    """
    payload = {"name": "Acme", "founded_year": 2008, "headquarters": "Berlin"}
    model = _CapturingScripted([ScriptedTurn(text=json.dumps(payload))])
    # The scripted/capturing fake doesn't claim native support by
    # default, so flip the flag manually for the test scope.
    model.supports_native_structured_output = True  # type: ignore[attr-defined]

    agent = Agent("base instructions", model=model)
    result = await agent.run("...", output_schema=CompanyInfo)

    # Parsing still works.
    assert isinstance(result.parsed, CompanyInfo)

    # The system message the model saw must NOT contain the JSON
    # Schema directive. Look at what we captured.
    sent_messages = model.captured[0]
    system_text = next(
        (m.content for m in sent_messages if m.role == Role.SYSTEM),
        "",
    )
    assert "STRUCTURED OUTPUT REQUIRED" not in system_text
    assert "json_schema" not in system_text.lower()
    # The base instructions ARE there — we only stripped the directive.
    assert "base instructions" in system_text


async def test_non_native_model_still_gets_in_prompt_schema_directive() -> None:
    """Custom user-supplied adapters that don't declare
    ``supports_native_structured_output`` keep the prompt-augmentation
    safety net — the model needs the schema in the prompt to know
    what to emit, since there's no decode-time constraint."""
    payload = {"name": "Acme", "founded_year": 2008, "headquarters": "Berlin"}
    model = _CapturingScripted([ScriptedTurn(text=json.dumps(payload))])
    # Default = not native; nothing to flip.

    agent = Agent("base instructions", model=model)
    result = await agent.run("...", output_schema=CompanyInfo)

    assert isinstance(result.parsed, CompanyInfo)
    sent_messages = model.captured[0]
    system_text = next(
        (m.content for m in sent_messages if m.role == Role.SYSTEM),
        "",
    )
    # Directive present for the non-native fallback path.
    assert "STRUCTURED OUTPUT REQUIRED" in system_text


# ---------------------------------------------------------------------------
# RunResult.value — smart accessor for "the answer", parsed when set
# ---------------------------------------------------------------------------


async def test_result_value_returns_parsed_when_schema_succeeds() -> None:
    """``result.value`` is the recommended accessor: when the model
    emitted schema-valid JSON, you get the typed Pydantic instance
    directly — no need to remember whether to use ``.parsed`` or
    ``.output``. This was the ergonomic wart that sent users to
    ``type(result.output) -> str`` and made them think the schema
    was ignored."""
    payload = {"name": "Acme", "founded_year": 2008, "headquarters": "Berlin"}
    model = ScriptedModel([ScriptedTurn(text=json.dumps(payload))])
    agent = Agent("...", model=model)

    result = await agent.run("...", output_schema=CompanyInfo)

    assert isinstance(result.value, CompanyInfo)
    assert result.value is result.parsed
    # ``output`` is still the raw string for logging / display.
    assert isinstance(result.output, str)


async def test_result_value_falls_back_to_output_without_schema() -> None:
    """No ``output_schema`` means no ``.parsed``; ``.value`` then
    returns the raw string so existing call sites that read
    ``result.value`` keep working in both modes."""
    model = ScriptedModel([ScriptedTurn(text="just text")])
    agent = Agent("...", model=model)

    result = await agent.run("...")

    assert result.parsed is None
    assert result.value == "just text"
    assert result.value == result.output


# ---------------------------------------------------------------------------
# Agent(output_schema=...) — agent-bound default schema
# ---------------------------------------------------------------------------


async def test_agent_default_output_schema_applied_to_run() -> None:
    """``Agent(output_schema=Receipt)`` applies the schema to every
    ``agent.run()`` without per-call repetition. Pydantic AI calls
    this ``output_type=`` on Agent; we keep the same kwarg name on
    both ``Agent.__init__`` and ``run()`` so users only learn one."""
    payload = {"name": "Acme", "founded_year": 2008, "headquarters": "Berlin"}
    model = ScriptedModel([ScriptedTurn(text=json.dumps(payload))])
    agent = Agent("...", model=model, output_schema=CompanyInfo)

    result = await agent.run("...")

    assert isinstance(result.parsed, CompanyInfo)
    assert result.parsed.name == "Acme"


async def test_per_call_output_schema_overrides_agent_default() -> None:
    """A per-call ``output_schema=`` on ``run()`` wins over the
    agent's default. Lets users have a "usual" schema on the agent
    but switch shapes for one-off calls."""

    class Other(BaseModel):
        flag: bool

    payload = {"flag": True}
    model = ScriptedModel([ScriptedTurn(text=json.dumps(payload))])
    agent = Agent("...", model=model, output_schema=CompanyInfo)

    result = await agent.run("...", output_schema=Other)

    assert isinstance(result.parsed, Other)
    assert result.parsed.flag is True


# ---------------------------------------------------------------------------
# Tagged unions — ``output_schema=A | B``
# ---------------------------------------------------------------------------


class _Success(BaseModel):
    kind: str = "success"
    value: int


class _Failure(BaseModel):
    kind: str = "failure"
    reason: str


async def test_tagged_union_picks_first_matching_member() -> None:
    """``output_schema=A | B`` lets the agent return one of multiple
    shapes per call. Validation tries each member in declaration
    order and accepts the first that fits — so the model can decide
    "valid result vs structured error" at decode time without the
    framework needing a discriminator field."""
    payload = {"kind": "success", "value": 42}
    model = ScriptedModel([ScriptedTurn(text=json.dumps(payload))])
    agent = Agent("...", model=model)

    result = await agent.run("...", output_schema=_Success | _Failure)

    assert isinstance(result.parsed, _Success)
    assert result.parsed.value == 42


async def test_tagged_union_falls_back_to_second_member() -> None:
    """When the first member doesn't fit, the second is tried.
    ``_Success`` requires ``value: int``; the failure-shaped payload
    fails on _Success and validates against _Failure."""
    payload = {"kind": "failure", "reason": "boom"}
    model = ScriptedModel([ScriptedTurn(text=json.dumps(payload))])
    agent = Agent("...", model=model)

    result = await agent.run("...", output_schema=_Success | _Failure)

    assert isinstance(result.parsed, _Failure)
    assert result.parsed.reason == "boom"


async def test_tagged_union_failure_after_retries_raises() -> None:
    """If neither member validates, ``OutputValidationError`` is
    still raised — unions widen the accepted shape, they don't
    suppress validation."""
    bad = json.dumps({"unrelated": "shape"})
    model = ScriptedModel([
        ScriptedTurn(text=bad),
        ScriptedTurn(text=bad),
    ])
    agent = Agent("...", model=model)

    with pytest.raises(OutputValidationError):
        await agent.run(
            "...",
            output_schema=_Success | _Failure,
            output_validation_retries=1,
        )
