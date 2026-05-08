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

from jeevesagent import Agent, OutputValidationError
from jeevesagent.model.scripted import ScriptedModel, ScriptedTurn

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
