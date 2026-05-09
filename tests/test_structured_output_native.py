"""Native structured-output tests — the OpenAI ``response_format``
and Anthropic forced-tool-call paths.

These tests assert the **payload the adapter sends to the
provider**, not the response handling. The agent loop's
prompt-augmentation + validate-with-retry is already covered by
``tests/test_structured_output.py``; the goal here is to prove
that adapters with native support translate the schema correctly
so the model is constrained at decode time and the retry path
becomes a fallback that almost never fires.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from loomflow.core.types import Message, Role
from loomflow.model.anthropic import AnthropicModel
from loomflow.model.openai import OpenAIModel

pytestmark = pytest.mark.anyio


class Receipt(BaseModel):
    amount: float
    currency: str


# ---------------------------------------------------------------------------
# OpenAI: response_format=json_schema with strict=True
# ---------------------------------------------------------------------------


class _OpenAIRecorder:
    """Minimal fake of the OpenAI client. Records the kwargs each
    create() call receives so tests can assert on the payload."""

    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] = {}
        self.chat = self  # mimic .chat.completions.create
        self.completions = self

    async def create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs

        class _Msg:
            content = json.dumps({"amount": 42.0, "currency": "USD"})
            tool_calls = None

        class _Choice:
            message = _Msg()
            finish_reason = "stop"

        class _Resp:
            choices = [_Choice()]
            usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()

        return _Resp()


async def test_openai_complete_attaches_response_format_json_schema() -> None:
    """When ``output_schema=Receipt`` is passed, the adapter must
    add a strict json_schema response_format to the create() call.
    Without it, the model is unconstrained and the agent loop pays
    for occasional validation retries."""
    fake = _OpenAIRecorder()
    model = OpenAIModel("gpt-4o", client=fake)

    text, _, _, _ = await model.complete(
        [Message(role=Role.USER, content="give me a receipt")],
        output_schema=Receipt,
    )

    assert text == json.dumps({"amount": 42.0, "currency": "USD"})
    rf = fake.last_kwargs.get("response_format")
    assert rf is not None, "adapter should attach response_format"
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["name"] == "Receipt"
    schema = rf["json_schema"]["schema"]
    assert "amount" in schema["properties"]
    assert "currency" in schema["properties"]


async def test_openai_complete_omits_response_format_when_no_schema() -> None:
    """No ``output_schema`` means no ``response_format`` — the
    adapter must not send the kwarg unless the caller asked for it."""
    fake = _OpenAIRecorder()
    model = OpenAIModel("gpt-4o", client=fake)
    await model.complete([Message(role=Role.USER, content="hello")])
    assert "response_format" not in fake.last_kwargs


async def test_openai_complete_handles_non_pydantic_schema() -> None:
    """If the caller passes something that's not a Pydantic model
    (e.g. a dict, defensive against duck-typing slip-ups), the
    adapter degrades gracefully — no response_format, no crash."""
    fake = _OpenAIRecorder()
    model = OpenAIModel("gpt-4o", client=fake)
    await model.complete(
        [Message(role=Role.USER, content="hello")],
        output_schema={"not": "pydantic"},  # type: ignore[arg-type]
    )
    assert "response_format" not in fake.last_kwargs


# ---------------------------------------------------------------------------
# Anthropic: forced tool call with the schema as the input_schema
# ---------------------------------------------------------------------------


class _AnthropicRecorder:
    """Fake Anthropic client. Captures kwargs and emits a
    ``tool_use`` block matching the synthetic ``__output__`` tool
    so the adapter's tool-args → text path is exercised."""

    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] = {}
        self.messages = self

    async def create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs

        class _Block:
            type = "tool_use"
            name = "__output__"
            id = "toolu_01"
            input = {"amount": 42.0, "currency": "USD"}

        class _Msg:
            content = [_Block()]
            stop_reason = "tool_use"
            usage = type("U", (), {"input_tokens": 10, "output_tokens": 5})()

        return _Msg()


async def test_anthropic_complete_forces_synthetic_tool_call() -> None:
    """When ``output_schema=Receipt`` is passed, the Anthropic
    adapter must:

    * add a synthetic ``__output__`` tool whose ``input_schema`` IS
      the requested schema, AND
    * set ``tool_choice`` to force the model to invoke it.

    Together these constrain the model to emit JSON matching the
    schema as the tool's args — Anthropic's idiomatic structured-
    output pattern.
    """
    fake = _AnthropicRecorder()
    model = AnthropicModel("claude-opus-4-7", client=fake)

    text, tool_calls, _, _ = await model.complete(
        [Message(role=Role.USER, content="give me a receipt")],
        output_schema=Receipt,
    )

    # The synthetic tool's args were promoted to the message text.
    assert json.loads(text) == {"amount": 42.0, "currency": "USD"}
    # The synthetic tool is NOT surfaced as a real tool_call to the
    # agent loop — it's the structured output, not a tool dispatch.
    assert tool_calls == []

    # The request payload must include the synthetic tool + force
    # the model to invoke it.
    sent_tools = fake.last_kwargs.get("tools", [])
    synthetic = next((t for t in sent_tools if t["name"] == "__output__"), None)
    assert synthetic is not None
    assert "amount" in synthetic["input_schema"]["properties"]
    assert fake.last_kwargs.get("tool_choice") == {
        "type": "tool",
        "name": "__output__",
    }


async def test_anthropic_complete_omits_synthetic_when_no_schema() -> None:
    fake = _AnthropicRecorder()
    model = AnthropicModel("claude-opus-4-7", client=fake)

    # Provide a synthetic non-tool_use response so the test doesn't
    # accidentally think there's a synthetic tool result to extract.
    async def _create_text_only(**kwargs: Any) -> Any:
        fake.last_kwargs = kwargs

        class _Block:
            type = "text"
            text = "plain answer"

        class _Msg:
            content = [_Block()]
            stop_reason = "end_turn"
            usage = type("U", (), {"input_tokens": 10, "output_tokens": 5})()

        return _Msg()

    fake.create = _create_text_only  # type: ignore[method-assign]

    await model.complete([Message(role=Role.USER, content="hello")])

    # No schema → no synthetic tool, no tool_choice override.
    assert "tool_choice" not in fake.last_kwargs
    sent_tools = fake.last_kwargs.get("tools", []) or []
    assert all(t.get("name") != "__output__" for t in sent_tools)
