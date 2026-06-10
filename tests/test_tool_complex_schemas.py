"""Recursive tool schemas — complex param annotations.

``@tool`` now derives a full recursive JSON schema for non-primitive
parameters (Pydantic models, ``list[T]``, ``dict``, ``Literal``,
optionals) via :class:`pydantic.TypeAdapter`. Nested ``$defs`` are
hoisted to the tool schema root so ``#/$defs/...`` refs stay valid.
At call time ``Tool.execute`` validates the model's raw dict /
JSON-string args back into the annotated Python types via a per-param
``TypeAdapter``; primitives keep the old cheap string-coercion path.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from loomflow import Agent, ScriptedModel, ScriptedTurn
from loomflow.core.types import ToolCall
from loomflow.tools import tool
from loomflow.tools.registry import InProcessToolHost

pytestmark = pytest.mark.anyio


class _Address(BaseModel):
    city: str
    zip: str


class _Person(BaseModel):
    name: str
    address: _Address


# ---------------------------------------------------------------------------
# schema derivation
# ---------------------------------------------------------------------------


def test_basemodel_param_schema_is_object_with_hoisted_defs() -> None:
    @tool
    async def greet(person: _Person) -> str:
        """Greet a person."""
        return person.name

    schema = greet.input_schema
    person_schema = schema["properties"]["person"]
    assert person_schema["type"] == "object"
    # The nested model is hoisted to the tool-schema root under $defs.
    assert "$defs" in schema
    assert "_Address" in schema["$defs"]
    # The address field $refs the hoisted def.
    addr_ref = person_schema["properties"]["address"]["$ref"]
    assert addr_ref == "#/$defs/_Address"


def test_list_int_param_schema() -> None:
    @tool
    async def total(nums: list[int]) -> int:
        """Sum a list of integers."""
        return sum(nums)

    schema = total.input_schema["properties"]["nums"]
    assert schema == {"type": "array", "items": {"type": "integer"}}


def test_primitive_params_schema_unchanged_and_no_adapters() -> None:
    @tool
    async def add(a: int, b: str) -> str:
        """Mix of primitives."""
        return f"{a}{b}"

    schema = add.input_schema
    assert schema["properties"]["a"] == {"type": "integer"}
    assert schema["properties"]["b"] == {"type": "string"}
    # Primitives never get a TypeAdapter — they stay on the cheap path.
    assert add.param_adapters == {}


def test_basemodel_param_gets_adapter() -> None:
    @tool
    async def greet(person: _Person) -> str:
        """Greet."""
        return person.name

    assert "person" in greet.param_adapters


# ---------------------------------------------------------------------------
# execute — validation back into Python types
# ---------------------------------------------------------------------------


async def test_execute_dict_becomes_basemodel_instance() -> None:
    captured: dict[str, object] = {}

    @tool
    async def greet(person: _Person) -> str:
        """Greet."""
        captured["person"] = person
        return person.name

    out = await greet.execute(
        {"person": {"name": "Ada", "address": {"city": "London", "zip": "E1"}}}
    )
    assert out == "Ada"
    assert isinstance(captured["person"], _Person)
    assert captured["person"].address.city == "London"


async def test_execute_json_string_becomes_basemodel_instance() -> None:
    captured: dict[str, object] = {}

    @tool
    async def greet(person: _Person) -> str:
        """Greet."""
        captured["person"] = person
        return person.name

    # Models sometimes serialise nested objects as a JSON string.
    out = await greet.execute(
        {"person": '{"name": "Bo", "address": {"city": "Paris", "zip": "75"}}'}
    )
    assert out == "Bo"
    assert isinstance(captured["person"], _Person)


async def test_execute_list_int_coerces_string_items() -> None:
    @tool
    async def total(nums: list[int]) -> int:
        """Sum."""
        return sum(nums)

    # ["1", "2"] → [1, 2] via the list[int] adapter.
    out = await total.execute({"nums": ["1", "2"]})
    assert out == 3


async def test_execute_primitive_string_coercion_still_works() -> None:
    @tool
    async def read(path: str, offset: int = 0) -> str:
        """Read with a typed offset."""
        return f"{path}@{offset}"

    out = await read.execute({"path": "f.py", "offset": "10"})
    assert out == "f.py@10"


# ---------------------------------------------------------------------------
# invalid complex value — original value passes through
# ---------------------------------------------------------------------------


async def test_invalid_value_passes_original_through_via_host() -> None:
    """A dict missing a required field can't validate → the ORIGINAL
    value is handed to the function, whose own error surfaces as a
    ToolResult error (the host never crashes)."""

    @tool
    async def greet(person: _Person) -> str:
        """Greet."""
        return person.name

    host = InProcessToolHost([greet])
    # Missing 'address' → adapter validation fails → raw dict passes
    # through → the function hits .name on a dict → AttributeError.
    result = await host.call("greet", {"person": {"name": "X"}}, call_id="c1")
    assert not result.ok
    assert result.error is not None


# ---------------------------------------------------------------------------
# end-to-end through Agent
# ---------------------------------------------------------------------------


async def test_basemodel_tool_call_end_to_end() -> None:
    captured: dict[str, object] = {}

    @tool
    async def greet(person: _Person) -> str:
        """Greet."""
        captured["person"] = person
        return f"hi {person.name}"

    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        tool="greet",
                        args={
                            "person": {
                                "name": "Ada",
                                "address": {"city": "London", "zip": "E1"},
                            }
                        },
                    )
                ]
            ),
            ScriptedTurn(text="greeted Ada"),
        ]
    )
    agent = Agent("greeter", model=model, tools=[greet])

    result = await agent.run("greet Ada")
    assert "greeted Ada" in result.output
    assert isinstance(captured["person"], _Person)
    assert captured["person"].name == "Ada"
