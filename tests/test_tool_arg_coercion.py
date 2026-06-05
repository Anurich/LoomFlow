"""Schema-driven coercion of stringified tool arguments.

Models routinely emit numeric / boolean tool args as strings
(``rate_pct="8"``, ``replace_all="true"``). Before this fix loomflow
passed them straight to the typed Python function, which crashed
(``"8" / 100`` → TypeError) and made the agent loop burn turns
retrying. ``Tool.execute`` now coerces string args to their declared
JSON-schema type first.
"""

from __future__ import annotations

import pytest

from loomflow.tools import tool
from loomflow.tools.registry import _coerce_to_json_type

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# The unit: _coerce_to_json_type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("val,jt,expected", [
    ("8", "integer", 8),
    ("8.0", "integer", 8),
    ("-3", "integer", -3),
    ("8", "number", 8.0),
    ("8.5", "number", 8.5),
    ("true", "boolean", True),
    ("True", "boolean", True),
    ("1", "boolean", True),
    ("false", "boolean", False),
    ("0", "boolean", False),
    ("no", "boolean", False),
    ("hello", "string", "hello"),
])
def test_coerce_happy(val, jt, expected) -> None:
    assert _coerce_to_json_type(val, jt) == expected


def test_coerce_non_string_passthrough() -> None:
    # Already-correct types are returned untouched.
    assert _coerce_to_json_type(8, "integer") == 8
    assert _coerce_to_json_type(8.5, "number") == 8.5
    assert _coerce_to_json_type(True, "boolean") is True


def test_coerce_unparseable_passes_original_through() -> None:
    # Garbage that can't be coerced surfaces as-is, so the function's
    # own error is what the model sees (not a masked coercion error).
    assert _coerce_to_json_type("not-a-number", "integer") == "not-a-number"
    assert _coerce_to_json_type("maybe", "boolean") == "maybe"


# ---------------------------------------------------------------------------
# The bug, reproduced end-to-end through Tool.execute
# ---------------------------------------------------------------------------

async def test_typed_tool_survives_stringified_arg() -> None:
    # This is the exact crash from the framework shootout:
    # add_tax(amount, rate_pct) with rate_pct sent as the string "8".
    @tool
    def add_tax(amount: float, rate_pct: float) -> float:
        "Add tax percent to an amount."
        return round(amount * (1 + rate_pct / 100), 2)

    # Pre-fix this raised TypeError("'str'/'int'"); now it coerces.
    out = await add_tax.execute({"amount": 50.0, "rate_pct": "8"})
    assert out == 54.0


async def test_int_param_coerced() -> None:
    @tool
    def read(path: str, offset: int = 0, limit: int = 100) -> str:
        "Read with typed pagination."
        return f"{path}[{offset}:{offset + limit}]"

    out = await read.execute({"path": "f.py", "offset": "10", "limit": "5"})
    assert out == "f.py[10:15]"


async def test_bool_param_coerced() -> None:
    @tool
    def write(path: str, create_parents: bool = False) -> str:
        "Write, optionally creating parent dirs."
        return f"{path} parents={create_parents}"

    out = await write.execute({"path": "a/b.txt", "create_parents": "true"})
    assert out == "a/b.txt parents=True"
    # the false-y string must become False, NOT Python's truthy bool("false")
    out2 = await write.execute({"path": "x", "create_parents": "false"})
    assert out2 == "x parents=False"


async def test_string_param_untouched() -> None:
    @tool
    def echo(text: str) -> str:
        "Echo."
        return text

    assert await echo.execute({"text": "8"}) == "8"  # stays a string
