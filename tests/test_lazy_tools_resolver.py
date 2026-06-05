"""``resolve_lazy_tools`` — the Tuning(lazy_tools=...) spec normaliser."""

from __future__ import annotations

import pytest

from loomflow.core.errors import ConfigError
from loomflow.tools.lazy import resolve_lazy_tools


def test_true_means_all_lazy() -> None:
    eager, meta = resolve_lazy_tools(True)
    assert eager == set()
    assert meta == "expand_tool"


def test_list_form_sets_eager() -> None:
    eager, meta = resolve_lazy_tools(["read", "grep"])
    assert eager == {"read", "grep"}
    assert meta == "expand_tool"


def test_dict_form_eager_and_meta_name() -> None:
    eager, meta = resolve_lazy_tools(
        {"eager": ["read"], "meta_tool_name": "show_tool"}
    )
    assert eager == {"read"}
    assert meta == "show_tool"


def test_dict_unknown_key_raises() -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        resolve_lazy_tools({"eagr": ["read"]})


def test_dict_eager_not_list_raises() -> None:
    with pytest.raises(ConfigError, match="'eager' must be a list"):
        resolve_lazy_tools({"eager": "read"})


def test_dict_meta_name_empty_raises() -> None:
    with pytest.raises(ConfigError, match="meta_tool_name"):
        resolve_lazy_tools({"meta_tool_name": ""})


def test_list_non_string_raises() -> None:
    with pytest.raises(ConfigError, match="tool-name strings"):
        resolve_lazy_tools([1, 2])


def test_bad_type_raises() -> None:
    with pytest.raises(ConfigError, match="bool | list"):
        resolve_lazy_tools(42)
