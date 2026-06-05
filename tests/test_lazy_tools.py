"""Lazy tool loading — :class:`LazyToolHost` + Agent integration.

The load-bearing invariant is **cache stability**: ``list_tools()`` must
return a byte-identical list across turns so the prompt-cache tool
breakpoint is never invalidated. That assertion is the one this whole
feature lives or dies on.
"""

from __future__ import annotations

import pytest

from loomflow import Agent, Tuning
from loomflow.core.errors import ConfigError
from loomflow.model.echo import EchoModel
from loomflow.tools import InProcessToolHost, LazyToolHost, tool

pytestmark = pytest.mark.anyio


@tool
def alpha(x: str) -> str:
    "Search alpha things."
    return x


@tool
def beta(y: int) -> int:
    "Compute a beta value."
    return y


def _host(eager: set[str] | None = None, meta: str = "expand_tool") -> LazyToolHost:
    return LazyToolHost(InProcessToolHost([alpha, beta]), eager=eager, meta_tool_name=meta)


# ---------------------------------------------------------------------------
# Exposure + cache stability
# ---------------------------------------------------------------------------

async def test_only_eager_and_meta_exposed() -> None:
    h = _host(eager={"alpha"})
    names = sorted(d.name for d in await h.list_tools())
    assert names == ["alpha", "expand_tool"]  # beta hidden


async def test_all_lazy_exposes_only_meta() -> None:
    h = _host(eager=set())
    names = sorted(d.name for d in await h.list_tools())
    assert names == ["expand_tool"]


async def test_list_tools_is_byte_stable_across_calls() -> None:
    # THE cache-safety invariant: identical defs every turn.
    h = _host(eager={"alpha"})
    a = await h.list_tools()
    b = await h.list_tools()
    assert [(d.name, d.description, d.input_schema) for d in a] == [
        (d.name, d.description, d.input_schema) for d in b
    ]


async def test_meta_tool_schema_enumerates_lazy_names() -> None:
    h = _host(eager=set())
    meta = next(d for d in await h.list_tools() if d.name == "expand_tool")
    enum = meta.input_schema["properties"]["name"]["enum"]
    assert set(enum) == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# expand + dispatch
# ---------------------------------------------------------------------------

async def test_expand_returns_full_schema() -> None:
    h = _host(eager=set())
    r = await h.call("expand_tool", {"name": "beta"}, call_id="c")
    assert r.ok
    assert "beta" in r.output
    assert "y" in r.output  # the arg name from the schema


async def test_expand_unknown_errors_with_valid_list() -> None:
    h = _host(eager=set())
    r = await h.call("expand_tool", {"name": "nope"}, call_id="c")
    assert not r.ok
    assert "alpha" in (r.error or "") and "beta" in (r.error or "")


async def test_real_tool_executes_without_prior_expand() -> None:
    # Expansion is advisory, not a gate — the real tool runs regardless.
    h = _host(eager=set())
    r = await h.call("beta", {"y": 7}, call_id="c")
    assert r.ok
    assert r.output == 7


async def test_catalog_section_lists_lazy_tools() -> None:
    h = _host(eager=set())
    cat = h.catalog_section()
    assert "alpha" in cat and "beta" in cat
    assert "Search alpha things." in cat


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------

def test_non_inprocess_base_raises() -> None:
    class FakeHost:  # not an InProcessToolHost
        pass

    with pytest.raises(ConfigError, match="InProcessToolHost"):
        LazyToolHost(FakeHost())  # type: ignore[arg-type]


def test_meta_name_collision_raises() -> None:
    base = InProcessToolHost([alpha])
    with pytest.raises(ConfigError, match="collides"):
        LazyToolHost(base, meta_tool_name="alpha")


def test_unknown_eager_name_raises() -> None:
    base = InProcessToolHost([alpha, beta])
    with pytest.raises(ConfigError, match="unknown tool"):
        LazyToolHost(base, eager={"gamma"})


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------

async def test_agent_lazy_on_wraps_and_injects_catalog() -> None:
    a = Agent("help", model=EchoModel(), tools=[alpha, beta], tuning=Tuning(lazy_tools=True))
    assert type(a._tool_host).__name__ == "LazyToolHost"
    names = sorted(d.name for d in await a._tool_host.list_tools())
    assert names == ["expand_tool"]
    assert "expand_tool" in a._instructions
    assert "alpha" in a._instructions  # catalog body


async def test_agent_lazy_off_leaves_host_untouched() -> None:
    # Regression guard: default behaviour byte-identical to before.
    a = Agent("help", model=EchoModel(), tools=[alpha, beta])
    assert isinstance(a._tool_host, InProcessToolHost)
    names = sorted(d.name for d in await a._tool_host.list_tools())
    assert names == ["alpha", "beta"]


async def test_agent_eager_list_keeps_named_tools_eager() -> None:
    a = Agent(
        "help", model=EchoModel(), tools=[alpha, beta],
        tuning=Tuning(lazy_tools=["alpha"]),
    )
    names = sorted(d.name for d in await a._tool_host.list_tools())
    assert names == ["alpha", "expand_tool"]
