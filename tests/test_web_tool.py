"""Tests for ``loomflow.tools.web_tool``.

Backend selection + config errors are pure (no mocking). The
network-bound paths (Serper HTTP, DuckDuckGo) are mocked at the
SDK boundary so tests are deterministic + offline.

Two failure modes worth locking down hard:

* HTTP error → tool returns ``"(web search failed: ...)"`` string,
  NOT a raised exception. The agent must be able to see the
  failure as a tool result and decide what to do next, not have
  the whole turn explode.
* The result-formatting shape is the model's interface. Both
  backends must produce IDENTICAL markdown shapes — same numbered
  list, same ``[title](url)`` form, same indented snippet — so
  switching backends doesn't change what the model has learned to
  parse.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from loomflow import ConfigError
from loomflow.tools import Tool, web_tool
from loomflow.tools.web import _format_results

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Config errors (pure — no mocking)
# ---------------------------------------------------------------------------


def test_unknown_backend_raises_configerror() -> None:
    with pytest.raises(ConfigError, match="unknown web backend"):
        web_tool(backend="bing")  # type: ignore[arg-type]


def test_serper_without_key_raises_configerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No api_key= AND no SERPER_API_KEY env → ConfigError with a
    # message that mentions both ways to supply the key (otherwise
    # the user has to read source to figure out how to fix it).
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="api_key|SERPER_API_KEY"):
        web_tool(backend="serper")


def test_serper_picks_up_env_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    t = web_tool(backend="serper")
    assert isinstance(t, Tool)
    assert t.name == "web_search"


def test_serper_picks_up_explicit_key() -> None:
    t = web_tool(backend="serper", api_key="explicit-test-key")
    assert t.name == "web_search"


def test_ddg_default_needs_no_key() -> None:
    # No env, no kwarg. DDG is the no-friction default.
    t = web_tool()
    assert t.name == "web_search"


def test_both_backends_produce_identical_tool_shape() -> None:
    # The model should not see a different tool when the user
    # swaps backends. Same name, same input schema.
    serper = web_tool(backend="serper", api_key="k")
    ddg = web_tool(backend="duckduckgo")
    assert serper.name == ddg.name == "web_search"
    assert serper.input_schema == ddg.input_schema
    assert serper.description == ddg.description


# ---------------------------------------------------------------------------
# Result formatting (pure helper — covers both backends' output shape)
# ---------------------------------------------------------------------------


def test_format_results_empty_says_no_results() -> None:
    # The model needs to KNOW the search returned nothing — don't
    # hand it an empty string it might mistake for a tool that
    # didn't run.
    assert _format_results([]) == "(no results)"


def test_format_results_renders_markdown_list() -> None:
    out = _format_results(
        [
            {"title": "A", "url": "https://a.example", "snippet": "snip A"},
            {"title": "B", "url": "https://b.example", "snippet": "snip B"},
        ]
    )
    # Numbered, [title](url), indented snippet — what the model
    # has learned to parse.
    assert "1. [A](https://a.example)" in out
    assert "   snip A" in out
    assert "2. [B](https://b.example)" in out
    assert "   snip B" in out


def test_format_results_handles_missing_fields() -> None:
    # Defensive: a backend that doesn't return a url/snippet
    # shouldn't crash — render what's there.
    out = _format_results([{"title": "T"}])
    assert "T" in out
    # No empty-bracket markdown noise.
    assert "[T]()" not in out


# ---------------------------------------------------------------------------
# Serper backend — mock at httpx.AsyncClient boundary
# ---------------------------------------------------------------------------


def _mock_httpx_response(payload: dict[str, Any]) -> MagicMock:
    """Build a fake httpx Response that returns ``payload`` from
    .json() and is a no-op on raise_for_status()."""
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _patched_httpx_client(response: Any) -> Any:
    """Patch context manager — `httpx.AsyncClient(...)` returns
    an AsyncMock acting as an async context manager whose
    ``.post(...)`` returns ``response``."""
    fake_client = AsyncMock()
    fake_client.__aenter__.return_value = fake_client
    fake_client.__aexit__.return_value = False
    fake_client.post = AsyncMock(return_value=response)
    return patch("httpx.AsyncClient", return_value=fake_client)


async def test_serper_happy_path_formats_results() -> None:
    payload = {
        "organic": [
            {
                "title": "Python docs",
                "link": "https://docs.python.org",
                "snippet": "The official docs.",
            },
            {
                "title": "PEP 8",
                "link": "https://peps.python.org/pep-0008/",
                "snippet": "Style guide.",
            },
        ]
    }
    with _patched_httpx_client(_mock_httpx_response(payload)):
        t = web_tool(backend="serper", api_key="k")
        out = await t.fn(query="python")

    assert "Python docs" in out
    assert "https://docs.python.org" in out
    assert "The official docs." in out
    assert "PEP 8" in out


async def test_serper_empty_organic_says_no_results() -> None:
    # Serper returned 200 but no organic results — the wrapper
    # must produce the same "(no results)" the helper does.
    with _patched_httpx_client(_mock_httpx_response({"organic": []})):
        t = web_tool(backend="serper", api_key="k")
        out = await t.fn(query="obscure")
    assert "(no results)" in out


async def test_serper_http_error_returns_failure_string() -> None:
    # 401 / 500 / network blip — the tool returns a "(web search
    # failed: ...)" string instead of raising. Agents can see the
    # failure as a tool result and act on it.
    import httpx

    fake_client = AsyncMock()
    fake_client.__aenter__.return_value = fake_client
    fake_client.__aexit__.return_value = False
    fake_client.post = AsyncMock(
        side_effect=httpx.HTTPError("boom")
    )
    with patch("httpx.AsyncClient", return_value=fake_client):
        t = web_tool(backend="serper", api_key="k")
        out = await t.fn(query="anything")

    assert "failed" in out.lower()
    assert "boom" in out


# ---------------------------------------------------------------------------
# DuckDuckGo backend — mock at DDGS class boundary
# ---------------------------------------------------------------------------


def _patched_ddgs(results: Any) -> Any:
    """Patch ``duckduckgo_search.DDGS`` so the ``with DDGS(...) as
    ddgs`` block returns a stub whose ``.text(...)`` yields
    ``results`` (either a list or a side-effect exception)."""
    instance = MagicMock()
    instance.__enter__.return_value = instance
    instance.__exit__.return_value = False
    if isinstance(results, Exception):
        instance.text.side_effect = results
    else:
        instance.text.return_value = results
    return patch(
        "duckduckgo_search.DDGS", return_value=instance
    )


async def test_ddg_happy_path_formats_results() -> None:
    # ``duckduckgo_search`` is an optional dep (``loomflow[web]``).
    # In CI environments without the ``web`` extra installed (e.g.
    # base/dev matrix) the patch target can't resolve and the test
    # can't run — skip gracefully rather than fail.
    pytest.importorskip("duckduckgo_search")
    raw = [
        {
            "title": "Async I/O",
            "href": "https://docs.python.org/asyncio.html",
            "body": "asyncio module.",
        }
    ]
    with _patched_ddgs(raw):
        t = web_tool(backend="duckduckgo")
        out = await t.fn(query="asyncio")
    assert "Async I/O" in out
    assert "asyncio.html" in out


async def test_ddg_error_returns_failure_string() -> None:
    # Same optional-dep guard as the happy-path test above.
    pytest.importorskip("duckduckgo_search")
    # DDG raises various — RatelimitException, etc. We catch
    # broadly and surface as a failure string.
    with _patched_ddgs(RuntimeError("rate limit")):
        t = web_tool(backend="duckduckgo")
        out = await t.fn(query="anything")
    assert "failed" in out.lower()
    assert "rate limit" in out
