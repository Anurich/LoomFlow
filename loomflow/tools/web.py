"""Web-search tool factory for agents.

``web_tool(backend="serper" | "duckduckgo")`` returns a single
:class:`Tool` the user wires into an Agent's ``tools=`` list. The
model calls it as ``web_search(query=...)`` and gets back a
markdown-formatted top-N result list — title, URL, and snippet per
result — directly readable, no JSON parsing required in the prompt.

Two backends ship:

* ``"serper"`` — Google search via https://serper.dev. Best quality.
  Needs an API key (env ``SERPER_API_KEY`` or ``api_key=`` kwarg).
  Pulls in ``httpx`` (optional extra: ``loomflow[serper]``).
* ``"duckduckgo"`` (default) — Free, no key. Quality varies; DDG
  rate-limits. Pulls in ``duckduckgo-search`` (optional extra:
  ``loomflow[duckduckgo]``).

Both extras together: ``loomflow[web]``.

SDK imports are lazy — ``import loomflow.tools`` doesn't require
either extra; the import fires when the tool actually runs. Same
convention as every other backend-bearing module in loomflow.

The ``backend=`` selector is the seam for future expansion (Brave,
Tavily, Google PSE) and for a future ``web_fetch_tool`` covering
the read-a-specific-URL case.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from ..core import ConfigError
from .registry import Tool

# Shared description so both backends present an identical
# interface to the model — switching backends behind the same
# tool name doesn't change the contract the model has learned.
_TOOL_DESCRIPTION = (
    "Search the web for information not in the local codebase: "
    "library docs, API references, error-message context, recent "
    "best practices, or anything time-sensitive the agent might "
    "not know from training. Returns a top-N list of results "
    "(title + URL + snippet) as markdown. Pick focused keyword "
    "queries — not full sentences."
)


def web_tool(
    *,
    backend: Literal["serper", "duckduckgo"] = "duckduckgo",
    api_key: str | None = None,
    max_results: int = 5,
    timeout: float = 10.0,
) -> Tool:
    """Build a :class:`Tool` that searches the web.

    Returns one tool named ``web_search``. The model calls it with
    a ``query`` string and gets back a markdown-formatted top-N
    result list.

    Args:
        backend: Which search backend to use.

            * ``"serper"`` — Google via https://serper.dev. Best
              quality. Requires an API key (see ``api_key``).
              Optional extra: ``pip install loomflow[serper]``.
            * ``"duckduckgo"`` (default) — Free, no key. Lower and
              more variable quality; DDG rate-limits. Optional
              extra: ``pip install loomflow[duckduckgo]``.

        api_key: Serper API key. If ``None`` and
            ``backend="serper"``, the ``SERPER_API_KEY`` env var is
            read. Ignored for ``duckduckgo``.
        max_results: Top-N results to return (default 5). Higher
            costs more tokens in the prompt; 5 is a reasonable
            balance.
        timeout: Network timeout in seconds (default 10). Applies
            to both backends.

    Returns:
        A :class:`Tool` named ``web_search`` with a single
        ``query: str`` parameter.

    Raises:
        ConfigError: If ``backend`` is not one of the two supported
            values, or ``backend="serper"`` is requested with no
            key available (neither ``api_key=`` nor
            ``SERPER_API_KEY`` env).

    Example::

        from loomflow import Agent
        from loomflow.tools import web_tool

        agent = Agent(
            instructions="Look up APIs and write code.",
            tools=[web_tool(backend="serper")],
        )
    """
    if backend == "serper":
        return _build_serper_tool(
            api_key=api_key,
            max_results=max_results,
            timeout=timeout,
        )
    if backend == "duckduckgo":
        return _build_ddg_tool(
            max_results=max_results,
            timeout=timeout,
        )
    raise ConfigError(
        f"unknown web backend {backend!r} "
        "(supported: 'serper', 'duckduckgo')"
    )


# ---------------------------------------------------------------------------
# Serper backend (Google via https://serper.dev)
# ---------------------------------------------------------------------------


def _build_serper_tool(
    *, api_key: str | None, max_results: int, timeout: float
) -> Tool:
    resolved_key = api_key or os.environ.get("SERPER_API_KEY")
    if not resolved_key:
        raise ConfigError(
            "web_tool(backend='serper') needs an API key. Pass "
            "api_key= or set SERPER_API_KEY in the environment. "
            "Get a free one at https://serper.dev"
        )

    async def web_search(query: str) -> str:
        return await _serper_search(
            query=query,
            api_key=resolved_key,
            max_results=max_results,
            network_timeout=timeout,
        )

    return Tool(
        name="web_search",
        description=_TOOL_DESCRIPTION,
        fn=web_search,
        input_schema=_QUERY_SCHEMA,
    )


async def _serper_search(
    *,
    query: str,
    api_key: str,
    max_results: int,
    network_timeout: float,
) -> str:
    # Lazy import — `import loomflow.tools` must not require httpx
    # for users who don't enable the serper backend.
    # `network_timeout` (not the more natural `timeout`) avoids
    # ruff's ASYNC109 — `timeout` as an async-fn param suggests a
    # sync-style block; we're just forwarding to httpx which has
    # its own timeout machinery.
    try:
        import httpx
    except ImportError as exc:
        raise ConfigError(
            "web_tool(backend='serper') needs httpx — "
            "`pip install loomflow[serper]`"
        ) from exc

    try:
        async with httpx.AsyncClient(timeout=network_timeout) as client:
            resp = await client.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": max_results},
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        # Return a tool-result string rather than raise — the agent
        # can see "search failed: <reason>" and decide what to do
        # next (retry, ask the user, try a different query).
        return f"(web search failed: {exc})"

    organic = data.get("organic", []) or []
    items: list[dict[str, Any]] = []
    for raw in organic[:max_results]:
        items.append(
            {
                "title": raw.get("title", "(no title)"),
                "url": raw.get("link", ""),
                "snippet": raw.get("snippet", ""),
            }
        )
    return _format_results(items)


# ---------------------------------------------------------------------------
# DuckDuckGo backend (free, no key)
# ---------------------------------------------------------------------------


def _build_ddg_tool(*, max_results: int, timeout: float) -> Tool:
    async def web_search(query: str) -> str:
        return await _ddg_search(
            query=query,
            max_results=max_results,
            network_timeout=timeout,
        )

    return Tool(
        name="web_search",
        description=_TOOL_DESCRIPTION,
        fn=web_search,
        input_schema=_QUERY_SCHEMA,
    )


async def _ddg_search(
    *, query: str, max_results: int, network_timeout: float
) -> str:
    # ``network_timeout`` (not ``timeout``) for the same reason as
    # ``_serper_search`` — ruff ASYNC109 dislikes ``timeout`` as
    # an async-fn param. We're forwarding to DDGS which has its
    # own timeout, not blocking ourselves.
    try:
        from duckduckgo_search import DDGS  # type: ignore[import-not-found, import-untyped]
    except ImportError as exc:
        raise ConfigError(
            "web_tool(backend='duckduckgo') needs duckduckgo-search "
            "— `pip install loomflow[duckduckgo]`"
        ) from exc

    import anyio

    # DDGS is synchronous (uses requests under the hood). Push the
    # call to a worker thread so it doesn't block the event loop —
    # standard loomflow pattern for sync SDKs.
    def _do() -> list[dict[str, Any]]:
        with DDGS(timeout=int(network_timeout)) as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    try:
        raw_results = await anyio.to_thread.run_sync(_do)
    except Exception as exc:  # noqa: BLE001 — DDG raises various
        # Same shape as serper's HTTP error — return a string so
        # the agent can react rather than crash.
        return f"(web search failed: {exc})"

    items: list[dict[str, Any]] = []
    for raw in raw_results:
        items.append(
            {
                "title": raw.get("title", "(no title)"),
                "url": raw.get("href", ""),
                "snippet": raw.get("body", ""),
            }
        )
    return _format_results(items)


# ---------------------------------------------------------------------------
# Shared schema + output formatting
# ---------------------------------------------------------------------------

_QUERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "A short, focused search query. Keywords beat "
                "full sentences."
            ),
        }
    },
    "required": ["query"],
}


def _format_results(items: list[dict[str, Any]]) -> str:
    """Render the search items as markdown the model can read at
    a glance — same shape regardless of backend, so switching
    backends doesn't change what the model sees."""
    if not items:
        return "(no results)"
    out: list[str] = []
    for i, item in enumerate(items, 1):
        title = item.get("title") or "(no title)"
        url = item.get("url") or ""
        snippet = item.get("snippet") or ""
        if url:
            out.append(f"{i}. [{title}]({url})")
        else:
            out.append(f"{i}. {title}")
        if snippet:
            out.append(f"   {snippet}")
        out.append("")
    return "\n".join(out).rstrip()
