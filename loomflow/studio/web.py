"""The Studio web app: browse recorded traces, agents and experiments.

``create_app(path)`` returns a plain ASGI-3 callable, like
:func:`loomflow.serve.create_app`: no web framework, run it under any
ASGI server. ``loom studio`` runs it with uvicorn on 127.0.0.1.

The app only reads the database (it opens it read-only) and serves no
scripts; every recorded value is HTML-escaped and the pages carry a
Content-Security-Policy that forbids scripts. It has no login: keep it
on localhost, or behind your own authentication, because traces hold
prompts, replies and tool payloads.

Routes (all ``GET``): ``/traces`` (filter with ``q``, ``agent``,
``user``, ``model``, ``status``, ``problems``, ``session``,
``version``, ``range``, ``project``, ``page``), ``/traces/{id}``
(``?span=`` selects a step), ``/traces/{id}.json``, ``/agents``,
``/agents/{key}``, ``/experiments``, ``/experiments/{id}``,
``/experiments/compare?a=&b=&show=all|changed|regressed``,
``/health``.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable, MutableMapping
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote

import anyio

from . import _pages as pages
from ._assets import CSS
from ._state import DEFAULT_PATH
from ._store import StudioStore

__all__ = ["create_app"]

_LOG = logging.getLogger("loomflow.studio")

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

PAGE_SIZE = 50
_HTML = "text/html; charset=utf-8"
_CSP = "default-src 'none'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; form-action 'self'; base-uri 'none'"


class _Response:
    def __init__(
        self,
        body: str | bytes,
        *,
        status: int = 200,
        content_type: str = _HTML,
        headers: list[tuple[bytes, bytes]] | None = None,
    ) -> None:
        self.body = body.encode("utf-8") if isinstance(body, str) else body
        self.status = status
        self.content_type = content_type
        self.headers = headers or []


def create_app(path: str | Path | None = None) -> ASGIApp:
    """The Studio web app over the database at ``path`` (default
    ``.loomflow/studio.db``, the file :func:`loomflow.studio.enable`
    writes by default)."""
    from .. import __version__

    db_path = Path(path).expanduser() if path is not None else DEFAULT_PATH
    shown_path = str(db_path)

    def page(title: str, active: str, body: str, project: str | None, store: StudioStore | None) -> str:
        projects = store.projects() if store is not None else []
        return pages.layout(
            title=title, active=active, body=body, project=project,
            projects=projects, db_path=shown_path, version=__version__,
        )

    def respond(path_info: str, query: dict[str, list[str]]) -> _Response:
        def arg(name: str) -> str | None:
            values = query.get(name)
            return values[0].strip() or None if values else None

        if path_info == "/":
            return _Response(b"", status=302, headers=[(b"location", b"/traces")])
        if path_info == "/static/studio.css":
            return _Response(CSS, content_type="text/css; charset=utf-8",
                             headers=[(b"cache-control", b"max-age=300")])
        if path_info == "/health":
            return _Response(json.dumps({"ok": True, "db": shown_path, "exists": db_path.exists()}),
                             content_type="application/json")
        project = arg("project")
        if not db_path.exists():
            return _Response(page("No runs yet", "traces", pages.empty_db(shown_path), None, None))
        store = StudioStore(db_path, readonly=True)
        labels = {a["agent_key"]: a["name"] for a in store.list_agents(None)}

        if path_info == "/traces":
            return _Response(page("Traces", "traces", _traces(store, query, arg, project, labels), project, store))
        if path_info.startswith("/traces/"):
            trace_id = path_info[len("/traces/"):]
            as_json = trace_id.endswith(".json")
            trace = store.get_trace(trace_id.removesuffix(".json"))
            if trace is None:
                return _Response(page("Not found", "traces", pages.not_found("Trace"), project, store), status=404)
            if as_json:
                return _Response(json.dumps(trace, indent=2, default=str), content_type="application/json")
            body = pages.trace_page(trace, selected=arg("span"), agent_labels=labels)
            return _Response(page(pages.first_line(trace.get("input"), 60) or "Trace", "traces", body,
                                  trace["project"], store))
        if path_info == "/agents" or path_info.startswith("/agents/"):
            agents = store.list_agents(project)
            key = unquote(path_info[len("/agents/"):]) if path_info.startswith("/agents/") else None
            selected = next((a for a in agents if a["agent_key"] == key), None) if key else (agents[0] if agents else None)
            if key and selected is None:
                return _Response(page("Not found", "agents", pages.not_found("Agent"), project, store), status=404)
            versions = store.agent_versions(selected["project"], selected["agent_key"]) if selected else []
            body = pages.agents_page(agents, selected=selected, versions=versions, project=project)
            return _Response(page("Agents", "agents", body, project, store))
        if path_info == "/experiments":
            body = pages.experiments_page(store.list_experiments(project), project=project)
            return _Response(page("Experiments", "experiments", body, project, store))
        if path_info == "/experiments/compare":
            a = store.get_experiment(arg("a") or "")
            b = store.get_experiment(arg("b") or "")
            if a is None or b is None:
                return _Response(page("Not found", "experiments", pages.not_found("Experiment"), project, store),
                                 status=404)
            show = arg("show") if arg("show") in ("all", "changed", "regressed") else "all"
            body = pages.compare_page(a, b, show=show or "all")
            return _Response(page("Compare", "experiments", body, b["project"], store))
        if path_info.startswith("/experiments/"):
            exp = store.get_experiment(path_info[len("/experiments/"):])
            if exp is None:
                return _Response(page("Not found", "experiments", pages.not_found("Experiment"), project, store),
                                 status=404)
            return _Response(page(exp["name"], "experiments", pages.experiment_page(exp), exp["project"], store))
        return _Response(page("Not found", "traces", pages.not_found("Page"), project, store), status=404)

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        method = scope.get("method", "GET")
        if method not in ("GET", "HEAD"):
            response = _Response("Studio is read-only.", status=405, content_type="text/plain; charset=utf-8",
                                 headers=[(b"allow", b"GET, HEAD")])
        else:
            query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
            try:
                response = await anyio.to_thread.run_sync(respond, scope.get("path", "/"), query)
            except Exception:  # noqa: BLE001 — one bad page mustn't take the app down
                _LOG.exception("Studio page failed: %s", scope.get("path"))
                response = _Response("Studio couldn't render this page; see the server log.", status=500,
                                     content_type="text/plain; charset=utf-8")
        headers = [
            (b"content-type", response.content_type.encode("latin-1")),
            (b"content-length", str(len(response.body)).encode("latin-1")),
            (b"content-security-policy", _CSP.encode("latin-1")),
            (b"x-content-type-options", b"nosniff"),
            (b"referrer-policy", b"no-referrer"),
            *response.headers,
        ]
        await send({"type": "http.response.start", "status": response.status, "headers": headers})
        await send({"type": "http.response.body", "body": b"" if method == "HEAD" else response.body})

    return app


def _traces(
    store: StudioStore,
    query: dict[str, list[str]],
    arg: Callable[[str], str | None],
    project: str | None,
    labels: dict[str, str],
) -> str:
    try:
        page_no = max(int(arg("page") or 0), 0)
    except ValueError:
        page_no = 0
    try:
        version = int(arg("version") or "") if arg("version") else None
    except ValueError:
        version = None
    rng = arg("range") if arg("range") in pages.RANGES else "all"
    seconds = pages.RANGES[rng or "all"][1]
    filters: dict[str, Any] = {
        "q": arg("q"), "agent": arg("agent"), "user": arg("user"), "model": arg("model"),
        "status": arg("status"), "problems": "1" if arg("problems") == "1" else None,
        "session": arg("session"), "version": version, "range": rng if rng != "all" else None,
    }
    data = store.list_traces(
        project=project,
        search=filters["q"],
        agent=filters["agent"],
        user=filters["user"],
        model=filters["model"],
        session=filters["session"],
        status=filters["status"],
        version=version,
        since=time.time() - seconds if seconds else None,
        problems=bool(filters["problems"]),
        limit=PAGE_SIZE,
        offset=page_no * PAGE_SIZE,
    )
    if not data["total"] and not any(v for k, v in filters.items() if k != "range") and not store.projects():
        return pages.empty_db(str(store.path))
    return pages.traces_page(
        data,
        filters={k: v for k, v in filters.items() if v not in (None, "")},
        options=store.filter_options(project),
        project=project,
        page=page_no,
        page_size=PAGE_SIZE,
        agent_labels=labels,
    )
