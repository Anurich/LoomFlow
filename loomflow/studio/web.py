"""The Studio web app: browse recorded traces, agents and experiments.

``create_app(path)`` returns a plain ASGI-3 callable, like
:func:`loomflow.serve.create_app`: no web framework, run it under any
ASGI server. ``loom studio`` runs it with uvicorn on 127.0.0.1.

The app reads recorded runs and changes only what people add on its
pages: scores, and cases in datasets (``read_only=True`` turns that
off). It serves no scripts; every recorded value is HTML-escaped and
the pages carry a Content-Security-Policy that forbids scripts. It has
no login: keep it on localhost, or behind your own authentication,
because traces hold prompts, replies and tool payloads.

Forms post back with a token only this app's pages carry, and a
request is refused unless its ``Host`` is a name the app answers to
(``allowed_hosts``, default the loopback names), so another site open
in your browser can neither post to it nor, by pointing its own name
at 127.0.0.1, read it.

Routes (``GET``): ``/traces`` (filter with ``q``, ``agent``, ``user``,
``model``, ``status``, ``problems``, ``unscored``, ``session``,
``version``, ``range``, ``project``, ``page``), ``/traces/{id}``
(``?span=`` selects a step), ``/traces/{id}.json``, ``/agents``,
``/agents/{key}``, ``/experiments``, ``/experiments/{id}``,
``/experiments/compare?a=&b=&show=all|changed|regressed``,
``/datasets``, ``/datasets/{id}``, ``/datasets/{id}.jsonl``,
``/evals/new``, ``/health``. ``POST``: ``/traces/{id}/scores``,
``/traces/{id}/datasets``, ``/scores/{n}/delete``,
``/datasets/{id}/cases/{case}/delete``, ``/evals``,
``/experiments/{id}/cancel``.

With ``agents=``, the page can run evals on those agents (see
:mod:`loomflow.studio._evals`).
"""

from __future__ import annotations

import hmac
import json
import logging
import re
import secrets
import time
from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping, MutableMapping
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlsplit

import anyio

from . import _evals
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
MAX_FORM = 1_000_000
"""Bytes accepted in one form post."""
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")
_HTML = "text/html; charset=utf-8"
_TEXT = "text/plain; charset=utf-8"
_ID = r"[A-Za-z0-9_-]{1,80}"
_CSP = "default-src 'none'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; form-action 'self'; base-uri 'none'"


class _Response:
    def __init__(
        self,
        body: str | bytes,
        *,
        status: int = 200,
        content_type: str = _HTML,
        headers: list[tuple[bytes, bytes]] | None = None,
        then: Callable[[], _Response] | None = None,
    ) -> None:
        self.body = body.encode("utf-8") if isinstance(body, str) else body
        self.status = status
        self.content_type = content_type
        self.headers = headers or []
        # Work that has to run on the event loop (starting or cancelling
        # an eval); its response is the one sent.
        self.then = then


def create_app(
    path: str | Path | None = None,
    *,
    read_only: bool = False,
    allowed_hosts: Collection[str] | None = None,
    agents: Iterable[Any] | Mapping[str, Any] = (),
) -> ASGIApp:
    """The Studio web app over the database at ``path`` (default
    ``.loomflow/studio.db``, the file :func:`loomflow.studio.enable`
    writes by default).

    ``agents`` are the agents the page may run evals on: ``Agent``
    objects, or ``"module:attribute"`` names to import. The app runs
    them in its own process, so their models need keys here.
    ``read_only=True`` hides the forms and refuses every change.
    ``allowed_hosts`` are the host names the app answers to (default:
    ``localhost``, ``127.0.0.1``, ``::1`` and ``*.localhost``); pass
    the name you reach it by when it runs elsewhere, or ``["*"]`` to
    answer to any.
    """
    from .. import __version__

    db_path = Path(path).expanduser() if path is not None else DEFAULT_PATH
    shown_path = str(db_path)
    token = secrets.token_urlsafe(24)
    form_token = None if read_only else token
    hosts = {h.lower().strip("[]") for h in (allowed_hosts if allowed_hosts is not None else LOOPBACK_HOSTS)}
    targets = _evals.targets(agents)
    runner = _evals.Runner(db_path, targets) if targets and not read_only else None

    def host_ok(headers: dict[str, str]) -> bool:
        host = headers.get("host")
        if host is None or "*" in hosts:
            return True
        name = _host_name(host)
        return name in hosts or (allowed_hosts is None and name.endswith(".localhost"))

    def page(
        title: str, active: str, body: str, project: str | None, store: StudioStore | None,
        refresh: int | None = None,
    ) -> str:
        projects = store.projects() if store is not None else []
        return pages.layout(
            title=title, active=active, body=body, project=project,
            projects=projects, db_path=shown_path, version=__version__, refresh=refresh,
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
            structure = None
            if trace.get("agent_key") and trace.get("agent_version"):
                definition = store.version_definition(trace["project"], trace["agent_key"], trace["agent_version"])
                structure = (definition or {}).get("structure")
            notice = None
            if arg("saved") == "score":
                notice = "Score saved."
            elif arg("added"):
                added = store.get_dataset(arg("added") or "")
                if added is not None:
                    notice = pages.added_notice(added)
            body = pages.trace_page(
                trace, selected=arg("span"), agent_labels=labels, structure=structure, token=form_token,
                notice=notice, score_keys=store.score_keys(trace["project"]),
                datasets=[d["name"] for d in sorted(store.list_datasets(trace["project"]),
                                                    key=lambda d: d.get("updated_at") or 0, reverse=True)],
            )
            return _Response(page(pages.first_line(trace.get("input"), 60) or "Trace", "traces", body,
                                  trace["project"], store))
        if path_info == "/agents" or path_info.startswith("/agents/"):
            agents = store.list_agents(project)
            key = unquote(path_info[len("/agents/"):]) if path_info.startswith("/agents/") else None
            selected = next((a for a in agents if a["agent_key"] == key), None) if key else (agents[0] if agents else None)
            if key and selected is None:
                return _Response(page("Not found", "agents", pages.not_found("Agent"), project, store), status=404)
            versions = store.agent_versions(selected["project"], selected["agent_key"]) if selected else []
            body = pages.agents_page(
                agents, selected=selected, versions=versions, project=project, shown_version=_int(arg("version")),
                evaluable=runner is not None and selected is not None and selected["agent_key"] in runner.targets,
            )
            return _Response(page("Agents", "agents", body, project, store))
        if path_info == "/experiments":
            body = pages.experiments_page(store.list_experiments(project), project=project, can_run=runner is not None)
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
            exp_id = path_info[len("/experiments/"):]
            exp = store.get_experiment(exp_id)
            run = runner.runs.get(exp_id) if runner is not None else None
            if exp is None and run is not None:
                body = pages.starting_page({"name": run.name, "agent": run.target.name, "dataset": run.dataset})
                return _Response(page(run.name, "experiments", body, project, store, refresh=2))
            if exp is None:
                return _Response(page("Not found", "experiments", pages.not_found("Experiment"), project, store),
                                 status=404)
            # Until the file says it ended: one this app started is about
            # to; one started elsewhere might have stopped without saying.
            refresh = None
            if exp.get("ended_at") is None:
                if run is not None:
                    refresh = 3 if run.running else 1
                elif time.time() - exp["started_at"] < 3600:
                    refresh = 5
            body = pages.experiment_page(
                exp, token=form_token, cancellable=run is not None and run.running,
                previous=store.previous_experiment(exp), refreshing=refresh is not None,
            )
            return _Response(page(exp["name"], "experiments", body, exp["project"], store, refresh=refresh))
        if path_info == "/evals/new":
            if runner is None:
                why = ("This Studio is read-only." if read_only and targets else
                       "Start Studio with the agents it may run, for example "
                       "<span class=\"mono\">loom studio --agent app.agents:billing</span>, or list them under "
                       "<span class=\"mono\">[tool.loomflow.studio] agents</span> in pyproject.toml.")
                body = f'<div class="head"><h1>Run an eval</h1></div><div class="panel empty"><p>{why}</p></div>'
                return _Response(page("Run an eval", "experiments", body, project, store))
            datasets = store.list_datasets(None)
            chosen = next((d for d in datasets if d["id"] == arg("dataset")), None) or next(
                (d for d in datasets if d.get("cases")), None)
            cost_project = chosen["project"] if chosen else project
            info = [
                {"key": t.key, "name": t.name, "spec": t.spec,
                 "cost": store.average_cost(cost_project, t.key) if cost_project else None}
                for t in runner.targets.values()
            ]
            busy = runner.busy()
            body = pages.eval_form(
                info, datasets, token=token, dataset_id=chosen["id"] if chosen else None, agent_key=arg("agent"),
                metrics=_evals.METRICS, busy={"id": busy.id, "name": busy.name} if busy else None,
            )
            return _Response(page("Run an eval", "experiments", body, project, store))
        if path_info == "/datasets":
            body = pages.datasets_page(store.list_datasets(project), project=project, writable=form_token is not None)
            return _Response(page("Datasets", "datasets", body, project, store))
        if path_info.startswith("/datasets/"):
            dataset_id = path_info[len("/datasets/"):]
            as_jsonl = dataset_id.endswith(".jsonl")
            dataset = store.get_dataset(dataset_id.removesuffix(".jsonl"))
            if dataset is None:
                return _Response(page("Not found", "datasets", pages.not_found("Dataset"), project, store),
                                 status=404)
            if as_jsonl:
                return _Response(
                    pages.dataset_jsonl(dataset),
                    content_type="application/jsonl; charset=utf-8",
                    headers=[(b"content-disposition",
                              f'attachment; filename="{pages.safe_name(dataset["name"])}.jsonl"'.encode("latin-1"))],
                )
            body = pages.dataset_page(dataset, token=form_token, can_run=runner is not None)
            return _Response(page(dataset["name"], "datasets", body, dataset["project"], store))
        return _Response(page("Not found", "traces", pages.not_found("Page"), project, store), status=404)

    def respond_post(path_info: str, form: dict[str, list[str]]) -> _Response:
        def field(name: str, limit: int) -> str:
            values = form.get(name)
            return values[0].strip()[:limit] if values else ""

        if not db_path.exists():
            return _form_error("There's no Studio file to save to yet.", "/traces", status=404)
        if not hmac.compare_digest(field("token", 200), token):
            return _form_error("This page is out of date (Studio restarted since it loaded). Reload it and try "
                               "again.", "/traces", status=403)
        reader = StudioStore(db_path, readonly=True)
        if match := re.fullmatch(rf"/traces/({_ID})/scores", path_info):
            trace = reader.get_trace(match[1])
            if trace is None:
                return _form_error("That run isn't recorded any more.", "/traces", status=404)
            back = f"/traces/{quote(trace['trace_id'])}"
            key = field("key", 64) or "review"
            verdict = field("verdict", 10)
            value: bool | float | str
            if verdict in ("good", "bad"):
                value = verdict == "good"
            else:
                raw = field("value", 200)
                if not raw:
                    return _form_error("Enter a value to score with: a number or a label.", back)
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
            from ._hooks import _score_row

            row = _score_row(trace.get("run_id"), key, value, field("comment", 2000) or None, "ui")
            with StudioStore(db_path).write() as conn:
                StudioStore.add_score(conn, {**row, "trace_id": trace["trace_id"]})
            return _redirect(back + "?saved=score")
        if match := re.fullmatch(rf"/traces/({_ID})/datasets", path_info):
            trace = reader.get_trace(match[1])
            if trace is None:
                return _form_error("That run isn't recorded any more.", "/traces", status=404)
            back = f"/traces/{quote(trace['trace_id'])}"
            name, case_input = field("dataset", 100), field("input", 100_000)
            if not name or not case_input:
                return _form_error("A case needs a dataset name and an input.", back)
            with StudioStore(db_path).write() as conn:
                dataset_id = StudioStore.add_dataset_case(
                    conn, project=trace["project"], dataset=name, input=case_input,
                    expected=field("expected", 100_000) or None, trace_id=trace["trace_id"],
                )
            return _redirect(f"{back}?added={quote(dataset_id)}")
        if match := re.fullmatch(r"/scores/(\d{1,18})/delete", path_info):
            trace_id = reader.score_trace(int(match[1]))
            with StudioStore(db_path).write() as conn:
                StudioStore.delete_score(conn, int(match[1]))
            return _redirect(f"/traces/{quote(trace_id)}" if trace_id else "/traces")
        if path_info == "/evals":
            return start_eval(form, field, reader)
        if match := re.fullmatch(rf"/experiments/({_ID})/cancel", path_info):
            exp_id = match[1]

            def cancel() -> _Response:
                if runner is not None:
                    runner.cancel(exp_id)
                return _redirect(f"/experiments/{quote(exp_id)}")

            return _Response(b"", then=cancel)
        if match := re.fullmatch(rf"/datasets/({_ID})/cases/({_ID})/delete", path_info):
            if reader.get_dataset(match[1]) is None:
                return _form_error("That dataset doesn't exist.", "/datasets", status=404)
            with StudioStore(db_path).write() as conn:
                StudioStore.delete_dataset_case(conn, match[1], match[2])
            return _redirect(f"/datasets/{quote(match[1])}")
        return _Response("Not found.", status=404, content_type=_TEXT)

    def start_eval(
        form: dict[str, list[str]], field: Callable[[str, int], str], reader: StudioStore
    ) -> _Response:
        if runner is None:
            return _form_error("This Studio can't run evals: start it with --agent.", "/experiments", status=404)
        target = runner.targets.get(field("agent", 200))
        dataset = reader.get_dataset(field("dataset", 100))
        back = "/evals/new" + (f"?dataset={quote(dataset['id'])}" if dataset else "")
        if target is None:
            return _form_error("Pick one of the agents Studio was started with.", back)
        if dataset is None or not dataset["cases"]:
            return _form_error("Pick a dataset that has cases.", back)
        try:
            metrics = _evals.build_metrics(form.get("metrics", []), field("judge_model", 100) or None)
        except ValueError as exc:
            return _form_error(str(exc), back)
        try:
            concurrency = min(max(int(field("concurrency", 3) or 4), 1), 8)
        except ValueError:
            concurrency = 4
        raw_cap = field("max_cost_usd", 12).lstrip("$")
        try:
            cap = float(raw_cap) if raw_cap else None
        except ValueError:
            cap = -1.0
        if cap is not None and not cap > 0:
            return _form_error("The spending limit has to be an amount above 0, or empty for no limit.", back)
        name = field("name", 100) or f"{target.name} {time.strftime('%b %d %H:%M')}"

        def start() -> _Response:
            busy = runner.busy()
            if busy is not None:
                return _form_error(f"An eval is already running ({busy.name}). Wait for it, or cancel it, first.",
                                   f"/experiments/{quote(busy.id)}", status=409)
            run = runner.start(
                target=target, dataset=dataset["name"], project=dataset["project"], total=len(dataset["cases"]),
                metrics=metrics, name=name, concurrency=concurrency, max_cost_usd=cap,
            )
            return _redirect(f"/experiments/{quote(run.id)}")

        return _Response(b"", then=start)

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    if runner is not None:
                        await runner.aclose()
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        method = scope.get("method", "GET")
        request = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        allowed = b"GET, HEAD" if read_only else b"GET, HEAD, POST"
        if not host_ok(request):
            response = _Response("Studio doesn't answer to this host name. Open it at http://127.0.0.1, or start "
                                 "it with the name you use.", status=421, content_type=_TEXT)
        elif method == "POST" and not read_only:
            response = await _post(scope, receive, request, respond_post)
        elif method not in ("GET", "HEAD"):
            response = _Response("Studio is read-only." if read_only else "Method not allowed.", status=405,
                                 content_type=_TEXT, headers=[(b"allow", allowed)])
        else:
            query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
            try:
                response = await anyio.to_thread.run_sync(respond, scope.get("path", "/"), query)
            except Exception:  # noqa: BLE001 — one bad page mustn't take the app down
                _LOG.exception("Studio page failed: %s", scope.get("path"))
                response = _Response("Studio couldn't render this page; see the server log.", status=500,
                                     content_type=_TEXT)
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


async def _post(
    scope: Scope,
    receive: Receive,
    headers: dict[str, str],
    handle: Callable[[str, dict[str, list[str]]], _Response],
) -> _Response:
    """Check and read a form post, then hand it to ``handle``."""
    # A browser says where a request comes from; refuse other sites.
    if headers.get("sec-fetch-site") == "cross-site":
        return _Response("Studio only accepts forms from its own pages.", status=403, content_type=_TEXT)
    origin = headers.get("origin")
    if origin and origin != "null" and urlsplit(origin).netloc.lower() != headers.get("host", "").lower():
        return _Response("Studio only accepts forms from its own pages.", status=403, content_type=_TEXT)
    if headers.get("content-type", "").split(";")[0].strip().lower() != "application/x-www-form-urlencoded":
        return _Response("Studio only accepts its own forms.", status=415, content_type=_TEXT)
    body = b""
    more = True
    while more:
        message = await receive()
        if message["type"] == "http.disconnect":
            return _Response("", status=400, content_type=_TEXT)
        body += message.get("body", b"")
        more = bool(message.get("more_body"))
        if len(body) > MAX_FORM:
            return _Response("That's too much to save at once.", status=413, content_type=_TEXT)
    form = parse_qs(body.decode("utf-8", errors="replace"), keep_blank_values=True)
    try:
        response = await anyio.to_thread.run_sync(handle, scope.get("path", "/"), form)
        return response.then() if response.then is not None else response
    except Exception:  # noqa: BLE001 — one bad form mustn't take the app down
        _LOG.exception("Studio couldn't save: %s", scope.get("path"))
        return _Response("Studio couldn't save that; see the server log.", status=500, content_type=_TEXT)


def _redirect(location: str) -> _Response:
    return _Response(b"", status=303, headers=[(b"location", location.encode("utf-8"))])


def _form_error(message: str, back: str, *, status: int = 400) -> _Response:
    return _Response(pages.form_error(message, back), status=status)


def _host_name(host: str) -> str:
    """``host`` (a ``Host`` header) without its port."""
    host = host.strip().lower()
    if host.startswith("["):
        return host[1:].split("]", 1)[0]
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value else None
    except ValueError:
        return None


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
        "unscored": "1" if arg("unscored") == "1" else None,
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
        unscored=bool(filters["unscored"]),
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
