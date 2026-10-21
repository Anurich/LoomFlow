"""HTML for the Studio web app: one function per page, plain strings,
every value escaped. No templates engine, no JavaScript."""

from __future__ import annotations

import html
import json
import re
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any
from urllib.parse import quote, urlencode

from ._diagram import render as render_diagram

KIND_LABELS = {
    "agent": "Agent run",
    "workflow": "Workflow run",
    "step": "Workflow step",
    "turn": "Turn",
    "model": "Model call",
    "tool": "Tool call",
    "event": "Event",
    "span": "Step",
}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def href(path: str, **params: Any) -> str:
    clean = {k: v for k, v in params.items() if v not in (None, "", False)}
    return e(path + ("?" + urlencode(clean) if clean else ""))


def fmt_ts(ts: float | None) -> str:
    if ts is None:
        return ""
    moment = datetime.fromtimestamp(ts)
    now = datetime.now()
    if moment.date() == now.date():
        return moment.strftime("%H:%M:%S")
    if moment.year == now.year:
        return moment.strftime("%b %d %H:%M")
    return moment.strftime("%Y-%m-%d %H:%M")


def fmt_ms(ms: float | None) -> str:
    if ms is None:
        return ""
    if ms < 10:
        return f"{ms:.1f} ms"
    if ms < 1000:
        return f"{ms:.0f} ms"
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes} m {rest} s"


def fmt_cost(cost: float | None) -> str:
    if not cost:
        return "$0"
    if cost >= 1:
        return f"${cost:,.2f}"
    if cost >= 0.01:
        return f"${cost:.3f}"
    return f"${cost:.4f}" if cost >= 0.0001 else f"${cost:.6f}"


def fmt_int(n: Any) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return ""


def fmt_score(score: Mapping[str, Any]) -> str:
    if score.get("value") is not None:
        return str(score["value"])
    value = score.get("score")
    if value is None:
        return ""
    return f"{value:.2f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)


def status_html(status: str | None) -> str:
    status = status or "ok"
    if status == "ok":
        return '<span class="status"><span class="dot" aria-hidden="true"></span>ok</span>'
    return f'<span class="status {e(status)}">{e(status)}</span>'


def pretty(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def current(flag: bool, value: str = "true") -> str:
    """An ``aria-current`` attribute when ``flag`` is set."""
    return f' aria-current="{value}"' if flag else ""


def version_tag(version: Any) -> str:
    return f' <span class="ver">v{e(version)}</span>' if version else ""


def trace_link(trace_id: str | None, text: str = "trace") -> str:
    return f'<a href="{href("/traces/" + quote(trace_id))}">{e(text)}</a>' if trace_id else ""


def first_line(text: Any, limit: int = 120) -> str:
    raw = "" if text is None else str(text)
    line = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

_MARK = (
    '<svg width="22" height="22" viewBox="0 0 28 28" fill="none" aria-hidden="true">'
    '<line x1="9" y1="3" x2="9" y2="25" stroke="var(--line-strong)" stroke-width="1.5"/>'
    '<line x1="19" y1="3" x2="19" y2="25" stroke="var(--line-strong)" stroke-width="1.5"/>'
    '<line x1="3" y1="14" x2="25" y2="14" stroke="var(--madder)" stroke-width="1.5"/>'
    '<rect x="6.6" y="11.6" width="4.8" height="4.8" fill="var(--bg)" stroke="var(--indigo)" stroke-width="1.5"/>'
    '<rect x="16.6" y="11.6" width="4.8" height="4.8" fill="var(--indigo)"/></svg>'
)


def layout(
    *,
    title: str,
    active: str,
    body: str,
    project: str | None,
    projects: Sequence[str],
    db_path: str,
    version: str,
) -> str:
    def item(path: str, label: str, key: str) -> str:
        current = ' aria-current="page"' if key == active else ""
        return f'<a class="item" href="{href(path, project=project)}"{current}>{e(label)}</a>'

    project_links = "".join(
        f'<a class="item" href="{href("/traces", project=p)}"'
        f'{current(p == project, "page")}>{e(p)}</a>'
        for p in projects
    )
    all_link = (
        f'<a class="item" href="/traces"{current(not project, "page")}>'
        "All projects</a>"
        if len(projects) > 1
        else ""
    )
    projects_html = (
        f'<div class="nav-group"><span class="nav-label">Projects</span>{all_link}{project_links}</div>'
        if projects
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)} · Loomflow Studio</title>
<link rel="stylesheet" href="/static/studio.css">
</head>
<body>
<div class="shell">
<nav class="nav" aria-label="Studio">
  <a class="brand" href="{href("/traces", project=project)}">{_MARK}<span>Loomflow Studio</span></a>
  <div class="nav-group">
    {item("/traces", "Traces", "traces")}
    {item("/agents", "Agents", "agents")}
    {item("/experiments", "Experiments", "experiments")}
    {item("/datasets", "Datasets", "datasets")}
  </div>
  {projects_html}
  <div class="nav-foot"><span>Reading</span><span class="mono soft">{e(db_path)}</span><span>loomflow {e(version)}</span></div>
</nav>
<main>
{body}
</main>
</div>
</body>
</html>"""


def empty_db(db_path: str) -> str:
    return f"""<div class="head"><h1>No runs recorded yet</h1></div>
<div class="panel empty">
  <p>Studio reads <span class="mono">{e(db_path)}</span>, which has no runs in it yet.</p>
  <p>Turn recording on in the app you want to watch, then run it:</p>
  <pre class="code">export LOOMFLOW_STUDIO=1          # or, in code:
import loomflow.studio
loomflow.studio.enable()</pre>
  <p class="soft">Runs show up here as they finish. Start <span class="mono">loom studio</span> from the same folder,
  or pass <span class="mono">--db</span> with the file's path.</p>
</div>"""


def not_found(what: str) -> str:
    return f'<div class="head"><h1>{e(what)} not found</h1></div><p><a href="/traces">Back to traces</a></p>'


def form_error(message: str, back: str) -> str:
    """A whole page for a form that couldn't be saved."""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Not saved · Loomflow Studio</title><link rel="stylesheet" href="/static/studio.css"></head>
<body><main><div class="head"><h1>Not saved</h1></div>
<div class="notice">{e(message)}</div><p><a href="{e(back)}">Go back</a></p></main></body></html>"""


def _token_field(token: str) -> str:
    return f'<input type="hidden" name="token" value="{e(token)}">'


def _options(list_id: str, values: Sequence[str]) -> str:
    return f'<datalist id="{e(list_id)}">' + "".join(f'<option value="{e(v)}">' for v in values) + "</datalist>"


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------

RANGES = {"24h": ("Last 24 hours", 86400.0), "7d": ("Last 7 days", 7 * 86400.0), "all": ("All time", None)}


def _select(name: str, label: str, values: Sequence[str], current: str | None) -> str:
    options = "".join(
        f'<option value="{e(v)}"{" selected" if v == current else ""}>{e(v)}</option>' for v in values
    )
    return (
        f'<label class="sr" for="f-{e(name)}">{e(label)}</label>'
        f'<select id="f-{e(name)}" name="{e(name)}"><option value="">{e(label)}: any</option>{options}</select>'
    )


def traces_page(
    data: Mapping[str, Any],
    *,
    filters: Mapping[str, Any],
    options: Mapping[str, Sequence[str]],
    project: str | None,
    page: int,
    page_size: int,
    agent_labels: Mapping[str, str],
) -> str:
    total, traces = data["total"], data["traces"]
    rng = filters.get("range") or "all"
    median = fmt_ms(data["median_ms"]) if data["median_ms"] is not None else "n/a"
    noun = "run" if total == 1 else "runs"
    summary = (
        f'{fmt_int(total)} {noun}, {fmt_int(data["errors"])} with errors, '
        f'{fmt_int(data["interrupted"])} interrupted, {fmt_cost(data["cost_usd"])} spent, median {median}.'
    )
    range_links = "".join(
        f'<a class="chip" href="{href("/traces", project=project, **{**filters, "range": key, "page": None})}"'
        f'{current(key == rng)}>{e(label)}</a>'
        for key, (label, _) in RANGES.items()
    )
    hidden = "".join(
        f'<input type="hidden" name="{e(k)}" value="{e(v)}">'
        for k, v in (("project", project), ("range", rng), ("session", filters.get("session")),
                     ("version", filters.get("version")))
        if v
    )
    form = f"""<form class="filters" method="get" action="/traces" role="search">
  {hidden}
  <label class="sr" for="f-q">Search traces</label>
  <input id="f-q" type="search" name="q" value="{e(filters.get("q"))}" placeholder="Search inputs, outputs, messages, tool calls">
  {_select("agent", "Agent", options.get("agents", []), filters.get("agent"))}
  {_select("user", "User", options.get("users", []), filters.get("user"))}
  {_select("model", "Model", options.get("models", []), filters.get("model"))}
  {_select("status", "Status", ["ok", "error", "interrupted", "running"], filters.get("status"))}
  <label class="check"><input type="checkbox" name="problems" value="1"{" checked" if filters.get("problems") else ""}> Problems only</label>
  <label class="check"><input type="checkbox" name="unscored" value="1"{" checked" if filters.get("unscored") else ""}> Not scored</label>
  <button class="btn" type="submit">Filter</button>
  <a class="btn" href="{href("/traces", project=project)}">Clear</a>
</form>"""
    session_note = ""
    if filters.get("session"):
        session_note = (
            f'<p class="lede">Thread <span class="mono">{e(filters["session"])}</span>: every run in this '
            f'conversation, newest first. <a href="{href("/traces", project=project)}">Show all runs</a></p>'
        )
    rows = "".join(_trace_row(t, agent_labels) for t in traces)
    if not traces:
        rows = '<tr><td colspan="10" class="soft">No runs match these filters.</td></tr>'
    start = page * page_size
    newer = (
        f'<a href="{href("/traces", project=project, **{**filters, "page": page - 1 or None})}">Newer</a>'
        if page > 0
        else "<span></span>"
    )
    older = (
        f'<a href="{href("/traces", project=project, **{**filters, "page": page + 1})}">Older</a>'
        if start + len(traces) < total
        else "<span></span>"
    )
    shown = f"Showing {start + 1 if traces else 0}–{start + len(traces)} of {fmt_int(total)}"
    return f"""<div class="head"><div class="row-gap" style="flex-direction:column;align-items:flex-start">
<h1>Traces</h1><p class="lede">{e(summary)}</p></div>
<div class="row-gap">{range_links}</div></div>
{session_note}
{form}
<div class="panel scroll">
<table>
<thead><tr><th>Status</th><th>Started</th><th>Agent</th><th>Input</th><th>User</th><th>Model</th>
<th class="num">Tokens</th><th class="num">Cost</th><th class="num">Time</th><th>Score</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<div class="pager">{newer}<span>{e(shown)}</span>{older}</div>
</div>"""


def _trace_row(t: Mapping[str, Any], agent_labels: Mapping[str, str]) -> str:
    link = href(f"/traces/{quote(t['trace_id'])}")
    name = agent_labels.get(t.get("agent_key") or "", t.get("name") or "")
    version = f'<span class="ver">v{e(t["agent_version"])}</span>' if t.get("agent_version") else ""
    scores = t.get("scores") or []
    score = f"{scores[0]['key']}: {fmt_score(scores[0])}" if scores else ""
    if len(scores) > 1:
        score += f" (+{len(scores) - 1})"
    tokens = (t.get("tokens_in") or 0) + (t.get("tokens_out") or 0)

    def cell(content: str, cls: str = "") -> str:
        return f'<td class="{cls}"><a class="cell" href="{link}" tabindex="-1">{content}</a></td>'

    return (
        '<tr class="link">'
        f'<td><a class="cell" href="{link}">{status_html(t.get("status"))}</a></td>'
        + cell(f'<span class="mono soft">{e(fmt_ts(t.get("started_at")))}</span>')
        + cell(f"<strong>{e(name)}</strong>{version}")
        + cell(e(first_line(t.get("input"))), "clip")
        + cell(e(t.get("user_id")), "soft")
        + cell(f'<span class="mono soft">{e(t.get("model"))}</span>')
        + cell(e(fmt_int(tokens) if tokens else ""), "num")
        + cell(e(fmt_cost(t.get("cost_usd")) if t.get("cost_usd") else ""), "num")
        + cell(e(fmt_ms(t.get("duration_ms"))), "num")
        + cell(e(score), "soft clip")
        + "</tr>"
    )


# ---------------------------------------------------------------------------
# One trace
# ---------------------------------------------------------------------------


def _ordered(spans: Sequence[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], int]]:
    ids = {s["span_id"] for s in spans}
    children: dict[str | None, list[Mapping[str, Any]]] = defaultdict(list)
    for s in spans:
        parent = s.get("parent_id") if s.get("parent_id") in ids else None
        children[parent].append(s)
    for group in children.values():
        group.sort(key=lambda s: (s["started_at"], s["kind"] == "event"))
    out: list[tuple[Mapping[str, Any], int]] = []

    def walk(parent: str | None, depth: int) -> None:
        for s in children.get(parent, []):
            out.append((s, depth))
            if len(out) < 20_000:
                walk(s["span_id"], depth + 1)

    walk(None, 0)
    return out


def _used(spans: Sequence[Mapping[str, Any]]) -> set[str]:
    """The names a trace's steps go by: agents (and their catalog
    keys), workflow nodes, tools. These light up in the diagram."""
    used: set[str] = set()
    for span in spans:
        if span["kind"] in ("agent", "workflow", "step", "tool"):
            used.add(span["name"])
        key = (span.get("attributes") or {}).get("agent_key")
        if key:
            used.add(key)
    return used


def trace_page(
    trace: Mapping[str, Any],
    *,
    selected: str | None,
    agent_labels: Mapping[str, str],
    structure: Mapping[str, Any] | None = None,
    token: str | None = None,
    notice: tuple[str, str] | None = None,
    score_keys: Sequence[str] = (),
    datasets: Sequence[str] = (),
) -> str:
    """The trace page. With a form ``token`` it has forms to score the
    run and add it to a dataset; ``notice`` is ``(panel, html)``, a
    confirmation shown in the ``"scores"`` or ``"datasets"`` panel."""
    spans = trace["spans"]
    ordered = _ordered(spans)
    by_id = {s["span_id"]: s for s in spans}
    chosen = by_id.get(selected or "") or (ordered[0][0] if ordered else None)
    t0 = min((s["started_at"] for s in spans), default=trace["started_at"])
    t1 = max((s.get("ended_at") or s["started_at"] for s in spans), default=t0)
    window = max(t1 - t0, 1e-6)
    base = f"/traces/{quote(trace['trace_id'])}"

    rows = []
    for span, depth in ordered:
        kind = span["kind"]
        left = (span["started_at"] - t0) / window * 100
        css = "k-error" if span.get("status") == "error" else f"k-{kind}"
        if kind == "event":
            graphic = f'<span class="mark {css}" style="left:{left:.2f}%" aria-hidden="true"></span>'
        else:
            width = max(((span.get("ended_at") or span["started_at"]) - span["started_at"]) / window * 100, 0.4)
            root = " root" if depth == 0 else ""
            graphic = (
                f'<span class="bar {css}{root}" style="left:{left:.2f}%;width:{width:.2f}%" aria-hidden="true"></span>'
            )
        current = ' aria-current="true"' if chosen is not None and span["span_id"] == chosen["span_id"] else ""
        label_cls = "mono" if kind in ("model", "tool") else ""
        weight = "font-weight:700;" if depth == 0 else ""
        status = "" if span.get("status") in (None, "ok") else f" {status_html(span.get('status'))}"
        rows.append(
            f'<a class="tl-row" href="{href(base, span=span["span_id"])}#detail"{current}>'
            f'<span class="tl-name" style="padding-left:{depth * 16}px">'
            f'<span class="dot {css}" aria-hidden="true"></span>'
            f'<span class="label {label_cls}" style="{weight}">{e(span["name"])}</span>{status}</span>'
            f'<span class="track">{graphic}</span>'
            f'<span class="num soft">{e(fmt_ms(span.get("duration_ms")) if kind != "event" else "")}</span></a>'
        )
    ticks = "".join(
        f'<span style="position:absolute;left:{p * 100 - (3 if p else 0):.1f}%">{e(fmt_ms(window * p * 1000))}</span>'
        for p in (0.0, 1 / 3, 2 / 3)
    )
    timeline = f"""<section class="panel scroll wide" aria-label="Timeline">
<div class="tl">
<div class="panel-head"><h2>Timeline</h2><div class="legend">
<span><span class="swatch k-model" aria-hidden="true"></span>Model call</span>
<span><span class="swatch k-tool" aria-hidden="true"></span>Tool call</span>
<span><span class="swatch k-step" aria-hidden="true"></span>Framework step</span></div></div>
<div class="tl-row tl-axis"><span>Step</span><span class="track">{ticks}</span><span class="num">Time</span></div>
{"".join(rows) or '<p class="soft" style="padding:16px">No steps were recorded.</p>'}
</div></section>"""

    name = agent_labels.get(trace.get("agent_key") or "", trace.get("name") or "run")
    version = f' <span class="ver">v{e(trace["agent_version"])}</span>' if trace.get("agent_version") else ""
    meta = [
        ("Agent" if trace.get("kind") == "agent" else "Workflow", f"<strong>{e(name)}</strong>{version}"),
        ("User", e(trace.get("user_id") or "anonymous")),
    ]
    if trace.get("session_id"):
        runs = trace.get("thread_runs") or 1
        meta.append((
            "Thread",
            f'<a class="mono" href="{href("/traces", project=trace["project"], session=trace["session_id"])}">'
            f'{e(trace["session_id"])}</a> <span class="faint">({runs} run{"s" if runs != 1 else ""})</span>',
        ))
    if trace.get("model"):
        meta.append(("Model", f'<span class="mono">{e(trace["model"])}</span>'))
    meta += [
        ("Started", e(fmt_ts(trace.get("started_at")))),
        ("Duration", e(fmt_ms(trace.get("duration_ms")) or "running")),
        ("Tokens", e(fmt_int((trace.get("tokens_in") or 0) + (trace.get("tokens_out") or 0)))),
        ("Cost", e(fmt_cost(trace.get("cost_usd")))),
        ("Status", status_html(trace.get("status"))),
    ]
    if trace.get("experiment_id"):
        meta.append(("Experiment", f'<a href="{href("/experiments/" + quote(trace["experiment_id"]))}">'
                                   f'case {e(trace.get("case_key"))}</a>'))
    if trace.get("tags"):
        meta.append(("Tags", e(", ".join(trace["tags"]))))
    meta_html = "".join(f"<div><dt>{e(k)}</dt><dd>{v}</dd></div>" for k, v in meta)
    error = ""
    if trace.get("error"):
        error = f'<div class="notice"><strong>Error:</strong> {e(trace["error"])}</div>'
    elif trace.get("interruption_reason"):
        error = f'<div class="notice"><strong>Interrupted:</strong> {e(trace["interruption_reason"])}</div>'
    title = first_line(trace.get("input"), 140) or name
    detail = _span_detail(chosen) if chosen is not None else ""
    architecture = ""
    if structure:
        agents_link = href("/agents/" + quote(trace["agent_key"]), project=trace["project"],
                           version=trace.get("agent_version"))
        architecture = (
            '<section class="panel pad" aria-label="Architecture"><div class="row-gap" '
            'style="justify-content:space-between"><h2>Architecture</h2>'
            '<span class="faint" style="font-size:13px">Highlighted: the agents, steps and tools this run used. '
            f'<a href="{agents_link}">All versions</a></span></div>'
            + render_diagram(structure, uid="trace", title=f"{name} architecture", active=_used(spans))
            + "</section>"
        )
    return f"""<div class="crumbs"><a href="{href("/traces", project=trace["project"])}">Traces</a>
<span aria-hidden="true">/</span><span class="mono">{e(trace["trace_id"][:12])}</span></div>
<div class="head"><h1>{e(title)}</h1><div class="row-gap">
<a class="btn" href="{href(base + ".json")}">JSON</a></div></div>
<dl class="meta">{meta_html}</dl>
{error}
<div class="split">
{timeline}
<div class="side">
{detail}
{_scores_panel(trace, token=token, keys=score_keys, notice=_notice(notice, "scores"))}
{_dataset_panel(trace, token=token, datasets=datasets, notice=_notice(notice, "datasets")) if token else ""}
</div></div>
{architecture}"""


def _attr(attrs: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if attrs.get(key) is not None:
            return attrs[key]
    return None


def _span_detail(span: Mapping[str, Any]) -> str:
    kind, attrs, content = span["kind"], span.get("attributes") or {}, span.get("content") or {}
    meta: list[tuple[str, Any]] = []
    if kind == "agent":
        meta = [
            ("Model", attrs.get("model")),
            ("Turns", attrs.get("turns")),
            ("Tokens", _tokens(attrs.get("tokens_in"), attrs.get("tokens_out"))),
            ("Cost", fmt_cost(attrs.get("cost_usd")) if attrs.get("cost_usd") else None),
            ("Fingerprint", attrs.get("fingerprint")),
            ("Run", attrs.get("run_id")),
        ]
    elif kind == "model":
        meta = [
            ("Model", attrs.get("model")),
            ("Input tokens", fmt_int(_attr(attrs, "gen_ai.usage.input_tokens"))),
            ("Output tokens", fmt_int(_attr(attrs, "gen_ai.usage.output_tokens"))),
            ("Cache read", fmt_int(_attr(attrs, "gen_ai.usage.cache_read.input_tokens"))),
            ("Finish", _listed(_attr(attrs, "gen_ai.response.finish_reasons"))),
            ("Cost", fmt_cost(attrs.get("cost_usd")) if attrs.get("cost_usd") else None),
        ]
    elif kind == "tool":
        permission = attrs.get("loom.permission")
        if permission and attrs.get("loom.permission.reason"):
            permission = f"{permission} ({attrs['loom.permission.reason']})"
        meta = [
            ("Tool", attrs.get("tool")),
            ("Permission", permission),
            ("Approval", attrs.get("loom.approval")),
            ("Edited by approver", "yes" if attrs.get("loom.approval.edited") else None),
            ("Denied", "yes" if content.get("denied") else None),
            ("Error", attrs.get("error.type")),
        ]
    elif kind in ("step", "workflow"):
        meta = [("Node" if kind == "step" else "Workflow", span["name"]),
                ("Interrupted", attrs.get("interruption_reason"))]
    meta.append(("Duration", fmt_ms(span.get("duration_ms")) if kind != "event" else None))
    meta.append(("Status", span.get("status") if span.get("status") not in (None, "ok") else None))
    meta_html = "".join(
        f"<div><dt>{e(k)}</dt><dd>{e(v)}</dd></div>" for k, v in meta if v not in (None, "")
    )

    blocks: list[str] = []
    if span.get("error"):
        blocks.append(f'<div class="notice">{e(span["error"])}</div>')
    if kind == "model":
        messages = content.get("messages") or []
        if messages:
            items = "".join(
                f'<div class="msg"><span class="role">{e(m.get("role"))}</span><div class="body">'
                f"{e(m.get('content') if m.get('content') is not None else '(not recorded)')}"
                + "".join(
                    f'<div class="mono soft">calls {e(c.get("tool"))}({e(pretty(c.get("args")))})</div>'
                    for c in m.get("tool_calls") or []
                )
                + "</div></div>"
                for m in messages
            )
            blocks.append(f'<div><h3>Messages sent ({len(messages)})</h3>{items}</div>')
        reply = content.get("reply")
        calls = content.get("tool_calls") or []
        reply_text = reply or ""
        if calls:
            reply_text += ("\n" if reply_text else "") + "\n".join(
                f"tool call  {c.get('tool')}({json.dumps(c.get('args'), ensure_ascii=False, default=str)})"
                for c in calls
            )
        if reply_text or "reply" in content:
            blocks.append(_block("Reply", reply_text or "(empty)"))
    else:
        for key, title in (("input", "Input"), ("args", "Arguments"), ("output", "Output"),
                           ("parsed", "Parsed output"), ("result", "Result"), ("error", "Error"),
                           ("reason", "Reason"), ("message", "Message"), ("data", "Details")):
            if key in content and content[key] not in (None, "", {}, []):
                blocks.append(_block(title, pretty(content[key])))
    if not blocks and kind not in ("turn", "span"):
        blocks.append('<p class="soft">Nothing more was recorded for this step.</p>')
    extra = {k: v for k, v in attrs.items() if not k.startswith("_")}
    attrs_html = (
        f'<details><summary>All attributes</summary>{_block("", pretty(extra))}</details>' if extra else ""
    )
    return f"""<section class="panel pad" id="detail" aria-label="Selected step">
<div><span class="faint" style="font-size:13px">{e(KIND_LABELS.get(kind, kind))}</span>
<h2 class="{"mono" if kind in ("model", "tool") else ""}" style="font-size:19px">{e(span["name"])}</h2></div>
{f'<dl class="grid">{meta_html}</dl>' if meta_html else ""}
{"".join(blocks)}
{attrs_html}
</section>"""


def _listed(value: Any) -> Any:
    return ", ".join(str(v) for v in value) if isinstance(value, list | tuple) else value


def _tokens(tokens_in: Any, tokens_out: Any) -> str | None:
    if tokens_in is None and tokens_out is None:
        return None
    return f"{fmt_int(tokens_in or 0)} in, {fmt_int(tokens_out or 0)} out"


def _block(title: str, text: str) -> str:
    heading = f"<h3>{e(title)}</h3>" if title else ""
    return f'<div style="display:flex;flex-direction:column;gap:6px">{heading}<pre class="code">{e(text)}</pre></div>'


def _notice(notice: tuple[str, str] | None, panel: str) -> str:
    if notice is None or notice[0] != panel:
        return ""
    return f'<div class="notice ok" role="status">{notice[1]}</div>'


def _scores_panel(trace: Mapping[str, Any], *, token: str | None, keys: Sequence[str], notice: str = "") -> str:
    scores = trace.get("scores") or []
    if not scores:
        hint = "Score it below, or" if token else "Add one with"
        body = (f'<p class="soft">No scores yet. {hint} <span class="mono">'
                "loomflow.studio.score(result, key, value)</span> in code.</p>")
    else:
        parts = []
        for s in scores:
            remove = (
                f'<form class="inline" method="post" action="/scores/{e(s["id"])}/delete">{_token_field(token)}'
                f'<button class="link-btn" type="submit" aria-label="Remove the {e(s["key"])} score">Remove</button>'
                "</form>"
                if token and s.get("id") is not None
                else "<span></span>"
            )
            parts.append(
                f'<div class="score-row"><span class="mono" style="font-size:13px">{e(s["key"])}</span>'
                f'<span><strong>{e(fmt_score(s))}</strong></span>{remove}<span></span>'
                f'<span class="faint" style="font-size:13px">{e(s.get("source"))}'
                f'{" · " + e(s["comment"]) if s.get("comment") else ""}</span></div>'
            )
        body = "".join(parts)
    form = ""
    if token:
        suggestions = list(dict.fromkeys(["review", *keys]))
        form = f"""<form class="stack" method="post" action="/traces/{e(quote(trace["trace_id"]))}/scores">
{_token_field(token)}
<label for="s-key">Score</label>
<input id="s-key" type="text" name="key" value="review" list="s-keys" maxlength="64" required>
{_options("s-keys", suggestions)}
<div class="choices" role="radiogroup" aria-label="Verdict">
<label><input type="radio" name="verdict" value="good"> Good</label>
<label><input type="radio" name="verdict" value="bad"> Bad</label>
<label for="s-value">or a value</label>
<input id="s-value" type="text" name="value" maxlength="200" placeholder="4, 0.8 or a label" style="width:auto;flex:1 1 140px">
</div>
<label for="s-comment">Comment</label>
<input id="s-comment" type="text" name="comment" maxlength="2000" placeholder="Optional">
<div><button class="btn primary" type="submit">Save score</button></div>
</form>"""
    return (f'<section class="panel pad" id="scores" aria-label="Scores"><h2>Scores</h2>{notice}<div>{body}</div>'
            f"{form}</section>")


def _dataset_panel(
    trace: Mapping[str, Any], *, token: str | None, datasets: Sequence[str], notice: str = ""
) -> str:
    def text(value: Any) -> str:
        return value if isinstance(value, str) else pretty(value) if value is not None else ""

    return f"""<section class="panel pad" id="datasets" aria-label="Add to a dataset"><h2>Add to a dataset</h2>
{notice}
<form class="stack" method="post" action="/traces/{e(quote(trace["trace_id"]))}/datasets">
{_token_field(token or "")}
<label for="d-name">Dataset</label>
<input id="d-name" type="text" name="dataset" list="d-names" maxlength="100" required placeholder="billing-regressions">
{_options("d-names", datasets)}
<label for="d-input">Input</label>
<textarea id="d-input" name="input" rows="3" required>{e(text(trace.get("input")))}</textarea>
<label for="d-expected">Expected output</label>
<textarea id="d-expected" name="expected" rows="4">{e(text(trace.get("output")))}</textarea>
<p class="hint">Edit it into the answer the agent should give: evals compare outputs with it.</p>
<div><button class="btn" type="submit">Add to dataset</button></div>
</form></section>"""


def added_notice(dataset: Mapping[str, Any]) -> str:
    count = len(dataset.get("cases") or [])
    return (f'Added to <a href="/datasets/{e(quote(dataset["id"]))}">{e(dataset["name"])}</a>, '
            f'which has {count} case{"" if count == 1 else "s"}.')


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


def agents_page(
    agents: Sequence[Mapping[str, Any]],
    *,
    selected: Mapping[str, Any] | None,
    versions: Sequence[Mapping[str, Any]],
    project: str | None,
    shown_version: int | None = None,
) -> str:
    intro = ('<div class="head"><div class="row-gap" style="flex-direction:column;align-items:flex-start">'
             '<h1>Agents</h1><p class="lede">Every agent and workflow that has run, how it\'s put together, '
             "and its versions. A new version is recorded automatically whenever its instructions, model, "
             "tools, team or settings change.</p></div></div>")
    if not agents:
        return intro + '<div class="panel empty"><p>No agents have run yet.</p></div>'
    chosen = (selected["project"], selected["agent_key"]) if selected else None
    picks = "".join(
        f'<a class="agent-pick" href="{href("/agents/" + quote(a["agent_key"]), project=a["project"])}"'
        f'{current((a["project"], a["agent_key"]) == chosen)}>'
        f'<span style="display:flex;justify-content:space-between;gap:8px"><strong>{e(a["name"])}</strong>'
        f'<span class="ver">v{e(a.get("current"))}</span></span>'
        f'<span class="soft" style="font-size:13px">{e(a["kind"].capitalize())}, {fmt_int(a.get("runs_week") or 0)}'
        f' runs this week{"" if project else ", " + e(a["project"])}</span></a>'
        for a in agents
    )
    if selected is None or not versions:
        return intro + f'<div class="split"><section class="panel list" aria-label="Agent list">{picks}</section></div>'
    score_keys: list[str] = []
    for v in versions:
        for key in v["scores"]:
            if key not in score_keys:
                score_keys.append(key)
    main_score = score_keys[0] if score_keys else None
    row_parts: list[str] = []
    for i, v in enumerate(versions):
        badge = ' <span class="status interrupted">current</span>' if i == 0 else ""
        runs_link = href("/traces", project=selected["project"], agent=selected["agent_key"], version=v["version"])
        error_rate = f"{100 * v['errors'] / v['runs']:.1f}%" if v["runs"] else ""
        score = v["scores"].get(main_score) if main_score else None
        row_parts.append(
            f"<tr><td><strong>v{e(v['version'])}</strong>{badge}"
            f'<br><span class="ver" style="margin:0">{e(v["fingerprint"])}</span></td>'
            f'<td class="soft">{e(fmt_ts(v["first_seen"]))}</td>'
            f'<td class="num"><a href="{runs_link}">{fmt_int(v["runs"])}</a></td>'
            f'<td class="num">{e(error_rate)}</td>'
            f'<td class="num"><strong>{e(_num(score) if score is not None else "")}</strong></td>'
            f"<td>{e(v['changes'])}</td></tr>"
        )
    rows = "".join(row_parts)
    latest = next((v for v in versions if v["version"] == shown_version), versions[0])
    definition = latest["definition"] or {}
    if definition.get("kind") == "workflow":
        pairs = [("Nodes", ", ".join(definition.get("nodes") or [])), ("Max steps", definition.get("max_steps"))]
        text_title, text = "", ""
    else:
        tools = ", ".join(
            t["name"] + (" (destructive)" if t.get("destructive") else "") for t in definition.get("tools") or []
        )
        pairs = [
            ("Model", definition.get("model")),
            ("Architecture", f"{definition.get('architecture')}, up to {definition.get('max_turns')} turns"),
            ("Tools", tools or "none"),
            ("Permissions", definition.get("permissions")),
            ("Guardrails", ", ".join(definition.get("guardrails") or []) or "none"),
            ("Memory", definition.get("memory")),
            ("Output schema", definition.get("output_schema")),
            ("Effort", definition.get("effort")),
        ]
        text_title, text = "Instructions", definition.get("instructions") or ""
    pairs_html = "".join(f"<div><dt>{e(k)}</dt><dd>{e(v)}</dd></div>" for k, v in pairs if v not in (None, ""))
    base = "/agents/" + quote(selected["agent_key"])
    tabs = "".join(
        f'<a class="chip" href="{href(base, project=selected["project"], version=v["version"])}"'
        f'{current(v["version"] == latest["version"])}>v{e(v["version"])}</a>'
        for v in versions
    )
    diagram = render_diagram(
        definition.get("structure"), uid=f"a{latest['version']}", title=f"{selected['name']} v{latest['version']}"
    )
    return intro + f"""<div class="split">
<section class="panel list" aria-label="Agent list">{picks}</section>
<div class="wide" style="display:flex;flex-direction:column;gap:20px">
<section class="panel pad" aria-label="Architecture">
<div class="row-gap" style="justify-content:space-between"><h2>Architecture of {e(selected["name"])} v{e(latest["version"])}</h2>
<div class="ver-tabs" aria-label="Versions">{tabs if len(versions) > 1 else ""}</div></div>
{diagram}
</section>
<section class="panel scroll" aria-label="Versions">
<div class="panel-head"><h2>{e(selected["name"])} versions</h2></div>
<table><thead><tr><th>Version</th><th>First seen</th><th class="num">Runs</th><th class="num">Errors</th>
<th class="num">{e(main_score or "Score")}</th><th>What changed</th></tr></thead><tbody>{rows}</tbody></table>
</section>
<section class="panel pad" aria-label="Current definition">
<div class="row-gap" style="justify-content:space-between"><h2>Definition of v{e(latest["version"])}</h2>
<span class="faint" style="font-size:13px">Recorded from the running code</span></div>
<dl class="grid">{pairs_html}</dl>
{_block(text_title, text) if text else ""}
</section></div></div>"""


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def _metric_means(exp: Mapping[str, Any]) -> dict[str, float]:
    metrics = (exp.get("summary") or {}).get("metrics") or {}
    return {name: float(v["mean"]) for name, v in metrics.items() if isinstance(v, Mapping) and "mean" in v}


def experiments_page(experiments: Sequence[Mapping[str, Any]], *, project: str | None) -> str:
    intro = ('<div class="head"><div class="row-gap" style="flex-direction:column;align-items:flex-start">'
             '<h1>Experiments</h1><p class="lede">Each <span class="mono">EvalHarness</span> run, grouped by '
             "dataset. Compare two runs on the same dataset to see what improved and what regressed.</p></div></div>")
    if not experiments:
        return intro + ('<div class="panel empty"><p>No experiments yet. Run an eval with Studio on:</p>'
                        '<pre class="code">loomflow.studio.enable()\nreport = await EvalHarness(agent, metrics).run(dataset)</pre>'
                        '<p class="soft">or <span class="mono">LOOMFLOW_STUDIO=1 loom eval cases.jsonl --agent app:agent</span></p></div>')
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for exp in experiments:
        groups[(exp["project"], exp["dataset"])].append(exp)
    sections = []
    for (proj, dataset), exps in groups.items():
        metric_names = sorted({m for exp in exps for m in _metric_means(exp)})
        head = "".join(f'<th class="num">{e(m)}</th>' for m in metric_names)
        rows = "".join(
            f'<tr class="link"><td><a href="{href("/experiments/" + quote(x["id"]))}"><strong>{e(x["name"])}</strong></a>'
            f'{version_tag(x.get("agent_version"))}</td>'
            f'<td class="soft">{e(fmt_ts(x["started_at"]))}</td><td class="num">{fmt_int(x.get("cases"))}</td>'
            + "".join(
                f'<td class="num">{e(f"{_metric_means(x)[m]:.2f}" if m in _metric_means(x) else "")}</td>'
                for m in metric_names
            )
            + "</tr>"
            for x in exps
        )
        options_a = "".join(
            f'<option value="{e(x["id"])}"{" selected" if i == 1 else ""}>{e(x["name"])}, {e(fmt_ts(x["started_at"]))}</option>'
            for i, x in enumerate(exps)
        )
        options_b = "".join(
            f'<option value="{e(x["id"])}"{" selected" if i == 0 else ""}>{e(x["name"])}, {e(fmt_ts(x["started_at"]))}</option>'
            for i, x in enumerate(exps)
        )
        compare = (
            f'<form class="filters" method="get" action="/experiments/compare" style="padding:14px 16px">'
            f'<label class="check" for="a-{e(dataset)}">Baseline</label><select id="a-{e(dataset)}" name="a">{options_a}</select>'
            f'<label class="check" for="b-{e(dataset)}">Against</label><select id="b-{e(dataset)}" name="b">{options_b}</select>'
            f'<button class="btn primary" type="submit">Compare</button></form>'
            if len(exps) > 1
            else ""
        )
        sections.append(
            f'<section class="panel scroll" aria-label="{e(dataset)}"><div class="panel-head">'
            f'<h2>Dataset {e(dataset)}</h2><span class="soft" style="font-size:13px">'
            f'{len(exps)} run{"s" if len(exps) != 1 else ""}{"" if project else ", " + e(proj)}</span></div>'
            f'<table><thead><tr><th>Experiment</th><th>Started</th><th class="num">Cases</th>{head}</tr></thead>'
            f"<tbody>{rows}</tbody></table>{compare}</section>"
        )
    return intro + "".join(sections)


def _case_scores(case: Mapping[str, Any]) -> str:
    if case.get("error"):
        return f'<span class="status error">error</span> <span class="soft">{e(case["error"])}</span>'
    return ", ".join(f"{e(k)} {e(_num(v))}" for k, v in (case.get("scores") or {}).items())


def _num(value: Any) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)


def experiment_page(exp: Mapping[str, Any]) -> str:
    means = _metric_means(exp)
    metrics = "".join(f"<div><dt>{e(k)}</dt><dd><strong>{e(f'{v:.2f}')}</strong></dd></div>" for k, v in means.items())
    rows = "".join(
        f'<tr><td class="num faint">{e(c["position"] + 1)}</td><td>{e(first_line(c.get("input"), 160))}</td>'
        f'<td class="soft">{e(first_line(c.get("expected"), 120))}</td><td>{e(first_line(c.get("output"), 160))}</td>'
        f"<td>{_case_scores(c)}</td>"
        f'<td>{trace_link(c.get("trace_id"))}</td></tr>'
        for c in exp["cases"]
    )
    version = f' <span class="ver">v{e(exp["agent_version"])}</span>' if exp.get("agent_version") else ""
    return f"""<div class="crumbs"><a href="{href("/experiments", project=exp["project"])}">Experiments</a>
<span aria-hidden="true">/</span><span>{e(exp["dataset"])}</span></div>
<div class="head"><div class="row-gap" style="flex-direction:column;align-items:flex-start">
<h1>{e(exp["name"])}{version}</h1>
<p class="lede">Dataset {e(exp["dataset"])}, {len(exp["cases"])} cases, started {e(fmt_ts(exp["started_at"]))}.</p></div></div>
{f'<dl class="meta">{metrics}</dl>' if metrics else ""}
<div class="panel scroll"><table><thead><tr><th class="num">#</th><th>Input</th><th>Expected</th><th>Output</th>
<th>Scores</th><th></th></tr></thead><tbody>{rows}</tbody></table></div>"""


def compare_page(a: Mapping[str, Any], b: Mapping[str, Any], *, show: str) -> str:
    means_a, means_b = _metric_means(a), _metric_means(b)
    order = list((a.get("metadata") or {}).get("metrics") or []) or sorted(set(means_a) | set(means_b))
    metric_names = [m for m in order if m in means_a or m in means_b]
    primary = metric_names[0] if metric_names else None

    def change(old: float | None, new: float | None, *, lower_better: bool = False, fmt: str = "{:+.2f}") -> str:
        if old is None or new is None:
            return '<span class="soft">n/a</span>'
        delta = new - old
        if abs(delta) < 1e-9:
            return '<span class="soft">no change</span>'
        good = delta < 0 if lower_better else delta > 0
        word = "better" if good else "worse"
        return f'<span class="{word}">{e(fmt.format(delta))} {word}</span>'

    def relative(old: float | None, new: float | None, *, floor: float) -> str:
        """Lower is better; under ``floor`` (absolute) or 5% it's noise."""
        if old is None or new is None:
            return '<span class="soft">n/a</span>'
        delta = new - old
        if abs(delta) < floor or (old and abs(delta) / old < 0.05):
            return '<span class="soft">no change</span>'
        word = "better" if delta < 0 else "worse"
        amount = f"{delta / old:+.0%}" if old else e(f"{delta:+g}")
        return f'<span class="{word}">{e(amount)} {word}</span>'

    metric_rows = "".join(
        f'<tr><td class="mono">{e(m)}</td><td class="num">{e(_num(means_a[m]) if m in means_a else "")}</td>'
        f'<td class="num"><strong>{e(_num(means_b[m]) if m in means_b else "")}</strong></td>'
        f"<td>{change(means_a.get(m), means_b.get(m))}</td></tr>"
        for m in metric_names
    )

    def mean_of(exp: Mapping[str, Any], key: str) -> float | None:
        values = [c[key] for c in exp["cases"] if c.get(key) is not None]
        return statistics.fmean(values) if values else None

    def p50(exp: Mapping[str, Any]) -> float | None:
        values = [c["duration_ms"] for c in exp["cases"] if c.get("duration_ms") is not None]
        return statistics.median(values) if values else None

    cost_a, cost_b = mean_of(a, "cost_usd"), mean_of(b, "cost_usd")
    if cost_a or cost_b:
        metric_rows += (
            f'<tr><td>cost per case</td><td class="num">{e(fmt_cost(cost_a))}</td>'
            f'<td class="num"><strong>{e(fmt_cost(cost_b))}</strong></td>'
            f'<td>{relative(cost_a, cost_b, floor=1e-7)}</td></tr>'
        )
    lat_a, lat_b = p50(a), p50(b)
    if lat_a is not None and lat_b is not None:
        metric_rows += (
            f'<tr><td>latency p50</td><td class="num">{e(fmt_ms(lat_a))}</td>'
            f'<td class="num"><strong>{e(fmt_ms(lat_b))}</strong></td>'
            f'<td>{relative(lat_a, lat_b, floor=50.0)}</td></tr>'
        )
    errors_a = sum(1 for c in a["cases"] if c.get("error"))
    errors_b = sum(1 for c in b["cases"] if c.get("error"))
    if errors_a or errors_b:
        metric_rows += (
            f'<tr><td>cases that errored</td><td class="num">{errors_a}</td><td class="num"><strong>{errors_b}</strong></td>'
            f'<td>{change(float(errors_a), float(errors_b), lower_better=True, fmt="{:+.0f}")}</td></tr>'
        )

    cases_a = {c["case_key"]: c for c in a["cases"]}
    joined: list[tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]] = [
        (cases_a.get(c["case_key"]), c) for c in b["cases"]
    ]
    in_b = {c["case_key"] for c in b["cases"]}
    joined += [(c, None) for c in a["cases"] if c["case_key"] not in in_b]

    def verdict(ca: Mapping[str, Any] | None, cb: Mapping[str, Any] | None) -> str:
        if ca is None:
            return "new"
        if cb is None:
            return "missing"
        if cb.get("error") and not ca.get("error"):
            return "regressed"
        if ca.get("error") and not cb.get("error"):
            return "improved"
        sa = (ca.get("scores") or {}).get(primary) if primary else None
        sb = (cb.get("scores") or {}).get(primary) if primary else None
        if sa is None or sb is None or abs(sb - sa) < 1e-9:
            return "same"
        return "improved" if sb > sa else "regressed"

    verdicts = [(ca, cb, verdict(ca, cb)) for ca, cb in joined]
    improved = sum(1 for *_, v in verdicts if v == "improved")
    regressed = sum(1 for *_, v in verdicts if v == "regressed")
    changed = sum(1 for *_, v in verdicts if v not in ("same",))
    shown = [x for x in verdicts if show == "all" or (show == "changed" and x[2] != "same")
             or (show == "regressed" and x[2] == "regressed")]

    def score_cell(case: Mapping[str, Any] | None) -> str:
        if case is None:
            return '<span class="soft">not run</span>'
        if case.get("error"):
            return '<span class="status error">error</span>'
        value = (case.get("scores") or {}).get(primary) if primary else None
        return e(_num(value)) if value is not None else ""

    labels = {"improved": '<span class="better">improved</span>',
              "regressed": '<span class="status error">regressed</span>',
              "same": '<span class="soft">same</span>', "new": '<span class="soft">new case</span>',
              "missing": '<span class="soft">not in this run</span>'}
    case_rows = "".join(
        f'<tr><td>{e(first_line((cb or ca or {}).get("input"), 150))}</td>'
        f'<td class="soft">{e(first_line((cb or ca or {}).get("expected"), 110))}</td>'
        f"<td>{score_cell(ca)}</td><td><strong>{score_cell(cb)}</strong></td><td>{labels[v]}</td>"
        f'<td>{trace_link(cb.get("trace_id") if cb else None)}</td></tr>'
        for ca, cb, v in shown
    ) or '<tr><td colspan="6" class="soft">No cases to show.</td></tr>'
    base = {"a": a["id"], "b": b["id"]}
    tabs = "".join(
        f'<a class="chip" href="{href("/experiments/compare", **base, show=key)}"'
        f'{current(key == show)}>{e(label)}</a>'
        for key, label in (("all", f"All {len(verdicts)}"), ("changed", f"Changed {changed}"),
                           ("regressed", f"Regressed {regressed}"))
    )

    def head(exp: Mapping[str, Any], role: str) -> str:
        version = f", agent v{exp['agent_version']}" if exp.get("agent_version") else ""
        return (f'<th class="num"><span style="display:flex;flex-direction:column;gap:2px;align-items:flex-end">'
                f'<span style="color:var(--ink);font-size:14px">{e(exp["name"])}</span>'
                f'<span class="faint" style="font-weight:400">{e(role + version)}, {e(fmt_ts(exp["started_at"]))}</span>'
                "</span></th>")

    note = "" if a["dataset"] == b["dataset"] else (
        f'<div class="notice">These runs used different datasets ({e(a["dataset"])} and {e(b["dataset"])}); '
        "only cases present in both are compared.</div>")
    return f"""<div class="crumbs"><a href="{href("/experiments", project=b["project"])}">Experiments</a>
<span aria-hidden="true">/</span><span>{e(b["dataset"])}</span></div>
<div class="head"><div class="row-gap" style="flex-direction:column;align-items:flex-start">
<h1>{e(a["name"])} against {e(b["name"])}</h1>
<p class="lede">Dataset {e(b["dataset"])}, {len(verdicts)} cases. {improved} improved, {regressed} regressed,
{len(verdicts) - changed} unchanged{f", judged on {primary}" if primary else ""}.</p></div></div>
{note}
<section class="panel scroll" aria-label="Metrics"><table><thead><tr><th>Metric</th>{head(a, "baseline")}{head(b, "this run")}
<th>Change</th></tr></thead><tbody>{metric_rows}</tbody></table></section>
<div class="head"><h2>Cases</h2><div class="row-gap">{tabs}</div></div>
<div class="panel scroll"><table><thead><tr><th>Input</th><th>Expected</th><th>Baseline</th><th>This run</th>
<th>Change</th><th></th></tr></thead><tbody>{case_rows}</tbody></table></div>"""


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def datasets_page(datasets: Sequence[Mapping[str, Any]], *, project: str | None, writable: bool) -> str:
    how = ("Add a run to one from its trace page, with the answer it should have given."
           if writable else "This Studio is read-only, so cases can't be added here.")
    intro = ('<div class="head"><div class="row-gap" style="flex-direction:column;align-items:flex-start">'
             f'<h1>Datasets</h1><p class="lede">Cases collected from recorded runs, to run evals on. {e(how)}</p>'
             "</div></div>")
    if not datasets:
        return intro + ('<div class="panel empty"><p>No datasets yet.</p><p class="soft">Open a trace whose answer '
                        "you want to check in future, and use <strong>Add to a dataset</strong>.</p></div>")
    rows = "".join(
        f'<tr><td><a href="/datasets/{e(quote(d["id"]))}"><strong>{e(d["name"])}</strong></a></td>'
        f'{"" if project else "<td>" + e(d["project"]) + "</td>"}'
        f'<td class="num">{fmt_int(d["cases"])}</td><td class="soft">{e(fmt_ts(d.get("updated_at")))}</td></tr>'
        for d in datasets
    )
    project_head = "" if project else "<th>Project</th>"
    return intro + f"""<div class="panel scroll"><table><thead><tr><th>Dataset</th>{project_head}
<th class="num">Cases</th><th>Last added</th></tr></thead><tbody>{rows}</tbody></table></div>"""


def dataset_page(dataset: Mapping[str, Any], *, token: str | None) -> str:
    cases = dataset.get("cases") or []
    base = f"/datasets/{quote(dataset['id'])}"
    rows = []
    for c in cases:
        source = trace_link(c["trace_id"], "run") if c.get("trace_id") and c.get("trace_exists") else (
            '<span class="faint">deleted</span>' if c.get("trace_id") else "")
        remove = (
            f'<form class="inline" method="post" action="{e(base)}/cases/{e(quote(c["id"]))}/delete">'
            f'{_token_field(token)}<button class="link-btn" type="submit">Remove</button></form>'
            if token
            else ""
        )
        rows.append(
            f'<tr><td>{e(first_line(c["input"], 140))}</td><td>{e(first_line(c.get("expected"), 140))}</td>'
            f'<td>{source}</td><td class="soft">{e(fmt_ts(c["created_at"]))}</td><td>{remove}</td></tr>'
        )
    table = "".join(rows) or '<tr><td colspan="5" class="soft">No cases. Add runs from their trace pages.</td></tr>'
    name = json.dumps(dataset["name"])
    usage = f"""import loomflow.studio
from loomflow.eval import Contains, EvalHarness

dataset = loomflow.studio.dataset({name})
await EvalHarness(agent, [Contains()]).run(dataset, experiment="my change")"""
    count = len(cases)
    return f"""<div class="crumbs"><a href="{href("/datasets", project=dataset["project"])}">Datasets</a>
<span aria-hidden="true">/</span><span>{e(dataset["name"])}</span></div>
<div class="head"><div class="row-gap" style="flex-direction:column;align-items:flex-start"><h1>{e(dataset["name"])}</h1>
<p class="lede">{count} case{"" if count == 1 else "s"} in project {e(dataset["project"])}.</p></div>
<div class="row-gap"><a class="btn" href="{e(base)}.jsonl">Download JSONL</a></div></div>
<div class="panel scroll"><table><thead><tr><th>Input</th><th>Expected</th><th>From</th><th>Added</th><th></th></tr>
</thead><tbody>{table}</tbody></table></div>
<section class="panel pad" aria-label="Run an eval on it"><h2>Run an eval on it</h2>
<pre class="code">{e(usage)}</pre>
<p class="soft">Or download the JSONL file and run <span class="mono">loom eval {e(safe_name(dataset["name"]))}.jsonl --agent app:agent --studio</span>.
Each case keeps its id, so experiments on this dataset compare case by case.</p></section>"""


def dataset_jsonl(dataset: Mapping[str, Any]) -> str:
    """The dataset as JSONL that ``Dataset.from_jsonl`` reads."""
    lines = [
        json.dumps({
            "id": c["id"],
            "input": c["input"],
            "expected": c.get("expected"),
            "metadata": {"studio_trace": c["trace_id"]} if c.get("trace_id") else {},
        }, ensure_ascii=False)
        for c in dataset.get("cases") or []
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def safe_name(name: str) -> str:
    """``name`` as a file name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "dataset"
