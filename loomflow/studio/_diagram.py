"""Draw an agent's or workflow's structure as SVG, on the server.

The input is :meth:`loomflow.graph.AgentGraph.to_dict` output. The
layout is layered top to bottom: each node sits one layer below the
lowest node that points at it, nodes in a layer are ordered to keep
edges short, and edges that point back up (an actor/critic loop, a
swarm handoff) are drawn as dashed curves on the side.

To keep the picture readable, a node's tools are listed inside it
rather than drawn as nodes, edges between the same two nodes are
merged, and a swarm's every-peer-to-every-peer handoffs become a note
on the group. With ``active`` names (from a recorded run), the parts
the run used are highlighted and the rest dimmed.

No scripts and no external requests: the SVG is inline in the page
and styled by the app's stylesheet.
"""

from __future__ import annotations

import html
from collections import defaultdict
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

_CHAR = 7.0          # px per character, 12px text
_TITLE_CHAR = 7.8    # px per character, 13px bold text
_MIN_W, _MAX_W = 120.0, 250.0
_LAYER_GAP = 70.0
_NODE_GAP = 34.0
_PAD = 24.0
_SIDE = 70.0         # room on the right for back edges
_LABEL_MAX = 30


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


@dataclass
class _Box:
    id: str
    label: str
    kind: str
    keys: tuple[str, ...]
    detail: str = ""
    tools: list[str] = field(default_factory=list)
    layer: int = 0
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    # (text, css class) per line of the box.
    lines: list[tuple[str, str]] = field(default_factory=list)

    @property
    def cx(self) -> float:
        return self.x + self.w / 2


@dataclass
class _Line:
    source: str
    target: str
    labels: list[str]
    dashed: bool
    back: bool = False


def _prepare(structure: Mapping[str, Any]) -> tuple[dict[str, _Box], list[_Line], list[dict[str, Any]]]:
    boxes = {
        n["id"]: _Box(id=n["id"], label=str(n.get("label") or ""), kind=str(n.get("kind") or "step"),
                      keys=tuple(n.get("keys") or ()), detail=str(n.get("detail") or ""))
        for n in structure.get("nodes", [])
    }
    raw_edges = [e for e in structure.get("edges", []) if e["source"] in boxes and e["target"] in boxes]
    subgraphs = [dict(g) for g in structure.get("subgraphs", [])]

    # Tools go inside the agent that uses them.
    incident: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in raw_edges:
        incident[edge["source"]].append(edge)
        incident[edge["target"]].append(edge)
    folded: set[str] = set()
    for box in boxes.values():
        if box.kind != "tool":
            continue
        edges = incident[box.id]
        if len(edges) == 1 and edges[0]["target"] == box.id and edges[0].get("label") == "uses":
            boxes[edges[0]["source"]].tools.append(box.label)
            folded.add(box.id)
    for box_id in folded:
        del boxes[box_id]
    raw_edges = [e for e in raw_edges if e["source"] in boxes and e["target"] in boxes]

    # A swarm's all-pairs handoffs become a note on its group.
    for group in subgraphs:
        members = [m for m in group.get("members", []) if m in boxes]
        handoffs = {(e["source"], e["target"]) for e in raw_edges
                    if e.get("label") == "handoff" and e["source"] in members and e["target"] in members}
        if len(members) > 1 and len(handoffs) == len(members) * (len(members) - 1):
            raw_edges = [e for e in raw_edges if (e["source"], e["target"]) not in handoffs or e.get("label") != "handoff"]
            group["label"] = f"{group.get('label', '')} (any can hand off to any other)".strip()

    # One line per pair of nodes.
    merged: dict[tuple[str, str], _Line] = {}
    for edge in raw_edges:
        key = (edge["source"], edge["target"])
        label = str(edge.get("label") or "")
        dashed = edge.get("style") == "dashed"
        line = merged.get(key)
        if line is None:
            merged[key] = _Line(edge["source"], edge["target"], [label] if label else [], dashed)
        else:
            if label and label not in line.labels:
                line.labels.append(label)
            line.dashed = line.dashed and dashed
    return boxes, list(merged.values()), subgraphs


def _layers(boxes: dict[str, _Box], lines: list[_Line]) -> dict[int, list[_Box]]:
    order = list(boxes)
    out: dict[str, list[str]] = defaultdict(list)
    incoming: dict[str, int] = defaultdict(int)
    for line in lines:
        if line.source != line.target:
            out[line.source].append(line.target)
            incoming[line.target] += 1
    # Edges back into a node on the current walk close a cycle: draw
    # them as back edges and lay the rest out as a DAG.
    state: dict[str, int] = {}
    back: set[tuple[str, str]] = set()

    def walk(node: str) -> None:
        state[node] = 1
        for nxt in out[node]:
            if state.get(nxt) == 1:
                back.add((node, nxt))
            elif nxt not in state:
                walk(nxt)
        state[node] = 2

    roots = [n for n in order if incoming[n] == 0] or order[:1]
    for node in roots + order:
        if node not in state:
            walk(node)
    for line in lines:
        line.back = (line.source, line.target) in back or line.source == line.target
    preds: dict[str, list[str]] = defaultdict(list)
    for line in lines:
        if not line.back:
            preds[line.target].append(line.source)
    layer: dict[str, int] = {}

    def depth(node: str, seen: frozenset[str] = frozenset()) -> int:
        if node in layer:
            return layer[node]
        if node in seen:
            return 0
        value = max((depth(p, seen | {node}) + 1 for p in preds[node]), default=0)
        layer[node] = value
        return value

    for node in order:
        boxes[node].layer = depth(node)
    by_layer: dict[int, list[_Box]] = defaultdict(list)
    for node in order:
        by_layer[boxes[node].layer].append(boxes[node])
    # Order each layer by the mean position of its neighbours above,
    # then below, a few times over.
    succs: dict[str, list[str]] = defaultdict(list)
    for line in lines:
        if not line.back:
            succs[line.source].append(line.target)
    def reorder(index: int, neighbour: int, links: dict[str, list[str]]) -> None:
        if neighbour not in by_layer:
            return
        pos = {b.id: i for i, b in enumerate(by_layer[neighbour])}
        here = {b.id: i for i, b in enumerate(by_layer[index])}
        by_layer[index].sort(key=lambda b: _mean([pos[n] for n in links[b.id] if n in pos], here[b.id]))

    for _ in range(4):
        for index in sorted(by_layer)[1:]:
            reorder(index, index - 1, preds)
        for index in sorted(by_layer, reverse=True)[1:]:
            reorder(index, index + 1, succs)
    return by_layer


def _label(labels: list[str], target: str) -> str:
    """One label for merged edges: ``delegate: fraud`` and ``forward:
    fraud`` read ``delegate, forward (fraud)``, or ``delegate, forward``
    on the edge into ``fraud``, which already says where it goes."""
    parts = [label.partition(": ") for label in labels]
    if all(sep for _, sep, _ in parts) and len({rest for _, _, rest in parts}) == 1:
        verbs = ", ".join(verb for verb, _, _ in parts)
        return verbs if parts[0][2] == target else f"{verbs} ({parts[0][2]})"
    return ", ".join(labels)


def _cubic(p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float],
           p3: tuple[float, float], t: float) -> tuple[float, float]:
    u = 1 - t
    return (
        u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
        u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1],
    )


def _mean(values: list[int], fallback: float) -> float:
    return sum(values) / len(values) if values else float(fallback)


def _wrap(text: str, width: int, limit: int) -> list[str]:
    """``text`` in lines of at most ``width`` characters, ``limit``
    lines at most (the last one ending in "…" if cut)."""
    words, out, line = text.split(), [], ""
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    if len(out) > limit:
        out = out[:limit]
        out[-1] = _clip(out[-1] + " …", width)
    return out


def _size(box: _Box) -> None:
    lines = [(_clip(box.label, _LABEL_MAX), "dg-title")]
    if box.kind in ("start", "end"):
        box.w, box.h, box.lines = 76.0, 30.0, lines
        return
    if box.detail:
        lines.append((_clip(box.detail, 36), "dg-detail"))
    if box.tools:
        more = f" +{len(box.tools) - 6} more" if len(box.tools) > 6 else ""
        text = "tools: " + ", ".join(box.tools[:6]) + more
        lines += [(ln, "dg-tools") for ln in _wrap(text, 32, 3)]
    widest = max([len(lines[0][0]) * _TITLE_CHAR] + [len(text) * _CHAR for text, _ in lines[1:]])
    box.w = min(max(widest + 28.0, _MIN_W), _MAX_W)
    box.h = 18.0 + 16.0 * len(lines) + 4.0
    box.lines = lines


def _place(by_layer: dict[int, list[_Box]], lines: list[_Line]) -> tuple[float, float]:
    for layer in by_layer.values():
        for box in layer:
            _size(box)
    y = _PAD
    for index in sorted(by_layer):
        tallest = max(b.h for b in by_layer[index])
        for box in by_layer[index]:
            box.y = y + (tallest - box.h) / 2
        y += tallest + _LAYER_GAP
    widths = {i: sum(b.w for b in layer) + _NODE_GAP * (len(layer) - 1) for i, layer in by_layer.items()}
    full = max(widths.values(), default=0.0)
    for index, layer in by_layer.items():
        x = _PAD + (full - widths[index]) / 2
        for box in layer:
            box.x = x
            x += box.w + _NODE_GAP
    # Pull each node towards the nodes that point at it, keeping order
    # and spacing.
    boxes = {b.id: b for layer in by_layer.values() for b in layer}
    preds: dict[str, list[str]] = defaultdict(list)
    for line in lines:
        if not line.back:
            preds[line.target].append(line.source)
    for _ in range(3):
        for index in sorted(by_layer)[1:]:
            layer = by_layer[index]
            wanted = [
                sum(boxes[p].cx for p in preds[b.id]) / len(preds[b.id]) - b.w / 2 if preds[b.id] else b.x
                for b in layer
            ]
            # Keep the order and the spacing, then centre the layer
            # where its nodes wanted to be.
            left = -1e9
            for box, want in zip(layer, wanted, strict=True):
                box.x = max(want, left)
                left = box.x + box.w + _NODE_GAP
            delta = sum(wanted) / len(wanted) - sum(b.x for b in layer) / len(layer)
            for box in layer:
                box.x += delta
    lowest = min((b.x for b in boxes.values()), default=_PAD)
    for box in boxes.values():
        box.x += _PAD - lowest
    width = max((b.x + b.w for b in boxes.values()), default=_PAD) + _PAD + _SIDE
    height = max((b.y + b.h for b in boxes.values()), default=_PAD) + _PAD
    return width, height


def _is_on(box: _Box, active: Collection[str]) -> bool:
    return bool(set(box.keys) & set(active)) or box.label in active


def render(
    structure: Mapping[str, Any] | None,
    *,
    uid: str,
    title: str,
    active: Collection[str] = (),
) -> str:
    """The SVG for ``structure``, plus a text version, as HTML."""
    if not structure or not structure.get("nodes"):
        return '<p class="soft">No architecture was recorded for this version.</p>'
    boxes, lines, subgraphs = _prepare(structure)
    by_layer = _layers(boxes, lines)
    width, height = _place(by_layer, lines)
    lit = bool(active)
    on = {b.id for b in boxes.values() if lit and _is_on(b, active)}
    if lit:
        # The entry and exit light up with the run.
        for box in boxes.values():
            if box.kind in ("start", "end"):
                on.add(box.id)
    # Groups are drawn first, edges over them, nodes on top.
    parts: list[str] = []
    placed: list[tuple[float, float, float]] = []
    for group in subgraphs:
        members = [boxes[m] for m in group.get("members", []) if m in boxes]
        if not members:
            continue
        # The label goes along the bottom, clear of the edges coming in.
        x0 = min(b.x for b in members) - 12
        y0 = min(b.y for b in members) - 10
        x1 = max(b.x + b.w for b in members) + 12
        y1 = max(b.y + b.h for b in members) + 26
        parts.append(
            f'<g class="dg-group"><rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" rx="10"/>'
            f'<text x="{x0 + 10:.1f}" y="{y1 - 8:.1f}">{_e(_clip(str(group.get("label", "")), 60))}</text></g>'
        )
    for line in lines:
        src, dst = boxes[line.source], boxes[line.target]
        edge_on = lit and src.id in on and dst.id in on
        cls = "dg-edge" + (" dashed" if line.dashed or line.back else "") + (" on" if edge_on else "")
        if line.back:
            sx, sy = src.x + src.w, src.y + src.h / 2
            tx, ty = dst.x + dst.w, dst.y + dst.h / 2
            reach = max(sx, tx) + _SIDE * 0.8
            path = f"M{sx:.1f},{sy:.1f} C{reach:.1f},{sy:.1f} {reach:.1f},{ty:.1f} {tx + 4:.1f},{ty:.1f}"
            mx, my = reach - 6, (sy + ty) / 2
        else:
            sx, sy = src.cx, src.y + src.h
            tx, ty = dst.cx, dst.y
            dy = max((ty - sy) / 2, 20.0)
            path = f"M{sx:.1f},{sy:.1f} C{sx:.1f},{sy + dy:.1f} {tx:.1f},{ty - dy:.1f} {tx:.1f},{ty - 3:.1f}"
            # Nearer the target, so labels of edges fanning out from one
            # node land apart.
            mx, my = _cubic((sx, sy), (sx, sy + dy), (tx, ty - dy), (tx, ty), 0.62)
        parts.append(f'<path class="{cls}" d="{path}" marker-end="url(#{uid}-arrow)"/>')
        if line.labels:
            text = _clip(_label(line.labels, dst.label), 30)
            w = len(text) * 6.2 + 10
            # Step a label down past any label already drawn where it
            # would go.
            for _ in range(4):
                if not any(abs(mx - px) * 2 < w + pw + 4 and abs(my - py) < 19 for px, py, pw in placed):
                    break
                my += 20
            placed.append((mx, my, w))
            parts.append(
                f'<g class="dg-label{" on" if edge_on else ""}"><rect x="{mx - w / 2:.1f}" y="{my - 9:.1f}" '
                f'width="{w:.1f}" height="17" rx="4"/><text x="{mx:.1f}" y="{my + 4:.1f}">{_e(text)}</text></g>'
            )
    for box in boxes.values():
        state = (" dg-on" if box.id in on else " dg-off") if lit else ""
        radius = box.h / 2 if box.kind in ("start", "end") else 8
        parts.append(
            f'<g class="dg-node dg-{_e(box.kind)}{state}"><title>{_e(box.label)}</title>'
            f'<rect x="{box.x:.1f}" y="{box.y:.1f}" width="{box.w:.1f}" height="{box.h:.1f}" rx="{radius:.1f}"/>'
        )
        for index, (text, cls) in enumerate(box.lines):
            ty = box.y + (box.h / 2 + 4 if box.kind in ("start", "end") else 20 + 16 * index)
            parts.append(f'<text class="{cls}" x="{box.cx:.1f}" y="{ty:.1f}">{_e(text)}</text>')
        parts.append("</g>")
    svg = (
        f'<svg class="diagram" xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-labelledby="{uid}-t">'
        f'<title id="{uid}-t">{_e(title)}</title>'
        f'<defs><marker id="{uid}-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="dg-arrow"/></marker></defs>'
        + "".join(parts)
        + "</svg>"
    )
    return f'<div class="diagram-box">{svg}</div>{_as_text(boxes, lines, on if lit else None)}'


def _as_text(boxes: dict[str, _Box], lines: list[_Line], on: set[str] | None) -> str:
    def name(box: _Box) -> str:
        used = " (used in this run)" if on is not None and box.id in on and box.kind not in ("start", "end") else ""
        tools = f"; tools: {', '.join(box.tools)}" if box.tools else ""
        return f"{box.label}{used}{tools}"

    items = "".join(
        f"<li>{_e(name(boxes[ln.source]))} → {_e(name(boxes[ln.target]))}"
        f"{' (' + _e(', '.join(ln.labels)) + ')' if ln.labels else ''}</li>"
        for ln in lines
    ) or "".join(f"<li>{_e(name(b))}</li>" for b in boxes.values())
    return f'<details class="diagram-text"><summary>As text</summary><ul>{items}</ul></details>'
