"""In-process tool registry.

A :class:`Tool` wraps a Python callable with a JSON-Schema-style input
description. :func:`tool` is a decorator that derives the schema from
type hints. :class:`InProcessToolHost` is the simplest
:class:`~loomflow.core.protocols.ToolHost`: a dict keyed by tool name.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, overload

import anyio

from ..core.types import ToolDef, ToolEvent, ToolResult

_PRIMITIVE_TO_JSON_SCHEMA: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _coerce_to_json_type(value: Any, json_type: str) -> Any:
    """Best-effort coerce a model-supplied arg to its declared JSON type.

    Models routinely emit numeric / boolean tool arguments as *strings*
    (``rate_pct="8"``, ``replace_all="true"``) even when the schema says
    ``integer`` / ``boolean``. Passing those straight to a typed Python
    function raises ``TypeError`` (``"8" / 100``), which the agent loop
    then burns turns retrying. We coerce here, matching what mature
    frameworks do at their schema-validation layer.

    Conservative by design: only string inputs are coerced, only when
    the schema names a primitive type, and any failure passes the
    ORIGINAL value through so the function's own error still surfaces
    rather than being masked.
    """
    if not isinstance(value, str):
        return value
    try:
        if json_type == "integer":
            # ``"8"`` and ``"8.0"`` both → 8; reject true floats.
            return int(value) if value.strip().lstrip("-").isdigit() else int(float(value))
        if json_type == "number":
            return float(value)
        if json_type == "boolean":
            low = value.strip().lower()
            if low in ("true", "1", "yes"):
                return True
            if low in ("false", "0", "no", ""):
                return False
            return value  # unrecognised → pass through
    except (ValueError, TypeError):
        return value
    return value


@dataclass
class Tool:
    """A registered tool: definition plus the callable that executes it."""

    name: str
    description: str
    fn: Callable[..., Any]
    input_schema: dict[str, Any] = field(default_factory=dict)
    destructive: bool = False

    def to_def(self) -> ToolDef:
        return ToolDef(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            destructive=self.destructive,
        )

    async def execute(self, args: Mapping[str, Any]) -> Any:
        """Invoke the underlying callable.

        Async functions are awaited; sync functions are dispatched to a
        worker thread via :func:`anyio.to_thread.run_sync` so they don't
        block the event loop.
        """
        kwargs = dict(args)
        # Coerce stringified args to their declared schema types before
        # calling — models often send numbers / bools as strings, which
        # would otherwise crash typed functions and trigger retry storms.
        props = self.input_schema.get("properties", {})
        if props:
            for name, val in list(kwargs.items()):
                spec = props.get(name)
                if isinstance(spec, dict) and "type" in spec:
                    kwargs[name] = _coerce_to_json_type(val, spec["type"])
        if inspect.iscoroutinefunction(self.fn):
            return await self.fn(**kwargs)
        return await anyio.to_thread.run_sync(lambda: self.fn(**kwargs))


# ---------------------------------------------------------------------------
# tool() decorator
# ---------------------------------------------------------------------------


@overload
def tool(fn: Callable[..., Any]) -> Tool: ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    destructive: bool = False,
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    destructive: bool = False,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Promote a callable to a :class:`Tool`.

    Use as ``@tool`` (bare) or ``@tool(name=..., description=..., destructive=...)``.
    The schema is derived from parameter annotations; primitive types map
    to their JSON-Schema equivalents, anything else falls back to ``string``.
    """

    def _make(f: Callable[..., Any]) -> Tool:
        # ``eval_str=True`` resolves PEP 563 stringized annotations
        # (``from __future__ import annotations`` turns ``offset: int``
        # into the *string* "int", which would otherwise fall back to
        # the "string" JSON type and defeat arg coercion). Fall back to
        # the raw signature if evaluation fails (forward refs to names
        # not importable at decoration time).
        try:
            sig = inspect.signature(f, eval_str=True)
        except (NameError, TypeError):
            sig = inspect.signature(f)
        schema = _schema_from_signature(sig)
        return Tool(
            name=name or f.__name__,
            description=(description or (f.__doc__ or "")).strip().split("\n")[0],
            fn=f,
            input_schema=schema,
            destructive=destructive,
        )

    if fn is not None:
        return _make(fn)
    return _make


def _schema_from_signature(sig: inspect.Signature) -> dict[str, Any]:
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        ann = param.annotation
        json_type = _PRIMITIVE_TO_JSON_SCHEMA.get(ann, "string")
        properties[pname] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


# ---------------------------------------------------------------------------
# InProcessToolHost
# ---------------------------------------------------------------------------


def _coerce_tool(item: Tool | Callable[..., Any]) -> Tool:
    if isinstance(item, Tool):
        return item
    if callable(item):
        return tool(item)
    raise TypeError(
        f"tools= entries must be Tool instances or callables "
        f"(decorate with @tool or pass a plain function); "
        f"got {type(item).__name__}: {item!r}.\n"
        f"Example:\n"
        f"  from loomflow.tools import tool\n"
        f"  @tool\n"
        f"  async def my_tool(query: str) -> str: ...\n"
        f"  agent = Agent(..., tools=[my_tool])"
    )


class InProcessToolHost:
    """A dict-backed :class:`~loomflow.core.protocols.ToolHost`."""

    def __init__(self, tools: list[Tool | Callable[..., Any]] | None = None) -> None:
        coerced = [_coerce_tool(t) for t in (tools or [])]
        self._tools: dict[str, Tool] = {t.name: t for t in coerced}

    def register(self, item: Tool | Callable[..., Any]) -> Tool:
        t = _coerce_tool(item)
        self._tools[t.name] = t
        return t

    def unregister(self, name: str) -> bool:
        """Remove a tool by name. Returns ``True`` if removed."""
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    async def list_tools(self, *, query: str | None = None) -> list[ToolDef]:
        defs = [t.to_def() for t in self._tools.values()]
        if query:
            q = query.lower()
            defs = [d for d in defs if q in d.name.lower() or q in d.description.lower()]
        return defs

    async def call(
        self,
        tool: str,
        args: Mapping[str, Any],
        *,
        call_id: str = "",
    ) -> ToolResult:
        registered = self._tools.get(tool)
        if registered is None:
            return ToolResult.error_(call_id=call_id, message=f"unknown tool: {tool}")
        try:
            output = await registered.execute(args)
        except Exception as exc:  # noqa: BLE001 — surface failure as ToolResult
            return ToolResult.error_(call_id=call_id, message=str(exc))
        return ToolResult.success(call_id=call_id, output=output)

    async def watch(self) -> AsyncIterator[ToolEvent]:
        """In-process registry is static; the generator yields nothing.

        Iterating over an empty tuple keeps this an async generator
        (so the return type is ``AsyncIterator``) without ever producing
        an event at runtime.
        """
        empty: tuple[ToolEvent, ...] = ()
        for ev in empty:
            yield ev


# Public alias used by Agent
ToolCallable = Callable[..., Awaitable[Any]] | Callable[..., Any]
