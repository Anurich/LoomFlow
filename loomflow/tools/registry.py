"""In-process tool registry.

A :class:`Tool` wraps a Python callable with a JSON-Schema-style input
description. :func:`tool` is a decorator that derives the schema from
type hints. :class:`InProcessToolHost` is the simplest
:class:`~loomflow.core.protocols.ToolHost`: a dict keyed by tool name.
"""

from __future__ import annotations

import functools
import inspect
import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, overload

import anyio
from pydantic import TypeAdapter, ValidationError

from ..core.errors import ConfigError
from ..core.types import ToolDef, ToolEvent, ToolResult

_PRIMITIVE_TO_JSON_SCHEMA: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}
# Validators for parameters annotated with a primitive type: after the
# lenient string coercion (``"8"`` → 8), a value that still isn't the
# declared type (``n: int`` given ``"abc"``) fails here instead of
# reaching the function.
_PRIMITIVE_ADAPTERS: dict[type, TypeAdapter[Any]] = {
    t: TypeAdapter(t) for t in _PRIMITIVE_TO_JSON_SCHEMA
}
# How many validation problems of one parameter are reported back.
_MAX_ARG_ERRORS = 5


class ToolArgumentError(ValueError):
    """The arguments a model supplied don't match the tool's declared
    parameter types. Raised by :meth:`Tool.execute` BEFORE the wrapped
    function is called; the tool host turns it into a tool error that
    names the parameter and the problem, so the model can retry."""


def _short_repr(value: Any, limit: int = 80) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _describe_arg_error(name: str, exc: Exception, value: Any) -> str:
    """One parameter's validation failure, for the model: which
    parameter (and where inside it), what was wrong, what it sent."""
    if isinstance(exc, ValidationError):
        errors = exc.errors(include_url=False)
        parts = []
        for err in errors[:_MAX_ARG_ERRORS]:
            loc = ".".join(str(piece) for piece in err["loc"])
            where = f"{name}.{loc}" if loc else name
            parts.append(f"parameter {where!r}: {err['msg']}")
        if len(errors) > _MAX_ARG_ERRORS:
            parts.append(f"(+{len(errors) - _MAX_ARG_ERRORS} more)")
        detail = "; ".join(parts)
    else:
        detail = f"parameter {name!r}: {type(exc).__name__}: {exc}"
    return f"{detail} (got {_short_repr(value)})"


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
            # ``"8"`` and ``"8.0"`` both → 8. Non-integral floats
            # (``"8.5"``) are NOT silently truncated — the original
            # string passes through so validation / the function's
            # own error surfaces instead of a wrong answer.
            stripped = value.strip()
            if stripped.lstrip("+-").isdigit():
                return int(stripped)
            as_float = float(stripped)
            if as_float.is_integer():
                return int(as_float)
            return value
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
    # End the calling agent's turn once this tool succeeds (see
    # ToolDef.ends_turn) — for control-transfer tools like handoffs.
    ends_turn: bool = False
    # Validators for non-primitive params (Pydantic models, lists,
    # dicts, Literals, ...) keyed by param name — populated by the
    # ``@tool`` decorator so ``execute`` can turn the model's raw
    # dict / JSON-string args back into the annotated Python types.
    param_adapters: dict[str, TypeAdapter[Any]] = field(
        default_factory=dict, repr=False
    )
    # Params annotated with a primitive type (``n: int``), keyed by
    # name — populated by ``@tool`` so ``execute`` can reject a value
    # that isn't that type. Unannotated params are not listed (and
    # not validated).
    param_types: dict[str, type] = field(default_factory=dict, repr=False)
    # Wall-clock cap for one call, in seconds (see ToolDef.timeout):
    # ``None`` uses the agent's default, ``float("inf")`` means no cap.
    timeout: float | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function directly, exactly as written (no
        coercion, validation, thread hop or timeout) — so a
        ``@tool``-decorated function can still be unit-tested like a
        plain function. An ``async def`` tool returns its coroutine."""
        return self.fn(*args, **kwargs)

    def to_def(self) -> ToolDef:
        return ToolDef(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            destructive=self.destructive,
            ends_turn=self.ends_turn,
            timeout=self.timeout,
            # A sync function runs in a worker thread (see ``execute``),
            # which a timeout cannot stop.
            cancellable=_is_async_callable(self.fn),
        )

    async def execute(self, args: Mapping[str, Any]) -> Any:
        """Invoke the underlying callable.

        Async callables (functions, ``functools.partial`` of one, or
        objects with an ``async def __call__``) are awaited; sync ones
        run in a worker thread via :func:`anyio.to_thread.run_sync` so
        they don't block the event loop. A sync callable that returns
        an awaitable has it awaited. A cancelled or timed-out sync
        call returns at once; its thread finishes in the background.
        """
        kwargs = dict(args)
        required = set(self.input_schema.get("required", ()))
        problems: list[str] = []
        # Coerce stringified args to their declared schema types before
        # calling — models often send numbers / bools as strings, which
        # would otherwise crash typed functions and trigger retry storms.
        # A value that still isn't the annotated type afterwards
        # (``n: int`` given ``"abc"``) is a problem reported to the
        # model; the function is not called with it.
        props = self.input_schema.get("properties", {})
        for name, val in list(kwargs.items()):
            if name in self.param_adapters:
                continue  # complex params validate below
            spec = props.get(name)
            if isinstance(spec, dict) and "type" in spec:
                val = kwargs[name] = _coerce_to_json_type(val, spec["type"])
            declared = self.param_types.get(name)
            if declared is None:
                continue
            try:
                kwargs[name] = _PRIMITIVE_ADAPTERS[declared].validate_python(val)
            except ValidationError as exc:
                if val is None and name not in required:
                    # ``null`` for an optional param: use its default.
                    del kwargs[name]
                else:
                    problems.append(_describe_arg_error(name, exc, val))
        # Complex params: validate the model-supplied value back into
        # the annotated type (dict → BaseModel instance, ["8"] →
        # [8], ...). Models also sometimes send nested objects as a
        # JSON *string*; decode and retry once. A value that validates
        # neither way is reported, never passed through raw.
        for name, adapter in self.param_adapters.items():
            if name not in kwargs:
                continue
            val = kwargs[name]
            try:
                kwargs[name] = adapter.validate_python(val)
                continue
            except Exception as exc:  # noqa: BLE001 — fall through to JSON retry
                failure: Exception = exc
            if isinstance(val, str):
                try:
                    kwargs[name] = adapter.validate_python(json.loads(val))
                    continue
                except ValidationError as exc:
                    # It WAS JSON: the decoded value's errors are the
                    # useful ones.
                    failure = exc
                except Exception:  # noqa: BLE001 — not JSON; keep first error
                    pass
            if val is None and name not in required:
                del kwargs[name]  # ``null`` for an optional param
                continue
            problems.append(_describe_arg_error(name, failure, val))
        if problems:
            raise ToolArgumentError(
                f"invalid arguments for tool {self.name!r}: "
                + "; ".join(problems)
                + ". The tool was not run; fix the arguments and call it again."
            )
        if _is_async_callable(self.fn):
            return await self.fn(**kwargs)
        result = await anyio.to_thread.run_sync(
            lambda: self.fn(**kwargs), abandon_on_cancel=True
        )
        if inspect.isawaitable(result):
            result = await result
        return result


def _is_async_callable(fn: Any) -> bool:
    while isinstance(fn, functools.partial):
        fn = fn.func
    return inspect.iscoroutinefunction(fn) or (
        not inspect.isfunction(fn)
        and inspect.iscoroutinefunction(getattr(fn, "__call__", None))  # noqa: B004
    )


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
    ends_turn: bool = False,
    timeout: float | None = None,
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    destructive: bool = False,
    ends_turn: bool = False,
    timeout: float | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Promote a callable to a :class:`Tool`.

    Use as ``@tool`` (bare) or ``@tool(name=..., description=..., destructive=...)``.
    ``ends_turn=True`` makes a successful call end the agent's turn (the
    model is not called again), for tools that hand control elsewhere.
    ``timeout=`` (seconds) overrides the agent's per-tool timeout
    (``Tuning(tool_timeout=...)``, 120s by default) for this tool;
    ``float("inf")`` removes the cap. The result is still directly
    callable (``my_tool(x)`` runs the function as written), so it can
    be unit-tested.
    The schema is derived from parameter annotations: primitive types map
    to their JSON-Schema equivalents, and complex types (Pydantic models,
    ``list[...]``, ``dict[...]``, ``Literal``, optionals, ...) get a full
    recursive schema via :class:`pydantic.TypeAdapter` — at call time the
    model's raw args are validated back into the annotated Python types.
    Annotations Pydantic cannot handle fall back to ``string``.
    """

    if timeout is not None and (isinstance(timeout, bool) or not timeout > 0):
        raise ConfigError(
            f"@tool(timeout=...) must be a positive number of seconds "
            f'(float("inf") for no cap), not {timeout!r}'
        )

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
        schema, adapters, primitives = _schema_from_signature(sig)
        doc_text, param_docs = _parse_docstring(f.__doc__ or "")
        for pname, text in param_docs.items():
            prop = schema["properties"].get(pname)
            if prop is not None and "description" not in prop:
                prop["description"] = text
        return Tool(
            name=name or f.__name__,
            # An explicit description is used as written; otherwise the
            # docstring, minus its Args / Returns / ... sections (whose
            # parameter entries describe the parameters instead).
            description=description.strip() if description else doc_text,
            fn=f,
            input_schema=schema,
            destructive=destructive,
            ends_turn=ends_turn,
            param_adapters=adapters,
            param_types=primitives,
            timeout=timeout,
        )

    if fn is not None:
        return _make(fn)
    return _make


_DOC_SECTION_RE = re.compile(
    r"^(Args|Arguments|Parameters|Params|Keyword Args|Keyword Arguments|"
    r"Returns?|Yields?|Raises|Examples?|Notes?|See Also|Attributes)\s*:?\s*$",
    re.IGNORECASE,
)
# Google ``name (type): text`` / ``name: text``; NumPy ``name : type``
# (a space before the colon — the text follows on indented lines).
_DOC_PARAM_RE = re.compile(r"^(\*{0,2}\w+)( ?)(\([^)]*\))?\s*:\s*(.*)$")


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """``(description, {param: text})`` from a Google- or NumPy-style
    docstring: the text before the first section heading, and the
    entries of an ``Args:`` / ``Parameters`` section."""
    lines = inspect.cleandoc(doc).splitlines() if doc else []
    description: list[str] = []
    params: dict[str, str] = {}
    section: str | None = None
    current: str | None = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        underline = i + 1 < len(lines) and set(lines[i + 1].strip()) == {"-"}
        if _DOC_SECTION_RE.match(stripped) and (stripped.endswith(":") or underline):
            section = stripped.rstrip(":").lower()
            current = None
            continue
        if set(stripped) == {"-"} and section is not None:
            continue  # NumPy heading underline
        if section is None:
            description.append(line)
            continue
        if section not in ("args", "arguments", "parameters", "params", "keyword args", "keyword arguments"):
            continue
        match = _DOC_PARAM_RE.match(stripped)
        indent = len(line) - len(line.lstrip())
        if match and indent <= 4 and not (current and indent > 4):
            current = match.group(1).lstrip("*")
            numpy_style = bool(match.group(2)) and not match.group(3)
            params[current] = "" if numpy_style else match.group(4).strip()
        elif current and stripped:
            params[current] = f"{params[current]} {stripped}".strip()
    return "\n".join(description).strip(), {k: v for k, v in params.items() if v}


def _schema_from_signature(
    sig: inspect.Signature,
) -> tuple[dict[str, Any], dict[str, TypeAdapter[Any]], dict[str, type]]:
    """Derive ``(input_schema, param_adapters, param_types)`` from a
    signature.

    Primitives map straight to their JSON-Schema type (and stay on the
    cheap string-coercion path in :meth:`Tool.execute`). Anything else
    goes through :class:`pydantic.TypeAdapter` for a full recursive
    schema, all parameters in one generation pass so two different
    models sharing a class name get distinct ``$defs`` entries; a
    parameter that is a model is inlined, and ``$defs`` sit at the
    schema root so ``#/$defs/...`` refs stay valid inside the tool
    definition. Annotations Pydantic cannot model fall back to
    ``string`` exactly as before.
    """
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    defs: dict[str, Any] = {}
    adapters: dict[str, TypeAdapter[Any]] = {}
    primitives: dict[str, type] = {}
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        ann = param.annotation
        if ann is inspect.Parameter.empty:
            properties[pname] = {"type": "string"}
        elif ann in _PRIMITIVE_TO_JSON_SCHEMA:
            properties[pname] = {"type": _PRIMITIVE_TO_JSON_SCHEMA[ann]}
            primitives[pname] = ann
        else:
            try:
                adapter: TypeAdapter[Any] = TypeAdapter(ann)
                adapter.json_schema()
            except Exception:  # noqa: BLE001 — unmodellable → legacy fallback
                properties[pname] = {"type": "string"}
            else:
                properties[pname] = {}  # filled below, in signature order
                adapters[pname] = adapter
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    if adapters:
        by_key, top = TypeAdapter.json_schemas(
            [(name, "validation", a) for name, a in adapters.items()]
        )
        defs = dict(top.get("$defs", {}))
        for name in adapters:
            piece = dict(by_key[(name, "validation")])
            ref = piece.get("$ref")
            if len(piece) == 1 and isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = defs.get(ref.removeprefix("#/$defs/"))
                if target is not None:
                    piece = dict(target)
            properties[name] = piece
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
    }
    if defs:
        schema["$defs"] = defs
    return schema, adapters, primitives


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
        self._tools: dict[str, Tool] = {}
        for item in tools or []:
            self._add(_coerce_tool(item), replace=False)

    def _add(self, t: Tool, *, replace: bool) -> None:
        existing = self._tools.get(t.name)
        if existing is not None and existing is not t and not replace:
            # The model addresses tools by name: a second tool under
            # the same name would silently replace the first.
            raise ConfigError(
                f"duplicate tool name {t.name!r}: another tool is already "
                f"registered under this name and would be silently "
                f"replaced. Rename one (@tool(name=...)), or call "
                f"register(..., replace=True) to replace it on purpose."
            )
        self._tools[t.name] = t

    def register(
        self, item: Tool | Callable[..., Any], *, replace: bool = False
    ) -> Tool:
        """Add a tool. A name already taken by a different tool raises
        :class:`~loomflow.core.errors.ConfigError` unless
        ``replace=True``."""
        t = _coerce_tool(item)
        self._add(t, replace=replace)
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
            return ToolResult.from_exception(call_id, exc)
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
