"""Code mode: let the model call tools by *writing code* instead of
emitting one tool_use block per call.

The two-tool pattern (Cloudflare "code mode" / Anthropic "code
execution with MCP"): instead of putting N tool schemas in the model's
context, register exactly two tools —

* ``search_api(query)`` — returns typed Python signature stubs for the
  matching tools, so the model discovers the API progressively without
  any tool definitions in context.
* ``run_code(code)`` — executes the model's Python. Each underlying
  tool is available as a real ``async`` callable; intermediate tool
  results stay **out of the model's context** — only what the code
  ``print()``s or assigns to ``result`` comes back (capped at 20k
  chars). This kills the 50k-token intermediate-result problem: the
  model writes ``rows = await query_db(...); result = len(rows)`` and
  sees ``"8231"``, not the rows.

Two execution modes
-------------------

* **Tool-binding mode (default, in-process).** ``executor=None``.
  The code runs via ``exec`` inside the agent process with restricted
  builtins (no ``open``/``eval``/``exec``; imports limited to ``json``,
  ``re``, ``math``, ``datetime``) and each tool bound to
  ``host.call(...)``. BE HONEST about the trust model: this restriction
  is an accident guard, **not** a security boundary — in-process code
  is exactly as trusted as any direct tool call (same process, same
  memory; a determined payload can escape a builtins allowlist).
  Permission/hook layers wrapped around the host still see every
  inner ``host.call``.
* **Data-transform mode (out-of-process).** Pass ``executor=`` (e.g.
  :class:`~loomflow.tools.executor.SubprocessExecutor`). The code runs
  in the executor **without tool bindings** — use it to let the model
  crunch untrusted data (parse, aggregate, transform) with process
  isolation, a hard timeout, and a minimal environment.

Out-of-process *tool dispatch* (running the code remotely while its
tool calls round-trip back to the host over a JSONL pipe) is future
work; v1 deliberately ships the simple, correct halves of the split.

Timeout caveat (in-process mode): ``anyio.fail_after`` cancels at
``await`` points. Model code that busy-loops synchronously (no await)
cannot be interrupted — another reason the executor mode exists.
"""

from __future__ import annotations

import ast
import builtins
import datetime
import inspect
import io
import json
import math
import re
import types
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import anyio

from ..core.protocols import ToolHost
from ..core.types import ToolDef
from .executor import CodeExecutor
from .registry import InProcessToolHost, Tool

#: Max characters of ``run_code`` output returned to the model.
RESULT_CAP = 20_000

_JSON_TO_PY_HINT: dict[str, str] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
    "null": "None",
}

# Builtins available to in-process run_code. An accident guard (no
# open / eval / exec / __import__ / globals), NOT a security boundary
# — see the module docstring.
_SAFE_BUILTIN_NAMES: tuple[str, ...] = (
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
    "callable", "chr", "classmethod", "complex", "dict", "divmod",
    "enumerate", "filter", "float", "format", "frozenset", "getattr",
    "hasattr", "hash", "hex", "id", "int", "isinstance", "issubclass",
    "iter", "len", "list", "map", "max", "min", "next", "object",
    "oct", "ord", "pow", "property", "range", "repr", "reversed",
    "round", "set", "slice", "sorted", "staticmethod", "str", "sum",
    "super", "tuple", "type", "zip", "__build_class__",
    # exceptions the code may reasonably raise / catch
    "ArithmeticError", "AssertionError", "AttributeError",
    "BaseException", "Exception", "GeneratorExit", "ImportError",
    "IndexError", "KeyError", "KeyboardInterrupt", "LookupError",
    "NameError", "NotImplementedError", "OverflowError",
    "RecursionError", "RuntimeError", "StopAsyncIteration",
    "StopIteration", "TypeError", "ValueError", "ZeroDivisionError",
)

_ALLOWED_MODULES: dict[str, Any] = {
    "json": json,
    "re": re,
    "math": math,
    "datetime": datetime,
}


# ---------------------------------------------------------------------------
# Stub rendering — the "typed module" the model discovers via search_api
# ---------------------------------------------------------------------------


def _py_hint(spec: Any) -> str:
    if isinstance(spec, dict):
        hint = _JSON_TO_PY_HINT.get(spec.get("type", ""))
        if hint:
            return hint
    return "Any"


def _render_stub(d: ToolDef) -> str:
    """Render one tool as a readable async signature + docstring."""
    props = d.input_schema.get("properties", {}) or {}
    required = set(d.input_schema.get("required", []) or [])
    # Required params first (no default), then optional ones.
    ordered = sorted(props, key=lambda p: (p not in required, list(props).index(p)))
    params: list[str] = []
    arg_docs: list[str] = []
    for pname in ordered:
        spec = props.get(pname, {})
        hint = _py_hint(spec)
        params.append(f"{pname}: {hint}" + ("" if pname in required else " = ..."))
        desc = spec.get("description", "") if isinstance(spec, dict) else ""
        if desc:
            first = " ".join(str(desc).split())
            if len(first) > 160:
                first = first[:157] + "..."
            arg_docs.append(f"        {pname} ({hint}): {first}")
    sig = f"async def {d.name}({', '.join(params)}) -> Any:"
    doc_lines = ['    """' + " ".join(d.description.split())]
    if arg_docs:
        doc_lines.append("")
        doc_lines.append("    Args:")
        doc_lines.extend(arg_docs)
    doc_lines.append('    """')
    return "\n".join([sig, *doc_lines])


def _score(d: ToolDef, terms: list[str]) -> int:
    if not terms:
        return 1  # empty query lists everything
    name = d.name.lower()
    hay = f"{name} {d.description.lower()}"
    score = 0
    for t in terms:
        if t in name:
            score += 5
        if t in hay:
            score += 1
    return score


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_code_mode_tools(
    host_or_tools: ToolHost | Sequence[Tool | Callable[..., Any]],
    executor: CodeExecutor | None = None,
    *,
    module_name: str = "tools",
    timeout_s: float = 30.0,
) -> list[Tool]:
    """Build the ``search_api`` + ``run_code`` tool pair.

    ``host_or_tools`` is either an existing
    :class:`~loomflow.core.protocols.ToolHost` (permission/sandbox
    wrappers included — inner calls go through it) or a plain list of
    tools/callables, which gets wrapped in an
    :class:`~loomflow.tools.registry.InProcessToolHost`.

    ``executor=None`` (default) → in-process tool-binding mode;
    ``executor=SubprocessExecutor()`` (or any :class:`CodeExecutor`)
    → out-of-process data-transform mode WITHOUT tool bindings. See
    the module docstring for the trust-model tradeoffs.

    ``module_name`` is the namespace the stubs advertise: inside
    ``run_code`` every tool is bound both as a bare name
    (``await read(...)``) and as an attribute (``await tools.read(...)``).
    """
    host: ToolHost
    if hasattr(host_or_tools, "list_tools") and hasattr(host_or_tools, "call"):
        host = cast(ToolHost, host_or_tools)
    else:
        host = InProcessToolHost(list(host_or_tools))

    async def _search_api(query: str) -> str:
        defs = await host.list_tools()
        terms = [t for t in re.split(r"\W+", query.lower()) if t]
        scored = [(s, d.name, _render_stub(d)) for d in defs if (s := _score(d, terms)) > 0]
        if not scored:
            names = ", ".join(sorted(d.name for d in defs)) or "(none)"
            return (
                f"No API functions match {query!r}. "
                f"Available functions: {names}. Try search_api('') to list all."
            )
        scored.sort(key=lambda x: (-x[0], x[1]))
        header = (
            f"# Python API (module {module_name!r}). Call these from run_code, e.g.\n"
            f"#   data = await {scored[0][1]}(...)\n"
            f"#   result = <what you want returned>\n"
        )
        body = "\n\n".join(stub for _, _, stub in scored)
        return _cap(header + body)

    if executor is not None:
        run_code_fn = _make_executor_run_code(executor, timeout_s)
        run_code_desc = (
            "Execute Python code out-of-process in a sandboxed executor "
            "(fresh scratch dir, minimal env, hard timeout). NO tool "
            "functions are available in this mode — use it for pure "
            "data transformation. print() what you want returned "
            "(the `result` variable is NOT read in this mode). Write "
            "file outputs under ./out/."
        )
    else:
        run_code_fn = _make_inprocess_run_code(host, module_name, timeout_s)
        run_code_desc = (
            "Execute Python code that can call the API functions "
            "discovered via search_api (they are async — use `await`). "
            "Top-level `await` is allowed. Assign your answer to a "
            "variable named `result` (or print() it); ONLY that comes "
            "back to you, so filter/aggregate large tool outputs in "
            "code instead of returning them raw. Imports are limited "
            "to json, re, math, datetime. Timeout: "
            f"{timeout_s:.0f}s."
        )

    search_tool = Tool(
        name="search_api",
        description=(
            "Search the Python tool API available inside run_code. "
            "Returns matching async function signatures with docs. "
            "Pass keywords (e.g. 'weather forecast'); pass '' to list "
            "every function."
        ),
        fn=_search_api,
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords to match against function names and docs.",
                },
            },
            "required": ["query"],
        },
    )
    run_tool = Tool(
        name="run_code",
        description=run_code_desc,
        fn=run_code_fn,
        input_schema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source to execute.",
                },
            },
            "required": ["code"],
        },
        destructive=True,  # arbitrary code / can invoke destructive tools
    )
    return [search_tool, run_tool]


# ---------------------------------------------------------------------------
# run_code — executor (data-transform) mode
# ---------------------------------------------------------------------------


def _make_executor_run_code(
    executor: CodeExecutor, timeout_s: float
) -> Callable[[str], Any]:
    async def _run_code(code: str) -> str:
        res = await executor.run(code, language="python", timeout_s=timeout_s)
        if res.timed_out:
            return f"ERROR: code timed out after {timeout_s:.1f}s"
        sections: list[str] = []
        if res.stdout:
            sections.append(res.stdout.rstrip())
        if res.returncode != 0:
            sections.append(f"[exit={res.returncode}]")
        if res.stderr:
            sections.append("--- stderr ---\n" + res.stderr.rstrip())
        if res.artifacts:
            names = ", ".join(sorted(res.artifacts))
            sections.append(f"[artifacts under ./out/: {names}]")
        return _cap("\n".join(sections)) if sections else "(no output — use print())"

    return _run_code


# ---------------------------------------------------------------------------
# run_code — in-process (tool-binding) mode
# ---------------------------------------------------------------------------


def _make_inprocess_run_code(
    host: ToolHost, module_name: str, timeout_s: float
) -> Callable[[str], Any]:
    def _limited_import(
        name: str,
        globals: Mapping[str, Any] | None = None,  # noqa: A002 — __import__ signature
        locals: Mapping[str, Any] | None = None,  # noqa: A002
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        if level == 0 and name in _ALLOWED_MODULES:
            return _ALLOWED_MODULES[name]
        raise ImportError(
            f"import of {name!r} is not allowed in run_code; "
            f"available modules: {', '.join(sorted(_ALLOWED_MODULES))}"
        )

    def _make_shim(tool_name: str) -> Callable[..., Any]:
        async def _shim(**kwargs: Any) -> Any:
            result = await host.call(tool_name, kwargs, call_id=f"code_mode:{tool_name}")
            if not result.ok:
                raise RuntimeError(
                    f"tool {tool_name!r} failed: {result.error or result.reason or 'denied'}"
                )
            return result.output

        _shim.__name__ = tool_name
        _shim.__qualname__ = tool_name
        return _shim

    async def _run_code(code: str) -> str:
        try:
            code_obj = compile(
                code, "<run_code>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
            )
        except SyntaxError as exc:
            return f"ERROR: SyntaxError: {exc}"

        buf = io.StringIO()

        def _print(*args: Any, **kwargs: Any) -> None:
            kwargs.pop("file", None)  # always capture; no host streams
            print(*args, file=buf, **kwargs)

        safe_builtins: dict[str, Any] = {
            n: getattr(builtins, n) for n in _SAFE_BUILTIN_NAMES if hasattr(builtins, n)
        }
        safe_builtins["__import__"] = _limited_import
        safe_builtins["print"] = _print

        namespace: dict[str, Any] = {
            "__builtins__": safe_builtins,
            "__name__": "<run_code>",
            **_ALLOWED_MODULES,
        }
        defs = await host.list_tools()
        shims = {d.name: _make_shim(d.name) for d in defs}
        namespace[module_name] = types.SimpleNamespace(**shims)
        for tool_name, shim in shims.items():
            if tool_name.isidentifier():
                namespace[tool_name] = shim

        unset = object()
        try:
            with anyio.fail_after(timeout_s):
                # exec-mode code object; with PyCF_ALLOW_TOP_LEVEL_AWAIT
                # eval() returns a coroutine when the code awaits.
                maybe_coro = eval(code_obj, namespace)  # noqa: S307
                if inspect.iscoroutine(maybe_coro):
                    await maybe_coro
        except TimeoutError:
            return (
                f"ERROR: run_code timed out after {timeout_s:.1f}s "
                "(note: only code with await points can be interrupted)"
            )
        except Exception as exc:  # noqa: BLE001 — surface to the model as text
            printed = buf.getvalue()
            msg = f"ERROR: {type(exc).__name__}: {exc}"
            if printed:
                msg += "\n--- printed before error ---\n" + printed.rstrip()
            return _cap(msg)

        value = namespace.get("result", unset)
        if value is not unset:
            text = value if isinstance(value, str) else repr(value)
        else:
            text = buf.getvalue().rstrip()
        if not text:
            return "(no output — assign to `result` or use print())"
        return _cap(text)

    return _run_code


def _cap(text: str) -> str:
    if len(text) <= RESULT_CAP:
        return text
    return text[:RESULT_CAP] + f"\n... (truncated, {len(text) - RESULT_CAP} more chars)"


__all__ = [
    "RESULT_CAP",
    "make_code_mode_tools",
]
