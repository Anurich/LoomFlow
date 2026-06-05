"""Lazy tool loading — send a compact catalog, expand schemas on demand.

A large tool roster is expensive: every request ships every tool's full
JSON schema in the ``tools`` field. :class:`LazyToolHost` cuts that by
mirroring the skill progressive-disclosure pattern
(:func:`loomflow.skills.tools.make_load_skill_tool`):

* The model sees only the **eager** tools plus a single ``expand_tool``
  meta-tool whose schema is *static* (its ``name`` arg is a fixed
  ``enum`` of every lazy tool name).
* A compact **catalog** (name + one-line description per lazy tool)
  lives in the system prompt — i.e. the cacheable prefix — at a cost of
  ~tens of tokens per tool instead of a full schema each.
* When the model wants a tool's full schema it calls
  ``expand_tool(name)``; the schema comes back as a tool *result*, not
  as a new entry in the ``tools`` array.

**Why the schema comes back as a result, not a new tool def:** the
prompt cache keys on the tool array. Anthropic places a cache breakpoint
on the last tool definition; OpenAI prefix-caches the serialized tools.
If lazy loading *appended* the expanded tool to the array, the array
would change between turns and invalidate the cache — the exact failure
mode the research warns against. Keeping ``list_tools()`` byte-stable
across turns preserves the breakpoint. The real tool still executes via
:meth:`call` regardless of whether it was expanded first; expansion is
advisory schema disclosure, not a gate.

v1 supports an :class:`~loomflow.tools.registry.InProcessToolHost` base
only (same capability constraint as ``Agent.add_tool`` /
``remove_tool``). Wrapping a non-InProcess host raises ``ConfigError``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

from ..core.errors import ConfigError
from ..core.types import ToolDef, ToolEvent, ToolResult
from .registry import InProcessToolHost, Tool

# Sentinel returned by the resolver to mean "lazy loading disabled".
_DISABLED = object()


class LazyToolHost:
    """Wrap an ``InProcessToolHost`` so only eager tools + ``expand_tool``
    are exposed; the rest are disclosed via a cached catalog and expanded
    on demand. Implements the ``ToolHost`` protocol."""

    def __init__(
        self,
        base: InProcessToolHost,
        *,
        eager: set[str] | None = None,
        meta_tool_name: str = "expand_tool",
    ) -> None:
        if not isinstance(base, InProcessToolHost):
            raise ConfigError(
                "lazy tools (v1) require an InProcessToolHost base; got "
                f"{type(base).__name__}. Pass tools as a list / callables, "
                "or disable lazy_tools for MCP-backed hosts."
            )
        self._base = base
        # Snapshot the catalog once, at construction, so the exposed
        # tool list is deterministic for the life of the host (cache
        # stability depends on list_tools() never varying per turn).
        all_names = list(base._tools.keys())
        if meta_tool_name in all_names:
            raise ConfigError(
                f"meta_tool_name {meta_tool_name!r} collides with an "
                "existing tool; pass a different meta_tool_name."
            )
        eager = eager or set()
        unknown = eager - set(all_names)
        if unknown:
            raise ConfigError(
                "lazy_tools eager set names unknown tool(s): "
                + ", ".join(sorted(unknown))
            )
        self._meta_tool_name = meta_tool_name
        self._eager = {n for n in all_names if n in eager}
        self._lazy = [n for n in all_names if n not in eager]
        self._meta_def = self._build_meta_def()

    # -- catalog --------------------------------------------------------

    def _build_meta_def(self) -> ToolDef:
        names = self._lazy
        catalog = self.catalog_section()
        description = (
            "Return the full input schema for a tool you want to call. "
            "The tool catalog below lists every available tool with a "
            "one-line summary; call expand_tool(name) to get the full "
            "argument schema before invoking that tool. You can also call "
            "a catalogued tool directly once you know its arguments."
        )
        if catalog:
            description += f"\n\n{catalog}"
        return ToolDef(
            name=self._meta_tool_name,
            description=description,
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": names,
                        "description": (
                            "The tool to expand. One of: "
                            f"{', '.join(names) or '(none)'}."
                        ),
                    }
                },
                "required": ["name"],
            },
            destructive=False,
        )

    def catalog_section(self) -> str:
        """Markdown bullet list of the lazy tools — meant for the system
        prompt (the cached prefix). One line per tool: name + the first
        line of its description."""
        if not self._lazy:
            return ""
        lines = ["Available tools (call expand_tool to see arguments):"]
        for name in self._lazy:
            t = self._base.get(name)
            desc = (t.description if t else "").strip().split("\n")[0]
            lines.append(f"- `{name}` — {desc}" if desc else f"- `{name}`")
        return "\n".join(lines)

    # -- ToolHost protocol ----------------------------------------------

    async def list_tools(self, *, query: str | None = None) -> list[ToolDef]:
        """Eager tools + the ``expand_tool`` meta-tool. Deterministic and
        stable across turns so the prompt-cache tool breakpoint holds."""
        defs: list[ToolDef] = []
        for name in self._base._tools:
            if name in self._eager:
                t = self._base.get(name)
                if t is not None:
                    defs.append(t.to_def())
        defs.append(self._meta_def)
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
        if tool == self._meta_tool_name:
            return self._expand(args, call_id=call_id)
        # Real tools dispatch to the base unchanged — permission checks
        # already ran in the architecture loop before this call.
        return await self._base.call(tool, args, call_id=call_id)

    def _expand(self, args: Mapping[str, Any], *, call_id: str) -> ToolResult:
        name = args.get("name", "")
        t = self._base.get(name)
        if t is None:
            valid = ", ".join(self._lazy) or "(none)"
            return ToolResult.error_(
                call_id=call_id,
                message=f"unknown tool: {name!r}. Available: {valid}",
            )
        import json

        schema = json.dumps(t.input_schema, indent=2)
        body = (
            f"Tool `{t.name}` — {t.description}\n\n"
            f"Arguments (JSON schema):\n{schema}\n\n"
            f"Call `{t.name}` directly with these arguments."
        )
        return ToolResult.success(call_id=call_id, output=body)

    async def watch(self) -> AsyncIterator[ToolEvent]:
        async for ev in self._base.watch():
            yield ev

    # Pass through register/get so skill / workspace wiring still works
    # when LazyToolHost is the outermost wrapper.
    def register(self, item: Tool | Any) -> Tool:
        return self._base.register(item)

    def get(self, name: str) -> Tool | None:
        return self._base.get(name)


def resolve_lazy_tools(spec: Any) -> tuple[set[str] | None, str]:
    """Normalise the ``Tuning.lazy_tools`` spec.

    Returns ``(eager, meta_tool_name)`` where ``eager`` is the set of
    tool names to keep eager (``None`` / empty means *all* lazy), or
    raises ``ConfigError`` on a malformed spec. Raises is caught by the
    sentinel check in the Agent: ``False`` / ``None`` never reach here.

    Accepted shapes:
    * ``True``            → all tools lazy, default meta-tool name.
    * ``list[str]``       → those names eager, rest lazy.
    * ``dict``            → ``{"eager": [...], "meta_tool_name": "..."}``.
    """
    meta = "expand_tool"
    if spec is True:
        return set(), meta
    if isinstance(spec, list):
        if not all(isinstance(x, str) for x in spec):
            raise ConfigError("lazy_tools list must contain tool-name strings")
        return set(spec), meta
    if isinstance(spec, dict):
        unknown = set(spec) - {"eager", "meta_tool_name"}
        if unknown:
            raise ConfigError(
                "lazy_tools dict has unknown key(s): " + ", ".join(sorted(unknown))
            )
        eager = spec.get("eager", [])
        if not isinstance(eager, list) or not all(isinstance(x, str) for x in eager):
            raise ConfigError("lazy_tools 'eager' must be a list of tool-name strings")
        meta = spec.get("meta_tool_name", meta)
        if not isinstance(meta, str) or not meta:
            raise ConfigError("lazy_tools 'meta_tool_name' must be a non-empty string")
        return set(eager), meta
    raise ConfigError(
        f"lazy_tools must be bool | list[str] | dict; got {type(spec).__name__}"
    )
