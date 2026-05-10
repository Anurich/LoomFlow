"""Tool factories that surface skills to the agent's model.

We inject ONE tool — ``load_skill(name)`` — into the agent's tool
host whenever a non-empty :class:`SkillRegistry` is configured. The
tool's input schema enumerates the registered skill names as an
``enum`` so:

* Strict-schema providers (Anthropic / OpenAI strict mode) reject
  hallucinated skill names at the API boundary
* The model sees every available skill name in the schema docs
* Typos return a tool error with the valid set listed

The tool's *description* also lists every skill with its short
description, giving the model the full catalog at metadata cost
without loading any bodies.

When a skill ships pending Tools (Mode B from ``tools.py`` or
Mode C from frontmatter ``tools:`` manifest), ``load_skill`` ALSO
registers those Tools with the agent's tool host on the first call.
The model sees the new tools in its toolset on the next turn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..tools.registry import InProcessToolHost, Tool
from .registry import SkillRegistry
from .skill import SkillError

if TYPE_CHECKING:
    from ..core.protocols import ToolHost


def make_load_skill_tool(
    registry: SkillRegistry,
    *,
    host: ToolHost | None = None,
    tool_name: str = "load_skill",
) -> Tool:
    """Build the ``load_skill`` tool for a given registry.

    When ``host`` is provided, the tool will register a skill's
    pending Tools (from Mode B / Mode C) with the host on first
    load — making them callable on subsequent turns. Without a
    host, ``load_skill`` only returns the body (skill brings no
    tools, or the framework integration handles registration
    elsewhere).
    """
    skill_names = registry.names()
    catalog_lines = "\n".join(
        s.metadata.to_catalog_line() for s in registry
    )
    description = (
        "Load the full instructions for a packaged skill. Call this "
        "ONCE per task when the user's request matches one of the "
        "available skills' descriptions. The tool returns the "
        "skill's full markdown body — follow its instructions step "
        "by step using the standard tools (read / write / bash / "
        "etc.). When a skill brings its own tools (marked '+N "
        "tools' in the catalog below), those tools also become "
        "callable on subsequent turns."
    )
    if catalog_lines:
        description += f"\n\nAvailable skills:\n{catalog_lines}"

    async def _load(name: str) -> str:
        # Pull the live :class:`RunContext` so a skill's
        # ``build_tools(ctx)`` factory can read ``ctx.metadata`` /
        # ``ctx.user_id`` and close over caller-supplied state
        # (a vectorstore, DB connection, API client). Cheap when
        # the skill has no factory — the ctx is just ignored.
        from ..core.context import get_run_context

        ctx = get_run_context()
        try:
            body, pending = registry.load_with_tools(name, ctx=ctx)
        except SkillError as exc:
            return f"Error: {exc}"

        # Register any skill-shipped Tools so the model can use
        # them on the next turn. Only fires once per skill —
        # load_with_tools is idempotent.
        if pending and host is not None:
            for tool in pending:
                if isinstance(host, InProcessToolHost):
                    host.register(tool)
                elif hasattr(host, "register"):
                    host.register(tool)  # type: ignore[attr-defined]
                # Hosts without register() (e.g. an immutable MCP
                # adapter) silently skip; we already validated the
                # registration path during Agent construction by
                # using ExtendedToolHost when needed.

        if pending:
            tool_list = ", ".join(t.name for t in pending)
            footer = (
                f"\n\n---\n_{len(pending)} tool(s) now available: "
                f"{tool_list}_"
            )
            return body + footer
        return body

    return Tool(
        name=tool_name,
        description=description,
        fn=_load,
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": skill_names,
                    "description": (
                        "The skill name to load. Must be one of: "
                        f"{', '.join(skill_names) or '(no skills registered)'}."
                    ),
                }
            },
            "required": ["name"],
        },
    )
