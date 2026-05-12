"""The :class:`Skill` class — one loadable agent skill.

A Skill is a directory on disk containing a ``SKILL.md`` file plus
optional supporting resources. The ``SKILL.md`` has YAML frontmatter
(metadata that loads at startup) and a markdown body (loaded only
when the skill is triggered).

Three flavours of "tools" a skill can ship — none required, freely
mixable in one skill:

* **Mode A** (markdown only): the body teaches the model how to use
  the agent's existing built-in tools (``read``, ``write``, ``bash``,
  etc.). No tool manifest, no Python imports. Pure instructions.
* **Mode C** (frontmatter manifest → subprocess Tool): SKILL.md's
  ``tools:`` block declares a script as a typed tool. At skill load
  the framework wraps the script in a Tool that executes via
  subprocess and returns stdout. Works for ANY language — Python,
  bash, Node, Go.
* **Mode B** (``tools.py`` auto-discovery): if a ``tools.py`` file
  sits in the skill folder, it's imported at construction. Any
  callable decorated with ``@tool`` becomes a registered Tool when
  the skill is loaded. In-process, Python-only.

Every Tool ships from a skill is **prefixed with the skill name**
(``web_research__fetch`` rather than ``fetch``) so multiple skills
loaded simultaneously don't collide.
"""

from __future__ import annotations

import asyncio
import importlib.util
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..tools.registry import Tool
from ._frontmatter import FrontmatterError, parse_frontmatter

_NAME_RE = re.compile(r"^[a-z0-9-]+$")
_RESERVED_WORDS = ("anthropic", "claude")
_MAX_NAME_LEN = 64
_MAX_DESC_LEN = 1024


class SkillError(ValueError):
    """Raised on invalid skill construction or frontmatter."""


@dataclass
class SkillMetadata:
    """Lightweight skill descriptor — what loads at startup.

    The body is NOT in here; it's read on demand via
    :meth:`Skill.load_body`. Keep this small — it lives in the
    system prompt for the entire agent's lifetime."""

    name: str
    description: str
    license: str | None = None
    compatibility: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    allowed_tools: list[str] | None = None
    requires: list[str] = field(default_factory=list)
    """Names of other skills this one depends on. Loading this skill
    also loads every transitive requirement (dependencies first,
    then this skill) and materialises their tools. Cycles raise
    :class:`SkillError`."""
    source_label: str | None = None
    has_python_tools: bool = False    # tools.py was found
    declared_tool_count: int = 0      # # of frontmatter `tools:` entries

    def to_catalog_line(self) -> str:
        """One-line catalog entry for the system prompt."""
        n_tools = (
            (1 if self.has_python_tools else 0) + self.declared_tool_count
        )
        suffix = f" [+{n_tools} tools]" if n_tools else ""
        if self.source_label:
            return (
                f"  - {self.name}{suffix} "
                f"[{self.source_label}]: {self.description}"
            )
        return f"  - {self.name}{suffix}: {self.description}"


@dataclass(frozen=True)
class ToolSpec:
    """One subprocess-tool declaration parsed from frontmatter.

    Mode C — the user declared this script as a tool. The skill load
    flow turns it into a real ``Tool`` whose ``fn`` execs the script
    via subprocess and returns stdout."""

    name: str
    description: str
    script: str  # path relative to skill folder
    args: dict[str, dict[str, Any]] = field(default_factory=dict)


class Skill:
    """A loadable agent skill."""

    def __init__(
        self,
        path: str | Path,
        *,
        source_label: str | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if self.path.is_file():
            skill_md = self.path
            self.path = self.path.parent
        else:
            skill_md = self.path / "SKILL.md"
        if not skill_md.exists():
            raise SkillError(
                f"No SKILL.md found at {skill_md}. A skill is a "
                "directory containing SKILL.md plus optional "
                "supporting files (REFERENCE.md, scripts/, etc.)."
            )
        text = skill_md.read_text()
        meta, body, tool_specs = _parse_skill(
            text, source_label=source_label
        )
        self.metadata = meta
        self._body = body
        self._tool_specs = tool_specs
        self._skill_md_path = skill_md
        # Pending tools — Mode C subprocess wrappers are built
        # eagerly because they don't execute user code (just spawn
        # subprocesses at call time, which is fine in any context).
        self._pending_tools: list[Tool] = []
        self._pending_tools.extend(
            _build_subprocess_tools(tool_specs, self.path, self.name)
        )
        # Mode B (Python ``tools.py``) is detected here but NOT
        # imported — the import is deferred to first use via
        # :meth:`materialize_tools` below. Two reasons:
        #
        # 1. ``tools.py`` may run module-level setup code that needs
        #    an event loop (e.g. ``asyncio.run(setup())``). Importing
        #    at ``Skill(...)`` construction inside Jupyter — which
        #    has its own running loop — fails with ``RuntimeError:
        #    asyncio.run() cannot be called from a running event
        #    loop``. Deferring import to load_skill time means we
        #    import only when the agent loop is already executing.
        #
        # 2. Aligns with progressive disclosure: a skill costs
        #    nothing in the catalog until the model decides to load
        #    it. If we never load it, we never pay the cost of
        #    parsing / importing its tools.
        self._tools_py_path: Path | None = None
        # Cache: filled on first materialize_tools() call.
        self._materialized_python_tools: list[Tool] | None = None
        if str(self.path) != "<inline>":
            tools_py = self.path / "tools.py"
            if tools_py.exists():
                self._tools_py_path = tools_py
                # Set the catalog-line flag based on file presence;
                # we accept that a tools.py with zero @tool defs
                # would lie about it, but that's a degenerate case
                # and avoids parsing AST at construction.
                self.metadata.has_python_tools = True

    @classmethod
    def from_text(
        cls, text: str, *, source_label: str | None = None
    ) -> Skill:
        """Build an inline skill from a SKILL.md-formatted string.

        No filesystem path; bundled scripts and ``tools.py`` aren't
        accessible. Useful for one-off skill definitions in code."""
        instance = cls.__new__(cls)
        instance.path = Path("<inline>")
        instance._skill_md_path = Path("<inline>")
        meta, body, tool_specs = _parse_skill(
            text, source_label=source_label
        )
        # Inline skills can't reference scripts on disk; reject any
        # `tools:` manifest entry that would dangle.
        if tool_specs:
            raise SkillError(
                "Inline skills (Skill.from_text) cannot declare "
                "subprocess tools — they have no filesystem path "
                "to reference scripts from. Put the skill on disk."
            )
        instance.metadata = meta
        instance._body = body
        instance._tool_specs = []
        instance._pending_tools = []
        # Inline skills have no on-disk ``tools.py`` to import, so
        # the lazy-import slots stay empty.
        instance._tools_py_path = None
        instance._materialized_python_tools = []
        return instance

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def description(self) -> str:
        return self.metadata.description

    @property
    def pending_tools(self) -> list[Tool]:
        """The Tool instances this skill will register on load.

        For Mode C (subprocess wrappers from frontmatter ``tools:``
        manifest): always returned eagerly; they don't execute user
        code at construction.

        For Mode B (Python ``@tool`` from ``tools.py``): only
        returned if :meth:`materialize_tools` has already been
        called. Otherwise the import is still pending and this
        property returns an empty Mode B portion. Use
        :meth:`materialize_tools` to force the import (with optional
        :class:`~loomflow.RunContext` for the
        ``build_tools(ctx)`` factory protocol).
        """
        out = list(self._pending_tools)
        if self._materialized_python_tools is not None:
            out.extend(self._materialized_python_tools)
        return out

    def materialize_tools(
        self, ctx: Any | None = None
    ) -> list[Tool]:
        """Resolve all :class:`Tool` instances this skill ships,
        importing ``tools.py`` lazily on first call.

        ``ctx`` (a :class:`~loomflow.RunContext` or any object with
        ``metadata`` / ``user_id`` / ``session_id`` attributes) is
        forwarded to the skill's ``build_tools(ctx)`` factory if
        defined. The factory pattern lets a skill's tools close over
        caller-supplied state — a vectorstore, a DB connection, an
        API client — without globals or module-level setup:

        .. code-block:: python

            # skills/pdf-retrieval/tools.py
            from loomflow import tool

            def build_tools(ctx):
                vs = ctx.metadata["vectorstore"]
                @tool
                async def retreiver(query: str) -> list:
                    return await vs.search_hybrid(query=query)
                return [retreiver]

        When ``tools.py`` does *not* export ``build_tools``, the
        framework falls back to discovering module-level
        ``@tool``-decorated globals (back-compat behaviour).

        Idempotent: subsequent calls reuse the cached result. Pass
        a different ``ctx`` and it's ignored — re-import the skill
        registry if you need a fresh build.
        """
        if self._materialized_python_tools is None and self._tools_py_path is not None:
            self._materialized_python_tools = _import_python_tools(
                self._tools_py_path, self.name, ctx
            )
        if self._materialized_python_tools is None:
            self._materialized_python_tools = []
        return list(self._pending_tools) + list(
            self._materialized_python_tools
        )

    def load_body(self) -> str:
        """Return the full SKILL.md body (without frontmatter)."""
        return self._body

    def list_files(self) -> list[Path]:
        """Enumerate every file bundled with this skill."""
        if str(self.path) == "<inline>":
            return []
        return sorted(p for p in self.path.rglob("*") if p.is_file())

    def __repr__(self) -> str:
        return (
            f"Skill(name={self.name!r}, "
            f"path={self.path}, "
            f"label={self.metadata.source_label!r}, "
            f"pending_tools={len(self._pending_tools)})"
        )


# ---------------------------------------------------------------------------
# Parsing — frontmatter → SkillMetadata + ToolSpec list + body
# ---------------------------------------------------------------------------


def _parse_skill(
    text: str, *, source_label: str | None
) -> tuple[SkillMetadata, str, list[ToolSpec]]:
    try:
        meta, body = parse_frontmatter(text)
    except FrontmatterError as exc:
        raise SkillError(f"Bad SKILL.md frontmatter: {exc}") from exc

    name = meta.get("name")
    if not isinstance(name, str) or not name:
        raise SkillError(
            "SKILL.md frontmatter must include a non-empty 'name' string."
        )
    if len(name) > _MAX_NAME_LEN:
        raise SkillError(
            f"Skill name {name!r} exceeds {_MAX_NAME_LEN} chars."
        )
    if not _NAME_RE.fullmatch(name):
        raise SkillError(
            f"Skill name {name!r} must match {_NAME_RE.pattern} "
            "(lowercase letters, digits, hyphens)."
        )
    lower = name.lower()
    for reserved in _RESERVED_WORDS:
        if reserved in lower:
            raise SkillError(
                f"Skill name {name!r} contains reserved word "
                f"{reserved!r}."
            )

    description = meta.get("description")
    if not isinstance(description, str) or not description.strip():
        raise SkillError(
            "SKILL.md frontmatter must include a non-empty "
            "'description' string."
        )
    if len(description) > _MAX_DESC_LEN:
        raise SkillError(
            f"Skill description exceeds {_MAX_DESC_LEN} chars "
            f"(got {len(description)})."
        )

    allowed_tools = meta.get("allowed_tools")
    if allowed_tools is not None and (
        not isinstance(allowed_tools, list)
        or not all(isinstance(t, str) for t in allowed_tools)
    ):
        raise SkillError("'allowed_tools' must be a list of strings.")

    requires = meta.get("requires") or []
    if not isinstance(requires, list) or not all(
        isinstance(r, str) for r in requires
    ):
        raise SkillError("'requires' must be a list of skill name strings.")
    # Reject self-reference and duplicates here so SkillRegistry's
    # cycle-detector doesn't have to special-case the trivial case.
    if name in requires:
        raise SkillError(
            f"Skill {name!r} cannot list itself in 'requires'."
        )
    seen: set[str] = set()
    deduped: list[str] = []
    for r in requires:
        if r and r not in seen:
            seen.add(r)
            deduped.append(r)

    extra = meta.get("metadata") or {}
    if extra and not isinstance(extra, dict):
        raise SkillError("'metadata' must be a mapping if provided.")

    tool_specs = _parse_tool_manifest(meta.get("tools"))

    metadata = SkillMetadata(
        name=name,
        description=description.strip(),
        license=_optional_str(meta, "license"),
        compatibility=_optional_str(meta, "compatibility"),
        extra=dict(extra),
        allowed_tools=list(allowed_tools) if allowed_tools else None,
        requires=deduped,
        source_label=source_label,
        declared_tool_count=len(tool_specs),
    )
    return metadata, body.strip(), tool_specs


def _parse_tool_manifest(raw: Any) -> list[ToolSpec]:
    """Parse the `tools:` block in frontmatter.

    Expected shape::

        tools:
          tool_name:
            description: What it does.
            script: scripts/foo.py
            args:
              arg_name:
                type: string
                description: ...
    """
    if raw is None:
        return []
    if not isinstance(raw, dict):
        raise SkillError(
            "Frontmatter 'tools:' must be a mapping of "
            "tool-name → spec dict."
        )
    specs: list[ToolSpec] = []
    for tool_name, spec in raw.items():
        if not isinstance(tool_name, str) or not _NAME_RE.fullmatch(
            tool_name.replace("_", "-")
        ):
            # Allow underscores in tool names (Python-style); the
            # name regex allows lowercase + hyphens — accept either
            # by normalizing.
            if not re.fullmatch(r"[a-z0-9_-]+", tool_name):
                raise SkillError(
                    f"Tool name {tool_name!r} must contain only "
                    "lowercase letters, digits, hyphens, or "
                    "underscores."
                )
        if not isinstance(spec, dict):
            raise SkillError(
                f"Tool {tool_name!r} spec must be a mapping."
            )
        description = spec.get("description", "")
        script = spec.get("script")
        if not isinstance(script, str) or not script:
            raise SkillError(
                f"Tool {tool_name!r} must declare a 'script' path."
            )
        args = spec.get("args") or {}
        if not isinstance(args, dict):
            raise SkillError(
                f"Tool {tool_name!r}: 'args' must be a mapping."
            )
        # Each arg's spec is itself a dict: {type, description, ...}.
        validated_args: dict[str, dict[str, Any]] = {}
        for arg_name, arg_spec in args.items():
            if not isinstance(arg_spec, dict):
                # Allow shorthand: arg_name: string (no description)
                if isinstance(arg_spec, str):
                    arg_spec = {"type": arg_spec}
                else:
                    raise SkillError(
                        f"Tool {tool_name!r}: arg {arg_name!r} "
                        "must be a string or mapping."
                    )
            validated_args[arg_name] = dict(arg_spec)
        specs.append(
            ToolSpec(
                name=tool_name,
                description=str(description),
                script=script,
                args=validated_args,
            )
        )
    return specs


def _optional_str(meta: dict[str, Any], key: str) -> str | None:
    val = meta.get(key)
    if val is None:
        return None
    if not isinstance(val, str):
        raise SkillError(f"'{key}' must be a string if provided.")
    return val


# ---------------------------------------------------------------------------
# Mode C — wrap ToolSpec entries in subprocess Tool objects
# ---------------------------------------------------------------------------


def _normalize_skill_name(name: str) -> str:
    """Skill name (with hyphens) → safe tool-name prefix.

    ``web-research`` → ``web_research`` so the prefixed tool name
    ``web_research__fetch`` is a valid identifier."""
    return name.replace("-", "_")


def _build_subprocess_tools(
    specs: list[ToolSpec],
    skill_path: Path,
    skill_name: str,
) -> list[Tool]:
    """Build one Tool per ToolSpec, wrapping its script in a
    subprocess invocation.

    The Tool's ``fn`` is a closure that knows the skill's path and
    the spec's args. When invoked it builds an argv list (positional,
    in declaration order), execs the script, captures stdout, and
    returns the captured text. Stderr is folded into stdout so
    failures surface in the model's tool result."""
    prefix = f"{_normalize_skill_name(skill_name)}__"
    return [
        _make_subprocess_tool(spec, skill_path, prefix) for spec in specs
    ]


def _make_subprocess_tool(
    spec: ToolSpec, skill_path: Path, prefix: str
) -> Tool:
    script_full = (skill_path / spec.script).resolve()
    interpreter = _interpreter_for(script_full)
    arg_order = list(spec.args.keys())

    async def _run(**kwargs: Any) -> str:
        # Convert kwargs to positional argv in declaration order.
        argv = [str(kwargs.get(arg_name, "")) for arg_name in arg_order]
        cmd = [*interpreter, str(script_full), *argv]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except FileNotFoundError as exc:
            return f"Error: failed to launch {shlex.join(cmd)}: {exc}"
        stdout_bytes, _ = await proc.communicate()
        out = stdout_bytes.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            return f"Error (exit={proc.returncode}):\n{out}"
        return out

    return Tool(
        name=f"{prefix}{spec.name}",
        description=(
            spec.description
            or f"Run the bundled script {spec.script}."
        ),
        fn=_run,
        input_schema={
            "type": "object",
            "properties": {
                arg_name: {
                    k: v
                    for k, v in arg_spec.items()
                    if k in {"type", "description", "enum"}
                }
                for arg_name, arg_spec in spec.args.items()
            },
            "required": list(spec.args.keys()),
        },
    )


def _interpreter_for(script: Path) -> list[str]:
    """Pick the interpreter to run a script. Recognised by suffix:
    ``.py`` → current Python; ``.sh`` → bash; otherwise assume the
    file is directly executable (shebang line / native binary)."""
    suffix = script.suffix.lower()
    if suffix == ".py":
        return [sys.executable]
    if suffix in {".sh", ".bash"}:
        return ["bash"]
    if suffix in {".js", ".mjs"}:
        return ["node"]
    return []  # rely on the script's shebang or executable bit


# ---------------------------------------------------------------------------
# Mode B — auto-discover @tool functions in tools.py
# ---------------------------------------------------------------------------


def _import_python_tools(
    tools_py: Path,
    skill_name: str,
    ctx: Any | None = None,
) -> list[Tool]:
    """Import a skill's ``tools.py`` and collect its :class:`Tool`
    instances.

    Two ways to expose tools, in priority order:

    1. **Factory protocol** — ``tools.py`` exports a callable
       ``build_tools(ctx)`` that returns a ``list[Tool]``. Called
       with ``ctx`` (typically a :class:`RunContext`) so tools can
       close over caller-supplied state. This is the recommended
       pattern for tools that need a vectorstore / DB / API
       client — no globals, no module-level setup, no event-loop
       collisions.

    2. **Module-level discovery** — when no ``build_tools`` is
       exported, scan module attributes for :class:`Tool` instances
       (typically the result of ``@tool``-decorating a function at
       module level). Back-compatible with skills written before
       the factory protocol existed.

    Tool names are prefixed with ``{skill_name}__`` so multiple
    skills exposing the same tool name don't clash when registered
    with one agent.

    Errors during import surface as :class:`SkillError` with a
    contextual message — including a hint when the failure looks
    like an event-loop collision (``asyncio.run`` at module level),
    which is a frequent first-time mistake.
    """
    if not tools_py.exists():
        return []
    module_name = (
        f"_loomflow_skill_tools__{_normalize_skill_name(skill_name)}"
    )
    module_spec = importlib.util.spec_from_file_location(
        module_name, tools_py
    )
    if module_spec is None or module_spec.loader is None:
        raise SkillError(
            f"Could not load skill module at {tools_py}"
        )
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    try:
        module_spec.loader.exec_module(module)
    except RuntimeError as exc:
        # Surface the most common gotcha — module-level work that
        # needs an event loop — with a pointer to the fix instead
        # of the raw asyncio traceback.
        if "event loop" in str(exc):
            raise SkillError(
                f"Error importing {tools_py}: {exc}\n"
                f"\n"
                f"Hint: ``tools.py`` runs when the skill is loaded "
                f"by the agent (which already has a running event "
                f"loop). Don't call ``asyncio.run(...)`` or do "
                f"other event-loop work at module level. Either:\n"
                f"  • Move the setup outside the skill folder "
                f"(into your main script).\n"
                f"  • Use a ``build_tools(ctx)`` factory in "
                f"tools.py and read setup state from "
                f"``ctx.metadata``."
            ) from exc
        raise SkillError(
            f"Error importing {tools_py}: {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 — surface ANY import error
        raise SkillError(
            f"Error importing {tools_py}: {exc}"
        ) from exc

    prefix = f"{_normalize_skill_name(skill_name)}__"

    # 1. Factory protocol — preferred when state is needed.
    factory = getattr(module, "build_tools", None)
    if callable(factory):
        try:
            built = factory(ctx)
        except Exception as exc:  # noqa: BLE001
            raise SkillError(
                f"Error calling build_tools(ctx) in {tools_py}: "
                f"{exc}"
            ) from exc
        if not isinstance(built, list) or not all(
            isinstance(t, Tool) for t in built
        ):
            raise SkillError(
                f"build_tools(ctx) in {tools_py} must return a "
                f"list[Tool]; got {type(built).__name__}."
            )
        return [
            Tool(
                name=f"{prefix}{t.name}",
                description=t.description,
                fn=t.fn,
                input_schema=t.input_schema,
            )
            for t in built
        ]

    # 2. Back-compat — module-level @tool discovery.
    tools: list[Tool] = []
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        obj = getattr(module, attr_name)
        if isinstance(obj, Tool):
            tools.append(
                Tool(
                    name=f"{prefix}{obj.name}",
                    description=obj.description,
                    fn=obj.fn,
                    input_schema=obj.input_schema,
                )
            )
    return tools
