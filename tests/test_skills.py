"""Tests for the skills module.

Covers:

* The minimal YAML frontmatter parser (strings, lists, nested
  dicts, multi-line literals, edge cases).
* :class:`Skill` construction from disk and from inline text;
  frontmatter validation (name regex / length / reserved words /
  description length); listing bundled files.
* :class:`SkillSource` coercion (path / Path / tuple) + recursive
  discovery of multiple skills under one source.
* :class:`SkillRegistry` last-source-wins override semantics + the
  catalog section format injected into system prompts.
* The ``load_skill`` tool: enum on its ``name`` arg, returns the
  full body, returns a clean error for unknown names.
* End-to-end: ``Agent(skills=[...])`` adds the catalog to the
  system prompt and surfaces ``load_skill`` in the tool list.
* ``Team.supervisor(skills=[...])`` forwards skills to the
  coordinator agent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loomflow import Agent, ScriptedModel, ScriptedTurn
from loomflow.skills import Skill, SkillError, SkillRegistry, SkillSource
from loomflow.skills._frontmatter import (
    FrontmatterError,
    parse_frontmatter,
)
from loomflow.skills.tools import make_load_skill_tool
from loomflow.team import Team

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
# Frontmatter parser
# ---------------------------------------------------------------------------


def test_frontmatter_parses_basic_scalars() -> None:
    text = (
        "---\n"
        "name: my-skill\n"
        "description: Does a thing.\n"
        "---\n\n"
        "Body here.\n"
    )
    meta, body = parse_frontmatter(text)
    assert meta == {"name": "my-skill", "description": "Does a thing."}
    assert body.strip() == "Body here."


def test_frontmatter_parses_nested_metadata_dict() -> None:
    text = (
        "---\n"
        "name: x\n"
        "description: y\n"
        "metadata:\n"
        "  author: alice\n"
        "  version: \"1.0\"\n"
        "---\n"
    )
    meta, _ = parse_frontmatter(text)
    assert meta["metadata"] == {"author": "alice", "version": "1.0"}


def test_frontmatter_parses_inline_and_block_lists() -> None:
    inline = (
        "---\n"
        "name: x\n"
        "description: y\n"
        "allowed_tools: [bash, read, write]\n"
        "---\n"
    )
    meta_inline, _ = parse_frontmatter(inline)
    assert meta_inline["allowed_tools"] == ["bash", "read", "write"]

    block = (
        "---\n"
        "name: x\n"
        "description: y\n"
        "allowed_tools:\n"
        "  - bash\n"
        "  - read\n"
        "  - write\n"
        "---\n"
    )
    meta_block, _ = parse_frontmatter(block)
    assert meta_block["allowed_tools"] == ["bash", "read", "write"]


def test_frontmatter_handles_quoted_strings() -> None:
    text = (
        "---\n"
        "name: x\n"
        'description: "A description with: colons and # hashes"\n'
        "---\n"
    )
    meta, _ = parse_frontmatter(text)
    assert meta["description"] == "A description with: colons and # hashes"


def test_frontmatter_handles_booleans_and_nulls() -> None:
    text = (
        "---\n"
        "name: x\n"
        "description: y\n"
        "active: true\n"
        "deprecated: false\n"
        "extra: null\n"
        "---\n"
    )
    meta, _ = parse_frontmatter(text)
    assert meta["active"] is True
    assert meta["deprecated"] is False
    assert meta["extra"] is None


def test_frontmatter_missing_fence_raises() -> None:
    with pytest.raises(FrontmatterError, match="frontmatter"):
        parse_frontmatter("# Just a markdown file with no frontmatter")


def test_frontmatter_unclosed_fence_raises() -> None:
    text = "---\nname: x\ndescription: y\n# no closing fence\n"
    with pytest.raises(FrontmatterError):
        parse_frontmatter(text)


# ---------------------------------------------------------------------------
# Skill class
# ---------------------------------------------------------------------------


def _make_skill_dir(
    tmp_path: Path,
    name: str,
    description: str,
    extra: str = "",
) -> Path:
    """Helper: write a minimal SKILL.md to a fresh subdirectory."""
    folder = tmp_path / name
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n{extra}---\n\n"
        f"# {name}\n\nInstructions for {name}.\n"
    )
    return folder


def test_skill_loads_from_directory(tmp_path: Path) -> None:
    folder = _make_skill_dir(tmp_path, "my-skill", "Does a thing.")
    skill = Skill(folder)
    assert skill.name == "my-skill"
    assert skill.description == "Does a thing."
    assert "Instructions for my-skill" in skill.load_body()


def test_skill_loads_from_skill_md_path_directly(tmp_path: Path) -> None:
    folder = _make_skill_dir(tmp_path, "my-skill", "Does a thing.")
    # Pointing at SKILL.md should also work (parent becomes bundle root).
    skill = Skill(folder / "SKILL.md")
    assert skill.name == "my-skill"
    assert skill.path == folder.resolve()


def test_skill_from_text_inline() -> None:
    skill = Skill.from_text(
        "---\nname: inline\ndescription: An inline skill.\n---\n# Inline\nDo it inline."
    )
    assert skill.name == "inline"
    assert "Do it inline" in skill.load_body()


def test_skill_rejects_missing_skill_md(tmp_path: Path) -> None:
    folder = tmp_path / "no-skill"
    folder.mkdir()
    with pytest.raises(SkillError, match="No SKILL.md"):
        Skill(folder)


def test_skill_rejects_invalid_name(tmp_path: Path) -> None:
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "SKILL.md").write_text(
        "---\nname: BadName_With_Underscores\ndescription: x\n---\n# x"
    )
    with pytest.raises(SkillError, match="must match"):
        Skill(bad)


def test_skill_rejects_reserved_word(tmp_path: Path) -> None:
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "SKILL.md").write_text(
        "---\nname: anthropic-helper\ndescription: x\n---\n# x"
    )
    with pytest.raises(SkillError, match="reserved word"):
        Skill(bad)


def test_skill_rejects_oversized_name(tmp_path: Path) -> None:
    bad = tmp_path / "bad"
    bad.mkdir()
    long_name = "a" * 65
    (bad / "SKILL.md").write_text(
        f"---\nname: {long_name}\ndescription: x\n---\n# x"
    )
    with pytest.raises(SkillError, match="exceeds 64"):
        Skill(bad)


def test_skill_rejects_oversized_description(tmp_path: Path) -> None:
    bad = tmp_path / "bad"
    bad.mkdir()
    long_desc = "x" * 1025
    (bad / "SKILL.md").write_text(
        f"---\nname: ok-name\ndescription: {long_desc}\n---\n# x"
    )
    with pytest.raises(SkillError, match="description exceeds"):
        Skill(bad)


def test_skill_captures_optional_metadata(tmp_path: Path) -> None:
    folder = _make_skill_dir(
        tmp_path,
        "my-skill",
        "x",
        extra=(
            "license: MIT\n"
            "compatibility: requires internet\n"
            "metadata:\n"
            "  author: alice\n"
            "  version: \"2.1\"\n"
            "allowed_tools: [bash, read]\n"
        ),
    )
    skill = Skill(folder)
    assert skill.metadata.license == "MIT"
    assert skill.metadata.compatibility == "requires internet"
    assert skill.metadata.extra == {"author": "alice", "version": "2.1"}
    assert skill.metadata.allowed_tools == ["bash", "read"]


def test_skill_list_files_includes_bundled_resources(
    tmp_path: Path,
) -> None:
    folder = _make_skill_dir(tmp_path, "my-skill", "x")
    (folder / "REFERENCE.md").write_text("# Reference")
    (folder / "scripts").mkdir()
    (folder / "scripts" / "helper.py").write_text("print('hi')")
    skill = Skill(folder)
    files = {p.name for p in skill.list_files()}
    assert "SKILL.md" in files
    assert "REFERENCE.md" in files
    assert "helper.py" in files


# ---------------------------------------------------------------------------
# SkillSource
# ---------------------------------------------------------------------------


def test_skill_source_coerce_accepts_str_and_path(tmp_path: Path) -> None:
    src1 = SkillSource.coerce(str(tmp_path))
    src2 = SkillSource.coerce(tmp_path)
    assert src1.path == tmp_path
    assert src2.path == tmp_path
    assert src1.label is None


def test_skill_source_coerce_accepts_labelled_tuple(
    tmp_path: Path,
) -> None:
    src = SkillSource.coerce((tmp_path, "Project"))
    assert src.path == tmp_path
    assert src.label == "Project"


def test_skill_source_discovers_multiple_skills(
    tmp_path: Path,
) -> None:
    _make_skill_dir(tmp_path, "alpha", "First.")
    _make_skill_dir(tmp_path, "beta", "Second.")
    _make_skill_dir(tmp_path, "gamma", "Third.")
    source = SkillSource(tmp_path, label="Test")
    skills = source.discover()
    names = {s.name for s in skills}
    assert names == {"alpha", "beta", "gamma"}
    assert all(s.metadata.source_label == "Test" for s in skills)


def test_skill_source_rejects_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    with pytest.raises(SkillError, match="does not exist"):
        SkillSource(missing).discover()


# ---------------------------------------------------------------------------
# SkillRegistry — override semantics
# ---------------------------------------------------------------------------


def test_registry_last_source_wins_on_duplicate_name(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    user = tmp_path / "user"
    base.mkdir()
    user.mkdir()
    (base / "shared").mkdir()
    (base / "shared" / "SKILL.md").write_text(
        "---\nname: shared\ndescription: BASE version.\n---\nbase body"
    )
    (user / "shared").mkdir()
    (user / "shared" / "SKILL.md").write_text(
        "---\nname: shared\ndescription: USER override.\n---\nuser body"
    )
    registry = SkillRegistry([base, user])
    assert len(registry) == 1
    skill = registry.get("shared")
    assert skill is not None
    assert skill.description == "USER override."
    assert "user body" in skill.load_body()


def test_registry_catalog_section_lists_all_names(
    tmp_path: Path,
) -> None:
    _make_skill_dir(tmp_path, "alpha", "First skill.")
    _make_skill_dir(tmp_path, "beta", "Second skill.")
    registry = SkillRegistry([tmp_path])
    section = registry.catalog_section()
    assert "## Available skills" in section
    assert "alpha: First skill." in section
    assert "beta: Second skill." in section


def test_registry_catalog_section_empty_when_no_skills() -> None:
    assert SkillRegistry([]).catalog_section() == ""


def test_registry_load_unknown_returns_clear_error() -> None:
    registry = SkillRegistry([])
    with pytest.raises(SkillError, match="Unknown skill"):
        registry.load("does-not-exist")


def test_registry_add_and_remove(tmp_path: Path) -> None:
    folder = _make_skill_dir(tmp_path, "my-skill", "x")
    registry = SkillRegistry([])
    registry.add(Skill(folder))
    assert "my-skill" in registry
    removed = registry.remove("my-skill")
    assert removed is not None
    assert "my-skill" not in registry


# ---------------------------------------------------------------------------
# load_skill tool
# ---------------------------------------------------------------------------


def test_load_skill_tool_enumerates_skill_names(tmp_path: Path) -> None:
    _make_skill_dir(tmp_path, "alpha", "First.")
    _make_skill_dir(tmp_path, "beta", "Second.")
    registry = SkillRegistry([tmp_path])
    tool = make_load_skill_tool(registry)
    name_schema = tool.input_schema["properties"]["name"]
    assert "enum" in name_schema
    assert set(name_schema["enum"]) == {"alpha", "beta"}
    # Tool description includes the catalog.
    assert "alpha" in tool.description
    assert "beta" in tool.description


async def test_load_skill_tool_returns_full_body(tmp_path: Path) -> None:
    _make_skill_dir(tmp_path, "alpha", "First.")
    registry = SkillRegistry([tmp_path])
    tool = make_load_skill_tool(registry)
    body = await tool.fn(name="alpha")
    assert "Instructions for alpha" in body


async def test_load_skill_tool_unknown_returns_error_string(
    tmp_path: Path,
) -> None:
    _make_skill_dir(tmp_path, "alpha", "First.")
    registry = SkillRegistry([tmp_path])
    tool = make_load_skill_tool(registry)
    out = await tool.fn(name="nope")
    assert out.startswith("Error:")
    assert "Unknown skill" in out


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------


async def test_agent_appends_skill_catalog_to_instructions(
    tmp_path: Path,
) -> None:
    _make_skill_dir(tmp_path, "alpha", "First skill.")
    agent = Agent(
        "You are a helper.",
        model=ScriptedModel([ScriptedTurn(text="ok")]),
        skills=[tmp_path],
    )
    assert "## Available skills" in agent.instructions
    assert "alpha: First skill." in agent.instructions


async def test_agent_registers_load_skill_tool(tmp_path: Path) -> None:
    _make_skill_dir(tmp_path, "alpha", "First skill.")
    agent = Agent(
        "test",
        model=ScriptedModel([ScriptedTurn(text="ok")]),
        skills=[tmp_path],
    )
    tools = await agent.tool_host.list_tools()
    tool_names = {t.name for t in tools}
    assert "load_skill" in tool_names


async def test_agent_no_skills_no_load_skill_tool() -> None:
    agent = Agent(
        "test",
        model=ScriptedModel([ScriptedTurn(text="ok")]),
    )
    tools = await agent.tool_host.list_tools()
    assert all(t.name != "load_skill" for t in tools)
    assert agent.skills is None


async def test_agent_skills_property_exposes_registry(
    tmp_path: Path,
) -> None:
    _make_skill_dir(tmp_path, "alpha", "First.")
    agent = Agent(
        "test",
        model=ScriptedModel([ScriptedTurn(text="ok")]),
        skills=[tmp_path],
    )
    assert agent.skills is not None
    assert "alpha" in agent.skills


async def test_agent_with_inline_skill() -> None:
    skill = Skill.from_text(
        "---\nname: standup\ndescription: Format a daily standup.\n---\n# Standup format"
    )
    agent = Agent(
        "test",
        model=ScriptedModel([ScriptedTurn(text="ok")]),
        skills=[skill],
    )
    assert "standup: Format a daily standup." in agent.instructions


# ---------------------------------------------------------------------------
# Team integration — skills forward to the coordinator
# ---------------------------------------------------------------------------


async def test_team_supervisor_accepts_skills(tmp_path: Path) -> None:
    _make_skill_dir(tmp_path, "alpha", "First.")
    worker = Agent(
        "worker", model=ScriptedModel([ScriptedTurn(text="ok")])
    )
    team = Team.supervisor(
        workers={"a": worker},
        instructions="Manage.",
        model=ScriptedModel([ScriptedTurn(text="ok")]),
        skills=[tmp_path],
    )
    # Coordinator agent has the skill catalog.
    assert "alpha: First." in team.instructions
    tool_names = {t.name for t in await team.tool_host.list_tools()}
    assert "load_skill" in tool_names


async def test_team_router_accepts_skills(tmp_path: Path) -> None:
    from loomflow.architecture import RouterRoute

    _make_skill_dir(tmp_path, "alpha", "First.")
    specialist = Agent(
        "specialist", model=ScriptedModel([ScriptedTurn(text="ok")])
    )
    team = Team.router(
        routes=[
            RouterRoute(
                name="r1", description="x", agent=specialist
            )
        ],
        instructions="Triage.",
        model=ScriptedModel([ScriptedTurn(text="ok")]),
        skills=[tmp_path],
    )
    assert "alpha: First." in team.instructions


# ---------------------------------------------------------------------------
# Layered sources — system → user → project override
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Mode C — frontmatter `tools:` manifest → subprocess Tool wrappers
# ---------------------------------------------------------------------------


def _make_mode_c_skill(tmp_path: Path) -> Path:
    """A skill that ships one Python script as a tool via frontmatter."""
    folder = tmp_path / "calc"
    (folder / "scripts").mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        "---\n"
        "name: calc\n"
        "description: Arithmetic helpers.\n"
        "tools:\n"
        "  add:\n"
        "    description: Sum two integers.\n"
        "    script: scripts/add.py\n"
        "    args:\n"
        "      a:\n"
        "        type: string\n"
        "        description: First int\n"
        "      b:\n"
        "        type: string\n"
        "        description: Second int\n"
        "---\n\n"
        "# Calc\nUse `add(a, b)` to sum two integers.\n"
    )
    (folder / "scripts" / "add.py").write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "print(int(sys.argv[1]) + int(sys.argv[2]))\n"
    )
    return folder


async def test_skill_parses_tools_manifest(tmp_path: Path) -> None:
    """The Skill should recognise the manifest entry and stash a
    pending Tool with the correct name + schema."""
    skill = Skill(_make_mode_c_skill(tmp_path))
    assert skill.metadata.declared_tool_count == 1
    pending = skill.pending_tools
    assert len(pending) == 1
    tool = pending[0]
    # Auto-prefixed.
    assert tool.name == "calc__add"
    # Args lifted into the JSON schema.
    schema = tool.input_schema["properties"]
    assert "a" in schema and "b" in schema
    assert schema["a"]["type"] == "string"


async def test_mode_c_subprocess_tool_runs(tmp_path: Path) -> None:
    """Calling the wrapped Tool execs the script via subprocess
    and returns its stdout."""
    skill = Skill(_make_mode_c_skill(tmp_path))
    add = skill.pending_tools[0]
    out = await add.fn(a="2", b="3")
    assert out.strip() == "5"


async def test_mode_c_subprocess_tool_surfaces_failures(
    tmp_path: Path,
) -> None:
    """Non-zero exit → tool result starts with 'Error' so the model
    sees the failure clearly."""
    folder = tmp_path / "boom"
    (folder / "scripts").mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        "---\n"
        "name: boom\n"
        "description: Always crashes.\n"
        "tools:\n"
        "  crash:\n"
        "    description: Crash.\n"
        "    script: scripts/crash.py\n"
        "---\n\n"
        "# Crash\n"
    )
    (folder / "scripts" / "crash.py").write_text(
        "import sys\nprint('oops', file=sys.stderr)\nsys.exit(1)\n"
    )
    skill = Skill(folder)
    out = await skill.pending_tools[0].fn()
    assert out.startswith("Error (exit=1)")
    assert "oops" in out


def test_inline_skill_rejects_tools_manifest() -> None:
    """``Skill.from_text`` can't reference scripts on disk, so a
    manifest entry must fail loudly."""
    with pytest.raises(SkillError, match="Inline skills"):
        Skill.from_text(
            "---\nname: x\ndescription: y\n"
            "tools:\n"
            "  foo:\n"
            "    description: z\n"
            "    script: nope.py\n"
            "---\n# x"
        )


# ---------------------------------------------------------------------------
# Mode B — tools.py auto-discovery → in-process Python Tool registration
# ---------------------------------------------------------------------------


def _make_mode_b_skill(tmp_path: Path) -> Path:
    """A skill that ships a Python @tool function via tools.py."""
    folder = tmp_path / "greeter"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nname: greeter\ndescription: Greet people.\n---\n"
        "# Greeter\nUse the `say_hi` tool to greet someone.\n"
    )
    (folder / "tools.py").write_text(
        "from loomflow import tool\n"
        "@tool\n"
        "async def say_hi(name: str) -> str:\n"
        "    \"\"\"Say hi to NAME.\"\"\"\n"
        "    return f'hi, {name}!'\n"
    )
    return folder


async def test_mode_b_imports_tools_py(tmp_path: Path) -> None:
    """``tools.py`` import is deferred — ``Skill()`` only detects
    the file. Mode B tools materialize on first
    ``materialize_tools()`` call (which the framework triggers from
    inside ``load_skill``)."""
    skill = Skill(_make_mode_b_skill(tmp_path))
    assert skill.metadata.has_python_tools is True
    # Before materialize_tools() the Mode B portion is empty.
    assert skill.pending_tools == []
    # First materialize triggers the import; tools become available.
    materialized = skill.materialize_tools()
    assert len(materialized) == 1
    tool = materialized[0]
    assert tool.name == "greeter__say_hi"
    out = await tool.fn(name="Alice")
    assert out == "hi, Alice!"
    # After materialize, ``pending_tools`` reflects the cached set.
    assert {t.name for t in skill.pending_tools} == {"greeter__say_hi"}


async def test_mode_b_import_error_surfaces_at_load_time(
    tmp_path: Path,
) -> None:
    """Import errors in ``tools.py`` are now deferred to
    ``materialize_tools()`` (i.e. when ``load_skill`` fires) — the
    deferral lets ``tools.py`` use the live event loop, but means
    construction succeeds silently and the error appears on first
    use. ``Skill()`` constructor still completes; the SkillError
    raises when materialization is attempted."""
    folder = tmp_path / "broken"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nname: broken\ndescription: y\n---\n# x"
    )
    (folder / "tools.py").write_text(
        "import nonexistent_module_xyz\n"
    )
    # Construction succeeds — the file isn't imported yet.
    skill = Skill(folder)
    # materialize_tools() triggers the import → SkillError.
    with pytest.raises(SkillError, match="Error importing"):
        skill.materialize_tools()


async def test_tools_py_with_event_loop_work_no_longer_crashes_at_construction(
    tmp_path: Path,
) -> None:
    """A ``tools.py`` that does ``asyncio.run(...)`` at module level
    used to crash at ``Skill()`` construction (because Skill ran the
    import there, and the construction may happen inside a Jupyter
    event loop). The deferred-import design means construction
    succeeds; the import only happens when the framework calls
    ``materialize_tools()`` from inside the agent loop, where doing
    event-loop work is fine.

    The test verifies the *construction-time* behaviour: no crash.
    What materialize does later is up to the user's ``tools.py`` —
    if they STILL do ``asyncio.run(...)`` at module level, our new
    error handler gives them a hint instead of the raw asyncio
    traceback.
    """
    folder = tmp_path / "loopy"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nname: loopy\ndescription: y\n---\n# x"
    )
    (folder / "tools.py").write_text(
        "import asyncio\n"
        "async def _setup():\n"
        "    pass\n"
        "# Would crash at construction under eager-import; now\n"
        "# only crashes when materialize_tools() runs inside a\n"
        "# live event loop. The Skill construction itself is fine.\n"
        "asyncio.run(_setup())\n"
    )
    # Inside this test we're already in an anyio event loop, so
    # materialize_tools() WILL fail — but Skill() construction
    # must succeed regardless.
    skill = Skill(folder)
    assert skill.metadata.name == "loopy"
    # The hint kicks in only when the import error is an event-loop
    # collision; verify the SkillError mentions the fix.
    with pytest.raises(SkillError) as excinfo:
        skill.materialize_tools()
    msg = str(excinfo.value)
    assert "event loop" in msg
    assert "build_tools(ctx)" in msg


async def test_build_tools_factory_protocol_passes_ctx(
    tmp_path: Path,
) -> None:
    """When ``tools.py`` exports ``build_tools(ctx)`` instead of
    module-level ``@tool`` definitions, the framework calls the
    factory with the supplied context and uses its returned list.
    This is the dependency-injection pattern: tools close over
    state (a vectorstore, a DB, etc.) that the caller passes via
    ``ctx.metadata`` rather than module-level globals."""
    folder = tmp_path / "factory"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nname: factory\ndescription: y\n---\n# x"
    )
    (folder / "tools.py").write_text(
        "from loomflow import tool\n"
        "\n"
        "def build_tools(ctx):\n"
        "    secret = ctx.metadata['secret']\n"
        "    @tool\n"
        "    async def reveal() -> str:\n"
        "        return f'secret is {secret}'\n"
        "    return [reveal]\n"
    )
    skill = Skill(folder)

    # Forge a minimal ctx-like object — RunContext works, but for
    # this unit test we just need ``.metadata``.
    class _Ctx:
        metadata = {"secret": "foo42"}

    materialized = skill.materialize_tools(_Ctx())
    assert len(materialized) == 1
    tool = materialized[0]
    assert tool.name == "factory__reveal"
    assert (await tool.fn()) == "secret is foo42"


async def test_build_tools_factory_must_return_list_of_tool(
    tmp_path: Path,
) -> None:
    """A ``build_tools(ctx)`` that returns the wrong type should
    raise a clear ``SkillError`` so the user sees the contract,
    not a downstream TypeError when registration tries to iterate."""
    folder = tmp_path / "badfactory"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nname: badfactory\ndescription: y\n---\n# x"
    )
    (folder / "tools.py").write_text(
        "def build_tools(ctx):\n"
        "    return 'not a list'\n"
    )
    skill = Skill(folder)
    with pytest.raises(SkillError, match="must return a list"):
        skill.materialize_tools(None)


async def test_mode_b_no_tools_py_means_no_python_tools(
    tmp_path: Path,
) -> None:
    folder = _make_skill_dir(tmp_path, "plain", "Just markdown.")
    skill = Skill(folder)
    assert skill.metadata.has_python_tools is False
    assert skill.pending_tools == []


# ---------------------------------------------------------------------------
# Mixed mode — one skill with BOTH a manifest tool AND a tools.py tool
# ---------------------------------------------------------------------------


async def test_skill_can_mix_mode_b_and_mode_c(tmp_path: Path) -> None:
    folder = tmp_path / "mixed"
    (folder / "scripts").mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        "---\nname: mixed\ndescription: Both kinds.\n"
        "tools:\n"
        "  shout:\n"
        "    description: Yell.\n"
        "    script: scripts/shout.py\n"
        "    args:\n"
        "      msg:\n"
        "        type: string\n"
        "        description: Message\n"
        "---\n\n# Mixed\n"
    )
    (folder / "scripts" / "shout.py").write_text(
        "import sys\nprint(sys.argv[1].upper())\n"
    )
    (folder / "tools.py").write_text(
        "from loomflow import tool\n"
        "@tool\n"
        "async def whisper(msg: str) -> str:\n"
        "    return msg.lower()\n"
    )
    skill = Skill(folder)
    # Mode C (subprocess wrappers) is eager — appears immediately.
    eager_names = {t.name for t in skill.pending_tools}
    assert eager_names == {"mixed__shout"}
    assert skill.metadata.has_python_tools is True
    assert skill.metadata.declared_tool_count == 1
    # Mode B (Python @tool from tools.py) is lazy — materialize
    # to pull in the second tool.
    full_names = {t.name for t in skill.materialize_tools()}
    assert full_names == {"mixed__shout", "mixed__whisper"}


# ---------------------------------------------------------------------------
# Catalog annotation — "+N tools" suffix per skill
# ---------------------------------------------------------------------------


def test_catalog_annotation_counts_pending_tools(tmp_path: Path) -> None:
    """A skill with N pending tools should show '+N tools' in its
    catalog line so the model knows loading it expands the toolset."""
    _make_mode_c_skill(tmp_path)
    registry = SkillRegistry([tmp_path])
    line = next(s.metadata.to_catalog_line() for s in registry)
    assert "[+1 tools]" in line


def test_catalog_no_annotation_for_pure_markdown(tmp_path: Path) -> None:
    _make_skill_dir(tmp_path, "alpha", "Plain.")
    registry = SkillRegistry([tmp_path])
    line = next(s.metadata.to_catalog_line() for s in registry)
    assert "+0 tools" not in line
    assert "[+" not in line  # no annotation when zero


# ---------------------------------------------------------------------------
# Lazy registration — pending tools register only on load_skill()
# ---------------------------------------------------------------------------


async def test_pending_tools_not_registered_until_load_skill(
    tmp_path: Path,
) -> None:
    """At Agent construction the pending tools exist on the skill
    but are NOT in the agent's tool host yet. Calling load_skill
    is what registers them."""
    _make_mode_b_skill(tmp_path)
    agent = Agent(
        "test",
        model=ScriptedModel([ScriptedTurn(text="ok")]),
        skills=[tmp_path],
    )
    tools_before = {t.name for t in await agent.tool_host.list_tools()}
    assert "load_skill" in tools_before
    assert "greeter__say_hi" not in tools_before  # not yet

    # Simulate the model calling load_skill via the actual Tool
    # instance (list_tools returns ToolDefs, not Tools, so use
    # InProcessToolHost.get to grab the runnable instance).
    real_tool = agent.tool_host.get("load_skill")  # type: ignore[attr-defined]
    body = await real_tool.fn(name="greeter")
    assert "Greeter" in body  # body content
    assert "greeter__say_hi" in body  # footer lists the new tool

    tools_after = {t.name for t in await agent.tool_host.list_tools()}
    assert "greeter__say_hi" in tools_after  # now registered


async def test_load_skill_idempotent_on_repeat_call(
    tmp_path: Path,
) -> None:
    """Calling load_skill twice for the same skill registers the
    tools only once — second call returns the body without re-
    registering."""
    _make_mode_b_skill(tmp_path)
    agent = Agent(
        "test",
        model=ScriptedModel([ScriptedTurn(text="ok")]),
        skills=[tmp_path],
    )
    load_tool = agent.tool_host.get("load_skill")  # type: ignore[attr-defined]
    await load_tool.fn(name="greeter")
    count_after_first = len(await agent.tool_host.list_tools())
    await load_tool.fn(name="greeter")
    count_after_second = len(await agent.tool_host.list_tools())
    assert count_after_first == count_after_second


def test_three_source_layered_override_pattern(tmp_path: Path) -> None:
    """The full DeepAgents layered pattern: a base set, a user
    override layer, and a project override layer. Last source wins
    by name across all three."""
    system = tmp_path / "system"
    user = tmp_path / "user"
    project = tmp_path / "project"
    for d in (system, user, project):
        d.mkdir()

    (system / "shared").mkdir()
    (system / "shared" / "SKILL.md").write_text(
        "---\nname: shared\ndescription: SYSTEM.\n---\nsystem"
    )
    (user / "shared").mkdir()
    (user / "shared" / "SKILL.md").write_text(
        "---\nname: shared\ndescription: USER.\n---\nuser"
    )
    (project / "shared").mkdir()
    (project / "shared" / "SKILL.md").write_text(
        "---\nname: shared\ndescription: PROJECT.\n---\nproject"
    )
    # Plus one skill only in system to verify we don't drop non-overridden.
    (system / "system-only").mkdir()
    (system / "system-only" / "SKILL.md").write_text(
        "---\nname: system-only\ndescription: Only in system.\n---\nbody"
    )

    registry = SkillRegistry(
        [system, user, (project, "Project")]
    )
    assert "shared" in registry
    assert "system-only" in registry
    shared = registry.get("shared")
    assert shared is not None
    assert shared.description == "PROJECT."
    # The labelled source attaches its label to the skill metadata.
    assert shared.metadata.source_label == "Project"
