"""Regression tests for the WSF1 review fixes in ``loomflow/skills/``.

Covers:

* Re-wrapped (prefixed) skill tools keep ``destructive`` and
  ``param_adapters`` instead of being silently laundered past the
  permission gate / arg coercion.
* Mode C subprocess tools run via anyio (trio-safe) and get killed
  after the wall-clock timeout instead of hanging the agent loop.
* ``load_skill``'s schema no longer bakes the skill names into an
  ``enum``: skills added after tool construction are loadable, and
  unknown names fail at call time with the valid set listed.
* A skill's ``tools.py`` is imported in a worker thread when
  ``load_skill`` fires — module-level code that needs a loop-free
  context (e.g. ``asyncio.run``) no longer collides with the agent's
  running event loop.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import loomflow.skills.skill as skill_mod
from loomflow.skills import Skill, SkillRegistry
from loomflow.skills.tools import make_load_skill_tool

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_skill_dir(
    tmp_path: Path,
    name: str,
    description: str,
    extra: str = "",
) -> Path:
    folder = tmp_path / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n{extra}---\n\n"
        f"# {name}\n\nInstructions for {name}.\n"
    )
    return folder


# ---------------------------------------------------------------------------
# Fix 5 — prefixed re-wrap preserves Tool metadata
# ---------------------------------------------------------------------------


async def test_module_level_tools_keep_destructive_and_adapters(
    tmp_path: Path,
) -> None:
    folder = _make_skill_dir(tmp_path, "wiper", "Wipes things.")
    (folder / "tools.py").write_text(
        "from loomflow import tool\n"
        "\n"
        "@tool(destructive=True)\n"
        "async def wipe(items: list[int]) -> str:\n"
        '    """Wipe items."""\n'
        "    return str(items)\n"
    )
    skill = Skill(folder)
    (wipe,) = skill.materialize_tools()
    assert wipe.name == "wiper__wipe"
    # destructive must survive the prefix re-wrap or the permission
    # gate never sees it.
    assert wipe.destructive is True
    # param_adapters must survive too or list[int] args stop coercing.
    assert "items" in wipe.param_adapters
    assert await wipe.execute({"items": ["1", "2"]}) == "[1, 2]"


async def test_build_tools_factory_tools_keep_destructive(
    tmp_path: Path,
) -> None:
    folder = _make_skill_dir(tmp_path, "nuker", "Factory tools.")
    (folder / "tools.py").write_text(
        "from loomflow import tool\n"
        "\n"
        "def build_tools(ctx):\n"
        "    @tool(destructive=True)\n"
        "    def nuke(target: str) -> str:\n"
        '        """Nuke a target."""\n'
        "        return target\n"
        "    return [nuke]\n"
    )
    skill = Skill(folder)
    (nuke,) = skill.materialize_tools(None)
    assert nuke.name == "nuker__nuke"
    assert nuke.destructive is True


# ---------------------------------------------------------------------------
# Fix 6a — subprocess tools run via anyio, with a kill-switch timeout
# ---------------------------------------------------------------------------

_MODE_C_FRONTMATTER = (
    "tools:\n"
    "  {tool_name}:\n"
    "    description: {desc}\n"
    "    script: scripts/{script}\n"
    "    args:\n"
    "      msg:\n"
    "        type: string\n"
    "        description: Message\n"
)


async def test_subprocess_tool_runs_under_anyio(tmp_path: Path) -> None:
    folder = _make_skill_dir(
        tmp_path,
        "shouter",
        "Yells.",
        extra=_MODE_C_FRONTMATTER.format(
            tool_name="shout", desc="Yell.", script="shout.py"
        ),
    )
    (folder / "scripts").mkdir()
    (folder / "scripts" / "shout.py").write_text(
        "import sys\nprint(sys.argv[1].upper())\n"
    )
    skill = Skill(folder)
    (shout,) = skill.pending_tools
    out = await shout.fn(msg="hi there")
    assert out.strip() == "HI THERE"


async def test_subprocess_tool_killed_after_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(skill_mod, "_SUBPROCESS_TIMEOUT_S", 0.5)
    folder = _make_skill_dir(
        tmp_path,
        "sleeper",
        "Naps.",
        extra=_MODE_C_FRONTMATTER.format(
            tool_name="nap", desc="Nap.", script="nap.py"
        ),
    )
    (folder / "scripts").mkdir()
    (folder / "scripts" / "nap.py").write_text(
        "import time\ntime.sleep(30)\nprint('done')\n"
    )
    skill = Skill(folder)
    (nap,) = skill.pending_tools
    out = await nap.fn(msg="zzz")
    assert "timed out" in out
    assert "done" not in out


# ---------------------------------------------------------------------------
# Fix 7 — no enum; late-registered skills loadable; call-time validation
# ---------------------------------------------------------------------------


async def test_load_skill_accepts_skill_added_after_tool_construction(
    tmp_path: Path,
) -> None:
    _make_skill_dir(tmp_path / "early", "alpha", "First.")
    registry = SkillRegistry([tmp_path / "early"])
    load_tool = make_load_skill_tool(registry)

    name_schema = load_tool.input_schema["properties"]["name"]
    assert "enum" not in name_schema  # a baked enum would reject 'gamma'

    gamma_folder = _make_skill_dir(tmp_path / "late", "gamma", "Late arrival.")
    registry.add(Skill(gamma_folder))

    out = await load_tool.fn(name="gamma")
    assert "Instructions for gamma" in out


async def test_load_skill_unknown_name_lists_valid_set(tmp_path: Path) -> None:
    _make_skill_dir(tmp_path, "alpha", "First.")
    registry = SkillRegistry([tmp_path])
    load_tool = make_load_skill_tool(registry)
    out = await load_tool.fn(name="nope")
    assert out.startswith("Error:")
    assert "nope" in out
    assert "alpha" in out  # the valid set is spelled out for the model


# ---------------------------------------------------------------------------
# Fix 6b — tools.py imports run off the event loop
# ---------------------------------------------------------------------------


async def test_load_skill_imports_tools_py_off_event_loop(
    tmp_path: Path,
) -> None:
    """Module-level ``asyncio.run(...)`` in a skill's ``tools.py``
    used to explode when load_skill fired (import executed on the
    agent's running loop). The import now happens in a worker
    thread, so it succeeds."""
    folder = _make_skill_dir(tmp_path, "loopy", "Needs loop-free import.")
    (folder / "tools.py").write_text(
        "import asyncio\n"
        "from loomflow import tool\n"
        "\n"
        "async def _setup():\n"
        "    return 'ready'\n"
        "\n"
        "_STATE = asyncio.run(_setup())\n"
        "\n"
        "@tool\n"
        "def ping() -> str:\n"
        '    """Ping."""\n'
        "    return _STATE\n"
    )
    registry = SkillRegistry([folder])
    load_tool = make_load_skill_tool(registry)
    out = await load_tool.fn(name="loopy")
    assert not out.startswith("Error"), out
    assert "loopy__ping" in out  # footer announces the new tool
