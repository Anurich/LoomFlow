"""Agent.from_config — load a TOML config into an Agent."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from loomflow import Agent, EchoModel
from loomflow.core.errors import ConfigError
from loomflow.governance.budget import StandardBudget

pytestmark = pytest.mark.anyio


def _write_toml(path: Path, body: str) -> None:
    path.write_text(body)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_minimal_config_loads(tmp_path: Path) -> None:
    cfg = tmp_path / "agent.toml"
    _write_toml(
        cfg,
        """
        instructions = "You are a research assistant."
        model = "echo"
        """,
    )
    agent = Agent.from_config(cfg)
    assert isinstance(agent, Agent)
    assert agent.model.name == "echo"


def test_max_turns_and_auto_consolidate_round_trip(tmp_path: Path) -> None:
    cfg = tmp_path / "agent.toml"
    _write_toml(
        cfg,
        """
        instructions = "You are helpful."
        model = "echo"
        max_turns = 13
        auto_consolidate = true
        """,
    )
    agent = Agent.from_config(cfg)
    assert agent._max_turns == 13  # noqa: SLF001
    assert agent._auto_consolidate is True  # noqa: SLF001


def test_budget_block_constructs_standard_budget(tmp_path: Path) -> None:
    cfg = tmp_path / "agent.toml"
    _write_toml(
        cfg,
        """
        instructions = "..."
        model = "echo"

        [budget]
        max_tokens = 1000
        max_cost_usd = 0.5
        max_wall_clock_minutes = 2
        soft_warning_at = 0.75
        """,
    )
    agent = Agent.from_config(cfg)
    budget = agent.budget
    assert isinstance(budget, StandardBudget)
    cfg_obj = budget._cfg  # noqa: SLF001
    assert cfg_obj.max_tokens == 1000
    assert cfg_obj.max_cost_usd == 0.5
    assert cfg_obj.max_wall_clock == timedelta(minutes=2)
    assert cfg_obj.soft_warning_at == 0.75


# ---------------------------------------------------------------------------
# Pass concrete instances for things TOML can't express
# ---------------------------------------------------------------------------


def test_caller_can_override_model_with_instance(tmp_path: Path) -> None:
    cfg = tmp_path / "agent.toml"
    _write_toml(
        cfg,
        """
        instructions = "..."
        model = "echo"
        """,
    )
    custom = EchoModel(prefix="OVERRIDE: ")
    agent = Agent.from_config(cfg, model=custom)
    assert agent.model is custom


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_missing_instructions_raises(tmp_path: Path) -> None:
    cfg = tmp_path / "agent.toml"
    _write_toml(cfg, 'model = "echo"\n')
    with pytest.raises(ConfigError, match="instructions"):
        Agent.from_config(cfg)


def test_missing_model_raises(tmp_path: Path) -> None:
    cfg = tmp_path / "agent.toml"
    _write_toml(cfg, 'instructions = "x"\n')
    with pytest.raises(ConfigError, match="model"):
        Agent.from_config(cfg)


# ---------------------------------------------------------------------------
# End-to-end run
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# from_dict
# ---------------------------------------------------------------------------


def test_from_dict_minimal_config() -> None:
    agent = Agent.from_dict(
        {"instructions": "be helpful", "model": "echo"}
    )
    assert agent.model.name == "echo"
    assert agent._max_turns == 50  # noqa: SLF001 — default


def test_from_dict_reads_timeout() -> None:
    """``timeout`` is float-coerced from the config dict and passed
    through to the wall-clock guard."""
    agent = Agent.from_dict(
        {"instructions": "x", "model": "echo", "timeout": 0.5}
    )
    assert agent._timeout == 0.5  # noqa: SLF001


def test_from_dict_timeout_absent_is_none() -> None:
    agent = Agent.from_dict({"instructions": "x", "model": "echo"})
    assert agent._timeout is None  # noqa: SLF001


def test_from_dict_with_budget() -> None:
    agent = Agent.from_dict(
        {
            "instructions": "x",
            "model": "echo",
            "auto_consolidate": True,
            "budget": {
                "max_tokens": 1234,
                "max_cost_usd": 0.25,
                "max_wall_clock_minutes": 1,
            },
        }
    )
    assert agent._auto_consolidate is True  # noqa: SLF001
    cfg_obj = agent.budget._cfg  # noqa: SLF001
    assert cfg_obj.max_tokens == 1234
    assert cfg_obj.max_cost_usd == 0.25
    assert cfg_obj.max_wall_clock == timedelta(minutes=1)


def test_from_dict_missing_instructions_raises() -> None:
    with pytest.raises(ConfigError, match="instructions"):
        Agent.from_dict({"model": "echo"})


def test_from_dict_missing_model_raises() -> None:
    with pytest.raises(ConfigError, match="model"):
        Agent.from_dict({"instructions": "x"})


def test_from_dict_caller_can_override_model() -> None:
    custom = EchoModel(prefix="OVER: ")
    agent = Agent.from_dict(
        {"instructions": "x", "model": "echo"},
        model=custom,
    )
    assert agent.model is custom


def test_from_config_preserves_path_in_error_message(tmp_path: Path) -> None:
    """When ``from_config`` rewraps a ``ConfigError`` from
    ``from_dict``, it should include the file path so callers know
    which TOML produced the error."""
    cfg = tmp_path / "agent.toml"
    cfg.write_text('model = "echo"\n')  # missing instructions
    with pytest.raises(ConfigError, match=str(cfg)):
        Agent.from_config(cfg)


async def test_loaded_agent_can_run(tmp_path: Path) -> None:
    cfg = tmp_path / "agent.toml"
    _write_toml(
        cfg,
        """
        instructions = "echo bot"
        model = "echo"
        """,
    )
    agent = Agent.from_config(cfg)
    result = await agent.run("hello")
    assert "hello" in result.output


# ---------------------------------------------------------------------------
# Full dict-form config — every backend wired from a single TOML
# ---------------------------------------------------------------------------


def test_from_config_wires_memory_runtime_telemetry_blocks(
    tmp_path: Path,
) -> None:
    """Memory, runtime, and telemetry can each be expressed as a
    ``[block]`` in TOML — the resolvers do the work."""
    from loomflow.memory import SqliteMemory
    from loomflow.observability import InMemoryTelemetry
    from loomflow.runtime import SqliteRuntime

    cfg = tmp_path / "agent.toml"
    _write_toml(
        cfg,
        f"""
        instructions = "research bot"
        model = "echo"

        [memory]
        backend = "sqlite"
        path = "{tmp_path / 'mem.db'}"

        [runtime]
        backend = "sqlite"
        path = "{tmp_path / 'journal.db'}"

        [telemetry]
        backend = "memory"
        """,
    )
    agent = Agent.from_config(cfg)
    assert isinstance(agent.memory, SqliteMemory)
    assert isinstance(agent.runtime, SqliteRuntime)
    assert isinstance(agent._telemetry, InMemoryTelemetry)  # noqa: SLF001


def test_from_config_audit_log_block(tmp_path: Path) -> None:
    """The audit_log resolver uses ``name = <path>`` (its existing
    dict-form convention from 0.9.x). Confirms the from_dict path
    threads the block through unchanged."""
    from loomflow.security import FileAuditLog

    out = tmp_path / "audit.jsonl"
    cfg = tmp_path / "agent.toml"
    _write_toml(
        cfg,
        f"""
        instructions = "x"
        model = "echo"

        [audit_log]
        name = "{out}"
        scope_full = true
        """,
    )
    agent = Agent.from_config(cfg)
    # FullTranscriptAuditLog wraps the FileAuditLog when scope_full=true.
    inner = getattr(agent._audit_log, "inner", agent._audit_log)  # noqa: SLF001
    assert isinstance(inner, FileAuditLog)


def test_from_config_permissions_block_with_mode_and_deny_list(
    tmp_path: Path,
) -> None:
    from loomflow.security import Mode, StandardPermissions

    cfg = tmp_path / "agent.toml"
    _write_toml(
        cfg,
        """
        instructions = "x"
        model = "echo"

        [permissions]
        backend = "standard"
        mode = "accept_edits"
        denied_tools = ["bash"]
        """,
    )
    agent = Agent.from_config(cfg)
    perms = agent._permissions  # noqa: SLF001
    assert isinstance(perms, StandardPermissions)
    assert perms._mode == Mode.ACCEPT_EDITS  # noqa: SLF001
    assert "bash" in perms._denied  # noqa: SLF001


def test_from_config_architecture_and_effort_strings(tmp_path: Path) -> None:
    from loomflow.architecture import Reflexion

    cfg = tmp_path / "agent.toml"
    _write_toml(
        cfg,
        """
        instructions = "x"
        model = "echo"
        architecture = "reflexion"
        effort = "high"
        response_tone = "concise"
        """,
    )
    agent = Agent.from_config(cfg)
    assert isinstance(agent.architecture, Reflexion)
    assert agent._default_effort == "high"  # noqa: SLF001
    assert agent._default_response_tone == "concise"  # noqa: SLF001


def test_from_config_skills_block_with_string_paths(tmp_path: Path) -> None:
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: Demo skill.\n---\nbody\n"
    )

    cfg = tmp_path / "agent.toml"
    _write_toml(
        cfg,
        f"""
        instructions = "x"
        model = "echo"
        skills = ["{skill_dir}"]
        """,
    )
    agent = Agent.from_config(cfg)
    assert agent.skills is not None
    assert "my-skill" in agent.skills


def test_from_config_skills_block_with_labelled_dict(tmp_path: Path) -> None:
    skill_dir = tmp_path / "labelled"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: labelled\ndescription: Labelled.\n---\nbody\n"
    )

    cfg = tmp_path / "agent.toml"
    _write_toml(
        cfg,
        f"""
        instructions = "x"
        model = "echo"

        [[skills]]
        path = "{skill_dir}"
        label = "Project"
        """,
    )
    agent = Agent.from_config(cfg)
    assert agent.skills is not None
    skill = agent.skills.get("labelled")
    assert skill is not None
    assert skill.metadata.source_label == "Project"


def test_from_config_mcp_block_wires_registry(tmp_path: Path) -> None:
    from loomflow.mcp import MCPRegistry

    cfg = tmp_path / "agent.toml"
    _write_toml(
        cfg,
        """
        instructions = "x"
        model = "echo"

        [[mcp]]
        name = "git"
        transport = "stdio"
        command = "uvx"
        args = ["mcp-server-git", "--repo", "."]

        [[mcp]]
        name = "remote"
        transport = "http"
        url = "https://example.com/mcp"
        """,
    )
    agent = Agent.from_config(cfg)
    # The MCPRegistry becomes the tool host (wrapped by ExtendedToolHost
    # only when skills are also present, which they aren't here).
    host = agent._tool_host  # noqa: SLF001
    # Either the registry directly, or its underlying registry attribute.
    registry = host if isinstance(host, MCPRegistry) else getattr(
        host, "_inner", host
    )
    assert isinstance(registry, MCPRegistry)
    assert sorted(registry.server_names) == ["git", "remote"]


def test_from_config_mcp_with_tools_kwarg_rejected(tmp_path: Path) -> None:
    """Passing both ``mcp`` config AND a ``tools=`` kwarg is
    ambiguous — the user has to decide which tool host wins. We
    refuse rather than silently dropping one."""
    from loomflow import tool

    @tool
    async def my_fn() -> str:
        """do thing"""
        return "ok"

    cfg = tmp_path / "agent.toml"
    _write_toml(
        cfg,
        """
        instructions = "x"
        model = "echo"

        [[mcp]]
        name = "git"
        transport = "stdio"
        command = "noop"
        """,
    )
    with pytest.raises(ConfigError, match="cannot mix"):
        Agent.from_config(cfg, tools=[my_fn])


def test_from_dict_mcp_rejects_unknown_transport() -> None:
    with pytest.raises(ConfigError, match="transport must be"):
        Agent.from_dict(
            {
                "instructions": "x",
                "model": "echo",
                "mcp": [{"name": "x", "transport": "ws"}],
            }
        )


def test_from_dict_kwarg_overrides_cfg_memory() -> None:
    """A ``memory=`` kwarg wins over a ``[memory]`` block in cfg.
    Callers wire their own ready-built backend without rewriting
    the TOML."""
    from loomflow import InMemoryMemory

    custom = InMemoryMemory()
    agent = Agent.from_dict(
        {
            "instructions": "x",
            "model": "echo",
            "memory": {"backend": "sqlite", "path": "/tmp/ignored.db"},
        },
        memory=custom,
    )
    assert agent.memory is custom


def test_from_config_full_stack(tmp_path: Path) -> None:
    """One TOML, every backend wired. Smoke test the end-to-end
    integration of every resolver."""
    from loomflow.memory import SqliteMemory
    from loomflow.observability import FileTelemetry
    from loomflow.runtime import SqliteRuntime
    from loomflow.security import (
        FileAuditLog,
        Mode,
        StandardPermissions,
    )

    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: skill\ndescription: x.\n---\nbody\n"
    )

    cfg = tmp_path / "agent.toml"
    _write_toml(
        cfg,
        f"""
        instructions = "everything"
        model = "echo"
        max_turns = 50
        architecture = "react"
        effort = "low"

        [memory]
        backend = "sqlite"
        path = "{tmp_path / 'mem.db'}"

        [runtime]
        backend = "sqlite"
        path = "{tmp_path / 'journal.db'}"

        [telemetry]
        backend = "file"
        path = "{tmp_path / 'spans.jsonl'}"

        [audit_log]
        name = "{tmp_path / 'audit.jsonl'}"

        [permissions]
        backend = "standard"
        mode = "default"
        denied_tools = ["bash"]

        [budget]
        max_tokens = 1000
        max_cost_usd = 0.25

        [[skills]]
        path = "{skill_dir}"
        """,
    )
    agent = Agent.from_config(cfg)
    assert isinstance(agent.memory, SqliteMemory)
    assert isinstance(agent.runtime, SqliteRuntime)
    assert isinstance(agent._telemetry, FileTelemetry)  # noqa: SLF001
    assert isinstance(agent._audit_log, FileAuditLog)  # noqa: SLF001
    perms = agent._permissions  # noqa: SLF001
    assert isinstance(perms, StandardPermissions)
    assert perms._mode == Mode.DEFAULT  # noqa: SLF001
    assert agent._max_turns == 50  # noqa: SLF001
    assert agent._default_effort == "low"  # noqa: SLF001
    assert "skill" in agent.skills  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Permissions resolver — string + dict + instance + errors
# ---------------------------------------------------------------------------


def test_permissions_resolver_string_forms() -> None:
    from loomflow.security import (
        AllowAll,
        Mode,
        StandardPermissions,
        resolve_permissions,
    )

    assert isinstance(resolve_permissions("allow_all"), AllowAll)
    strict = resolve_permissions("strict")
    assert isinstance(strict, StandardPermissions)
    assert strict._mode == Mode.DEFAULT  # noqa: SLF001
    bypass = resolve_permissions("bypass")
    assert isinstance(bypass, StandardPermissions)
    assert bypass._mode == Mode.BYPASS  # noqa: SLF001


def test_permissions_resolver_dict_form() -> None:
    from loomflow.security import (
        Mode,
        StandardPermissions,
        resolve_permissions,
    )

    perms = resolve_permissions(
        {
            "backend": "standard",
            "mode": "accept_edits",
            "allowed_tools": ["read"],
            "denied_tools": ["bash"],
        }
    )
    assert isinstance(perms, StandardPermissions)
    assert perms._mode == Mode.ACCEPT_EDITS  # noqa: SLF001
    assert perms._allowed == {"read"}  # noqa: SLF001
    assert perms._denied == {"bash"}  # noqa: SLF001


def test_permissions_resolver_passes_through_instance() -> None:
    from loomflow.security import (
        StandardPermissions,
        resolve_permissions,
    )

    perms = StandardPermissions()
    assert resolve_permissions(perms) is perms


def test_permissions_resolver_rejects_unknown_string() -> None:
    from loomflow.security import resolve_permissions

    with pytest.raises(ConfigError, match="unrecognised"):
        resolve_permissions("nope")


def test_permissions_resolver_rejects_unknown_mode() -> None:
    from loomflow.security import resolve_permissions

    with pytest.raises(ConfigError, match="unrecognised mode"):
        resolve_permissions({"backend": "standard", "mode": "wild"})
