"""Agent.from_config — load a TOML config into an Agent."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from jeevesagent import Agent, EchoModel
from jeevesagent.core.errors import ConfigError
from jeevesagent.governance.budget import StandardBudget

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
