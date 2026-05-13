"""``living_plan=`` resolver tests — bool / str / dict / instance /
None forms + error paths + Agent integration."""

from __future__ import annotations

import dataclasses

import pytest

from loomflow import Agent, LivingPlan, tool
from loomflow.core.errors import ConfigError
from loomflow.tools.plan_resolver import (
    ResolvedLivingPlan,
    resolve_living_plan,
)
from loomflow.workspace import InMemoryWorkspace

# ---------------------------------------------------------------------------
# None / bool
# ---------------------------------------------------------------------------


def test_none_disables() -> None:
    out = resolve_living_plan(None, workspace_present=False)
    assert out.enabled is False


def test_false_disables() -> None:
    out = resolve_living_plan(False, workspace_present=True)
    assert out.enabled is False


def test_true_no_workspace_enables_in_memory_only() -> None:
    out = resolve_living_plan(True, workspace_present=False)
    assert out.enabled is True
    assert out.mirror_to_workspace is False
    assert out.include_recall is False


def test_true_with_workspace_enables_mirror_and_recall() -> None:
    out = resolve_living_plan(True, workspace_present=True)
    assert out.enabled is True
    assert out.mirror_to_workspace is True
    assert out.include_recall is True


# ---------------------------------------------------------------------------
# String forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s", ["memory", "inmemory", "ephemeral"])
def test_string_memory_variants(s: str) -> None:
    out = resolve_living_plan(s, workspace_present=True)
    # Memory always disables mirror, even when workspace is present.
    assert out.enabled is True
    assert out.mirror_to_workspace is False
    assert out.include_recall is False


@pytest.mark.parametrize("s", ["workspace", "disk", "persist", "mirror"])
def test_string_workspace_variants(s: str) -> None:
    out = resolve_living_plan(s, workspace_present=True)
    assert out.enabled is True
    assert out.mirror_to_workspace is True
    assert out.include_recall is True


def test_string_workspace_without_workspace_raises() -> None:
    with pytest.raises(ConfigError, match="requires Agent.workspace"):
        resolve_living_plan("workspace", workspace_present=False)


def test_string_empty_raises() -> None:
    with pytest.raises(ConfigError, match="empty string"):
        resolve_living_plan("", workspace_present=False)


def test_string_unknown_raises() -> None:
    with pytest.raises(ConfigError, match="not recognised"):
        resolve_living_plan("frobnicate", workspace_present=False)


# ---------------------------------------------------------------------------
# Dict forms
# ---------------------------------------------------------------------------


def test_dict_enabled_default_true() -> None:
    out = resolve_living_plan({}, workspace_present=False)
    assert out.enabled is True


def test_dict_enabled_false() -> None:
    out = resolve_living_plan({"enabled": False}, workspace_present=True)
    assert out.enabled is False


def test_dict_mirror_workspace() -> None:
    out = resolve_living_plan(
        {"mirror": "workspace"}, workspace_present=True
    )
    assert out.mirror_to_workspace is True
    assert out.include_recall is True


def test_dict_mirror_none() -> None:
    out = resolve_living_plan(
        {"mirror": "none"}, workspace_present=True
    )
    assert out.mirror_to_workspace is False
    assert out.include_recall is False


def test_dict_mirror_workspace_without_workspace_raises() -> None:
    with pytest.raises(ConfigError, match="requires"):
        resolve_living_plan(
            {"mirror": "workspace"}, workspace_present=False
        )


def test_dict_include_recall_without_mirror_raises() -> None:
    with pytest.raises(ConfigError, match="requires a workspace mirror"):
        resolve_living_plan(
            {"include_recall": True, "mirror": "none"},
            workspace_present=True,
        )


def test_dict_include_recall_explicit_false() -> None:
    """When the mirror IS on, the user can still opt out of the
    recall tool to keep the tool surface tighter."""
    out = resolve_living_plan(
        {"mirror": "workspace", "include_recall": False},
        workspace_present=True,
    )
    assert out.mirror_to_workspace is True
    assert out.include_recall is False


def test_dict_task_id_passed_through() -> None:
    out = resolve_living_plan({"task_id": "task-007"}, workspace_present=False)
    assert out.task_id == "task-007"


def test_dict_author_passed_through() -> None:
    out = resolve_living_plan(
        {"author": "researcher"}, workspace_present=True
    )
    assert out.author == "researcher"


def test_dict_seed_plan_passthrough() -> None:
    seed = LivingPlan(goal="seed-goal")
    out = resolve_living_plan(
        {"seed_plan": seed}, workspace_present=False
    )
    assert out.seed_plan is seed


def test_dict_invalid_seed_plan_type_raises() -> None:
    with pytest.raises(ConfigError, match="seed_plan"):
        resolve_living_plan(
            {"seed_plan": {"not": "a plan"}}, workspace_present=False
        )


def test_dict_unknown_key_raises() -> None:
    with pytest.raises(ConfigError, match="unknown keys"):
        resolve_living_plan(
            {"frobnicate": True}, workspace_present=False
        )


def test_dict_unknown_mirror_value_raises() -> None:
    with pytest.raises(ConfigError, match="mirror"):
        resolve_living_plan(
            {"mirror": "frobnicate"}, workspace_present=True
        )


# ---------------------------------------------------------------------------
# Instance passthrough
# ---------------------------------------------------------------------------


def test_living_plan_instance_passthrough() -> None:
    seed = LivingPlan(goal="pre-seeded")
    out = resolve_living_plan(seed, workspace_present=True)
    assert out.enabled is True
    assert out.seed_plan is seed


# ---------------------------------------------------------------------------
# Bad input types
# ---------------------------------------------------------------------------


def test_invalid_type_raises() -> None:
    with pytest.raises(ConfigError, match="must be"):
        resolve_living_plan(42, workspace_present=False)


# ---------------------------------------------------------------------------
# Returned dataclass shape
# ---------------------------------------------------------------------------


def test_returns_resolved_living_plan_dataclass() -> None:
    out = resolve_living_plan(True, workspace_present=False)
    assert isinstance(out, ResolvedLivingPlan)
    # Frozen dataclass — assignment must raise FrozenInstanceError.
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.enabled = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Agent constructor integration
# ---------------------------------------------------------------------------


@tool
def _stub() -> str:
    return "ok"


def test_agent_accepts_string_living_plan() -> None:
    a = Agent("t", model="echo", tools=[_stub], living_plan="memory")
    assert a._living_plan_spec.enabled is True
    assert a._living_plan_spec.mirror_to_workspace is False


def test_agent_accepts_dict_living_plan() -> None:
    a = Agent(
        "t",
        model="echo",
        tools=[_stub],
        living_plan={"enabled": True, "mirror": "none"},
    )
    assert a._living_plan_spec.enabled is True
    assert a._living_plan_spec.mirror_to_workspace is False


def test_agent_accepts_workspace_string_with_workspace() -> None:
    ws = InMemoryWorkspace()
    a = Agent(
        "t",
        model="echo",
        tools=[_stub],
        workspace=ws,
        living_plan="workspace",
    )
    assert a._living_plan_spec.mirror_to_workspace is True
    names = [t.name for t in a._tool_host._tools.values()]
    assert "recall_past_plans" in names


def test_agent_rejects_workspace_living_plan_without_workspace() -> None:
    with pytest.raises(ConfigError):
        Agent(
            "t",
            model="echo",
            tools=[_stub],
            living_plan="workspace",
        )
