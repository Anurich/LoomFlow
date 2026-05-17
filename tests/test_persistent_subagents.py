"""Persistent-subagent registry + send_message tool — coverage.

Tests the v0.10.10 persistent-subagent primitives across all six
``Team.*`` builders:

* ``Team.supervisor`` / ``swarm`` / ``router`` / ``debate`` /
  ``actor_critic`` / ``blackboard`` accept ``persistent_subagents``
  (default ``True``) and stamp a ``_worker_registry`` onto the
  coordinator.
* Each registered worker has a stable
  ``session_id`` (``persistent_worker_<role>_<ULID>``) reused across
  every spawn site in the corresponding architecture.
* ``persistent_subagents=False`` opt-out leaves the registry empty.
* ``make_send_message_tool`` rejects unknown IDs with a tool-result
  error string (not a raise).
* ``_WorkerHandle.touch`` pins ``user_id`` on first call + updates
  ``last_used_at``.
* ``new_worker_id`` validates role names against the
  Python-identifier rule (matches ``Supervisor.add_worker``).

All tests are zero-dep (EchoModel / ScriptedModel only); no
``@pytest.mark.live`` calls. Async-marked at module top per the
project convention.
"""

from __future__ import annotations

import pytest

from loomflow import Agent, EchoModel, ScriptedModel, ScriptedTurn
from loomflow.agent.worker_registry import (
    _WorkerHandle,
    build_worker_registry,
    new_worker_id,
    resolve_persistent_session,
)
from loomflow.architecture.router import RouterRoute
from loomflow.memory.inmemory import InMemoryMemory
from loomflow.team import Team
from loomflow.tools.send_message import make_send_message_tool

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Worker-id + registry primitives
# ---------------------------------------------------------------------------


def test_new_worker_id_format() -> None:
    """ID is ``worker_<role>_<ULID>``; ULID makes two calls unique."""
    wid_a = new_worker_id("coder")
    wid_b = new_worker_id("coder")
    assert wid_a.startswith("worker_coder_")
    assert wid_b.startswith("worker_coder_")
    assert wid_a != wid_b


def test_new_worker_id_rejects_non_identifier_role() -> None:
    """Non-Python-identifier roles raise — mirror Supervisor's check."""
    with pytest.raises(ValueError, match="must be a Python identifier"):
        new_worker_id("not a role")
    with pytest.raises(ValueError):
        new_worker_id("123")
    with pytest.raises(ValueError):
        new_worker_id("hyphen-name")


def test_new_worker_id_accepts_dunder_roles() -> None:
    """Blackboard uses ``__coordinator`` / ``__decider``; must be OK."""
    assert new_worker_id("__coordinator").startswith(
        "worker___coordinator_"
    )
    assert new_worker_id("__decider").startswith(
        "worker___decider_"
    )


def test_build_worker_registry_assigns_stable_sessions() -> None:
    """Each handle gets a stable session_id; map covers every role."""
    a = Agent(instructions="", model=EchoModel())
    b = Agent(instructions="", model=EchoModel())
    registry, role_map = build_worker_registry({"alpha": a, "beta": b})

    assert set(role_map) == {"alpha", "beta"}
    assert registry[role_map["alpha"]].agent is a
    assert registry[role_map["beta"]].agent is b
    # Stable session IDs: prefix conveys persistence; ID is unique
    assert registry[role_map["alpha"]].session_id.startswith(
        "persistent_worker_alpha_"
    )
    # All handles start with user_id pinned to None (first-touch).
    assert registry[role_map["alpha"]].user_id is None


def test_handle_touch_pins_user_id_first_time() -> None:
    """First touch pins user_id; subsequent touches don't override."""
    a = Agent(instructions="", model=EchoModel())
    registry, role_map = build_worker_registry({"x": a})
    handle = registry[role_map["x"]]

    handle.touch(user_id="alice")
    assert handle.user_id == "alice"
    assert handle.last_used_at is not None

    handle.touch(user_id="bob")  # second touch must NOT override pin
    assert handle.user_id == "alice"


def test_resolve_persistent_session_falls_back_when_no_registry() -> None:
    """Without a registry, the fallback session_id is returned."""
    sid, handle = resolve_persistent_session(
        "anything",
        fallback="parent__sub_x",
        registry=None,
        role_to_id=None,
    )
    assert sid == "parent__sub_x"
    assert handle is None


def test_resolve_persistent_session_returns_handle_when_registered() -> None:
    """With a registry, the handle's stable session_id wins."""
    a = Agent(instructions="", model=EchoModel())
    registry, role_map = build_worker_registry({"r1": a})
    sid, handle = resolve_persistent_session(
        "r1",
        fallback="parent__sub_x",
        registry=registry,
        role_to_id=role_map,
    )
    assert handle is registry[role_map["r1"]]
    assert sid == handle.session_id
    assert sid != "parent__sub_x"


def test_resolve_persistent_session_unknown_role_falls_back() -> None:
    """Role not in the registry → fallback (NOT raise)."""
    a = Agent(instructions="", model=EchoModel())
    registry, role_map = build_worker_registry({"r1": a})
    sid, handle = resolve_persistent_session(
        "not_registered",
        fallback="parent__sub_x",
        registry=registry,
        role_to_id=role_map,
    )
    assert sid == "parent__sub_x"
    assert handle is None


# ---------------------------------------------------------------------------
# Team.* builders all stamp a registry by default
# ---------------------------------------------------------------------------


def _scripted(text: str) -> Agent:
    return Agent("", model=ScriptedModel(turns=[ScriptedTurn(text=text)]))


def test_team_supervisor_stamps_registry_by_default() -> None:
    coord = Team.supervisor(
        workers={"researcher": _scripted("ok"), "coder": _scripted("ok")},
        model="echo",
    )
    reg: dict[str, _WorkerHandle] = coord._worker_registry
    assert {h.role for h in reg.values()} == {"researcher", "coder"}


def test_team_supervisor_opt_out_leaves_registry_empty() -> None:
    coord = Team.supervisor(
        workers={"x": _scripted("ok"), "y": _scripted("ok")},
        model="echo",
        persistent_subagents=False,
    )
    assert coord._worker_registry == {}


def test_team_swarm_stamps_registry_by_default() -> None:
    coord = Team.swarm(
        agents={"alpha": _scripted("ok"), "beta": _scripted("ok")},
        entry_agent="alpha",
        model="echo",
    )
    reg: dict[str, _WorkerHandle] = coord._worker_registry
    assert {h.role for h in reg.values()} == {"alpha", "beta"}


def test_team_swarm_opt_out() -> None:
    coord = Team.swarm(
        agents={"alpha": _scripted("ok"), "beta": _scripted("ok")},
        entry_agent="alpha",
        model="echo",
        persistent_subagents=False,
    )
    assert coord._worker_registry == {}


def test_team_router_stamps_registry_by_default() -> None:
    coord = Team.router(
        routes=[
            RouterRoute(name="billing", agent=_scripted("ok")),
            RouterRoute(name="tech", agent=_scripted("ok")),
        ],
        model="echo",
    )
    reg: dict[str, _WorkerHandle] = coord._worker_registry
    assert {h.role for h in reg.values()} == {"billing", "tech"}


def test_team_router_opt_out() -> None:
    coord = Team.router(
        routes=[RouterRoute(name="x", agent=_scripted("ok"))],
        model="echo",
        persistent_subagents=False,
    )
    assert coord._worker_registry == {}


def test_team_debate_stamps_registry_with_debaters_and_judge() -> None:
    coord = Team.debate(
        debaters=[_scripted("a"), _scripted("b")],
        judge=_scripted("verdict"),
        model="echo",
    )
    reg: dict[str, _WorkerHandle] = coord._worker_registry
    roles = {h.role for h in reg.values()}
    assert roles == {"debater_0", "debater_1", "judge"}


def test_team_debate_without_judge_only_registers_debaters() -> None:
    coord = Team.debate(
        debaters=[_scripted("a"), _scripted("b")], model="echo"
    )
    reg: dict[str, _WorkerHandle] = coord._worker_registry
    roles = {h.role for h in reg.values()}
    assert roles == {"debater_0", "debater_1"}


def test_team_actor_critic_stamps_actor_and_critic() -> None:
    coord = Team.actor_critic(
        actor=_scripted("draft"),
        critic=_scripted('{"issues": [], "score": 1.0, "summary": "ok"}'),
        model="echo",
    )
    reg: dict[str, _WorkerHandle] = coord._worker_registry
    assert {h.role for h in reg.values()} == {"actor", "critic"}


def test_team_blackboard_stamps_agents_coordinator_decider() -> None:
    coord = Team.blackboard(
        agents={
            "researcher": _scripted("r"),
            "writer": _scripted("w"),
        },
        coordinator=_scripted(
            '{"terminate": true, "next_agent": null, "instruction": null}'
        ),
        decider=_scripted("final"),
        model="echo",
    )
    reg: dict[str, _WorkerHandle] = coord._worker_registry
    roles = {h.role for h in reg.values()}
    assert roles == {"researcher", "writer", "__coordinator", "__decider"}


def test_team_blackboard_without_coordinator_decider() -> None:
    coord = Team.blackboard(
        agents={"x": _scripted("ok"), "y": _scripted("ok")},
        model="echo",
    )
    reg: dict[str, _WorkerHandle] = coord._worker_registry
    assert {h.role for h in reg.values()} == {"x", "y"}


# ---------------------------------------------------------------------------
# send_message tool — error paths (success paths exercised end-to-end
# by the supervisor integration tests in tests/test_supervisor.py)
# ---------------------------------------------------------------------------


class _FakeSession:
    id = "s1"
    session_id = "s1"


async def test_send_message_unknown_id_returns_error_string() -> None:
    """Unknown worker IDs return an error string (NOT raise)."""
    registry: dict[str, _WorkerHandle] = {}
    tool = make_send_message_tool(
        registry, session=_FakeSession(), memory=InMemoryMemory()
    )

    from loomflow.core.context import RunContext, set_run_context

    ctx = RunContext(
        user_id=None, session_id="s1", run_id="r1", metadata={}
    )
    async with set_run_context(ctx):
        out = await tool.fn(to="worker_ghost_X", content="hello")
    assert "unknown worker id" in out.lower()
    assert "worker_ghost_X" in out


async def test_send_message_cross_tenant_rejected() -> None:
    """Cross-user invocation surfaces a clear refusal string."""
    a = Agent(instructions="", model=EchoModel())
    registry, role_map = build_worker_registry({"alpha": a})
    handle = registry[role_map["alpha"]]
    handle.touch(user_id="alice")  # pin to alice

    tool = make_send_message_tool(
        registry, session=_FakeSession(), memory=InMemoryMemory()
    )

    from loomflow.core.context import RunContext, set_run_context

    ctx = RunContext(
        user_id="bob", session_id="s1", run_id="r1", metadata={}
    )
    async with set_run_context(ctx):
        out = await tool.fn(to=handle.worker_id, content="hello")
    assert "cross-tenant" in out.lower() or "rejected" in out.lower()
    assert "alice" in out and "bob" in out
