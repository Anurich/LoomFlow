"""Regression tests for the WSB2 review fixes.

* ``Workflow.route`` — the original input is carried through
  per-run state (``RunContext.metadata``), not a construction-time
  closure. Two concurrent ``run()`` calls on ONE Workflow instance
  (the normal build-once, run-per-request server pattern) must not
  leak one request's input into another request's handler.
* ``Workflow.parallel`` — ``return_exceptions=True`` gather
  semantics (partial results survive a failing branch) and
  per-branch telemetry spans on the fan-out node.
* ``Team`` builders — behaviour preserved after the
  ``_build_coordinator`` refactor: the worker registry stamped on
  the coordinator is the SAME dict the architecture holds, and the
  Tuning knobs still reach the Agent.
"""

from __future__ import annotations

import anyio
import anyio.lowlevel
import pytest

from loomflow import Agent, Workflow
from loomflow.model.scripted import ScriptedModel, ScriptedTurn
from loomflow.observability.tracing import InMemoryTelemetry
from loomflow.team import Team

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Workflow.route — no cross-request input leak
# ---------------------------------------------------------------------------


async def test_route_concurrent_runs_do_not_leak_inputs() -> None:
    """Deterministic interleave of the old bug: request A's
    classifier runs (capturing A's input), then request B runs to
    completion (with the old construction-time closure this
    overwrote the captured value), then A's handler reads the
    captured input. A must still see input-A."""
    a_classifier_entered = anyio.Event()
    release_a = anyio.Event()

    async def classify(text: str) -> str:
        if text == "input-A":
            a_classifier_entered.set()
            # Hold A between "input captured" and "handler reads it"
            # while B runs through the same Workflow instance.
            await release_a.wait()
        return "handled"

    async def handler(text: str) -> str:
        return f"handled:{text}"

    wf = Workflow.route(classify, {"handled": handler})

    results: dict[str, str] = {}

    async def run_a() -> None:
        results["a"] = (await wf.run("input-A")).output

    async def run_b() -> None:
        await a_classifier_entered.wait()
        results["b"] = (await wf.run("input-B")).output
        release_a.set()

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_a)
        tg.start_soon(run_b)

    assert results["b"] == "handled:input-B"
    # With the shared-closure bug, A's handler saw B's input and
    # produced "handled:input-B" — a cross-request data leak.
    assert results["a"] == "handled:input-A"


async def test_route_many_concurrent_runs_outputs_match_inputs() -> None:
    """Fan out many concurrent runs (including the default route)
    through one Workflow instance; every output must correspond to
    its own input."""

    async def classify(text: str) -> str:
        i = int(text.rsplit("-", 1)[1])
        # Stagger tasks so classifier/handler phases interleave.
        for _ in range(i % 5):
            await anyio.lowlevel.checkpoint()
        if i % 5 == 0:
            return "unmatched-key"
        return "even" if i % 2 == 0 else "odd"

    async def even_handler(text: str) -> str:
        return f"even:{text}"

    async def odd_handler(text: str) -> str:
        return f"odd:{text}"

    async def default_handler(text: str) -> str:
        return f"default:{text}"

    wf = Workflow.route(
        classify,
        {"even": even_handler, "odd": odd_handler},
        default=default_handler,
    )

    outputs: dict[int, str] = {}

    async def one(i: int) -> None:
        outputs[i] = (await wf.run(f"req-{i}")).output

    async with anyio.create_task_group() as tg:
        for i in range(24):
            tg.start_soon(one, i)

    for i in range(24):
        if i % 5 == 0:
            expected = f"default:req-{i}"
        elif i % 2 == 0:
            expected = f"even:req-{i}"
        else:
            expected = f"odd:req-{i}"
        assert outputs[i] == expected, f"run {i} leaked another run's input"


# ---------------------------------------------------------------------------
# Workflow.parallel — return_exceptions + per-branch telemetry
# ---------------------------------------------------------------------------


async def test_parallel_return_exceptions_keeps_partial_results() -> None:
    async def ok(x: int) -> int:
        return x + 1

    async def boom(_x: int) -> int:
        raise ValueError("boom")

    wf = Workflow.parallel([ok, boom, ok], return_exceptions=True)
    result = await wf.run(1)

    out = result.output
    assert out[0] == 2
    assert out[2] == 2
    assert isinstance(out[1], ValueError)
    assert str(out[1]) == "boom"


async def test_parallel_default_still_propagates_branch_exception() -> None:
    """Without ``return_exceptions``, a failing branch cancels the
    siblings and the exception propagates (possibly grouped by the
    anyio task group)."""

    async def boom(_x: int) -> int:
        raise ValueError("boom")

    async def never_finishes(x: int) -> int:
        await anyio.sleep(60)  # cancelled by boom's failure
        return x

    wf = Workflow.parallel([boom, never_finishes])
    with pytest.raises((ValueError, ExceptionGroup)):
        await wf.run(1)


async def test_parallel_return_exceptions_does_not_cancel_siblings() -> None:
    """With ``return_exceptions=True`` a failing branch must not
    cancel its siblings — they complete and contribute results."""
    finished: list[str] = []

    async def boom(_x: int) -> int:
        raise ValueError("early failure")

    async def slow_ok(x: int) -> int:
        # Several checkpoints: plenty of chances to be cancelled if
        # the failing branch (which raises immediately) tore down
        # the task group.
        for _ in range(5):
            await anyio.lowlevel.checkpoint()
        finished.append("slow_ok")
        return x * 10

    wf = Workflow.parallel([boom, slow_ok], return_exceptions=True)
    result = await wf.run(3)

    assert finished == ["slow_ok"]
    assert isinstance(result.output[0], ValueError)
    assert result.output[1] == 30


async def test_parallel_emits_per_branch_telemetry_spans() -> None:
    tel = InMemoryTelemetry()

    async def alpha(x: int) -> int:
        return x

    async def beta(x: int) -> int:
        return x

    wf = Workflow.parallel([alpha, beta], telemetry=tel)
    await wf.run(1)

    step_attrs = {
        s.attributes.get("step")
        for s in tel.spans()
        if s.name == "loom.workflow.step"
    }
    # The fan_out node span plus one synthetic span per branch.
    assert "fan_out" in step_attrs
    assert "fan_out.alpha" in step_attrs
    assert "fan_out.beta" in step_attrs


async def test_parallel_branch_names_disambiguated_in_spans() -> None:
    """Reusing the same callable across branches still produces
    distinct per-branch span names."""
    tel = InMemoryTelemetry()

    async def f(x: int) -> int:
        return x

    wf = Workflow.parallel([f, f], telemetry=tel)
    await wf.run(1)

    step_attrs = {
        s.attributes.get("step")
        for s in tel.spans()
        if s.name == "loom.workflow.step"
    }
    assert "fan_out.f" in step_attrs
    assert "fan_out.f_1" in step_attrs


# ---------------------------------------------------------------------------
# Team — refactored builders preserve behaviour
# ---------------------------------------------------------------------------


def _scripted(text: str) -> Agent:
    return Agent("", model=ScriptedModel(turns=[ScriptedTurn(text=text)]))


def test_team_supervisor_registry_is_same_dict_as_architecture() -> None:
    """The registry stamped on the coordinator Agent must be the
    SAME dict the Supervisor architecture holds, so send_message and
    external introspection observe one shared registry."""
    coord = Team.supervisor(
        workers={"w1": _scripted("ok"), "w2": _scripted("ok")},
        model="echo",
    )
    assert coord._worker_registry is coord.architecture._worker_registry
    assert {h.role for h in coord._worker_registry.values()} == {"w1", "w2"}


def test_team_builders_forward_tuning_and_agent_kwargs() -> None:
    """Spot-check that the shared ``_build_coordinator`` path still
    forwards both a plain Agent kwarg (max_turns) and a
    Tuning-folded kwarg (response_tone) for every builder."""
    from loomflow.architecture import RouterRoute

    teams = [
        Team.supervisor(
            workers={"w": _scripted("ok")},
            model="echo",
            max_turns=7,
            response_tone="concise",
        ),
        Team.swarm(
            agents={"a": _scripted("ok"), "b": _scripted("ok")},
            entry_agent="a",
            model="echo",
            max_turns=7,
            response_tone="concise",
        ),
        Team.router(
            routes=[RouterRoute(name="r", agent=_scripted("ok"))],
            model="echo",
            max_turns=7,
            response_tone="concise",
        ),
        Team.debate(
            debaters=[_scripted("a"), _scripted("b")],
            model="echo",
            max_turns=7,
            response_tone="concise",
        ),
        Team.actor_critic(
            actor=_scripted("draft"),
            critic=_scripted('{"issues": [], "score": 1.0, "summary": "ok"}'),
            model="echo",
            max_turns=7,
            response_tone="concise",
        ),
        Team.blackboard(
            agents={"x": _scripted("ok")},
            model="echo",
            max_turns=7,
            response_tone="concise",
        ),
    ]
    for team in teams:
        assert team._max_turns == 7
        assert team._default_response_tone == "concise"
