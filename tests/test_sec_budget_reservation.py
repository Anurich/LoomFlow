"""Security regression tests — budget caps under concurrency.

``allows_step`` (check) and ``consume`` (charge) used to be
check-then-act: N concurrent runs all passed the check before any was
charged, so a per-user cap of 1 call served N. Inside ``running()``
the check now reserves the coming call under the same lock.
"""

from __future__ import annotations

import warnings
from typing import Any

import anyio
import pytest

from loomflow import Agent
from loomflow.core.types import BudgetStatus, Usage
from loomflow.governance.budget import BudgetConfig, StandardBudget

pytestmark = pytest.mark.anyio


class _SlowModel:
    """Each call costs 1000 tokens / $1 and takes a little while."""

    name = "slow"

    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.fail = fail

    async def complete(self, messages: Any, **kwargs: Any) -> Any:
        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await anyio.sleep(0.02)
            if self.fail:
                raise RuntimeError("provider down")
            return (
                "ok",
                [],
                Usage(input_tokens=900, output_tokens=100, cost_usd=1.0),
                "stop",
            )
        finally:
            self.in_flight -= 1

    async def stream(self, messages: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def _agent(model: Any, budget: Any) -> Agent:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Agent("hi", model=model, budget=budget)


async def _run_concurrently(agent: Agent, n: int, user_id: str) -> list[Any]:
    results: list[Any] = []

    async def one(i: int) -> None:
        try:
            results.append(await agent.run(f"p{i}", user_id=user_id))
        except Exception as exc:  # noqa: BLE001 — collected for assertions
            results.append(exc)

    async with anyio.create_task_group() as tg:
        for i in range(n):
            tg.start_soon(one, i)
    return results


@pytest.mark.parametrize(
    "cfg",
    [
        BudgetConfig(per_user_max_tokens=1000),
        BudgetConfig(per_user_max_cost_usd=1.0),
        BudgetConfig(max_tokens=1000),
        BudgetConfig(max_cost_usd=1.0),
        BudgetConfig(per_user_max_input_tokens=900),
        BudgetConfig(max_output_tokens=100),
    ],
)
async def test_concurrent_runs_cannot_multiply_a_cap(cfg: BudgetConfig) -> None:
    budget = StandardBudget(cfg)
    model = _SlowModel()
    agent = _agent(model, budget)
    with anyio.fail_after(20):
        results = await _run_concurrently(agent, 50, "mallory")
    # One call's worth exhausts the cap; 50 concurrent runs used to
    # spend 50x (50_000 tokens / $50).
    assert model.calls == 1
    assert budget.usage_for("mallory")["tokens_total"] == 1000
    blocked = [r for r in results if getattr(r, "interrupted", False)]
    assert len(blocked) == 49
    assert all(str(r.interruption_reason).startswith("budget:") for r in blocked)


async def test_cap_overshoot_is_bounded_by_one_call() -> None:
    # Room for 3.5 calls: at most one call may straddle the cap.
    budget = StandardBudget(BudgetConfig(per_user_max_tokens=3500))
    model = _SlowModel()
    agent = _agent(model, budget)
    with anyio.fail_after(20):
        await _run_concurrently(agent, 40, "mallory")
    used = budget.usage_for("mallory")["tokens_total"]
    assert 3500 <= used <= 3500 + 1000
    assert model.calls == 4


async def test_headroom_still_allows_real_concurrency() -> None:
    budget = StandardBudget(BudgetConfig(per_user_max_tokens=1_000_000))
    model = _SlowModel()
    agent = _agent(model, budget)
    await agent.run("warm up", user_id="u")  # teaches the per-call estimate
    with anyio.fail_after(20):
        results = await _run_concurrently(agent, 20, "u")
    assert all(not r.interrupted for r in results)
    assert model.calls == 21
    assert model.max_in_flight > 1  # not serialised when there is room


async def test_users_do_not_wait_on_each_others_per_user_caps() -> None:
    budget = StandardBudget(BudgetConfig(per_user_max_tokens=1000))
    model = _SlowModel()
    agent = _agent(model, budget)
    results: list[Any] = []

    async def one(uid: str) -> None:
        results.append(await agent.run("p", user_id=uid))

    with anyio.fail_after(20):
        async with anyio.create_task_group() as tg:
            for i in range(10):
                tg.start_soon(one, f"user-{i}")
    assert all(not r.interrupted for r in results)
    assert model.calls == 10
    assert model.max_in_flight > 1


async def test_reservation_is_released_when_the_call_fails() -> None:
    budget = StandardBudget(BudgetConfig(per_user_max_tokens=1000))
    failing = _agent(_SlowModel(fail=True), budget)
    with anyio.fail_after(20):
        results = await _run_concurrently(failing, 5, "u")
    assert budget._holders == set()  # nothing left reserved
    assert budget.usage_for("u")["tokens_total"] == 0
    assert results  # every run finished (failed or was blocked), none hung

    # The failed calls consumed nothing, so the user still has budget.
    ok = _agent(_SlowModel(), budget)
    with anyio.fail_after(20):
        result = await ok.run("p", user_id="u")
    assert not result.interrupted
    assert budget.usage_for("u")["tokens_total"] == 1000


async def _hold_reservation(
    budget: StandardBudget, holding: anyio.Event, release: anyio.Event
) -> None:
    """A separate run that reserves the headroom and sits on it."""
    async with budget.running(user_id="holder"):
        assert (await budget.allows_step(user_id="holder")).state == "ok"
        holding.set()
        await release.wait()


async def test_reservation_is_released_on_cancellation() -> None:
    budget = StandardBudget(BudgetConfig(max_tokens=1000))
    holding, release = anyio.Event(), anyio.Event()
    waited: list[bool] = []
    with anyio.fail_after(10):
        async with anyio.create_task_group() as tg:
            tg.start_soon(_hold_reservation, budget, holding, release)
            await holding.wait()
            assert len(budget._holders) == 1

            # The headroom is fully reserved by the holder: another
            # run's check waits rather than passing.
            with anyio.move_on_after(0.05) as scope:
                async with budget.running(user_id="b"):
                    await budget.allows_step(user_id="b")
            waited.append(scope.cancelled_caught)

            # Cancel the holder mid-"call": its reservation must go.
            tg.cancel_scope.cancel()
    assert waited == [True]
    assert budget._holders == set()
    async with budget.running(user_id="c"):
        with anyio.fail_after(5):
            assert (await budget.allows_step(user_id="c")).state == "ok"


async def test_waiter_proceeds_when_the_holder_settles() -> None:
    budget = StandardBudget(BudgetConfig(max_tokens=1000))
    holding, release = anyio.Event(), anyio.Event()
    with anyio.fail_after(10):
        async with anyio.create_task_group() as tg:
            tg.start_soon(_hold_reservation, budget, holding, release)
            await holding.wait()

            async def _settle() -> None:
                await anyio.sleep(0.05)
                release.set()  # the holder gives up without consuming

            async with budget.running(user_id="b"):
                tg.start_soon(_settle)
                status = await budget.allows_step(user_id="b")
    assert status.state == "ok"


async def test_waiter_times_out_blocked_instead_of_hanging() -> None:
    budget = StandardBudget(BudgetConfig(max_tokens=1000), reserve_wait_seconds=0.05)
    holding, release = anyio.Event(), anyio.Event()
    with anyio.fail_after(10):
        async with anyio.create_task_group() as tg:
            tg.start_soon(_hold_reservation, budget, holding, release)
            await holding.wait()
            async with budget.running(user_id="b"):
                status = await budget.allows_step(user_id="b")
            release.set()
    assert status.blocked and status.reason == "max_tokens"


async def test_sub_run_does_not_deadlock_on_its_parents_reservation() -> None:
    budget = StandardBudget(BudgetConfig(max_tokens=10_000))
    with anyio.fail_after(5):
        async with budget.running(user_id="u"):
            assert (await budget.allows_step(user_id="u")).state == "ok"
            # A sub-agent sharing the budget, run before the parent consumes.
            async with budget.running(user_id="u"):
                assert (await budget.allows_step(user_id="u")).state == "ok"
                await budget.consume(
                    tokens_in=10, tokens_out=10, cost_usd=0.0, user_id="u"
                )
            await budget.consume(tokens_in=10, tokens_out=10, cost_usd=0.0, user_id="u")
    assert budget._holders == set()


async def test_direct_use_without_running_keeps_old_semantics() -> None:
    budget = StandardBudget(BudgetConfig(max_tokens=100))
    assert (await budget.allows_step()).state == "ok"
    assert (await budget.allows_step()).state == "ok"  # no reservation, no wait
    await budget.consume(tokens_in=60, tokens_out=30, cost_usd=0.0)
    assert (await budget.allows_step()).state == "warn"
    await budget.consume(tokens_in=10, tokens_out=0, cost_usd=0.0)
    status = await budget.allows_step()
    assert status.blocked and status.reason == "max_tokens"


async def test_legacy_two_method_budget_still_works() -> None:
    """A custom Budget with only allows_step / consume (and no
    user_id kwarg) is still driven correctly by the agent loop."""

    class Legacy:
        def __init__(self) -> None:
            self.checks = 0
            self.spent = 0

        async def allows_step(self) -> BudgetStatus:
            self.checks += 1
            if self.spent >= 1000:
                return BudgetStatus.blocked_("legacy_cap")
            return BudgetStatus.ok_()

        async def consume(
            self, *, tokens_in: int, tokens_out: int, cost_usd: float
        ) -> None:
            self.spent += tokens_in + tokens_out

    legacy = Legacy()
    agent = _agent(_SlowModel(), legacy)
    first = await agent.run("a", user_id="u")
    second = await agent.run("b", user_id="u")
    assert not first.interrupted
    assert second.interrupted and second.interruption_reason == "budget:legacy_cap"
    assert legacy.spent == 1000 and legacy.checks >= 2
