"""Signal channel tests (H2) — the ``signal``/``deliver`` path is now a
real per-session FIFO queue read by ``wait_for_signal`` / ``poll_signal``
on both :class:`InProcRuntime` and :class:`JournaledRuntime`."""

from __future__ import annotations

from typing import Any

import anyio
import pytest

from loomflow.runtime import InProcRuntime, JournaledRuntime

pytestmark = pytest.mark.anyio


@pytest.fixture(params=["inproc", "journaled"])
def runtime(request: pytest.FixtureRequest) -> InProcRuntime | JournaledRuntime:
    if request.param == "inproc":
        return InProcRuntime()
    return JournaledRuntime()


# ---------------------------------------------------------------------------
# deliver → wait_for_signal round-trip
# ---------------------------------------------------------------------------


async def test_wait_parks_then_wakes_on_signal(
    runtime: InProcRuntime | JournaledRuntime,
) -> None:
    got: list[Any] = []
    waiting = anyio.Event()

    async def waiter() -> None:
        waiting.set()
        got.append(await runtime.wait_for_signal("s1", "approval"))

    with anyio.fail_after(5):
        async with runtime.session("s1"):
            async with anyio.create_task_group() as tg:
                tg.start_soon(waiter)
                await waiting.wait()
                await anyio.sleep(0.01)  # let the waiter actually park
                assert got == []  # parked, not returned early
                await runtime.signal("s1", "approval", {"ok": True})

    assert got == [{"ok": True}]


async def test_signal_before_wait_is_queued(
    runtime: InProcRuntime | JournaledRuntime,
) -> None:
    async with runtime.session("s1"):
        await runtime.signal("s1", "go", "payload")
        with anyio.fail_after(5):
            assert await runtime.wait_for_signal("s1", "go") == "payload"


async def test_session_deliver_feeds_the_same_queue(
    runtime: InProcRuntime | JournaledRuntime,
) -> None:
    async with runtime.session("s1") as sess:
        await sess.deliver("evt", 42)
        assert runtime.poll_signal("s1", "evt") == 42


# ---------------------------------------------------------------------------
# FIFO ordering
# ---------------------------------------------------------------------------


async def test_same_name_signals_queue_fifo(
    runtime: InProcRuntime | JournaledRuntime,
) -> None:
    async with runtime.session("s1"):
        for i in range(3):
            await runtime.signal("s1", "n", i)
        with anyio.fail_after(5):
            assert await runtime.wait_for_signal("s1", "n") == 0
            assert await runtime.wait_for_signal("s1", "n") == 1
            assert runtime.poll_signal("s1", "n") == 2


# ---------------------------------------------------------------------------
# poll_signal is non-blocking
# ---------------------------------------------------------------------------


async def test_poll_signal_returns_none_when_empty(
    runtime: InProcRuntime | JournaledRuntime,
) -> None:
    async with runtime.session("s1"):
        assert runtime.poll_signal("s1", "missing") is None
        await runtime.signal("s1", "x", "p")
        assert runtime.poll_signal("s1", "x") == "p"
        # consumed — a second poll is empty again
        assert runtime.poll_signal("s1", "x") is None
    # unknown session never blocks either
    assert runtime.poll_signal("never-opened", "x") is None


# ---------------------------------------------------------------------------
# isolation
# ---------------------------------------------------------------------------


async def test_cross_session_isolation(
    runtime: InProcRuntime | JournaledRuntime,
) -> None:
    async with runtime.session("a"), runtime.session("b"):
        await runtime.signal("a", "sig", "for-a")
        assert runtime.poll_signal("b", "sig") is None
        assert runtime.poll_signal("a", "sig") == "for-a"


async def test_signal_names_are_isolated_within_a_session(
    runtime: InProcRuntime | JournaledRuntime,
) -> None:
    async with runtime.session("s1"):
        await runtime.signal("s1", "alpha", 1)
        assert runtime.poll_signal("s1", "beta") is None
        assert runtime.poll_signal("s1", "alpha") == 1


# ---------------------------------------------------------------------------
# cleanup on session exit
# ---------------------------------------------------------------------------


async def test_session_state_cleaned_up_on_exit(
    runtime: InProcRuntime | JournaledRuntime,
) -> None:
    async with runtime.session("s1"):
        await runtime.signal("s1", "sig", "pending")
    # last context exited: sessions dict no longer grows, and the
    # undelivered signal went with it.
    assert "s1" not in runtime._sessions
    assert runtime.poll_signal("s1", "sig") is None
    # poll on a cleaned session must not resurrect an entry
    assert "s1" not in runtime._sessions


async def test_nested_contexts_clean_up_only_after_last_exit(
    runtime: InProcRuntime | JournaledRuntime,
) -> None:
    async with runtime.session("s1"):
        async with runtime.session("s1"):
            await runtime.signal("s1", "sig", 1)
        # inner exit: the outer context is still open, state survives
        assert runtime.poll_signal("s1", "sig") == 1
        assert "s1" in runtime._sessions
    assert "s1" not in runtime._sessions


# ---------------------------------------------------------------------------
# concurrency
# ---------------------------------------------------------------------------


async def test_each_delivered_signal_wakes_exactly_one_waiter(
    runtime: InProcRuntime | JournaledRuntime,
) -> None:
    got: list[Any] = []

    async def waiter() -> None:
        got.append(await runtime.wait_for_signal("s1", "job"))

    with anyio.fail_after(5):
        async with runtime.session("s1"):
            async with anyio.create_task_group() as tg:
                tg.start_soon(waiter)
                tg.start_soon(waiter)
                await anyio.sleep(0.01)
                await runtime.signal("s1", "job", "first")
                await runtime.signal("s1", "job", "second")

    assert sorted(got) == ["first", "second"]
