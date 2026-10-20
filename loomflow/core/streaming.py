"""Yield what a concurrently running producer sends, safely abandonable.

The obvious shape, an async generator that opens a task group and
yields from inside it, breaks the moment a consumer does
``async for ev in agent.stream(...): break``. The generator is closed
later, from a different task, so the task group's cancel scope is torn
down in the wrong task and the cancellation lands on the *caller*: its
next unrelated ``await`` raises ``CancelledError``. anyio's docs list
"yielding in an async generator while enclosed in a cancel scope" as a
cause of exactly this corruption.

:func:`relay` and :func:`stream_from` keep no cancel scope open across
``yield``. On asyncio the producer runs as its own task, which closing
the generator cancels and then waits for, as a task group would. Other
backends keep a task group (trio has no unstructured spawn), so there
close the stream explicitly, e.g. with ``contextlib.aclosing``, before
abandoning it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing
from typing import TypeVar

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream

T = TypeVar("T")

Send = Callable[[T], Awaitable[None]]


async def relay(
    run: Callable[[], Awaitable[None]],
    receive_stream: MemoryObjectReceiveStream[T],
) -> AsyncGenerator[T, None]:
    """Run ``run()`` concurrently and yield what ``receive_stream`` gets.

    ``run`` must close the stream's senders when it finishes (normally
    or not); the stream then ends. If ``run`` raised, the items already
    sent are yielded first and then its exception is raised as-is.
    """
    async def runner() -> None:
        await run()

    if not _on_asyncio():
        async with anyio.create_task_group() as tg:
            tg.start_soon(runner)
            try:
                async with receive_stream:
                    async for item in receive_stream:
                        yield item
            except GeneratorExit:
                # Closed early. Not re-raised inside the task group,
                # which trio would wrap in an exception group.
                pass
            finally:
                tg.cancel_scope.cancel()
        return

    task = asyncio.ensure_future(runner())
    # A producer cancelled by an abandoned stream may still fail while
    # unwinding; read the result so asyncio doesn't log it as unretrieved.
    task.add_done_callback(lambda t: t.cancelled() or t.exception())
    try:
        async with receive_stream:
            async for item in receive_stream:
                yield item
        await task
    finally:
        if not task.done():
            task.cancel()
            # Let the producer unwind before the stream counts as
            # closed, so its cleanup (cancelled tools, sub-agents) has
            # run by the time ``aclose()`` returns.
            with anyio.CancelScope(shield=True):
                await asyncio.wait({task})


async def stream_from(
    produce: Callable[[Send[T]], Awaitable[None]],
    *,
    buffer_size: int = 128,
) -> AsyncGenerator[T, None]:
    """Run ``produce(send)`` concurrently and yield each item it sends.

    The stream ends when ``produce`` returns. If it raises, the items
    already sent are yielded first and then its exception is raised
    as-is (not wrapped in an ``ExceptionGroup``). A slow consumer
    blocks ``send`` once ``buffer_size`` items are queued.
    """
    send_stream, receive_stream = anyio.create_memory_object_stream[T](buffer_size)

    async def run() -> None:
        async with send_stream:
            await produce(send_stream.send)

    async with aclosing(relay(run, receive_stream)) as items:
        async for item in items:
            yield item


def _on_asyncio() -> bool:
    try:
        return asyncio.current_task() is not None
    except RuntimeError:  # no asyncio loop: trio or another backend
        return False
