"""Per-request wall-clock timeout helpers for the model adapters.

``request_timeout_s`` on the provider adapters is enforced two ways
(belt and suspenders):

* the SDK-native per-request ``timeout=`` option is forwarded so the
  HTTP layer applies its own connect/read timeouts, and
* the call is bounded by an ``anyio`` wall clock. Non-streaming calls
  run inside a single ``anyio.fail_after``; streaming iteration goes
  through :func:`iter_with_deadline`, which charges every await of
  the next SSE event against one shared absolute deadline. A hung
  stream — the socket stays open but no events ever arrive — is
  killed the moment the deadline passes.

Timeouts surface as
:class:`~loomflow.core.errors.TransientModelError` so
:class:`~loomflow.model.retrying.RetryingModel` retries them and
:class:`~loomflow.model.fallback.FallbackModel` fails over on them.
Mid-stream, the usual streaming discipline applies: once chunks have
been yielded neither wrapper replays, so the error propagates to the
consumer.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

import anyio

from ..core.errors import TransientModelError

__all__ = ["deadline_from", "iter_with_deadline", "timeout_error"]


def deadline_from(timeout_s: float | None) -> float | None:
    """Absolute deadline on anyio's clock, ``timeout_s`` from now.

    ``None`` in, ``None`` out — callers thread the adapter's optional
    ``request_timeout_s`` straight through.
    """
    if timeout_s is None:
        return None
    return anyio.current_time() + timeout_s


def timeout_error(
    label: str, timeout_s: float | None, cause: BaseException
) -> TransientModelError:
    """A classified transient error describing a wall-clock timeout.

    Transient by design: a timed-out request may well succeed on
    retry (or on the next model in a fallback chain).
    """
    return TransientModelError(
        f"{label}: request exceeded request_timeout_s={timeout_s}s",
        cause=cause,
    )


async def iter_with_deadline(
    source: AsyncIterable[Any],
    deadline: float | None,
    label: str,
    timeout_s: float | None,
) -> AsyncIterator[Any]:
    """Yield from ``source``, bounding EVERY next-item await by the
    shared absolute ``deadline``.

    The ``anyio.fail_after`` cancel scope is entered and exited
    entirely between yields (never held across one), so this is safe
    inside async generators. ``deadline=None`` disables the guard
    and yields straight through.
    """
    if deadline is None:
        async for item in source:
            yield item
        return
    iterator = source.__aiter__()
    while True:
        try:
            with anyio.fail_after(max(0.0, deadline - anyio.current_time())):
                item = await iterator.__anext__()
        except StopAsyncIteration:
            return
        except TimeoutError as exc:
            raise timeout_error(label, timeout_s, exc) from exc
        yield item
