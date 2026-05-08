"""Retry wrapper for any :class:`~jeevesagent.Model`.

:class:`RetryingModel` decorates an underlying model adapter
(``OpenAIModel``, ``AnthropicModel``, ``LiteLLMModel``, custom
implementations) with the framework's retry policy. The agent loop
sees a :class:`Model` like any other; the retry mechanics are
invisible above this layer.

What it does:

* On every :meth:`complete` / :meth:`stream` call, runs the
  underlying model up to :attr:`RetryPolicy.max_attempts` times.
* Catches the underlying SDK exception, runs it through
  :func:`~jeevesagent.governance.classify_model_error`.
* :class:`~jeevesagent.PermanentModelError` (auth, bad request,
  content filter) is re-raised immediately without backoff.
* :class:`~jeevesagent.TransientModelError` (rate limit, 5xx,
  network) is retried after a backoff computed by
  :func:`~jeevesagent.governance.compute_backoff`. Provider-supplied
  ``Retry-After`` hints set a floor on the wait.
* Unrecognised exceptions (anything :func:`classify_model_error`
  returns ``None`` for) propagate unchanged — better to let an
  unknown error bubble up than silently retry it.

Streaming retries are deliberately limited: once the first chunk
has been yielded to the consumer we cannot rewind, so retries only
fire while waiting for that first chunk. Errors mid-stream
propagate.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import anyio

from ..core.errors import (
    ModelError,
    PermanentModelError,
    TransientModelError,
)
from ..core.types import Message, ModelChunk, ToolCall, ToolDef, Usage
from ..governance.retry import (
    RetryPolicy,
    classify_model_error,
    compute_backoff,
)

__all__ = ["RetryingModel"]


class RetryingModel:
    """Wraps any :class:`~jeevesagent.Model` with retry semantics.

    Construction does not validate the inner model — anything that
    quacks like a Model (has ``name``, ``stream``, optional
    ``complete``) works. The wrapper keeps a stable ``name``
    matching the underlying model so telemetry and audit logs stay
    consistent.
    """

    def __init__(self, inner: Any, policy: RetryPolicy) -> None:
        self._inner = inner
        self._policy = policy
        self.name: str = getattr(inner, "name", "unknown")

    @property
    def inner(self) -> Any:
        """The wrapped model. Useful for tests + introspection."""
        return self._inner

    @property
    def policy(self) -> RetryPolicy:
        return self._policy

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> tuple[str, list[ToolCall], Usage, str]:
        """Single-shot completion with retry on transient failures."""

        async def _do_call() -> tuple[str, list[ToolCall], Usage, str]:
            result: tuple[str, list[ToolCall], Usage, str] = (
                await self._inner.complete(
                    messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
            return result

        result: tuple[str, list[ToolCall], Usage, str] = (
            await self._with_retry(_do_call)
        )
        return result

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ModelChunk]:
        """Streaming completion with retry-before-first-chunk.

        We can't roll back chunks already yielded to the consumer,
        so retry behaviour applies to errors that occur *before*
        the first :class:`ModelChunk` is produced. If the underlying
        ``stream`` raises mid-stream the error propagates unchanged.
        """
        attempt = 1
        while True:
            try:
                iterator = self._inner.stream(
                    messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                # First chunk — if this raises we can still retry.
                first_chunk: ModelChunk | None = None
                try:
                    first_chunk = await iterator.__anext__()
                except StopAsyncIteration:
                    return
                # Past the gate; from here on errors propagate.
                yield first_chunk
                async for chunk in iterator:
                    yield chunk
                return
            except Exception as exc:  # noqa: BLE001
                classified = classify_model_error(exc)
                if not isinstance(classified, TransientModelError):
                    if classified is not None:
                        raise classified from exc
                    raise
                if attempt >= self._policy.max_attempts:
                    raise classified from exc
                delay = compute_backoff(
                    self._policy,
                    attempt,
                    retry_after=classified.retry_after,
                )
                await anyio.sleep(delay)
                attempt += 1

    async def _with_retry(self, fn: Any) -> Any:
        """Generic retry loop for non-streaming calls.

        Catches SDK exceptions, classifies, and either retries with
        backoff (transient) or raises (permanent / unknown).
        """
        attempt = 1
        last_transient: TransientModelError | None = None
        while True:
            try:
                return await fn()
            except ModelError as exc:
                # Already classified — handle by family.
                if isinstance(exc, PermanentModelError):
                    raise
                if not isinstance(exc, TransientModelError):
                    raise
                last_transient = exc
                if attempt >= self._policy.max_attempts:
                    raise
                delay = compute_backoff(
                    self._policy,
                    attempt,
                    retry_after=exc.retry_after,
                )
                await anyio.sleep(delay)
                attempt += 1
            except Exception as exc:  # noqa: BLE001
                classified = classify_model_error(exc)
                if classified is None:
                    # Unknown error — don't paper over it.
                    raise
                if isinstance(classified, PermanentModelError):
                    raise classified from exc
                last_transient = (
                    classified
                    if isinstance(classified, TransientModelError)
                    else None
                )
                if attempt >= self._policy.max_attempts:
                    if classified is not None:
                        raise classified from exc
                    raise
                delay = compute_backoff(
                    self._policy,
                    attempt,
                    retry_after=(
                        last_transient.retry_after
                        if last_transient is not None
                        else None
                    ),
                )
                await anyio.sleep(delay)
                attempt += 1
