"""Fallback chains across an ordered list of models.

:class:`FallbackModel` wraps N models (usually provider adapters,
each optionally wrapped in
:class:`~loomflow.model.retrying.RetryingModel`). When the current
model fails with a *fallback-worthy* error — decided by the
``fall_on`` predicate, :func:`default_fall_on` by default — the
chain advances to the next model. The last model's errors always
propagate: there is nothing left to fall to.

Composition order matters and is part of the contract::

    FallbackModel([
        RetryingModel(AnthropicModel(...), RetryPolicy()),
        RetryingModel(OpenAIModel(...), RetryPolicy()),
    ])

Retries exhaust on the primary FIRST — ``RetryingModel`` only raises
its final :class:`~loomflow.TransientModelError` after its policy is
spent — and only then does ``FallbackModel`` advance to the
secondary, which gets its own retry budget. Wrapping the other way
round (``RetryingModel(FallbackModel([...]))``) would instead re-run
the whole chain on every retry attempt.

Streaming discipline mirrors ``RetryingModel``: fail over only while
waiting for the FIRST chunk. Once a chunk has been yielded to the
consumer we cannot rewind, so mid-stream errors propagate unchanged
— no silent model switch, no duplicated chunks.

Attribution: ``usage.cost_usd`` is always computed by the adapter
that actually served (each prices with its own name), and
:attr:`FallbackModel.last_served` records that model's name after
every successful call so telemetry can see when the chain failed
over. Capability flags (``supports_native_structured_output``,
``count_tokens``, ...) are delegated to the PRIMARY model — they are
read before we know which member will serve — so keep chains
capability-homogeneous when structured output or token counting is
in play.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from ..core.errors import (
    ConfigError,
    PermanentModelError,
    TransientModelError,
)
from ..core.types import Message, ModelChunk, ToolCall, ToolDef, Usage
from ..governance.retry import classify_model_error

__all__ = ["FallbackModel", "default_fall_on"]


def default_fall_on(exc: Exception) -> bool:
    """Default fallback trigger.

    Fail over on:

    * :class:`~loomflow.TransientModelError` (including
      :class:`~loomflow.RateLimitError`) — rate limits, 5xx /
      overloaded, network blips, per-request timeouts. When the
      chain member is a ``RetryingModel`` these only surface after
      its retry budget is exhausted, so advancing is the correct
      next escalation.
    * *plain* :class:`~loomflow.PermanentModelError` — provider
      statuses the classifier couldn't pin to a subclass (odd
      4xx/5xx); a different provider may well not share them.

    Never fail over on:

    * ``AuthenticationError`` / ``InvalidRequestError`` /
      ``ContentFilterError`` — caller/config problems another model
      won't fix, and silently rerouting a content-filter rejection
      to a different provider would be a policy bypass.
    * Unclassified exceptions (:func:`classify_model_error` returns
      ``None``) — programming errors should surface, not be papered
      over by a model switch.
    """
    classified = classify_model_error(exc)
    if classified is None:
        return False
    if isinstance(classified, TransientModelError):
        return True
    return type(classified) is PermanentModelError


class FallbackModel:
    """Serve from the first model in the chain that can.

    Anything that quacks like a :class:`~loomflow.Model` (has
    ``name`` + ``stream``, optional ``complete``) works as a chain
    member. Like ``RetryingModel``, ``complete`` is delegated
    directly, so every member should expose the same optional
    surface the caller relies on.

    ``fall_on`` decides, per exception, whether to advance to the
    next model; ``None`` selects :func:`default_fall_on`. The
    wrapper's stable ``name`` is the primary's (telemetry / audit
    consistency); :attr:`last_served` carries the honest
    per-call attribution.
    """

    def __init__(
        self,
        models: Sequence[Any],
        *,
        fall_on: Callable[[Exception], bool] | None = None,
    ) -> None:
        chain = list(models)
        if not chain:
            raise ConfigError("FallbackModel requires at least one model")
        self._models = chain
        self._fall_on = fall_on if fall_on is not None else default_fall_on
        self.name: str = getattr(chain[0], "name", "unknown")
        #: Name of the model that served the most recent successful
        #: call (``None`` until the first success). Cost is already
        #: honest — the serving adapter prices its own usage — this
        #: makes the *routing* visible too.
        self.last_served: str | None = None

    @property
    def models(self) -> list[Any]:
        """The chain, primary first. Copy — mutation-safe."""
        return list(self._models)

    @property
    def supports_native_structured_output(self) -> bool:
        """Delegated to the PRIMARY model (documented simplification).

        The agent loop reads this flag once, before the call — i.e.
        before we know which chain member will serve. Chains mixing
        native / non-native members should therefore rely on the
        loop's prompt-augmentation fallback (leave the primary's
        flag ``False``) or keep the chain homogeneous.
        """
        return bool(
            getattr(self._models[0], "supports_native_structured_output", False)
        )

    def __getattr__(self, item: str) -> Any:
        # Optional-capability duck-typing: methods discovered via
        # ``hasattr`` (e.g. ``count_tokens``) forward to the primary
        # model. Only fires for attributes FallbackModel itself
        # doesn't define. Private names never forward — that would
        # mask genuine bugs (and recurse during __init__).
        if item.startswith("_"):
            raise AttributeError(item)
        models = self.__dict__.get("_models")
        if not models:
            raise AttributeError(item)
        return getattr(models[0], item)

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        output_schema: Any | None = None,
        effort: str | None = None,
        strict_effort: bool = False,
        prompt_caching: Any = None,
    ) -> tuple[str, list[ToolCall], Usage, str]:
        """Single-shot completion with failover.

        Each model is tried in order; a failure matching ``fall_on``
        advances the chain, anything else (and any failure on the
        last model) raises.
        """
        last_idx = len(self._models) - 1
        for idx, model in enumerate(self._models):
            try:
                result: tuple[str, list[ToolCall], Usage, str] = (
                    await model.complete(
                        messages,
                        tools=tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        output_schema=output_schema,
                        effort=effort,
                        strict_effort=strict_effort,
                        prompt_caching=prompt_caching,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — predicate decides
                if idx == last_idx or not self._fall_on(exc):
                    raise
                continue
            self.last_served = getattr(model, "name", "unknown")
            return result
        raise AssertionError("unreachable: chain is never empty")

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        output_schema: Any | None = None,
        effort: str | None = None,
        strict_effort: bool = False,
        prompt_caching: Any = None,
    ) -> AsyncIterator[ModelChunk]:
        """Streaming completion with failover-before-first-chunk.

        Failures while waiting for the first chunk advance the chain
        (when ``fall_on`` matches). Once the first chunk has been
        yielded we are committed to that model: mid-stream errors
        propagate unchanged — a switch would duplicate or drop
        content the consumer already saw.
        """
        last_idx = len(self._models) - 1
        for idx, model in enumerate(self._models):
            iterator = model.stream(
                messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                output_schema=output_schema,
                effort=effort,
                strict_effort=strict_effort,
                prompt_caching=prompt_caching,
            )
            try:
                first: ModelChunk = await iterator.__anext__()
            except StopAsyncIteration:
                # Empty stream — a successful (if silent) serve.
                self.last_served = getattr(model, "name", "unknown")
                return
            except Exception as exc:  # noqa: BLE001 — predicate decides
                if idx == last_idx or not self._fall_on(exc):
                    raise
                continue
            # Past the gate: chunks are out the door.
            self.last_served = getattr(model, "name", "unknown")
            yield first
            async for chunk in iterator:
                yield chunk
            return
