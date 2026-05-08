"""M10.3 — observability for AutoExtractMemory.

Two new behaviours we need to lock in with tests:

1. When ``telemetry=`` is provided, every extraction emits a
   ``jeeves.auto_extract.duration_ms`` histogram and a
   ``jeeves.auto_extract.invocations`` counter — tagged with
   ``user_id`` and ``status`` so dashboards can slice by tenant /
   failure rate.
2. When ``auto_picked=True``, a one-time-per-process startup notice
   tells the user auto-extract is on by default. Without that ops
   teams only learn about it from their LLM bills.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from jeevesagent.core.types import Episode, Fact
from jeevesagent.memory.auto_extract import AutoExtractMemory, _log
from jeevesagent.memory.consolidator import Consolidator
from jeevesagent.memory.facts import InMemoryFactStore
from jeevesagent.memory.inmemory import InMemoryMemory

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _RecordingTelemetry:
    """Minimal Telemetry that records every metric for assertions."""

    def __init__(self) -> None:
        self.metrics: list[tuple[str, float, dict[str, Any]]] = []

    async def emit_metric(self, name: str, value: float, **attrs: Any) -> None:
        self.metrics.append((name, value, dict(attrs)))

    async def trace(self, name: str, **attrs: Any) -> Any:  # pragma: no cover
        # Not used by AutoExtractMemory; included for protocol-shape
        # parity if the test ever needs to plug it elsewhere.
        raise NotImplementedError


class _FixedExtractConsolidator(Consolidator):
    """Consolidator that always emits one synthetic fact, deterministically.

    Subclasses :class:`Consolidator` to keep type compatibility but
    swaps in a no-network ``consolidate`` so tests never call the
    real LLM-backed extractor.
    """

    def __init__(self, *, raise_exc: BaseException | None = None) -> None:
        # Skip parent __init__; we don't need a model.
        self._raise = raise_exc

    async def consolidate(
        self, episodes: list[Episode], *, store: Any
    ) -> None:
        if self._raise is not None:
            raise self._raise
        for ep in episodes:
            await store.append(
                Fact(
                    subject=ep.user_id or "anon",
                    predicate="said",
                    object=ep.input[:32],
                    confidence=1.0,
                    user_id=ep.user_id,
                )
            )


def _make_inner() -> InMemoryMemory:
    inner = InMemoryMemory()
    inner.facts = InMemoryFactStore()
    return inner


# ---------------------------------------------------------------------------
# Telemetry: duration + invocation metrics
# ---------------------------------------------------------------------------


async def test_emits_duration_metric_with_user_id_and_ok_status() -> None:
    tel = _RecordingTelemetry()
    mem = AutoExtractMemory(
        _make_inner(),
        _FixedExtractConsolidator(),
        telemetry=tel,
    )
    await mem.remember(
        Episode(session_id="s", input="hi", output="hi", user_id="alice")
    )

    names = [m[0] for m in tel.metrics]
    assert "jeeves.auto_extract.duration_ms" in names
    assert "jeeves.auto_extract.invocations" in names

    duration = next(
        m for m in tel.metrics if m[0] == "jeeves.auto_extract.duration_ms"
    )
    invocation = next(
        m for m in tel.metrics if m[0] == "jeeves.auto_extract.invocations"
    )
    assert duration[1] >= 0.0  # wall time in ms, non-negative
    assert duration[2]["user_id"] == "alice"
    assert duration[2]["status"] == "ok"
    assert invocation[1] == 1
    assert invocation[2]["status"] == "ok"


async def test_emits_error_status_when_consolidator_fails() -> None:
    """Failed extraction must still emit metrics — observability of
    failures is the point. The remember() call still succeeds."""
    tel = _RecordingTelemetry()
    mem = AutoExtractMemory(
        _make_inner(),
        _FixedExtractConsolidator(raise_exc=RuntimeError("boom")),
        telemetry=tel,
    )
    eid = await mem.remember(
        Episode(session_id="s", input="hi", output="hi", user_id="bob")
    )
    assert eid  # write still succeeded

    statuses = {m[2]["status"] for m in tel.metrics}
    assert statuses == {"error"}


async def test_no_telemetry_means_no_metrics_no_crash() -> None:
    """The wrapper must work the same as before when telemetry is
    None — single-tenant code that never wires OTel is fine."""
    mem = AutoExtractMemory(
        _make_inner(),
        _FixedExtractConsolidator(),
        telemetry=None,
    )
    eid = await mem.remember(
        Episode(session_id="s", input="hi", output="hi")
    )
    assert eid


async def test_telemetry_exporter_failure_does_not_fail_remember() -> None:
    """A broken exporter (e.g. OTel collector down) must NOT turn a
    successful extract into a failed remember(). The wrapper
    swallows the exporter exception."""

    class _BrokenTelemetry:
        async def emit_metric(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("collector down")

        async def trace(self, *_: Any, **__: Any) -> Any:  # pragma: no cover
            raise NotImplementedError

    mem = AutoExtractMemory(
        _make_inner(),
        _FixedExtractConsolidator(),
        telemetry=_BrokenTelemetry(),
    )
    eid = await mem.remember(
        Episode(session_id="s", input="hi", output="hi")
    )
    assert eid


# ---------------------------------------------------------------------------
# Default-on startup notice
# ---------------------------------------------------------------------------


async def test_auto_picked_emits_one_time_notice(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the agent default-picks auto-extract on (network-model
    heuristic), the wrapper logs a notice the *first* time it's
    constructed in the process."""
    # The flag is process-wide — reset it so the test is
    # deterministic regardless of other tests' order.
    import jeevesagent.memory.auto_extract as ae_mod

    ae_mod._DEFAULT_ON_NOTICE_EMITTED = False

    with caplog.at_level(logging.INFO, logger=_log.name):
        AutoExtractMemory(
            _make_inner(),
            _FixedExtractConsolidator(),
            auto_picked=True,
        )
        # A second instance in the same process must NOT re-emit.
        AutoExtractMemory(
            _make_inner(),
            _FixedExtractConsolidator(),
            auto_picked=True,
        )

    matching = [r for r in caplog.records if "by default" in r.getMessage()]
    assert len(matching) == 1


async def test_explicit_opt_in_does_not_emit_notice(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Caller passing ``auto_extract=True`` knows what they're doing
    — they don't need the default-on warning."""
    import jeevesagent.memory.auto_extract as ae_mod

    ae_mod._DEFAULT_ON_NOTICE_EMITTED = False

    with caplog.at_level(logging.INFO, logger=_log.name):
        AutoExtractMemory(
            _make_inner(),
            _FixedExtractConsolidator(),
            auto_picked=False,
        )

    matching = [r for r in caplog.records if "by default" in r.getMessage()]
    assert matching == []
