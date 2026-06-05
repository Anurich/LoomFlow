"""Cache-hit-rate measurement: ``Usage.cache_hit_rate``,
``RunResult.cache_hit_rate``, and the telemetry metrics emitted on the
hot path.

The cache fields already flowed through both model adapters and onto
``Usage`` / ``RunResult``; this pins the *derived* hit-rate accessors
plus the metric emission (which must stay behind the ``fast_telemetry``
guard so the no-op path costs nothing).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from loomflow import Agent, RunResult, Usage
from loomflow.model.scripted import ScriptedModel, ScriptedTurn
from loomflow.observability.tracing import InMemoryTelemetry, NoTelemetry

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Usage.cache_hit_rate
# ---------------------------------------------------------------------------

def test_usage_hit_rate_all_cached() -> None:
    u = Usage(input_tokens=0, cached_input_tokens=100, cache_write_tokens=0)
    assert u.cache_hit_rate == 1.0


def test_usage_hit_rate_all_miss() -> None:
    u = Usage(input_tokens=100, cached_input_tokens=0, cache_write_tokens=0)
    assert u.cache_hit_rate == 0.0


def test_usage_hit_rate_mixed() -> None:
    # 30 read / (50 miss + 20 write + 30 read) = 30/100
    u = Usage(input_tokens=50, cache_write_tokens=20, cached_input_tokens=30)
    assert u.cache_hit_rate == pytest.approx(0.30)


def test_usage_hit_rate_zero_tokens_is_zero_not_div_error() -> None:
    assert Usage().cache_hit_rate == 0.0


# ---------------------------------------------------------------------------
# RunResult.cache_hit_rate
# ---------------------------------------------------------------------------

def _rr(**kw: object) -> RunResult:
    base = dict(
        output="x",
        session_id="s",
        turns=1,
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
    )
    base.update(kw)
    return RunResult(**base)  # type: ignore[arg-type]


def test_runresult_hit_rate_mixed() -> None:
    r = _rr(tokens_in=50, cache_write_tokens=20, cached_tokens_in=30)
    assert r.cache_hit_rate == pytest.approx(0.30)


def test_runresult_hit_rate_zero_is_safe() -> None:
    assert _rr().cache_hit_rate == 0.0


def test_runresult_still_constructs_minimally() -> None:
    # Back-compat: the new property must not require new fields.
    assert _rr().cache_hit_rate == 0.0


# ---------------------------------------------------------------------------
# Telemetry emission on the hot path
# ---------------------------------------------------------------------------

def _names(tel: InMemoryTelemetry) -> set[str]:
    return {m.name for m in tel.metrics()}


async def test_cache_metrics_emitted_when_telemetry_on() -> None:
    tel = InMemoryTelemetry()
    model = ScriptedModel([
        ScriptedTurn(
            text="done",
            usage=Usage(input_tokens=40, cache_write_tokens=10, cached_input_tokens=50),
        )
    ])
    agent = Agent("you help", model=model, telemetry=tel)
    result = await agent.run("hi")

    names = _names(tel)
    assert "loom.tokens.cached" in names
    assert "loom.tokens.cache_write" in names
    assert "loom.cache.hit_rate" in names
    # Run-level rollup too.
    assert "loom.session.cache_hit_rate" in names

    # The hit-rate metric value matches the accessor: 50/(40+10+50)=0.5
    hit = next(m for m in tel.metrics() if m.name == "loom.cache.hit_rate")
    assert hit.value == pytest.approx(0.5)
    assert result.cache_hit_rate == pytest.approx(0.5)


async def test_no_cache_metrics_when_no_cache_tokens() -> None:
    tel = InMemoryTelemetry()
    model = ScriptedModel([
        ScriptedTurn(text="done", usage=Usage(input_tokens=40, output_tokens=5))
    ])
    agent = Agent("you help", model=model, telemetry=tel)
    await agent.run("hi")

    names = _names(tel)
    # No cache reads/writes → those two counters stay silent...
    assert "loom.tokens.cached" not in names
    assert "loom.tokens.cache_write" not in names
    # ...but hit_rate still fires (input_tokens > 0) and reads 0.0.
    assert "loom.cache.hit_rate" in names
    hit = next(m for m in tel.metrics() if m.name == "loom.cache.hit_rate")
    assert hit.value == 0.0


async def test_fast_path_emits_nothing() -> None:
    # NoTelemetry is the no-op fast path: zero metrics, zero cost.
    tel = NoTelemetry()
    model = ScriptedModel([
        ScriptedTurn(
            text="done",
            usage=Usage(input_tokens=40, cached_input_tokens=50),
        )
    ])
    agent = Agent("you help", model=model, telemetry=tel)
    result = await agent.run("hi")
    # No assertion on tel (it captures nothing) — the point is it
    # runs cleanly and the accessor still works off the result.
    assert result.cache_hit_rate == pytest.approx(50 / 90)
