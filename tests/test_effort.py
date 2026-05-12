"""Reasoning-effort dial — per-provider translation + warn/strict."""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from loomflow.model._effort import (
    EffortNotSupportedError,
    _reset_warned_cache_for_tests,
    anthropic_kwargs,
    openai_kwargs,
)


@pytest.fixture(autouse=True)
def _clear_warning_cache() -> None:
    # Module-global ``_WARNED`` set lives across tests; reset between
    # cases so the once-per-pair logic doesn't bleed.
    _reset_warned_cache_for_tests()


# ---------------------------------------------------------------------------
# OpenAI mapping
# ---------------------------------------------------------------------------


class TestOpenAIMapping:
    @pytest.mark.parametrize(
        "model_name",
        ["o1-mini", "o3", "o3-mini", "o4-mini", "gpt-5", "gpt-5-mini"],
    )
    @pytest.mark.parametrize(
        "effort,expected",
        [
            ("minimal", "minimal"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "high"),  # clamped — OpenAI tops out at high
            ("max", "high"),    # clamped
        ],
    )
    def test_supported_models_get_reasoning_effort(
        self, model_name: str, effort: str, expected: str
    ) -> None:
        out = openai_kwargs(effort, model_name, strict=False)
        assert out == {"reasoning_effort": expected}

    def test_none_effort_is_a_noop(self) -> None:
        assert openai_kwargs(None, "o3-mini", strict=False) == {}
        assert openai_kwargs(None, "gpt-4o", strict=False) == {}

    def test_unsupported_model_warns_and_drops(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = openai_kwargs("high", "gpt-4o", strict=False)
        assert out == {}
        assert any("does not support" in str(w.message) for w in caught)

    def test_strict_raises_on_unsupported_model(self) -> None:
        with pytest.raises(EffortNotSupportedError):
            openai_kwargs("high", "gpt-4o", strict=True)

    def test_warning_fires_only_once_per_pair(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            openai_kwargs("high", "gpt-4o", strict=False)
            openai_kwargs("high", "gpt-4o", strict=False)
            openai_kwargs("high", "gpt-4o", strict=False)
        relevant = [w for w in caught if "does not support" in str(w.message)]
        assert len(relevant) == 1

    def test_different_effort_gets_its_own_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            openai_kwargs("high", "gpt-4o", strict=False)
            openai_kwargs("low", "gpt-4o", strict=False)
        relevant = [w for w in caught if "does not support" in str(w.message)]
        assert len(relevant) == 2


# ---------------------------------------------------------------------------
# Anthropic mapping
# ---------------------------------------------------------------------------


class TestAnthropicMapping:
    @pytest.mark.parametrize(
        "model_name", ["claude-opus-4-7", "claude-opus-mythos"]
    )
    def test_opus_4_7_is_adaptive_only(self, model_name: str) -> None:
        out = anthropic_kwargs("high", model_name, strict=False)
        assert out == {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        }

    def test_opus_4_7_keeps_xhigh_and_max(self) -> None:
        # 4.7 / Mythos is the ONLY regime that accepts xhigh / max.
        for value in ("xhigh", "max"):
            out = anthropic_kwargs(value, "claude-opus-4-7", strict=False)
            assert out["output_config"]["effort"] == value

    @pytest.mark.parametrize(
        "model_name", ["claude-opus-4-6", "claude-sonnet-4-6"]
    )
    def test_4_6_clamps_xhigh_to_high(self, model_name: str) -> None:
        out = anthropic_kwargs("xhigh", model_name, strict=False)
        assert out == {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        }

    def test_4_6_passes_normal_efforts_through(self) -> None:
        out = anthropic_kwargs("medium", "claude-sonnet-4-6", strict=False)
        assert out["output_config"]["effort"] == "medium"

    @pytest.mark.parametrize(
        "model_name,effort,expected_budget",
        [
            ("claude-sonnet-4-5", "minimal", 1024),
            ("claude-sonnet-4-5", "low", 2048),
            ("claude-sonnet-4-5", "medium", 4096),
            ("claude-sonnet-4-5", "high", 8192),
            ("claude-sonnet-4-5", "xhigh", 16384),
            ("claude-sonnet-4-5", "max", 32768),
            ("claude-sonnet-3-7", "high", 8192),
            ("claude-sonnet-4", "medium", 4096),
        ],
    )
    def test_legacy_models_get_budget_tokens(
        self, model_name: str, effort: str, expected_budget: int
    ) -> None:
        out = anthropic_kwargs(effort, model_name, strict=False)
        assert out == {
            "thinking": {"type": "enabled", "budget_tokens": expected_budget}
        }

    def test_unsupported_model_warns_and_drops(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = anthropic_kwargs("high", "claude-haiku-3-5", strict=False)
        assert out == {}
        assert any("does not support" in str(w.message) for w in caught)

    def test_strict_raises_on_unsupported_model(self) -> None:
        with pytest.raises(EffortNotSupportedError):
            anthropic_kwargs("high", "claude-haiku-3-5", strict=True)

    def test_none_effort_is_a_noop(self) -> None:
        assert anthropic_kwargs(None, "claude-opus-4-7", strict=False) == {}
        assert anthropic_kwargs(None, "claude-haiku-3-5", strict=False) == {}


# ---------------------------------------------------------------------------
# Agent kwarg surface
# ---------------------------------------------------------------------------


class TestAgentSurface:
    """The Agent forwards ``effort`` per-call and stores a default."""

    def test_agent_accepts_effort_default(self) -> None:
        from loomflow import Agent

        agent = Agent("test", model="echo", effort="high")
        assert agent._default_effort == "high"
        assert agent._strict_effort is False

    def test_agent_accepts_strict_effort(self) -> None:
        from loomflow import Agent

        agent = Agent("test", model="echo", effort="high", strict_effort=True)
        assert agent._strict_effort is True

    @pytest.mark.anyio
    async def test_per_call_effort_overrides_default(self) -> None:
        from loomflow import Agent

        captured: dict[str, Any] = {}

        class _CapturingEcho:
            name = "echo"

            async def complete(
                self,
                messages: list[Any],
                **kwargs: Any,
            ) -> tuple[str, list[Any], Any, str]:
                captured.update(kwargs)
                from loomflow.core.types import Usage
                return "ok", [], Usage(), "stop"

            async def stream(  # pragma: no cover — complete path wins
                self, messages: list[Any], **kwargs: Any
            ) -> Any:
                captured.update(kwargs)
                from loomflow.core.types import ModelChunk, Usage
                yield ModelChunk(
                    kind="finish",
                    finish_reason="stop",
                    usage=Usage(),
                )

        agent = Agent("test", model=_CapturingEcho(), effort="low")
        await agent.run("hi", effort="high")
        assert captured.get("effort") == "high"

    def test_dict_model_spec_sets_effort_default(self) -> None:
        from loomflow import Agent

        agent = Agent(
            "test",
            model={"name": "echo", "effort": "high"},
        )
        assert agent._default_effort == "high"
        assert agent.model.name == "echo"

    def test_dict_model_spec_sets_strict_effort(self) -> None:
        from loomflow import Agent

        agent = Agent(
            "test",
            model={"name": "echo", "strict_effort": True},
        )
        assert agent._strict_effort is True

    def test_dict_model_spec_accepts_model_alias_for_name(self) -> None:
        from loomflow import Agent

        agent = Agent(
            "test",
            model={"model": "echo", "effort": "low"},
        )
        assert agent.model.name == "echo"
        assert agent._default_effort == "low"

    def test_explicit_top_level_effort_wins_over_dict(self) -> None:
        from loomflow import Agent

        # Dict says "low", explicit kwarg says "high" — explicit wins.
        agent = Agent(
            "test",
            model={"name": "echo", "effort": "low"},
            effort="high",
        )
        assert agent._default_effort == "high"

    def test_dict_without_name_raises(self) -> None:
        import pytest as _pytest

        from loomflow import Agent
        from loomflow.core.errors import ConfigError

        with _pytest.raises(ConfigError, match="requires a 'name' key"):
            Agent("t", model={"effort": "high"})

    def test_dict_with_unknown_key_raises(self) -> None:
        import pytest as _pytest

        from loomflow import Agent
        from loomflow.core.errors import ConfigError

        with _pytest.raises(ConfigError, match="unknown key"):
            Agent("t", model={"name": "echo", "bogus": 1})

    def test_dict_with_both_name_and_model_raises(self) -> None:
        import pytest as _pytest

        from loomflow import Agent
        from loomflow.core.errors import ConfigError

        with _pytest.raises(ConfigError, match="pick one"):
            Agent("t", model={"name": "echo", "model": "echo"})

    @pytest.mark.anyio
    async def test_agent_default_effort_used_when_no_per_call(self) -> None:
        from loomflow import Agent

        captured: dict[str, Any] = {}

        class _CapturingEcho:
            name = "echo"

            async def complete(
                self, messages: list[Any], **kwargs: Any
            ) -> tuple[str, list[Any], Any, str]:
                captured.update(kwargs)
                from loomflow.core.types import Usage
                return "ok", [], Usage(), "stop"

            async def stream(  # pragma: no cover — complete path wins
                self, messages: list[Any], **kwargs: Any
            ) -> Any:
                captured.update(kwargs)
                from loomflow.core.types import ModelChunk, Usage
                yield ModelChunk(
                    kind="finish",
                    finish_reason="stop",
                    usage=Usage(),
                )

        agent = Agent("test", model=_CapturingEcho(), effort="medium")
        await agent.run("hi")
        assert captured.get("effort") == "medium"


# ---------------------------------------------------------------------------
# LiteLLM passthrough
# ---------------------------------------------------------------------------


class TestLiteLLMMapping:
    """LiteLLM normalises reasoning_effort across providers, so the
    override always emits the kwarg without warning."""

    def test_litellm_emits_reasoning_effort_for_any_model(self) -> None:
        pytest.importorskip("litellm")
        from loomflow.model.litellm import LiteLLMModel

        m = LiteLLMModel.__new__(LiteLLMModel)
        m.name = "mistral-large"  # type: ignore[attr-defined]
        out = m._effort_kwargs("high", strict_effort=False)
        assert out == {"reasoning_effort": "high"}

    def test_litellm_clamps_xhigh_and_max(self) -> None:
        pytest.importorskip("litellm")
        from loomflow.model.litellm import LiteLLMModel

        m = LiteLLMModel.__new__(LiteLLMModel)
        m.name = "command-r-plus"  # type: ignore[attr-defined]
        assert m._effort_kwargs("xhigh", strict_effort=False) == {
            "reasoning_effort": "high"
        }
        assert m._effort_kwargs("max", strict_effort=False) == {
            "reasoning_effort": "high"
        }

    def test_litellm_none_is_noop(self) -> None:
        pytest.importorskip("litellm")
        from loomflow.model.litellm import LiteLLMModel

        m = LiteLLMModel.__new__(LiteLLMModel)
        m.name = "mistral-large"  # type: ignore[attr-defined]
        assert m._effort_kwargs(None, strict_effort=False) == {}
