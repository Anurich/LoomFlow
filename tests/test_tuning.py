"""The :class:`Tuning` config object + its back-compat deprecation shim.

Pins the public contract introduced when the rarely-touched Agent
knobs moved off the flat ``Agent(...)`` signature into one optional
``tuning=Tuning(...)`` dataclass:

* ``Tuning()`` alone is a complete, valid config (every field defaulted).
* Passing ``tuning=`` applies its fields with no warning.
* The pre-0.11 flat form (``Agent(retry_policy=...)``) still works for
  now, but emits a ``DeprecationWarning`` naming the exact fix.
* An explicit ``tuning=`` wins over a legacy flat kwarg of the same name.
* A genuinely unknown kwarg still raises ``TypeError`` — the shim only
  absorbs *recognised* tuning fields, so typos are not swallowed.
* The buried knobs keep their existing validation (e.g. the
  ``>= 0`` / ``>= 1`` bounds) when supplied via ``Tuning``.
"""

from __future__ import annotations

import warnings

import pytest

from loomflow import Agent, Tuning

# Every buried field and a non-default value to round-trip it through.
# (model="echo" keeps construction zero-key and offline.)
_BURIED_SAMPLES = {
    "tool_result_summary_threshold": 123,
    "auto_compact_keep_recent_turns": 7,
    "tool_transcript_max_bytes": 999,
    "max_stop_hook_iterations": 3,
    "auto_consolidate": True,
    "response_tone": "terse",
}


def test_tuning_alone_is_a_valid_config() -> None:
    """``Tuning()`` with no args is a complete config — all defaults."""
    t = Tuning()
    assert t.tool_result_summary_threshold == 500
    assert t.auto_compact_keep_recent_turns == 4
    assert t.tool_transcript_max_bytes == 50_000
    assert t.max_stop_hook_iterations == 15
    assert t.auto_consolidate is False
    assert t.response_tone is None
    assert t.retry_policy is None


def test_plain_agent_emits_no_deprecation_warning() -> None:
    """The common path (no buried knobs) must be silent."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Agent("be helpful", model="echo")
    deps = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deps == []


def test_tuning_kwarg_applies_without_warning() -> None:
    """Passing ``tuning=Tuning(...)`` is the new, non-deprecated path."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        a = Agent(
            "hi", model="echo",
            tuning=Tuning(max_stop_hook_iterations=3, auto_consolidate=True),
        )
    deps = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deps == []
    assert a._max_stop_hook_iterations == 3
    assert a._auto_consolidate is True


@pytest.mark.parametrize("name,value", list(_BURIED_SAMPLES.items()))
def test_legacy_flat_kwarg_warns_and_still_works(name: str, value: object) -> None:
    """Each buried knob, passed flat, warns but is forwarded correctly."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Agent("hi", model="echo", **{name: value})
    deps = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deps) == 1
    msg = str(deps[0].message)
    assert name in msg
    assert "tuning=Tuning(" in msg


def test_explicit_tuning_wins_over_legacy_flat_kwarg() -> None:
    """If both forms set the same field, the explicit ``tuning=`` wins."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = Agent(
            "hi", model="echo",
            tuning=Tuning(max_stop_hook_iterations=3),
            max_stop_hook_iterations=99,  # legacy form, should lose
        )
    assert a._max_stop_hook_iterations == 3


def test_unknown_kwarg_still_raises_type_error() -> None:
    """The shim absorbs only recognised tuning fields; typos must fail."""
    with pytest.raises(TypeError, match="not_a_real_kwarg"):
        Agent("hi", model="echo", not_a_real_kwarg=1)


def test_buried_knob_validation_preserved_via_tuning() -> None:
    """Bounds checks still fire when the knob arrives through Tuning."""
    with pytest.raises(
        ValueError, match="tool_result_summary_threshold must be >= 0"
    ):
        Agent("hi", model="echo", tuning=Tuning(tool_result_summary_threshold=-1))
    with pytest.raises(
        ValueError, match="auto_compact_keep_recent_turns must be >= 1"
    ):
        Agent(
            "hi", model="echo",
            tuning=Tuning(auto_compact_keep_recent_turns=0),
        )
