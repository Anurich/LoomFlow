"""Freshness and lineage policy tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from loomflow.core.errors import FreshnessError, LineageError
from loomflow.core.types import CertifiedValue
from loomflow.data import FreshnessPolicy, LineagePolicy
from loomflow.data.lineage import (
    check_freshness,
    check_lineage,
    require_freshness,
    require_lineage,
)


def _value(
    *,
    source: str = "gmail/messages",
    fetched_minutes_ago: int = 0,
    valid_until: datetime | None = None,
    lineage: tuple[str, ...] = (),
) -> CertifiedValue:
    fetched = datetime.now(UTC) - timedelta(minutes=fetched_minutes_ago)
    return CertifiedValue(
        value="anything",
        source=source,
        fetched_at=fetched,
        valid_until=valid_until,
        lineage=lineage,
    )


# ---------------------------------------------------------------------------
# FreshnessPolicy
# ---------------------------------------------------------------------------


def test_freshness_no_rule_means_fresh() -> None:
    policy = FreshnessPolicy()  # empty rules
    assert check_freshness(_value(fetched_minutes_ago=999), policy)


def test_freshness_default_rule_applies_when_no_prefix_matches() -> None:
    policy = FreshnessPolicy(default=timedelta(minutes=5))
    assert check_freshness(_value(fetched_minutes_ago=2), policy)
    assert not check_freshness(_value(fetched_minutes_ago=10), policy)


def test_freshness_per_source_rule_wins_over_default() -> None:
    policy = FreshnessPolicy.from_dict(
        per_source={"gmail/": timedelta(minutes=2)},
        default=timedelta(hours=1),
    )
    # gmail/ has 2-minute max age; 5 minutes is stale.
    assert not check_freshness(
        _value(source="gmail/messages", fetched_minutes_ago=5), policy
    )
    # Different source falls under default 1h.
    assert check_freshness(
        _value(source="drive/files", fetched_minutes_ago=5), policy
    )


def test_freshness_valid_until_overrides_age_check() -> None:
    policy = FreshnessPolicy(default=timedelta(days=365))
    expired = _value(
        fetched_minutes_ago=1,
        valid_until=datetime.now(UTC) - timedelta(seconds=1),
    )
    assert not check_freshness(expired, policy)


def test_freshness_first_matching_prefix_wins() -> None:
    policy = FreshnessPolicy(
        per_source=(
            ("gmail/important", timedelta(minutes=1)),
            ("gmail/", timedelta(minutes=10)),
        ),
        default=timedelta(hours=1),
    )
    val = _value(source="gmail/important/x", fetched_minutes_ago=5)
    # The "important" rule (1 minute) matches first; 5min > 1min.
    assert not check_freshness(val, policy)


def test_require_freshness_raises_on_stale_value() -> None:
    policy = FreshnessPolicy(default=timedelta(minutes=1))
    val = _value(fetched_minutes_ago=10)
    with pytest.raises(FreshnessError, match="stale"):
        require_freshness(val, policy)


def test_require_freshness_silent_on_fresh_value() -> None:
    policy = FreshnessPolicy(default=timedelta(minutes=10))
    require_freshness(_value(fetched_minutes_ago=5), policy)


# ---------------------------------------------------------------------------
# LineagePolicy
# ---------------------------------------------------------------------------


def test_empty_lineage_policy_accepts_everything() -> None:
    policy = LineagePolicy()
    val = _value(source="random/source", lineage=("anywhere",))
    assert check_lineage(val, policy)


def test_lineage_allows_when_all_sources_match_prefix() -> None:
    policy = LineagePolicy.from_iter(["gmail/", "drive/"])
    val = _value(
        source="gmail/messages",
        lineage=("gmail/messages", "drive/files"),
    )
    assert check_lineage(val, policy)


def test_lineage_denies_when_ancestor_not_allowed() -> None:
    policy = LineagePolicy.from_iter(["gmail/"])
    val = _value(
        source="gmail/messages",
        lineage=("gmail/messages", "untrusted/source"),
    )
    assert not check_lineage(val, policy)


def test_lineage_denies_when_value_source_not_allowed() -> None:
    policy = LineagePolicy.from_iter(["gmail/"])
    val = _value(source="hacker/feed", lineage=("gmail/x",))
    assert not check_lineage(val, policy)


def test_require_lineage_raises_on_disallowed_source() -> None:
    policy = LineagePolicy.from_iter(["gmail/"])
    val = _value(source="gmail/x", lineage=("untrusted",))
    with pytest.raises(LineageError):
        require_lineage(val, policy)


def test_require_lineage_silent_when_all_allowed() -> None:
    policy = LineagePolicy.from_iter(["gmail/"])
    val = _value(source="gmail/x", lineage=("gmail/y", "gmail/z"))
    require_lineage(val, policy)
