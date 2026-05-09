"""Freshness and lineage policies for :class:`CertifiedValue`.

A :class:`CertifiedValue` (defined in :mod:`loomflow.core.types`)
carries provenance metadata: ``source``, ``fetched_at``, optional
``valid_until``, ``schema_version``, and a tuple of upstream value IDs
in ``lineage``.

This module supplies the policy *types* and *helpers*. Two flavours:

* :class:`FreshnessPolicy` — declare a maximum age per source
  prefix; ``valid_until`` always wins when set on the value itself.
* :class:`LineagePolicy` — declare an allow-list of source prefixes
  every value in a lineage chain must originate from.

Two helper styles for each:

* ``check_*`` returns ``True``/``False`` so callers can branch.
* ``require_*`` raises the appropriate
  :class:`~loomflow.core.errors.FreshnessError` /
  :class:`~loomflow.core.errors.LineageError` so callers can rely
  on exception propagation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from ..core.errors import FreshnessError, LineageError
from ..core.types import CertifiedValue


@dataclass(frozen=True)
class FreshnessPolicy:
    """Maximum age for certified values from each source.

    ``per_source`` maps a source-prefix (matched with ``startswith``)
    to a ``timedelta``. The first prefix that matches wins. ``default``
    is used when no prefix matches; if also ``None``, the policy
    treats all values as fresh.
    """

    per_source: tuple[tuple[str, timedelta], ...] = ()
    default: timedelta | None = None

    def max_age_for(self, source: str) -> timedelta | None:
        for prefix, max_age in self.per_source:
            if source.startswith(prefix):
                return max_age
        return self.default

    @classmethod
    def from_dict(
        cls,
        per_source: dict[str, timedelta] | None = None,
        *,
        default: timedelta | None = None,
    ) -> FreshnessPolicy:
        items = tuple((k, v) for k, v in (per_source or {}).items())
        return cls(per_source=items, default=default)


@dataclass(frozen=True)
class LineagePolicy:
    """Allow-list of source prefixes for the entire lineage chain.

    A :class:`CertifiedValue` is acceptable if every entry in
    ``value.lineage`` (interpreted as a source prefix) starts with one
    of the allowed prefixes.
    """

    allowed_sources: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_iter(cls, sources: list[str] | tuple[str, ...]) -> LineagePolicy:
        return cls(allowed_sources=frozenset(sources))


# ---------------------------------------------------------------------------
# Freshness checks
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(UTC)


def check_freshness(
    value: CertifiedValue,
    policy: FreshnessPolicy,
    *,
    now: datetime | None = None,
) -> bool:
    """Return ``True`` if ``value`` satisfies ``policy`` at ``now``.

    Logic:

    1. If ``valid_until`` is set on the value, fail if ``now > valid_until``.
    2. Look up ``policy.max_age_for(source)``. If ``None`` (no rule),
       the value is fresh by default.
    3. Otherwise fail if ``now - fetched_at > max_age``.
    """
    moment = now if now is not None else _utcnow()
    if value.valid_until is not None and moment > value.valid_until:
        return False
    max_age = policy.max_age_for(value.source)
    if max_age is None:
        return True
    return (moment - value.fetched_at) <= max_age


def require_freshness(
    value: CertifiedValue,
    policy: FreshnessPolicy,
    *,
    now: datetime | None = None,
) -> None:
    """Raise :class:`FreshnessError` when :func:`check_freshness` fails."""
    if not check_freshness(value, policy, now=now):
        moment = now if now is not None else _utcnow()
        age = moment - value.fetched_at
        raise FreshnessError(
            f"value from {value.source!r} is stale "
            f"(age {age}, valid_until={value.valid_until})"
        )


# ---------------------------------------------------------------------------
# Lineage checks
# ---------------------------------------------------------------------------


def check_lineage(value: CertifiedValue, policy: LineagePolicy) -> bool:
    """Return ``True`` if every lineage source is allowed.

    The value's own ``source`` is also required to be in the allow-list
    — there's no point trusting a chain whose tip you don't.
    """
    if policy.allowed_sources and not _allowed(value.source, policy.allowed_sources):
        return False
    for ancestor in value.lineage:
        if not _allowed(ancestor, policy.allowed_sources):
            return False
    return True


def require_lineage(value: CertifiedValue, policy: LineagePolicy) -> None:
    """Raise :class:`LineageError` when :func:`check_lineage` fails."""
    if not check_lineage(value, policy):
        bad = next(
            (anc for anc in value.lineage if not _allowed(anc, policy.allowed_sources)),
            value.source,
        )
        raise LineageError(
            f"lineage source {bad!r} not in allow-list "
            f"({sorted(policy.allowed_sources)})"
        )


def _allowed(source: str, allow: frozenset[str]) -> bool:
    if not allow:
        return True
    return any(source.startswith(prefix) for prefix in allow)
