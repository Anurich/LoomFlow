"""Data certification: provenance, freshness, and lineage policies.

Every value the agent reasons over can carry a :class:`CertifiedValue`
wrapper from :mod:`loomflow.core.types` (already there). This
module adds the *policies* that decide whether a certified value is
acceptable in context.
"""

from .lineage import (
    FreshnessPolicy,
    LineagePolicy,
    check_freshness,
    check_lineage,
    require_freshness,
    require_lineage,
)

__all__ = [
    "FreshnessPolicy",
    "LineagePolicy",
    "check_freshness",
    "check_lineage",
    "require_freshness",
    "require_lineage",
]
