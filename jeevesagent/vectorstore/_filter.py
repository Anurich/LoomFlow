"""MongoDB-style metadata filter evaluator.

Used by every in-process backend (InMemory, FAISS post-filter) and as
a translation source for Chroma/Postgres which speak their own
flavor of the same algebra.

# Supported operators

**Comparison** (field-level):

* ``$eq`` — equal
* ``$ne`` — not equal
* ``$gt`` / ``$gte`` — greater than / greater-or-equal
* ``$lt`` / ``$lte`` — less than / less-or-equal
* ``$in`` — value is in the supplied list
* ``$nin`` — value is NOT in the supplied list
* ``$exists`` — key is present (true) or absent (false)

**Logical** (top-level or nested):

* ``$and`` — list of subfilters, all must match
* ``$or`` — list of subfilters, at least one must match
* ``$not`` — negation of a subfilter

# Shorthand

For ergonomics, three shapes are accepted as shorthand for ``$eq`` /
``$in``:

* ``{"key": "value"}``           → ``{"key": {"$eq": "value"}}``
* ``{"key": ["a", "b"]}``         → ``{"key": {"$in": ["a", "b"]}}``
* ``{"key": {"$gte": 5}}``        → operator form, used as-is
* ``{"$and": [{"a": 1}, {"b": 2}]}`` → composition

# Examples

::

    {"source": "report.pdf"}                  # source equals report.pdf
    {"page": {"$gte": 10}}                    # page >= 10
    {"tag": {"$in": ["draft", "final"]}}      # tag in {draft, final}
    {"$or": [{"a": 1}, {"b": 2}]}             # a==1 OR b==2
    {"$and": [
        {"page": {"$gte": 10}},
        {"$not": {"author": "alice"}}
    ]}                                          # page>=10 AND author!=alice
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Operator names — kept here so the per-backend translators (Chroma,
# Postgres) can validate against the canonical set.
COMPARISON_OPERATORS = frozenset(
    {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "$exists"}
)
LOGICAL_OPERATORS = frozenset({"$and", "$or", "$not"})
ALL_OPERATORS = COMPARISON_OPERATORS | LOGICAL_OPERATORS


class FilterError(ValueError):
    """Raised for malformed filter expressions."""


def evaluate_filter(
    filter: Mapping[str, Any] | None,
    metadata: Mapping[str, Any],
) -> bool:
    """Return True if ``metadata`` satisfies ``filter``.

    Empty / None filter always matches. Raises :class:`FilterError`
    on unknown operators or malformed structure.
    """
    if not filter:
        return True
    return _eval_node(filter, metadata)


def _eval_node(
    node: Mapping[str, Any], metadata: Mapping[str, Any]
) -> bool:
    """Evaluate a filter sub-expression. Handles top-level logical
    operators and field-level constraints; descends into nested
    structures."""
    for key, value in node.items():
        if key == "$and":
            if not isinstance(value, list):
                raise FilterError("$and expects a list of subfilters")
            if not all(_eval_node(sub, metadata) for sub in value):
                return False
        elif key == "$or":
            if not isinstance(value, list):
                raise FilterError("$or expects a list of subfilters")
            if not any(_eval_node(sub, metadata) for sub in value):
                return False
        elif key == "$not":
            if not isinstance(value, Mapping):
                raise FilterError("$not expects a subfilter dict")
            if _eval_node(value, metadata):
                return False
        elif key.startswith("$"):
            raise FilterError(f"Unknown operator at top level: {key}")
        else:
            # Field-level constraint.
            if not _eval_field(metadata.get(key), value, key in metadata):
                return False
    return True


def _eval_field(
    actual: Any, condition: Any, present: bool
) -> bool:
    """Evaluate one field-level constraint.

    ``present`` is needed to distinguish "key absent" from "key
    explicitly None" so that ``$exists`` and ``$ne`` work correctly.
    """
    # Operator form: dict whose keys all start with "$".
    if isinstance(condition, Mapping) and condition and all(
        k.startswith("$") for k in condition
    ):
        for op, expected in condition.items():
            if not _apply_op(op, actual, expected, present):
                return False
        return True

    # Shorthand: list → $in
    if isinstance(condition, list | tuple):
        return present and actual in condition

    # Shorthand: scalar → $eq
    return present and actual == condition


def _apply_op(
    op: str, actual: Any, expected: Any, present: bool
) -> bool:
    if op == "$eq":
        return present and actual == expected
    if op == "$ne":
        # NB: missing key is "not equal to anything", so it matches $ne.
        return not present or actual != expected
    if op == "$gt":
        return (
            present
            and actual is not None
            and _safe_cmp(actual, expected, lambda a, b: a > b)
        )
    if op == "$gte":
        return (
            present
            and actual is not None
            and _safe_cmp(actual, expected, lambda a, b: a >= b)
        )
    if op == "$lt":
        return (
            present
            and actual is not None
            and _safe_cmp(actual, expected, lambda a, b: a < b)
        )
    if op == "$lte":
        return (
            present
            and actual is not None
            and _safe_cmp(actual, expected, lambda a, b: a <= b)
        )
    if op == "$in":
        if not isinstance(expected, list | tuple):
            raise FilterError("$in expects a list")
        return present and actual in expected
    if op == "$nin":
        if not isinstance(expected, list | tuple):
            raise FilterError("$nin expects a list")
        return not present or actual not in expected
    if op == "$exists":
        return bool(expected) == present
    raise FilterError(f"Unknown field operator: {op}")


def _safe_cmp(a: Any, b: Any, fn: Any) -> bool:
    """Comparison that returns False (rather than raising) for
    incomparable types (e.g. str vs int). Lets a heterogeneous
    metadata corpus filter cleanly without surfacing TypeErrors."""
    try:
        return bool(fn(a, b))
    except TypeError:
        return False
