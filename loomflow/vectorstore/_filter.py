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

# Value semantics (the cross-backend contract)

The same rules hold in every backend (InMemory / FAISS evaluate this
module directly; Postgres and Chroma translate to it):

* **Types are strict.** A number never equals a numeric string
  (``{"year": 2020}`` does not match ``"2020"``), and a bool never
  equals a number (``{"flag": 1}`` does not match ``True``). ``1`` and
  ``1.0`` are the same number.
* **None means "no value".** ``{"k": None}`` (and ``$eq`` /
  ``$in: [None]``) matches chunks where ``k`` is missing *or*
  ``None``; ``{"k": {"$ne": None}}`` matches chunks that have a
  value. Use ``$exists`` to ask about the key itself.
* **Lists match by element.** If the stored value is a list,
  ``$eq`` / shorthand equality matches when any element equals the
  operand, and ``$in`` when any element is in the supplied list (so
  ``{"headers": {"$in": ["Intro"]}}`` finds chunks under that
  heading). ``{"$eq": [...]}`` compares the whole list.
* **Ordering** (``$gt`` / ``$gte`` / ``$lt`` / ``$lte``) compares a
  number with a number, or a string with a string (by code point);
  any other pairing — including a missing key, ``None``, bools and
  lists — simply doesn't match.
* A missing key is "not equal" for ``$ne`` / ``$nin`` / ``$not``.

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
    explicitly None" so that ``$exists`` works correctly.
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
        return _in(actual, condition)

    # Shorthand: scalar → $eq
    return values_equal(actual, condition)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _scalar_equal(actual: Any, expected: Any) -> bool:
    """Type-strict equality: bools only equal bools, numbers only
    numbers (``1 == 1.0``), strings only strings."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return (
            isinstance(actual, bool)
            and isinstance(expected, bool)
            and actual == expected
        )
    if _is_number(actual) != _is_number(expected):
        return False
    return bool(actual == expected)


def values_equal(actual: Any, expected: Any) -> bool:
    """``$eq`` under the shared contract (see the module docstring):
    ``None`` matches a missing / ``None`` value, a stored list matches
    by element, everything else is type-strict."""
    if expected is None:
        return actual is None
    if actual is None:
        return False
    if isinstance(actual, list | tuple) and not isinstance(expected, list | tuple):
        return any(_scalar_equal(item, expected) for item in actual)
    if isinstance(actual, list | tuple):
        return list(actual) == list(expected)
    return _scalar_equal(actual, expected)


def _in(actual: Any, expected: Any) -> bool:
    return any(values_equal(actual, item) for item in expected)


def _ordered(actual: Any, expected: Any, fn: Any) -> bool:
    """Ordering comparison: number vs number or string vs string;
    any other pairing doesn't match (never raises)."""
    if _is_number(actual) and _is_number(expected):
        return bool(fn(actual, expected))
    if isinstance(actual, str) and isinstance(expected, str):
        return bool(fn(actual, expected))
    return False


def _apply_op(
    op: str, actual: Any, expected: Any, present: bool
) -> bool:
    if op == "$eq":
        return values_equal(actual, expected)
    if op == "$ne":
        # NB: missing key is "not equal to anything" (but None).
        return not values_equal(actual, expected)
    if op == "$gt":
        return _ordered(actual, expected, lambda a, b: a > b)
    if op == "$gte":
        return _ordered(actual, expected, lambda a, b: a >= b)
    if op == "$lt":
        return _ordered(actual, expected, lambda a, b: a < b)
    if op == "$lte":
        return _ordered(actual, expected, lambda a, b: a <= b)
    if op == "$in":
        if not isinstance(expected, list | tuple):
            raise FilterError("$in expects a list")
        return _in(actual, expected)
    if op == "$nin":
        if not isinstance(expected, list | tuple):
            raise FilterError("$nin expects a list")
        return not _in(actual, expected)
    if op == "$exists":
        return bool(expected) == present
    raise FilterError(f"Unknown field operator: {op}")
