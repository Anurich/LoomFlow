"""Bounded per-key state container — LRU + TTL eviction.

Used by primitives that hold per-``user_id`` state in process
(:class:`~jeevesagent.governance.budget.StandardBudget` for token
totals, :class:`~jeevesagent.memory.InMemoryMemory` for working
blocks). Without bounds these dicts grow until the process OOMs
under multi-tenant production load — eviction is the
disqualifier-fix for "is this actually scalable?".

Two complementary policies, both off by default:

* ``max_keys`` — cap the active key count. Eviction is
  least-recently-touched (LRU); the bucket evicted has its data
  *dropped*, not flushed elsewhere. Callers that need durable
  spill-to-disk should use a backend that persists (SqliteMemory,
  PostgresMemory) instead of relying on the in-process bound.
* ``ttl_seconds`` — drop a bucket when its last touch is older
  than this. Eviction runs lazily on touch (cheap; we already have
  the lock). Callers reading periodically can call
  :meth:`evict_expired` to force a sweep.

Both knobs are tuneable per-construction. The default for both is
``None`` (unbounded, today's behaviour) — opt in via the primitive's
constructor.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Generic, TypeVar, overload

__all__ = ["BoundedDict"]


_K = TypeVar("_K")
_V = TypeVar("_V")
_D = TypeVar("_D")


@dataclass(slots=True)
class _Slot(Generic[_V]):
    value: _V
    last_touched: datetime


class BoundedDict(Generic[_K, _V]):
    """LRU + TTL bounded mapping.

    Behaves like a dict for the typical operations a per-user-state
    holder needs (``__getitem__`` / ``__setitem__`` / ``setdefault``
    / ``__contains__`` / ``__len__`` / ``__iter__``); every access
    refreshes the LRU position and the last-touched timestamp.

    Construct with::

        BoundedDict(max_keys=100_000, ttl_seconds=86_400)

    Both args optional; pass ``None`` for unbounded on either axis.
    Default-empty ``BoundedDict()`` is unbounded — opt into limits
    explicitly so single-tenant code that never hits the bound
    isn't paying the eviction cost.
    """

    __slots__ = ("_data", "_max_keys", "_ttl_seconds")

    def __init__(
        self,
        *,
        max_keys: int | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        if max_keys is not None and max_keys < 1:
            raise ValueError("max_keys must be >= 1 when set")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0 when set")
        self._data: OrderedDict[_K, _Slot[_V]] = OrderedDict()
        self._max_keys = max_keys
        self._ttl_seconds = ttl_seconds

    @property
    def max_keys(self) -> int | None:
        return self._max_keys

    @property
    def ttl_seconds(self) -> float | None:
        return self._ttl_seconds

    # ---- core mapping operations ----------------------------------------

    def __contains__(self, key: object) -> bool:
        self._evict_expired_locked()
        return key in self._data

    def __len__(self) -> int:
        self._evict_expired_locked()
        return len(self._data)

    def __iter__(self) -> Iterator[_K]:
        self._evict_expired_locked()
        return iter(list(self._data.keys()))

    @overload
    def get(self, key: _K) -> _V | None: ...
    @overload
    def get(self, key: _K, default: _V) -> _V: ...
    @overload
    def get(self, key: _K, default: _D) -> _V | _D: ...
    def get(self, key: _K, default: _D | None = None) -> _V | _D | None:
        self._evict_expired_locked()
        slot = self._data.get(key)
        if slot is None:
            return default
        # Touch refreshes LRU + timestamp.
        slot.last_touched = _now()
        self._data.move_to_end(key)
        return slot.value

    def __getitem__(self, key: _K) -> _V:
        self._evict_expired_locked()
        slot = self._data[key]
        slot.last_touched = _now()
        self._data.move_to_end(key)
        return slot.value

    def __setitem__(self, key: _K, value: _V) -> None:
        self._evict_expired_locked()
        existing = self._data.get(key)
        if existing is not None:
            existing.value = value
            existing.last_touched = _now()
            self._data.move_to_end(key)
            return
        self._data[key] = _Slot(value=value, last_touched=_now())
        self._enforce_max_keys_locked()

    def setdefault(self, key: _K, default_factory: _V) -> _V:
        """Return the existing value or insert ``default_factory`` and
        return it. Like ``dict.setdefault`` but the default is the
        already-constructed value (callers with expensive default
        construction can pre-check ``key in self`` instead)."""
        self._evict_expired_locked()
        slot = self._data.get(key)
        if slot is None:
            slot = _Slot(value=default_factory, last_touched=_now())
            self._data[key] = slot
            self._enforce_max_keys_locked()
            return slot.value
        slot.last_touched = _now()
        self._data.move_to_end(key)
        return slot.value

    @overload
    def pop(self, key: _K) -> _V | None: ...
    @overload
    def pop(self, key: _K, default: _V) -> _V: ...
    @overload
    def pop(self, key: _K, default: _D) -> _V | _D: ...
    def pop(self, key: _K, default: _D | None = None) -> _V | _D | None:
        slot = self._data.pop(key, None)
        return slot.value if slot is not None else default

    def items(self) -> list[tuple[_K, _V]]:
        """Snapshot of (key, value) pairs after eviction. Returns a
        list (not a view) so callers can iterate without worrying
        about mutation under their feet."""
        self._evict_expired_locked()
        return [(k, slot.value) for k, slot in self._data.items()]

    def keys(self) -> list[_K]:
        self._evict_expired_locked()
        return list(self._data.keys())

    def values(self) -> list[_V]:
        self._evict_expired_locked()
        return [slot.value for slot in self._data.values()]

    # ---- explicit eviction ---------------------------------------------

    def evict_expired(self) -> int:
        """Force a TTL sweep. Returns the number of keys evicted.

        Most callers don't need this — every read/write triggers a
        lazy sweep already. Use this from a periodic background
        task if your primitive sees long quiet periods (no
        accesses) and you'd rather have the dict shrink predictably
        than wait for the next user request to drive eviction.
        """
        return self._evict_expired_locked()

    # ---- internals -----------------------------------------------------

    def _evict_expired_locked(self) -> int:
        if self._ttl_seconds is None or not self._data:
            return 0
        cutoff = _now()
        ttl = self._ttl_seconds
        # OrderedDict iteration order is insertion order; LRU touches
        # move_to_end. We can't assume the oldest-touched is at the
        # head, so we scan once. ``list()`` materialises so we can
        # mutate the dict mid-iteration.
        evicted = 0
        for key, slot in list(self._data.items()):
            if (cutoff - slot.last_touched).total_seconds() >= ttl:
                self._data.pop(key, None)
                evicted += 1
        return evicted

    def _enforce_max_keys_locked(self) -> None:
        if self._max_keys is None:
            return
        while len(self._data) > self._max_keys:
            # ``OrderedDict.popitem(last=False)`` evicts the
            # least-recently-touched (head) item.
            self._data.popitem(last=False)


def _now() -> datetime:
    return datetime.now(UTC)
