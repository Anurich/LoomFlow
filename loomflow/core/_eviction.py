"""Bounded per-key state container — LRU + TTL eviction.

Used by primitives that hold per-``user_id`` state in process
(:class:`~loomflow.governance.budget.StandardBudget` for token
totals, :class:`~loomflow.memory.InMemoryMemory` for working
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
  than this. Eviction runs lazily: per-key operations check the
  touched key's expiry inline, and a full O(n) sweep runs (a)
  once every :data:`_SWEEP_EVERY` per-key operations and (b) on
  every O(n) view operation (``len`` / ``iter`` / ``items`` /
  ...). Callers reading rarely can call :meth:`evict_expired` to
  force a sweep.

Both knobs are tuneable per-construction. The default for both is
``None`` (unbounded, today's behaviour) — opt in via the primitive's
constructor.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Generic, TypeVar, cast, overload

__all__ = ["BoundedDict"]


_K = TypeVar("_K")
_V = TypeVar("_V")
_D = TypeVar("_D")

# Per-key operations between amortised full TTL sweeps. Sweeping is
# O(n); doing it on every access made each hot-path ``get``/``set``
# O(n) too. 128 keeps the amortised cost at O(n/128) per op while an
# inline single-key expiry check preserves read correctness in
# between. See ``BoundedDict._maybe_sweep``.
_SWEEP_EVERY = 128


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

    __slots__ = ("_data", "_max_keys", "_ops_since_sweep", "_ttl_seconds")

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
        # Per-key operations since the last full TTL sweep — see
        # ``_maybe_sweep``.
        self._ops_since_sweep = 0

    @property
    def max_keys(self) -> int | None:
        return self._max_keys

    @property
    def ttl_seconds(self) -> float | None:
        return self._ttl_seconds

    # ---- core mapping operations ----------------------------------------

    def __contains__(self, key: object) -> bool:
        self._maybe_sweep()
        return self._live_slot(cast("_K", key)) is not None

    def __len__(self) -> int:
        # Full sweep — ``len`` must not count expired entries, and a
        # correct answer requires visiting every slot anyway.
        self._sweep_expired()
        return len(self._data)

    def __iter__(self) -> Iterator[_K]:
        # Full sweep — iteration is O(n) regardless.
        self._sweep_expired()
        return iter(list(self._data.keys()))

    @overload
    def get(self, key: _K) -> _V | None: ...
    @overload
    def get(self, key: _K, default: _V) -> _V: ...
    @overload
    def get(self, key: _K, default: _D) -> _V | _D: ...
    def get(self, key: _K, default: _D | None = None) -> _V | _D | None:
        self._maybe_sweep()
        slot = self._live_slot(key)
        if slot is None:
            return default
        # Touch refreshes LRU + timestamp.
        slot.last_touched = _now()
        self._data.move_to_end(key)
        return slot.value

    def __getitem__(self, key: _K) -> _V:
        self._maybe_sweep()
        slot = self._live_slot(key)
        if slot is None:
            raise KeyError(key)
        slot.last_touched = _now()
        self._data.move_to_end(key)
        return slot.value

    def __setitem__(self, key: _K, value: _V) -> None:
        self._maybe_sweep()
        existing = self._data.get(key)
        if existing is not None:
            # An expired slot is simply overwritten — value and
            # timestamp both refresh, same outcome as evict + insert.
            existing.value = value
            existing.last_touched = _now()
            self._data.move_to_end(key)
            return
        self._data[key] = _Slot(value=value, last_touched=_now())
        self._enforce_max_keys()

    def setdefault(self, key: _K, default_factory: _V) -> _V:
        """Return the existing value or insert ``default_factory`` and
        return it. Like ``dict.setdefault`` but the default is the
        already-constructed value (callers with expensive default
        construction can pre-check ``key in self`` instead)."""
        self._maybe_sweep()
        slot = self._live_slot(key)
        if slot is None:
            slot = _Slot(value=default_factory, last_touched=_now())
            self._data[key] = slot
            self._enforce_max_keys()
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
        self._sweep_expired()
        return [(k, slot.value) for k, slot in self._data.items()]

    def keys(self) -> list[_K]:
        self._sweep_expired()
        return list(self._data.keys())

    def values(self) -> list[_V]:
        self._sweep_expired()
        return [slot.value for slot in self._data.values()]

    # ---- explicit eviction ---------------------------------------------

    def evict_expired(self) -> int:
        """Force a TTL sweep. Returns the number of keys evicted.

        Most callers don't need this — per-key reads/writes check the
        touched key's expiry inline and a full sweep runs every
        :data:`_SWEEP_EVERY` operations. Use this from a periodic
        background task if your primitive sees long quiet periods (no
        accesses) and you'd rather have the dict shrink predictably
        than wait for the next user request to drive eviction.
        """
        return self._sweep_expired()

    # ---- internals -----------------------------------------------------
    #
    # NOTE: BoundedDict itself holds no lock — it is synchronous,
    # in-process state. Callers that share an instance across tasks
    # (e.g. StandardBudget) wrap accesses in their own ``anyio.Lock``.

    def _live_slot(self, key: _K) -> _Slot[_V] | None:
        """Return the slot for ``key`` if present AND not expired.

        An expired slot is dropped eagerly. This single-key inline
        check is what keeps ``get`` / ``__getitem__`` /
        ``__contains__`` / ``setdefault`` correct between the
        amortised full sweeps (see ``_maybe_sweep``)."""
        slot = self._data.get(key)
        if slot is None:
            return None
        ttl = self._ttl_seconds
        if (
            ttl is not None
            and (_now() - slot.last_touched).total_seconds() >= ttl
        ):
            self._data.pop(key, None)
            return None
        return slot

    def _maybe_sweep(self) -> None:
        """Amortised TTL sweep for the hot per-key operations.

        A full sweep is O(n); running one on EVERY ``get`` / ``set``
        made each access O(n) under multi-tenant load. Instead we
        sweep once per :data:`_SWEEP_EVERY` per-key operations —
        expired-but-unswept keys can't be observed in the meantime
        because every per-key read goes through ``_live_slot``'s
        inline expiry check, and the O(n) view operations
        (``__len__`` / ``__iter__`` / ``items`` / ...) always sweep.
        """
        if self._ttl_seconds is None:
            return
        self._ops_since_sweep += 1
        if self._ops_since_sweep >= _SWEEP_EVERY:
            self._sweep_expired()

    def _sweep_expired(self) -> int:
        self._ops_since_sweep = 0
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

    def _enforce_max_keys(self) -> None:
        if self._max_keys is None:
            return
        while len(self._data) > self._max_keys:
            # ``OrderedDict.popitem(last=False)`` evicts the
            # least-recently-touched (head) item.
            self._data.popitem(last=False)


def _now() -> datetime:
    return datetime.now(UTC)
