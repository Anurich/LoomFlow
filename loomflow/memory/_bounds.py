"""Shared size limits for the memory backends.

Three things every backend bounds the same way:

* **recall limits** — ``recall(limit=...)`` must be a positive int
  (:func:`validate_limit`), matching ``VectorStore.search(k=...)``.
* **working-block growth** — ``append_block`` keeps only the newest
  :data:`DEFAULT_MAX_BLOCK_CHARS` characters (:func:`trim_block`), so
  a block that is appended to on every turn can't grow without end
  (it is pinned into every prompt).
* **episode counts** — the in-process backends keep episodes in an
  :class:`EpisodeIndex`, partitioned by ``user_id`` with an optional
  per-user cap.
"""

from __future__ import annotations

import heapq
from typing import Any

from ..core.types import Episode

__all__ = [
    "DEFAULT_MAX_BLOCK_CHARS",
    "EpisodeIndex",
    "resolve_max_block_chars",
    "trim_block",
    "validate_limit",
]

#: Default cap on a working block grown through ``append_block``, in
#: characters (roughly 5k tokens). Older content is dropped first.
DEFAULT_MAX_BLOCK_CHARS = 20_000


def validate_limit(limit: Any, *, name: str = "limit") -> int:
    """A recall limit must be a positive int — ``0`` and negatives
    raise :class:`ValueError` (a negative used to slice from the end
    and return almost everything)."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError(f"{name} must be a positive integer, got {limit!r}")
    if limit <= 0:
        raise ValueError(f"{name} must be >= 1, got {limit}")
    return int(limit)


def resolve_max_block_chars(max_block_chars: int | None) -> int | None:
    """Validate a ``max_block_chars=`` constructor argument
    (``None`` disables the cap)."""
    if max_block_chars is None:
        return None
    if (
        isinstance(max_block_chars, bool)
        or not isinstance(max_block_chars, int)
        or max_block_chars < 1
    ):
        raise ValueError(
            "max_block_chars must be a positive integer or None, "
            f"got {max_block_chars!r}"
        )
    return max_block_chars


def trim_block(content: str, max_chars: int | None) -> str:
    """Keep the newest ``max_chars`` characters of an appended block
    (oldest-first trimming); ``None`` keeps everything."""
    if max_chars is None or len(content) <= max_chars:
        return content
    return content[-max_chars:]


class EpisodeIndex:
    """Episodes partitioned by ``user_id``, with an optional per-user
    cap.

    ``episodes`` is the flat ``id -> Episode`` view; ``for_user``
    returns one tenant's episodes without touching anyone else's, so a
    recall costs what that user stores, not what the process stores.

    With ``max_per_user`` set, each user keeps at most that many
    episodes: adding one more evicts that user's oldest (by
    ``occurred_at``) — never another user's. Eviction pops a per-user
    heap, so a write is ``O(log n)`` rather than a scan of everything.
    """

    def __init__(self, max_per_user: int | None = None) -> None:
        if max_per_user is not None and (
            isinstance(max_per_user, bool)
            or not isinstance(max_per_user, int)
            or max_per_user < 1
        ):
            raise ValueError(
                "max_episodes must be a positive integer or None, "
                f"got {max_per_user!r}"
            )
        self.max_per_user = max_per_user
        self.episodes: dict[str, Episode] = {}
        self._by_user: dict[str | None, dict[str, Episode]] = {}
        # Per-user min-heaps of (occurred_at timestamp, sequence, id).
        # An entry is live while ``_live_sequence[id]`` is its
        # sequence; replacing or removing an episode leaves a stale
        # entry that is skipped on pop and swept when stale entries
        # outnumber live ones.
        self._heaps: dict[str | None, list[tuple[float, int, str]]] = {}
        self._live_sequence: dict[str, int] = {}
        self._sequence = 0

    def __len__(self) -> int:
        return len(self.episodes)

    def has_named_users(self) -> bool:
        """Whether any episode belongs to a non-anonymous user."""
        return any(uid is not None for uid in self._by_user)

    def for_user(self, user_id: str | None) -> list[Episode]:
        """A snapshot of one user's episodes (insertion order)."""
        return list(self._by_user.get(user_id, {}).values())

    def put(self, episode: Episode) -> list[str]:
        """Insert or replace ``episode``; returns the ids evicted to
        keep its user within the cap (never the episode itself)."""
        previous = self.episodes.get(episode.id)
        if previous is not None and previous.user_id != episode.user_id:
            self.remove(episode.id)  # the id moved to another user
        # (A same-user replacement keeps its place in insertion order.)
        self.episodes[episode.id] = episode
        bucket = self._by_user.setdefault(episode.user_id, {})
        bucket[episode.id] = episode
        if self.max_per_user is None:
            return []
        heap = self._heaps.setdefault(episode.user_id, [])
        self._push(heap, episode)

        evicted: list[str] = []
        held_back: list[tuple[float, int, str]] = []
        while len(bucket) > self.max_per_user and heap:
            entry = heapq.heappop(heap)
            _, sequence, eid = entry
            if self._live_sequence.get(eid) != sequence:
                continue  # stale entry
            if eid == episode.id:
                held_back.append(entry)  # never evict the new arrival
                continue
            del bucket[eid]
            del self.episodes[eid]
            del self._live_sequence[eid]
            evicted.append(eid)
        for entry in held_back:
            heapq.heappush(heap, entry)
        if len(heap) > 2 * len(bucket) + 16:
            self._rebuild_heap(episode.user_id)
        return evicted

    def remove(self, episode_id: str) -> Episode | None:
        """Drop one episode by id (its heap entry goes stale)."""
        episode = self.episodes.pop(episode_id, None)
        if episode is None:
            return None
        self._live_sequence.pop(episode_id, None)
        bucket = self._by_user.get(episode.user_id)
        if bucket is not None:
            bucket.pop(episode_id, None)
            if not bucket:
                del self._by_user[episode.user_id]
                self._heaps.pop(episode.user_id, None)
        return episode

    def _push(self, heap: list[tuple[float, int, str]], episode: Episode) -> None:
        self._sequence += 1
        self._live_sequence[episode.id] = self._sequence
        heapq.heappush(
            heap, (episode.occurred_at.timestamp(), self._sequence, episode.id)
        )

    def _rebuild_heap(self, user_id: str | None) -> None:
        heap: list[tuple[float, int, str]] = []
        self._heaps[user_id] = heap
        for episode in self._by_user.get(user_id, {}).values():
            self._push(heap, episode)
