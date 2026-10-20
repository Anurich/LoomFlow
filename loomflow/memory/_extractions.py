"""Background fact extractions that ``forget`` stops before erasing.

An :class:`~loomflow.memory.auto_extract.AutoExtractMemory` extracts
facts after ``remember`` returns. An extraction still running when the
user's data is erased would write the erased facts back, so every
backend's ``forget`` first calls :func:`stop_extractions` on itself.
That covers ``agent.memory.forget(...)``, which reaches the backend,
not the wrapper around it.
"""

from __future__ import annotations

import weakref
from datetime import datetime
from typing import Any

# Live wrappers. Weak, so a dropped Agent's wrapper doesn't linger.
_WRAPPERS: weakref.WeakSet[Any] = weakref.WeakSet()


def track(wrapper: Any) -> None:
    """Register an extraction wrapper (it must expose ``inner`` and an
    async ``stop_extractions``)."""
    _WRAPPERS.add(wrapper)


async def stop_extractions(
    memory: Any,
    *,
    user_id: str | None,
    session_id: str | None = None,
    before: datetime | None = None,
) -> None:
    """Stop the extractions, by any wrapper of ``memory``, of the
    episodes a ``forget`` with these arguments erases."""
    for wrapper in list(_WRAPPERS):
        if wrapper.inner is memory:
            await wrapper.stop_extractions(
                user_id=user_id, session_id=session_id, before=before
            )
