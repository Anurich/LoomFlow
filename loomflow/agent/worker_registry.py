"""Per-Agent registry of persistent subagents (workers).

Solves a structural gap in ``Team.supervisor``: workers are
stateless per-delegation today. Each ``delegate(target, ...)`` call
spawns a fresh worker conversation with no memory of prior
delegations to the same role. The coordinator has to re-explain
context every engagement, burning tokens and losing the worker's
feel for the project built up in earlier turns.

This module provides the concrete primitives:

* :class:`_WorkerHandle` — one registered worker. Carries the
  worker's :class:`Agent`, a STABLE ``session_id`` reused across
  every delegate/send_message touching it, the ``user_id`` the
  worker was pinned to on first touch (multi-tenant safety), and
  an :class:`anyio.Lock` for serialising concurrent calls to the
  same worker (concurrent delegate + send_message races would
  otherwise corrupt the worker's session).

* :func:`new_worker_id` — produces ``worker_<role>_<ULID>`` IDs.
  Universal ``worker_`` prefix makes audit/observability grep
  predictable; the role suffix gives humans something readable;
  the ULID disambiguates two instances of the same role.

The registry itself is a plain ``dict[str, _WorkerHandle]`` stored
on the coordinator :class:`Agent` instance. Concrete-first per
loomflow convention (see :class:`LivingPlan`): start with a dict;
extract a ``WorkerRegistry`` Protocol into ``core/protocols.py``
+ a resolver if a second backend appears (Postgres-backed for
durable ``/resume``; Redis-backed for distributed CLI). At v0.10.10
there's only one backend (in-process), so no Protocol yet.

Lifecycle: the registry is populated by ``Team.supervisor`` at
coordinator construction (eager). Each handle's ``session_id`` is
assigned then. ``send_message(to=<id>, ...)`` works from the very
first invocation — even before any ``delegate`` call touched the
worker — because Memory's ``(user_id, session_id)`` partition is
ready immediately. Handles die with the Agent (REPL ``/clear``
rebuilds the coordinator → old registry GC'd with old Agent → no
stale state).
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import anyio

from ..core.ids import new_id

if TYPE_CHECKING:
    from .api import Agent


class CrossUserWorkerError(RuntimeError):
    """A persistent worker pinned to one ``user_id`` was invoked by
    a run belonging to a different ``user_id``.

    Raised by :func:`acquire_worker_session` BEFORE the worker's
    lock is taken or its session touched — cross-tenant reuse of a
    worker's stable session (and therefore its memory partition)
    must never happen. Callers surface the message to the model /
    caller instead of crashing the run (Supervisor returns it as a
    tool-result error string; other architectures emit an
    architecture event and degrade per call site).
    """


# Mirror :func:`Supervisor.add_worker` identifier-safety check at
# ``loomflow/architecture/supervisor.py:154``. Role names land in
# generated IDs + log lines, so anything that survives a shell
# round-trip + grep is fine; the Python-identifier rule is the
# tightest reasonable cap.
_VALID_ROLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def new_worker_id(role: str) -> str:
    """Build a stable ID for a registered worker.

    Format: ``worker_<role>_<ULID>``. The ``worker_`` prefix is
    universal so audit / telemetry tools can grep for subagent
    activity without knowing each project's role names. The role
    body gives humans a scannable hint when reading logs. The
    ULID disambiguates multiple registrations of the same role
    (you can have two ``coder`` workers in a single team).

    Raises ``ValueError`` when ``role`` isn't a Python identifier
    — mirrors ``Supervisor.add_worker``'s validation so the
    surfaces stay consistent.
    """
    if not _VALID_ROLE.match(role):
        raise ValueError(
            f"worker role must be a Python identifier, got {role!r}"
        )
    return new_id(f"worker_{role}")


@dataclass(slots=True)
class _WorkerHandle:
    """One registered worker — owns its ID, session, and lock.

    Mutable (NOT frozen) because:

    * ``user_id`` is pinned at first-touch (we don't know the
      run's user_id until ``Agent.run()`` starts; pre-pinning at
      construction would lock workers to the wrong user in
      multi-tenant servers where the coordinator is shared).
    * ``last_used_at`` ticks on every invocation.
    * ``lock`` is set in ``__post_init__`` (default-factory on
      ``anyio.Lock`` is awkward to express cleanly).

    The registry's analogue isn't :class:`LivingPlan` (per-run
    state via contextvar) — it's :class:`Memory` itself
    (mutable cross-run state on the Agent instance, durable by
    design). The whole point of this registry IS durability
    across ``Agent.run`` calls. Future readers: don't "fix"
    this into per-run state.
    """

    worker_id: str
    role: str
    agent: Agent
    session_id: str
    user_id: str | None
    created_at: datetime
    last_used_at: datetime | None = None
    lock: anyio.Lock = field(default_factory=anyio.Lock)

    def touch(self, *, user_id: str | None) -> None:
        """Update ``last_used_at`` + pin ``user_id`` on first touch.

        First-touch pinning is the multi-tenant safety primitive:
        once a handle has been used by user X, subsequent calls
        from user Y must be rejected (by the caller — we just
        record the pin here). Touch-after-touch with the SAME
        user_id is the common path; the caller checks the mismatch
        case before invoking the worker.
        """
        if self.user_id is None:
            self.user_id = user_id
        self.last_used_at = datetime.now(UTC)


@asynccontextmanager
async def acquire_worker_session(
    handle: _WorkerHandle, caller_user: str | None
) -> AsyncIterator[_WorkerHandle]:
    """Cross-user check → lock → touch, in ONE place.

    Every architecture that invokes a persistent worker MUST enter
    the worker's session through this helper (Supervisor previously
    inlined this dance; Swarm / Router / Blackboard / Debate touched
    with no check and no lock — the cross-tenant reuse bug).

    Semantics:

    * **Mismatch check** — when the handle is already pinned to a
      user and the current run belongs to a DIFFERENT user, raise
      :class:`CrossUserWorkerError` before taking the lock. A
      ``None`` on either side is not a mismatch (anonymous runs and
      first-touch pinning keep working).
    * **Lock** — held for the duration of the ``async with`` body so
      concurrent invocations of the same worker serialise instead of
      interleaving writes into the worker's session.
    * **Touch** — pins ``user_id`` on first touch and ticks
      ``last_used_at``, under the lock.
    """
    if (
        handle.user_id is not None
        and caller_user is not None
        and handle.user_id != caller_user
    ):
        raise CrossUserWorkerError(
            f"worker {handle.role!r} ({handle.worker_id}) belongs "
            f"to user_id {handle.user_id!r} but the current run is "
            f"user_id {caller_user!r}. Cross-tenant delegation is "
            "rejected."
        )
    async with handle.lock:
        handle.touch(user_id=caller_user)
        yield handle


def resolve_persistent_session(
    role: str,
    *,
    fallback: str,
    registry: dict[str, _WorkerHandle] | None,
    role_to_id: dict[str, str] | None,
) -> tuple[str, _WorkerHandle | None]:
    """Pick a worker's ``session_id`` — persistent or per-run.

    Used at every architecture spawn site. When a worker has a
    handle in the registry, return the handle's stable
    ``session_id`` (and the handle, so the caller can acquire the
    lock + check user_id). Otherwise return ``fallback`` (the
    pre-existing per-run deterministic ID the architecture would
    have used) and ``None`` for the handle.

    Mechanical insertion point — every architecture's spawn site
    becomes::

        sid, handle = resolve_persistent_session(
            role, fallback=f"{session.id}__role",
            registry=self._worker_registry,
            role_to_id=self._role_to_worker_id,
        )
        # ... use sid as session_id; if handle: handle.touch + lock
    """
    if (
        registry is not None
        and role_to_id is not None
        and role in role_to_id
    ):
        handle = registry[role_to_id[role]]
        return handle.session_id, handle
    return fallback, None


def build_worker_registry(
    workers: dict[str, Agent],
) -> tuple[dict[str, _WorkerHandle], dict[str, str]]:
    """Build a fresh ``(registry, role_to_worker_id)`` pair.

    Used by every ``Team.*`` builder that wants persistent
    subagents — supervisor, swarm, router, debate, actor_critic,
    blackboard. DRYs the eager-registration ceremony so each
    builder is a one-liner.

    The returned tuple:

    * ``registry`` keys are worker IDs (``worker_<role>_<ULID>``);
      values are :class:`_WorkerHandle` instances ready to be
      handed to a tool or architecture.
    * ``role_to_worker_id`` maps the original role key (the dict
      key in ``workers``) to the generated worker ID — lets
      architectures translate model-emitted role names (e.g.
      ``delegate("coder", ...)``) back to the handle.
    """
    registry: dict[str, _WorkerHandle] = {}
    role_to_worker_id: dict[str, str] = {}
    now = datetime.now(UTC)
    for role, worker_agent in workers.items():
        worker_id = new_worker_id(role)
        handle = _WorkerHandle(
            worker_id=worker_id,
            role=role,
            agent=worker_agent,
            session_id=f"persistent_{worker_id}",
            user_id=None,
            created_at=now,
        )
        registry[worker_id] = handle
        role_to_worker_id[role] = worker_id
    return registry, role_to_worker_id
