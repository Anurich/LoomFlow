"""Per-Agent registry of persistent subagents (workers).

Solves a structural gap in ``Team.supervisor``: workers are
stateless per-delegation today. Each ``delegate(target, ...)`` call
spawns a fresh worker conversation with no memory of prior
delegations to the same role. The coordinator has to re-explain
context every engagement, burning tokens and losing the worker's
feel for the project built up in earlier turns.

This module provides the concrete primitives:

* :class:`_WorkerHandle` — one registered worker. Carries the
  worker's :class:`Agent`, a STABLE session per user reused across
  every delegate/send_message touching it (so one shared coordinator
  serves many users, each with their own worker conversation), and
  a lock per user for serialising concurrent calls to the same
  worker session (concurrent delegate + send_message races would
  otherwise corrupt the worker's session). Workers spawned mid-run
  are instead owned by the spawning run's user and refuse everyone
  else.

* :func:`new_worker_id` — produces ``worker_<role>_<ULID>`` IDs for
  workers spawned mid-run. Universal ``worker_`` prefix makes
  audit/observability grep predictable; the role suffix gives humans
  something readable; the ULID disambiguates two instances of the
  same role. Workers registered when a team is built get
  :func:`stable_worker_id` instead: ``worker_<role>_<hash>`` of the
  role and the worker's instructions, the same in every process, so
  a team rebuilt after a restart finds its workers' sessions — their
  memory and their journal (a crashed coordinator resumes without
  repeating a worker's side effects).

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

import hashlib
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import anyio

from ..core.context import get_run_context
from ..core.ids import new_id

if TYPE_CHECKING:
    from .api import Agent


class CrossUserWorkerError(RuntimeError):
    """A worker owned by one ``user_id`` (one spawned mid-run) was
    invoked by a run belonging to a different ``user_id``.

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


def stable_worker_id(role: str, agent: Agent | None = None, *, index: int = 0) -> str:
    """A registered worker's ID, identical across processes:
    ``worker_<role>_<16 hex>`` hashed from the role, the worker's
    instructions and (for repeated roles) its position."""
    if not _VALID_ROLE.match(role):
        raise ValueError(
            f"worker role must be a Python identifier, got {role!r}"
        )
    instructions = str(getattr(agent, "_instructions", "") or "")
    digest = hashlib.sha256(
        f"{role}\x00{instructions}\x00{index}".encode()
    ).hexdigest()
    return f"worker_{role}_{digest[:16]}"


@dataclass(slots=True)
class _WorkerHandle:
    """One registered worker — owns its ID, sessions, and locks.

    A registered worker keeps one stable session per user:
    :meth:`session_for` derives it from ``session_id``, so alice and
    bob each get their own conversation with the same worker, and an
    anonymous run (``user_id=None``) gets the base ``session_id``.
    Calls are serialised per user session, not across users.

    A spawned worker (``owner_only=True``) belongs to the run that
    created it: ``user_id`` is that run's user and every other caller,
    anonymous or not, is refused (its instructions came from that
    user's conversation).

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
    owner_only: bool = False
    _user_locks: dict[str, anyio.Lock] = field(default_factory=dict)

    def session_for(self, user_id: str | None) -> str:
        """The worker's stable session id for ``user_id``'s runs."""
        if self.owner_only or user_id is None:
            return self.session_id
        digest = hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()
        return f"{self.session_id}__u{digest[:16]}"

    def touch(self, *, user_id: str | None = None) -> None:
        """Tick ``last_used_at``. (``user_id`` is accepted for
        backwards compatibility; workers are no longer pinned on
        first touch.)"""
        self.last_used_at = datetime.now(UTC)


@asynccontextmanager
async def acquire_worker_session(
    handle: _WorkerHandle, caller_user: str | None
) -> AsyncIterator[_WorkerHandle]:
    """Ownership check → per-user lock → touch, in ONE place.

    Every architecture that invokes a persistent worker MUST enter
    the worker's session through this helper, and run the worker
    under ``handle.session_for(caller_user)``.

    Semantics:

    * **Ownership** — an ``owner_only`` (spawned) worker raises
      :class:`CrossUserWorkerError` for any caller whose ``user_id``
      differs from its owner's, ``None`` included, before taking the
      lock.
    * **Lock** — held for the duration of the ``async with`` body so
      concurrent invocations of the same worker session serialise
      instead of interleaving writes into it. Different users don't
      wait on each other.
    * **Touch** — ticks ``last_used_at``, under the lock.
    """
    if handle.owner_only and caller_user != handle.user_id:
        raise CrossUserWorkerError(
            f"worker {handle.role!r} ({handle.worker_id}) belongs "
            f"to user_id {handle.user_id!r} but the current run is "
            f"user_id {caller_user!r}. Cross-tenant delegation is "
            "rejected."
        )
    if handle.owner_only or caller_user is None:
        async with handle.lock:
            handle.touch()
            yield handle
        return
    key = str(caller_user)
    lock = handle._user_locks.setdefault(key, anyio.Lock())
    try:
        async with lock:
            handle.touch()
            yield handle
    finally:
        # Drop idle locks so a long-lived server doesn't keep one per
        # user it has ever seen.
        if not lock.locked() and lock.statistics().tasks_waiting == 0:
            handle._user_locks.pop(key, None)


def resolve_persistent_session(
    role: str,
    *,
    fallback: str,
    registry: dict[str, _WorkerHandle] | None,
    role_to_id: dict[str, str] | None,
    user_id: str | None = None,
) -> tuple[str, _WorkerHandle | None]:
    """Pick a worker's ``session_id`` — persistent or per-run.

    Used at every architecture spawn site. When a worker has a
    handle in the registry, return the handle's stable session for
    ``user_id`` (default: the current run's user) and the handle, so
    the caller can acquire it. Otherwise return ``fallback`` (the
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
        user = user_id if user_id is not None else get_run_context().user_id
        return handle.session_for(user), handle
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

    * ``registry`` keys are worker IDs (:func:`stable_worker_id`,
      the same in every process); values are :class:`_WorkerHandle`
      instances ready to be handed to a tool or architecture.
    * ``role_to_worker_id`` maps the original role key (the dict
      key in ``workers``) to the generated worker ID — lets
      architectures translate model-emitted role names (e.g.
      ``delegate("coder", ...)``) back to the handle.
    """
    registry: dict[str, _WorkerHandle] = {}
    role_to_worker_id: dict[str, str] = {}
    now = datetime.now(UTC)
    for role, worker_agent in workers.items():
        worker_id = stable_worker_id(role, worker_agent)
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
