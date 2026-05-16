"""20_persistent_subagents.py — workers remember across runs.

Default behaviour as of v0.10.10: every ``Team.*`` builder
registers each worker with a stable persistent ID + session_id.
Workers accumulate conversation memory across handoffs, rounds,
AND across multiple ``coordinator.run()`` invocations.

This example illustrates the new shape by inspecting what the
Team builder stamps onto the coordinator under both modes:

* Persistent (the default): the coordinator carries a populated
  ``_worker_registry`` mapping ``worker_<role>_<ULID>`` →
  ``_WorkerHandle``. Each handle has a stable ``session_id``
  (``persistent_worker_<role>_<ULID>``) that EVERY spawn site
  in the architecture reuses — so the worker's Memory partition
  is shared across all delegations + ``Agent.run()`` calls.
* Legacy (``persistent_subagents=False``): the registry is empty
  and each spawn generates a fresh ULID session — workers start
  cold every time.

Zero-key — runs offline via ``EchoModel``.

Run with::

    python examples/20_persistent_subagents.py
"""

from __future__ import annotations

from loomflow import Agent, EchoModel
from loomflow.team import Team


def _worker(label: str) -> Agent:
    return Agent(instructions=f"You are {label}.", model=EchoModel())


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Persistent (default) — registry populated, sessions stable.
    # ------------------------------------------------------------------
    persistent = Team.supervisor(
        workers={
            "researcher": _worker("a researcher"),
            "writer": _worker("a writer"),
        },
        model="echo",
    )
    print("PERSISTENT mode (default)")
    print(f"  registry size = {len(persistent._worker_registry)}")
    for handle in persistent._worker_registry.values():
        print(f"  - role={handle.role!r}")
        print(f"      worker_id  = {handle.worker_id}")
        print(f"      session_id = {handle.session_id}")
        print(
            f"      user_id    = {handle.user_id} "
            "(pinned on first delegate / send_message)"
        )
    print()
    print(
        "  → every delegate('researcher', ...) reuses the same "
        "session_id, so memory survives across runs."
    )
    print(
        "  → send_message(to='<worker_id>', content=...) is "
        "auto-wired and lets the coordinator follow up by ID."
    )
    print()

    # ------------------------------------------------------------------
    # 2. Legacy opt-out — registry empty, send_message NOT wired.
    # ------------------------------------------------------------------
    legacy = Team.supervisor(
        workers={
            "researcher": _worker("a researcher"),
            "writer": _worker("a writer"),
        },
        model="echo",
        persistent_subagents=False,
    )
    print("LEGACY mode (persistent_subagents=False)")
    print(f"  registry size = {len(legacy._worker_registry)}")
    print(
        "  → architecture generates a fresh "
        "f'{session.id}__{role}_<ULID>' session per delegate."
    )
    print("  → no send_message tool; coordinator can only delegate fresh.")
    print()

    # ------------------------------------------------------------------
    # 3. Every Team.* builder follows the same pattern.
    # ------------------------------------------------------------------
    print("OTHER TEAM ARCHITECTURES")
    debate = Team.debate(
        debaters=[_worker("debater A"), _worker("debater B")],
        judge=_worker("judge"),
        model="echo",
    )
    print(
        "  Team.debate    — registers debater_0, debater_1, judge: "
        f"{sorted(h.role for h in debate._worker_registry.values())}"
    )

    actor_critic = Team.actor_critic(
        actor=_worker("actor"),
        critic=_worker("critic"),
        model="echo",
    )
    print(
        "  Team.actor_critic — registers actor, critic: "
        f"{sorted(h.role for h in actor_critic._worker_registry.values())}"
    )


if __name__ == "__main__":
    main()
