"""Stop hooks — framework-level Ralph loop.

Solves a structural failure mode of ReAct-class architectures: the
loop exits when the model emits text without a tool call, even
when there's still work to do. For multi-step plans (``living_plan
=True``), the model often emits prose between delegations like
"Now let me scaffold step 3" — ReAct interprets that as a final
answer and exits, leaving the plan undone.

The Ralph-loop fix has converged across the industry (Claude Code's
``handleStopHooks``, Anthropic ``/goal``, AutoGen's
``TerminationCondition``, Cursor's judge agent): after the model's
loop says "done," run an external check that can OVERRIDE the
model's claim and force the loop to continue.

Loomflow's version lives at the **Agent** level, not the
Architecture level, for three reasons:

1. All architectures (built-in + 3rd-party + future) benefit from
   one implementation.
2. Setup/teardown lifecycle already lives in ``Agent`` (the
   ``started`` / ``completed`` events come from there, not
   architectures — see ``architecture/base.py:23-29``).
3. Architectures own their *internal* loop; "what happens between
   full architecture runs" is the Agent's concern.

Re-invocation strategy: when a hook returns a continue directive,
``Agent._loop`` calls ``architecture.run(session, deps,
inject_message)`` *with the same session*. Architectures consume
``prompt`` as a fresh user turn (a contract documented on
``Architecture.run``); the conversation history in
``session.messages`` carries forward naturally. Bounded by
``Agent.max_stop_hook_iterations`` (default 15) so a flaky hook
can't burn the user's budget unbounded.

Two ways hooks land on an Agent:

* **User-supplied** — ``Agent(stop_hooks=[my_hook, ...])``. Any
  async callable matching the Protocol.
* **Auto-registered** — when ``living_plan=True``, the framework
  prepends a hook that detects incomplete plan steps. Opt out with
  ``living_plan={"auto_stop_hook": False, ...}``.

Why not in ``core/protocols.py``: that module holds backend axes
(Memory, Model, Permissions) — full subsystems wired in via
resolvers. ``StopHook`` is a per-Agent callable, closer in spirit
to ``RetryPolicy`` (which also lives outside ``core/protocols``).
No resolver is needed since callables aren't TOML-expressible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    # Defer the import to avoid a circular cycle: base.py defines
    # ``AgentSession`` + ``Dependencies`` and already imports core
    # protocols. The Protocol only needs them at type-check time.
    from ..architecture.base import AgentSession, Dependencies


@dataclass(frozen=True, slots=True)
class StopHookResult:
    """One hook's "keep going" directive.

    Returned from a :class:`StopHook` to force ``Agent._loop`` to
    re-invoke the architecture instead of exiting. Frozen because
    once a hook returns this, the framework treats it as immutable
    (mirrors :class:`~loomflow.RunContext` and
    :class:`~loomflow.BudgetStatus`).

    ``inject_message`` becomes a fresh user turn in the running
    session — the architecture sees it as if the user had typed it.

    ``reason`` is a short stable token for telemetry / audit
    (``"plan_step_3_doing"``, ``"output_too_short"``, ...). It
    appears on emitted ``architecture_event`` payloads and audit
    rows so observability tools can group stop-hook firings by
    cause.
    """

    inject_message: str
    reason: str = ""


@runtime_checkable
class StopHook(Protocol):
    """Async callable that votes on whether the agent's run is done.

    Called by ``Agent._loop`` AFTER the architecture's ``run()``
    coroutine returns, and BEFORE the agent emits ``Event.completed``.
    Each registered hook is awaited in order; the **first** hook
    that returns a :class:`StopHookResult` (not ``None``) wins —
    the framework injects its message, re-invokes the architecture,
    and starts the next iteration. Remaining hooks for that
    iteration are skipped.

    Returning ``None`` votes "stop." When ALL hooks return ``None``
    in a given iteration, the loop terminates and the agent
    proceeds to teardown.

    Hook signature::

        async def __call__(
            self,
            session: AgentSession,
            deps: Dependencies,
            *,
            iteration: int,
        ) -> StopHookResult | None:
            ...

    Arguments:

    * ``session`` — the live ``AgentSession`` carrying messages,
      output, turns, metadata. Read-only by convention; mutating
      it inside a hook is undefined behaviour.
    * ``deps`` — the full :class:`Dependencies` bundle. Use
      ``deps.context.user_id`` for multi-tenant partitioning when
      the hook calls back into ``deps.memory`` /
      ``deps.tools``.
    * ``iteration`` — 0 for the first hook firing (after the
      initial architecture pass), 1 after the first re-invocation,
      etc. Lets hooks back off as iterations grow (e.g. "give up
      after 3 re-prompts").

    Hooks MUST be ``async``. Sync callables passed in trip a
    TypeError at the ``await hook(...)`` site — by design; we want
    that failure visible immediately rather than wrapping in
    ``asyncio.to_thread`` and hiding the misuse.
    """

    name: str
    """Short stable identifier used in telemetry + audit. E.g.
    ``"living_plan"``, ``"output_validator"``. Defaults are
    suggested by the implementation; collisions are tolerated."""

    async def __call__(
        self,
        session: AgentSession,
        deps: Dependencies,
        *,
        iteration: int,
    ) -> StopHookResult | None:
        ...
