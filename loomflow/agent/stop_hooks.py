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
  ``living_plan={"auto_stop_hook": False, ...}``. When ``run_until=``
  is set, the framework appends a :class:`GoalStopHook` that re-prompts
  the agent until a fast checker model confirms a measurable condition
  holds (the ``/goal`` pattern), guarded by max-iteration, no-progress,
  and cost caps.

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


# ---------------------------------------------------------------------------
# GoalStopHook — run-until-done loop gated by a fast checker model
# ---------------------------------------------------------------------------


DEFAULT_GOAL_CHECKER_PROMPT = """\
You are a strict completion checker for an autonomous agent loop. You
are given a STOP CONDITION and the agent's latest output. Decide
whether the condition is now fully satisfied.

Judge only what the evidence shows. Do NOT give the benefit of the
doubt: if the output does not clearly demonstrate the condition holds
(e.g. it claims tests pass but shows no run, or leaves a step
unfinished), the condition is NOT met.

Respond with exactly one of these on the first line:

  DONE        — the condition is fully and verifiably satisfied
  NOT_DONE    — the condition is not yet satisfied

Then, on the next line, one short sentence of justification."""


class GoalStopHook:
    """Re-prompt the agent until a fast checker confirms a measurable
    condition holds — loomflow's ``/goal`` / run-until-done primitive.

    Implements the :class:`StopHook` Protocol structurally. After each
    architecture pass, the hook:

    1. checks three guardrails (in order) and votes **stop** if any
       trips — ``max_iterations`` reached, no observable progress for
       ``max_no_progress`` consecutive firings, or the loop's
       cumulative cost exceeded ``max_cost_usd`` (or the Agent-wide
       :class:`~loomflow.core.protocols.Budget` blocked the step);
    2. otherwise runs a single text-only call against the **checker
       model** (``deps.goal_checker`` — a small fast model — falling
       back to ``deps.model`` when unset), asking DONE / NOT_DONE
       against ``condition``;
    3. votes **stop** (returns ``None``) when the checker says DONE,
       else returns a :class:`StopHookResult` that re-prompts the
       agent naming the unmet condition.

    Guardrails are first-class, not optional: an unbounded run-until
    loop is the #1 autonomous-agent failure mode. ``max_iterations``
    here is a *per-goal* cap tighter than the Agent-wide
    ``max_stop_hook_iterations`` backstop.

    **Concurrency:** one instance is shared across concurrent
    ``agent.run()`` calls, so all per-run state (iteration fingerprint,
    no-progress counter) lives in ``session.metadata`` under the
    ``run_until.*`` namespace — never on the instance.

    **Multi-tenancy:** every budget call forwards
    ``deps.context.user_id`` so per-user caps are enforced.
    """

    name = "run_until"

    def __init__(
        self,
        condition: str,
        *,
        name: str = "run_until",
        checker_prompt: str | None = None,
        max_iterations: int = 20,
        max_no_progress: int = 3,
        max_cost_usd: float | None = None,
    ) -> None:
        if not condition or not condition.strip():
            raise ValueError("GoalStopHook condition must be non-empty")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if max_no_progress < 1:
            raise ValueError("max_no_progress must be >= 1")
        if max_cost_usd is not None and max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be > 0 when set")
        self.name = name
        self._condition = condition.strip()
        self._checker_prompt = checker_prompt or DEFAULT_GOAL_CHECKER_PROMPT
        self._max_iterations = max_iterations
        self._max_no_progress = max_no_progress
        self._max_cost_usd = max_cost_usd

    def _exit(self, session: AgentSession, reason: str) -> None:
        """Record why the loop stopped, for telemetry / callers.

        ``condition_met`` is a clean success — the goal was reached, so
        ``interrupted`` stays False. Every other reason is a guardrail
        cutting the loop short BEFORE the goal was met; surface that as
        an interruption so it flows through to ``RunResult`` (callers
        like a ``/goal`` UI can then distinguish "goal met" from "hit
        the cap"). The precise reason always lives in metadata."""
        session.metadata["run_until.exit"] = reason
        if reason != "condition_met":
            session.interrupted = True
            session.interruption_reason = f"run_until:{reason}"

    async def __call__(
        self,
        session: AgentSession,
        deps: Dependencies,
        *,
        iteration: int,
    ) -> StopHookResult | None:
        # Lazy imports: stop_hooks loads before the architecture
        # package is fully importable (base.py imports back into the
        # agent package). Deferring here mirrors make_plan_stop_hook.
        from ..architecture.helpers import add_usage, text_only_model_call
        from ..core.types import Message, Role

        user_id = getattr(deps.context, "user_id", None)

        # --- Guardrail 1: per-goal iteration cap ---
        # ``iteration`` is 0 on the first firing; the (iteration+1)-th
        # architecture pass has already run by the time we're called.
        if iteration + 1 >= self._max_iterations:
            self._exit(session, "max_iterations")
            return None

        # --- Guardrail 2: hard cost cap (loop-local + Agent-wide) ---
        try:
            status = await deps.budget.allows_step(user_id=user_id)
        except TypeError:  # legacy Budget without user_id kwarg
            status = await deps.budget.allows_step()
        if status.blocked:
            self._exit(session, f"budget:{status.reason}")
            return None
        if (
            self._max_cost_usd is not None
            and session.cumulative_usage.cost_usd >= self._max_cost_usd
        ):
            self._exit(session, "cost_cap")
            return None

        # --- Guardrail 3: no-progress detection ---
        # Fingerprint observable state; if it's unchanged for
        # ``max_no_progress`` consecutive firings, the loop is spinning.
        fingerprint = (
            session.output,
            session.cumulative_usage.output_tokens,
        )
        last_fp = session.metadata.get("run_until.last_fp")
        stalls = int(session.metadata.get("run_until.stalls", 0) or 0)
        if last_fp == fingerprint:
            stalls += 1
        else:
            stalls = 0
        session.metadata["run_until.last_fp"] = fingerprint
        session.metadata["run_until.stalls"] = stalls
        if stalls >= self._max_no_progress:
            self._exit(session, "no_progress")
            return None

        # --- Checker: is the condition satisfied? ---
        checker = deps.goal_checker or deps.model
        messages = [
            Message(role=Role.SYSTEM, content=self._checker_prompt),
            Message(
                role=Role.USER,
                content=(
                    f"STOP CONDITION:\n{self._condition}\n\n"
                    f"AGENT'S LATEST OUTPUT:\n{session.output}"
                ),
            ),
        ]
        verdict, usage = await text_only_model_call(
            deps,
            f"run_until_check_{iteration}",
            messages,
            model=checker,
        )
        # Mirror react.py: skip the consume round-trip on the fast
        # budget path (NoBudget is a no-op anyway). ``allows_step``
        # above stays unconditional — it IS the cost-cap guardrail.
        if not deps.fast_budget:
            try:
                await deps.budget.consume(
                    tokens_in=usage.input_tokens,
                    tokens_out=usage.output_tokens,
                    cost_usd=usage.cost_usd,
                    user_id=user_id,
                )
            except TypeError:  # legacy Budget without user_id kwarg
                await deps.budget.consume(
                    tokens_in=usage.input_tokens,
                    tokens_out=usage.output_tokens,
                    cost_usd=usage.cost_usd,
                )
        session.cumulative_usage = add_usage(session.cumulative_usage, usage)

        first_line = verdict.strip().splitlines()[0].upper() if verdict.strip() else ""
        # DONE only when affirmatively stated and not negated. Checking
        # NOT_DONE first avoids the substring trap ("NOT_DONE" contains
        # "DONE").
        if "NOT_DONE" in first_line or "NOT DONE" in first_line:
            done = False
        else:
            done = "DONE" in first_line
        if done:
            self._exit(session, "condition_met")
            return None

        # Not done — re-prompt, naming the condition + the checker's reason.
        reason_text = verdict.strip()
        return StopHookResult(
            inject_message=(
                "Your run-until goal is NOT yet met. The stop condition "
                f"is:\n\n{self._condition}\n\n"
                f"Checker assessment:\n{reason_text}\n\n"
                "Keep working: take concrete actions to satisfy the "
                "condition, then verify it. Do not claim completion "
                "without evidence the condition holds."
            ),
            reason="condition_unmet",
        )
