"""Router: classify input → dispatch to one specialist Agent.

OpenAI Agents SDK March 2026 "Handoff" pattern, plus the
classify-and-route shape every framework reinvents (CrewAI sequential,
LangGraph conditional edges, ...). The simplest multi-agent pattern.

Pattern
-------

1. **Classify.** A small fast LLM call decides which route handles
   the input best.
2. **Dispatch.** The chosen specialist :class:`Agent` runs to
   completion with its own model / memory / tools / architecture.
3. **Return.** The specialist's output becomes the Router's output.
   No cross-specialist synthesis.

Compared to :class:`Supervisor`:

* Cheaper (1 classification + 1 specialist; no synthesis pass).
* Deterministic (single specialist owns the task).
* Less flexible (no multi-domain tasks; routing errors cascade).

When to use
-----------
* Customer support (route to billing / tech / sales / general).
* Helpdesks where each query has one right specialist.
* API-gateway-style intent routing.

Replay correctness
------------------
The specialist's :meth:`Agent.run` is invoked with a deterministic
``session_id`` derived from the parent session and the route name:
``{parent_session_id}__route_{route_name}``. On replay, the same
specialist session_id reproduces, and the specialist's own journal
(under its own session) takes over from there. The parent's journal
caches the classification step — replay flows cleanly through both
layers.

Specialists are full Agents
---------------------------
Each route is a fully-constructed :class:`Agent` instance. They are
NOT shared dependencies of the parent. If you want shared budget /
memory / telemetry, pass the same instances when building the
specialists.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ..core.context import inherit_ambient_memory
from ..core.types import Event, Message, Role
from .base import AgentSession, Dependencies
from .helpers import (
    SubagentInvocation,
    consume_usage,
    consume_worker_usage,
    text_only_model_call,
    usage_from_result_dict,
)

if TYPE_CHECKING:
    from ..agent.api import Agent


_RouterRegistryT = dict[str, Any]  # registry of worker handles
_RouterRoleMapT = dict[str, str]   # role → worker_id


DEFAULT_CLASSIFIER_PROMPT = """\
You are a routing classifier. Given the user's request, decide which
specialist handles it best.

Available routes:
{route_descriptions}

Output exactly two lines, in this order:
route: <one of the route names above>
confidence: <number between 0 and 1>

Then optionally one line of brief reasoning. The first two lines
must match the format exactly so they can be parsed.
"""


@dataclass(frozen=True)
class RouterRoute:
    """One specialist + classification metadata.

    ``name`` is what the classifier emits in its ``route:`` line and
    must be unique within a Router. ``description`` is shown to the
    classifier alongside the name — keep it specific and
    distinguishing so the classifier picks reliably.
    """

    name: str
    agent: Agent
    description: str = ""


class Router:
    """Classify input → dispatch to ONE specialist :class:`Agent`."""

    name = "router"

    def __init__(
        self,
        *,
        routes: list[RouterRoute],
        fallback_route: str | None = None,
        require_confidence_above: float = 0.0,
        classifier_prompt: str | None = None,
        worker_registry: _RouterRegistryT | None = None,
        role_to_worker_id: _RouterRoleMapT | None = None,
        conversation_scope: Literal["per_route", "shared"] = "per_route",
    ) -> None:
        if not routes:
            raise ValueError("Router requires at least one route")
        if not 0.0 <= require_confidence_above <= 1.0:
            raise ValueError(
                "require_confidence_above must be in [0.0, 1.0]"
            )
        names = [r.name for r in routes]
        if len(set(names)) != len(names):
            raise ValueError(
                f"Route names must be unique; got {names}"
            )
        if fallback_route is not None and fallback_route not in names:
            raise ValueError(
                f"fallback_route {fallback_route!r} not in routes "
                f"({', '.join(names)})"
            )
        if conversation_scope not in ("per_route", "shared"):
            raise ValueError(
                "conversation_scope must be 'per_route' or 'shared', "
                f"got {conversation_scope!r}"
            )
        self._routes = list(routes)
        self._routes_by_name = {r.name: r for r in routes}
        self._fallback_route = fallback_route
        # A classification whose confidence line is MISSING counts
        # as confidence 0.0 whenever this threshold is > 0 (falls
        # to the fallback route / unresolved) — otherwise a
        # confidence-less classifier output would bypass the gate.
        self._min_confidence = require_confidence_above
        self._classifier_prompt = (
            classifier_prompt or DEFAULT_CLASSIFIER_PROMPT
        )
        # Persistent-subagent wiring — when Team.router was built
        # with ``persistent_subagents=True``, the chosen specialist
        # runs under its registered handle's stable session_id so
        # successive routes to the same specialist (e.g. follow-ups
        # in the same REPL session) reuse memory. Has no effect when
        # ``conversation_scope='shared'`` — see ``run()`` for why.
        self._worker_registry = worker_registry
        self._role_to_worker_id = role_to_worker_id
        # ``per_route`` (default) — every route runs under its own
        # derived session_id (``{parent}__route_{name}``) and is its
        # own conversation. Right primitive for fan-out / isolated-
        # domain routing (multilingual, security domains, A/B).
        #
        # ``shared`` — every route runs under the PARENT session_id
        # and contributes to one conversation. Right primitive for
        # chat frontends where routes are an implementation detail
        # (loom-code's SIMPLE-vs-COMPLEX classifier) and the user
        # expects "I'm having ONE conversation regardless of which
        # specialist answered the last turn." In shared mode,
        # persistent-subagent session_ids are also bypassed so all
        # turns land under the same parent session for rehydration.
        self._conversation_scope: Literal["per_route", "shared"] = (
            conversation_scope
        )

    def declared_workers(self) -> dict[str, Agent]:
        return {r.name: r.agent for r in self._routes}

    async def run(
        self,
        session: AgentSession,
        deps: Dependencies,
        prompt: str,
    ) -> AsyncIterator[Event]:
        # === 1. Classify ===
        descriptions = "\n".join(
            f"  - {r.name}: {r.description or '(no description)'}"
            for r in self._routes
        )
        classifier_prompt = self._classifier_prompt.format(
            route_descriptions=descriptions
        )
        msgs = [
            Message(role=Role.SYSTEM, content=classifier_prompt),
            Message(role=Role.USER, content=prompt),
        ]
        classification_text, usage = await text_only_model_call(
            deps, "router_classify", msgs
        )
        await consume_usage(deps, session, usage)

        parsed_route, parsed_confidence = _parse_classification(
            classification_text
        )
        if parsed_confidence is None:
            # The classifier omitted (or garbled) the confidence
            # line. When a threshold is configured, treat missing as
            # BELOW threshold (0.0) so ``require_confidence_above``
            # keeps meaning — the pre-fix default of 1.0 let any
            # confidence-less classification sail past the gate.
            # Without a threshold, keep the legacy "assume sure"
            # 1.0 (the value is only reported, never compared).
            confidence = 0.0 if self._min_confidence > 0.0 else 1.0
        else:
            confidence = parsed_confidence
        yield Event.architecture_event(
            session.id,
            "router.classified",
            route=parsed_route,
            confidence=confidence,
            raw=classification_text,
        )

        # === 2. Resolve to a real route ===
        chosen = self._resolve_route(parsed_route, confidence)

        if chosen is None:
            yield Event.architecture_event(
                session.id,
                "router.unresolved",
                attempted=parsed_route,
                confidence=confidence,
                min_confidence=self._min_confidence,
            )
            session.output = (
                f"Could not route this request. "
                f"Parsed route: {parsed_route!r}, "
                f"confidence: {confidence:.2f}."
            )
            return

        yield Event.architecture_event(
            session.id,
            "router.dispatched",
            route=chosen.name,
        )

        # === 3. Dispatch to specialist ===
        # SubagentInvocation forwards the specialist's MODEL_CHUNK /
        # TOOL_CALL / TOOL_RESULT events into our generator so
        # token-by-token streaming surfaces in the outer
        # `agent.stream(...)` consumer.
        #
        # session_id resolution depends on ``conversation_scope``:
        #
        # * ``per_route`` (default): derive ``{parent}__route_{name}``
        #   so each route is its own conversation, layered under the
        #   parent's journal. Persistent-subagent registry can
        #   further override this with a stable per-worker session_id
        #   (``resolve_persistent_session`` returns that handle).
        #
        # * ``shared``: pass the PARENT session_id straight through
        #   so all routes contribute to ONE conversation. Skip the
        #   persistent-subagent registry entirely — its whole point
        #   is per-worker memory across delegations, and in shared
        #   mode the "memory" IS the parent's session_id. Re-running
        #   the same prompt on a different route still rehydrates the
        #   prior turns from the parent session — that's the feature.
        from ..agent.worker_registry import (
            CrossUserWorkerError,
            acquire_worker_session,
            resolve_persistent_session,
        )
        if self._conversation_scope == "shared":
            specialist_session_id = session.id
            handle = None
        else:
            specialist_session_id, handle = resolve_persistent_session(
                chosen.name,
                fallback=f"{session.id}__route_{chosen.name}",
                registry=self._worker_registry,
                role_to_id=self._role_to_worker_id,
            )
        invocation = SubagentInvocation(
            chosen.agent,
            prompt,
            session_id=specialist_session_id,
            rollup_into=session,
        )
        # Memory propagation — specialist inherits coordinator's
        # memory backend when it has no explicit memory= of its own.
        with inherit_ambient_memory(deps.memory):
            if handle is not None:
                # Cross-user check + lock + touch, held for the
                # specialist's whole run so concurrent dispatches
                # to the same persistent worker serialise.
                try:
                    async with acquire_worker_session(
                        handle, deps.context.user_id
                    ):
                        async for ev in invocation.events():
                            yield ev
                except CrossUserWorkerError as exc:
                    session.interrupted = True
                    session.interruption_reason = (
                        f"specialist:{chosen.name}:cross_user_worker"
                    )
                    session.output = f"Could not route: {exc}"
                    yield Event.architecture_event(
                        session.id,
                        "router.cross_user_rejected",
                        route=chosen.name,
                        error=str(exc),
                    )
                    return
            else:
                async for ev in invocation.events():
                    yield ev

        result_dict = invocation.result
        # Charge the specialist's spend against the parent budget
        # (cumulative_usage was already rolled up by
        # ``rollup_into=session``).
        await consume_worker_usage(
            deps, chosen.agent, usage_from_result_dict(result_dict)
        )
        session.output = str(result_dict.get("output", ""))
        session.turns += int(result_dict.get("turns", 0) or 0)
        interrupted = bool(result_dict.get("interrupted", False))
        interruption_reason = result_dict.get("interruption_reason")
        if interrupted:
            session.interrupted = True
            session.interruption_reason = (
                f"specialist:{chosen.name}:"
                f"{interruption_reason or 'unknown'}"
            )

        yield Event.architecture_event(
            session.id,
            "router.completed",
            route=chosen.name,
            specialist_session_id=specialist_session_id,
            specialist_turns=int(result_dict.get("turns", 0) or 0),
            specialist_interrupted=interrupted,
        )

    def _resolve_route(
        self, parsed_name: str, confidence: float
    ) -> RouterRoute | None:
        """Map (parsed_name, confidence) to a RouterRoute or None.

        Logic:
        * confidence < threshold → fallback if set, else None
        * parsed_name unknown → fallback if set, else None
        * otherwise → exact match
        """
        if confidence < self._min_confidence:
            if self._fallback_route is not None:
                return self._routes_by_name[self._fallback_route]
            return None
        match = self._routes_by_name.get(parsed_name)
        if match is None:
            if self._fallback_route is not None:
                return self._routes_by_name[self._fallback_route]
            return None
        return match


# ---------------------------------------------------------------------------
# Classifier output parsing
# ---------------------------------------------------------------------------

_ROUTE_RE = re.compile(r"route\s*[:=]\s*([\w\-./]+)", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(
    r"confidence\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE
)


def _parse_classification(text: str) -> tuple[str, float | None]:
    """Extract ``(route_name, confidence)`` from classifier output.

    Looks for ``route: X`` and ``confidence: Y`` lines (case
    insensitive, ``=`` accepted as separator). If the route line is
    missing returns ``("", 0.0)``. If confidence is missing (or
    unparsable) returns ``None`` for it — the Router decides what a
    missing confidence means: 0.0 (below-threshold) when
    ``require_confidence_above`` is configured, 1.0 otherwise.
    Present confidences are clamped to ``[0, 1]``.
    """
    route_match = _ROUTE_RE.search(text)
    confidence_match = _CONFIDENCE_RE.search(text)

    if route_match is None:
        return "", 0.0

    route = route_match.group(1).strip()
    if confidence_match is None:
        return route, None

    try:
        confidence = float(confidence_match.group(1))
    except ValueError:
        return route, None
    return route, max(0.0, min(1.0, confidence))
