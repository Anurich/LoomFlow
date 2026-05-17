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
from typing import TYPE_CHECKING, Any

from ..core.context import inherit_ambient_memory
from ..core.types import Event, Message, Role
from .base import AgentSession, Dependencies
from .helpers import SubagentInvocation, add_usage, text_only_model_call

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
        self._routes = list(routes)
        self._routes_by_name = {r.name: r for r in routes}
        self._fallback_route = fallback_route
        self._min_confidence = require_confidence_above
        self._classifier_prompt = (
            classifier_prompt or DEFAULT_CLASSIFIER_PROMPT
        )
        # Persistent-subagent wiring — when Team.router was built
        # with ``persistent_subagents=True``, the chosen specialist
        # runs under its registered handle's stable session_id so
        # successive routes to the same specialist (e.g. follow-ups
        # in the same REPL session) reuse memory.
        self._worker_registry = worker_registry
        self._role_to_worker_id = role_to_worker_id

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
        await deps.budget.consume(
            tokens_in=usage.input_tokens,
            tokens_out=usage.output_tokens,
            cost_usd=usage.cost_usd,
        )
        session.cumulative_usage = add_usage(
            session.cumulative_usage, usage
        )
        session.turns += 1

        parsed_route, confidence = _parse_classification(
            classification_text
        )
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
        # Deterministic specialist session_id: replay finds the same
        # session under the specialist's own journal.
        # SubagentInvocation forwards the specialist's MODEL_CHUNK /
        # TOOL_CALL / TOOL_RESULT events into our generator so
        # token-by-token streaming surfaces in the outer
        # `agent.stream(...)` consumer.
        from ..agent.worker_registry import resolve_persistent_session
        specialist_session_id, handle = resolve_persistent_session(
            chosen.name,
            fallback=f"{session.id}__route_{chosen.name}",
            registry=self._worker_registry,
            role_to_id=self._role_to_worker_id,
        )
        if handle is not None:
            handle.touch(user_id=deps.context.user_id)
        invocation = SubagentInvocation(
            chosen.agent,
            prompt,
            session_id=specialist_session_id,
            rollup_into=session,
        )
        # Memory propagation — specialist inherits coordinator's
        # memory backend when it has no explicit memory= of its own.
        with inherit_ambient_memory(deps.memory):
            async for ev in invocation.events():
                yield ev

        result_dict = invocation.result
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


def _parse_classification(text: str) -> tuple[str, float]:
    """Extract ``(route_name, confidence)`` from classifier output.

    Looks for ``route: X`` and ``confidence: Y`` lines (case
    insensitive, ``=`` accepted as separator). If the route line is
    missing returns ``("", 0.0)``. If confidence is missing defaults
    to ``1.0`` (assume the model is sure if it didn't say otherwise).
    Confidence is clamped to ``[0, 1]``.
    """
    route_match = _ROUTE_RE.search(text)
    confidence_match = _CONFIDENCE_RE.search(text)

    if route_match is None:
        return "", 0.0

    route = route_match.group(1).strip()
    if confidence_match is None:
        return route, 1.0

    try:
        confidence = float(confidence_match.group(1))
    except ValueError:
        confidence = 1.0
    confidence = max(0.0, min(1.0, confidence))
    return route, confidence
