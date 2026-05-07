"""Swarm: peer agents pass control via a ``handoff`` tool.

OpenAI Swarm reference (late 2024, experimental). Anthropic Agent
Teams (Feb 2026) is the production answer that improved on the
original swarm idea by adding lightweight coordination.

⚠ Production warning
---------------------
The 2026 production literature is unanimous: swarm has goal-drift
and deadlock failure modes that hierarchical / graph topologies
don't. Use only for **exploratory or research-mode systems where
flow can't be pre-specified**. For production, prefer
:class:`Supervisor` (clear authority) or :class:`Router` (single
specialist owns the answer).

Pattern
-------

1. **Setup.** N peer :class:`Agent` instances; one designated as
   the entry agent (receives the first user message).
2. **Active turn.** The active agent runs to completion with one
   or more handoff tools injected. The model can call them (or
   not) freely during the turn.
3. **Detect handoff.** After the agent's turn ends, Swarm checks
   whether a handoff tool was called. If yes, switch active agent
   to the target and continue. If not, the agent's output is the
   final answer.
4. **Cycle / cap protection.** :data:`max_handoffs` caps total
   handoffs; ``detect_cycles`` watches for ``A→B→A→B`` patterns.

Tool-shape modes (legacy vs typed)
----------------------------------

By default, peers given as plain :class:`Agent` instances get a
single legacy tool::

    handoff(target: str, message: str = "")

This is the v0.5 shape — backwards-compatible.

For typed handoffs (the 2026 best-practice shape per
OpenAI Agents SDK), wrap a peer in :class:`Handoff` and supply an
``input_type`` (a Pydantic model). Each typed peer then gets its
own per-target tool::

    transfer_to_<name>(field1, field2, ...)   # typed args from the model

This gives the model a typed schema per target instead of a string
``message`` blob, and lets you supply an ``input_filter`` callback
to prune / transform the conversation history that the receiving
agent sees.

Replay correctness
------------------
Each peer turn uses a deterministic session id —
``{parent}__swarm_<peer>_<handoff_count>``. Replays of the
parent journal cache the per-turn results.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..core.types import Event
from ..tools.registry import Tool
from .base import AgentSession, Dependencies
from .helpers import SubagentInvocation

if TYPE_CHECKING:
    from ..agent.api import Agent


# ---------------------------------------------------------------------------
# Handoff config — wraps a peer Agent with typed-handoff metadata
# ---------------------------------------------------------------------------


InputFilter = Callable[[list[str], dict[str, Any]], str]
"""``(history, payload) -> prompt_string``. Receives the full
running history (list of message strings — agent outputs plus
transition markers) and the validated handoff payload, returns
the prompt the receiving agent should see. Use this to prune
context, summarize past turns, or strip private metadata."""


@dataclass(frozen=True)
class Handoff:
    """Per-peer handoff configuration.

    * ``agent`` — the peer :class:`Agent`.
    * ``input_type`` — optional Pydantic model. When set, the
      generated handoff tool's input schema mirrors this model's
      fields, so the calling model gets a typed schema (instead of
      a string ``message``). The validated payload is exposed to
      ``input_filter`` and surfaces in the ``swarm.handoff`` event.
    * ``input_filter`` — optional callback ``(history, payload)
      → prompt`` for selective context forwarding. Default behavior
      respects the Swarm's ``pass_full_history`` flag.
    * ``description`` — override the generated tool's description.
      Useful when the agent's name is opaque ("billing_v2") but
      the description should be user-friendly.
    * ``tool_name`` — override the auto-generated tool name. Default
      is ``"transfer_to_<key>"`` where ``<key>`` is the peer's key
      in the swarm's ``agents`` dict.
    """

    agent: Agent
    input_type: type[BaseModel] | None = None
    input_filter: InputFilter | None = None
    description: str | None = None
    tool_name: str | None = None


class Swarm:
    """Peer agents passing control through handoff tools."""

    name = "swarm"

    def __init__(
        self,
        *,
        agents: dict[str, Agent | Handoff],
        entry_agent: str,
        max_handoffs: int = 8,
        detect_cycles: bool = True,
        pass_full_history: bool = True,
        handoff_tool_name: str = "handoff",
    ) -> None:
        if not agents:
            raise ValueError("Swarm requires at least one peer agent")
        if entry_agent not in agents:
            raise ValueError(
                f"entry_agent {entry_agent!r} not in agents "
                f"({', '.join(agents.keys())})"
            )
        if max_handoffs < 0:
            raise ValueError("max_handoffs must be >= 0")

        # Normalize all peers to Handoff configs internally; plain
        # Agent values get an empty config (legacy untyped behavior).
        self._handoffs: dict[str, Handoff] = {
            k: v if isinstance(v, Handoff) else Handoff(agent=v)
            for k, v in agents.items()
        }
        # Convenience view of just the agents (used by introspection
        # and the public ``declared_workers()`` helper).
        self._agents: dict[str, Agent] = {
            k: h.agent for k, h in self._handoffs.items()
        }
        self._entry_agent = entry_agent
        self._max_handoffs = max_handoffs
        self._detect_cycles = detect_cycles
        self._pass_full_history = pass_full_history
        self._handoff_tool_name = handoff_tool_name
        # Mode: typed (per-target tools) if ANY peer declares an
        # input_type; otherwise legacy single-tool.
        self._typed_mode = any(
            h.input_type is not None for h in self._handoffs.values()
        )

    def declared_workers(self) -> dict[str, Agent]:
        return dict(self._agents)

    async def run(
        self,
        session: AgentSession,
        deps: Dependencies,
        prompt: str,
    ) -> AsyncIterator[Event]:
        active_name = self._entry_agent
        handoff_count = 0
        recent_handoffs: deque[tuple[str, str]] = deque(maxlen=4)
        history: list[str] = [prompt]
        last_output = ""

        yield Event.architecture_event(
            session.id,
            "swarm.started",
            entry_agent=active_name,
            num_peers=len(self._agents),
            mode="typed" if self._typed_mode else "legacy",
        )

        while True:
            active_agent = self._agents[active_name]
            yield Event.architecture_event(
                session.id,
                "swarm.active",
                agent=active_name,
                handoff_count=handoff_count,
            )

            # Build the input prompt for the active agent.
            if self._pass_full_history:
                active_prompt = "\n\n".join(history)
            else:
                active_prompt = history[-1]

            # The handoff tool(s) record the request via a closure
            # variable; we read it after the agent's turn ends.
            handoff_request: dict[str, Any] = {}
            handoff_tools = self._build_handoff_tools(handoff_request)

            # Run the active agent with the handoff tool(s) injected.
            # SubagentInvocation forwards the worker's MODEL_CHUNK /
            # TOOL_CALL / TOOL_RESULT events into our generator so
            # token-by-token streaming surfaces in the outer
            # `agent.stream(...)` consumer.
            sub_session_id = (
                f"{session.id}__swarm_{active_name}_{handoff_count}"
            )
            invocation = SubagentInvocation(
                active_agent,
                active_prompt,
                session_id=sub_session_id,
                extra_tools=handoff_tools,
            )
            async for ev in invocation.events():
                yield ev
            session.turns += int(invocation.result.get("turns", 0) or 0)
            last_output = str(invocation.result.get("output", ""))
            interrupted = bool(invocation.result.get("interrupted", False))
            interruption_reason = invocation.result.get(
                "interruption_reason"
            )

            if interrupted:
                session.interrupted = True
                session.interruption_reason = (
                    f"swarm:{active_name}:"
                    f"{interruption_reason or 'unknown'}"
                )
                session.output = last_output
                yield Event.architecture_event(
                    session.id,
                    "swarm.peer_interrupted",
                    agent=active_name,
                )
                return

            # Did the model call handoff?
            if not handoff_request:
                # No handoff → agent produced the final answer.
                session.output = last_output
                yield Event.architecture_event(
                    session.id,
                    "swarm.completed",
                    agent=active_name,
                    handoffs=handoff_count,
                )
                return

            target = str(handoff_request["target"])
            payload = dict(handoff_request.get("payload") or {})
            message = str(handoff_request.get("message", ""))

            # Record + cycle check
            recent_handoffs.append((active_name, target))
            if self._detect_cycles and _is_cycling(recent_handoffs):
                yield Event.architecture_event(
                    session.id,
                    "swarm.cycle_detected",
                    recent=list(recent_handoffs),
                )
                session.output = last_output
                return

            # Append agent output + transition to history. If the
            # target peer has an input_filter, that callback will
            # rewrite the active_prompt on the next iteration —
            # below.
            history.append(last_output or "[no output]")
            if message:
                transition = (
                    f"[Handoff: {active_name} → {target}] {message}"
                )
            else:
                transition = f"[Handoff: {active_name} → {target}]"
            history.append(transition)

            yield Event.architecture_event(
                session.id,
                "swarm.handoff",
                from_agent=active_name,
                to_agent=target,
                message=message,
                payload=payload,
                handoff_count=handoff_count + 1,
            )

            # If the receiving peer has an input_filter, run it.
            target_handoff = self._handoffs[target]
            if target_handoff.input_filter is not None:
                filtered_prompt = target_handoff.input_filter(
                    list(history), payload
                )
                # Replace the running history with a single condensed
                # entry so subsequent peers see the filtered view.
                history = [filtered_prompt]

            active_name = target
            handoff_count += 1

            if handoff_count >= self._max_handoffs:
                yield Event.architecture_event(
                    session.id,
                    "swarm.max_handoffs",
                    handoffs=handoff_count,
                )
                # Run the new active agent ONE more time with no
                # handoff tool? That conflicts with the design. The
                # spec says: just emit max_handoffs and use the
                # latest output. We've already captured the
                # PREVIOUS agent's output as last_output; that's
                # what we return.
                session.output = last_output
                return

    # -----------------------------------------------------------------
    # Tool factories — legacy single-tool vs typed per-target
    # -----------------------------------------------------------------

    def _build_handoff_tools(
        self, handoff_request: dict[str, Any]
    ) -> list[Tool]:
        if self._typed_mode:
            return [
                self._build_typed_tool(name, h, handoff_request)
                for name, h in self._handoffs.items()
            ]
        return [self._build_legacy_tool(handoff_request)]

    def _build_legacy_tool(
        self, handoff_request: dict[str, Any]
    ) -> Tool:
        agents = self._agents

        async def _handoff(target: str, message: str = "") -> str:
            if target not in agents:
                return (
                    f"Error: unknown peer {target!r}. "
                    f"Known: {', '.join(agents.keys())}"
                )
            handoff_request["target"] = target
            handoff_request["message"] = message
            return f"[handoff requested → {target}]"

        peer_names = list(agents.keys())
        # Build a per-peer description list so the model sees not
        # just the names but each peer's role at a glance.
        peer_descriptions = "\n".join(
            f"  - {name}: {(a.instructions or '').strip()[:120]}"
            for name, a in agents.items()
        )

        return Tool(
            name=self._handoff_tool_name,
            description=(
                "Hand off the conversation to another peer agent. "
                "Pass `target` (one of the configured peer names) "
                "and an optional `message` describing context to "
                "carry over.\n\n"
                f"Available peers:\n{peer_descriptions}"
            ),
            fn=_handoff,
            input_schema={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        # Enum tells strict-schema providers
                        # (Anthropic, OpenAI strict mode) the only
                        # valid values, so a hallucinated peer name
                        # is rejected at the API boundary instead
                        # of bouncing through our error path.
                        "enum": peer_names,
                        "description": (
                            "Name of the peer to hand off to. "
                            "Must be one of: "
                            f"{', '.join(peer_names)}."
                        ),
                    },
                    "message": {
                        "type": "string",
                        "description": (
                            "Optional context to pass along."
                        ),
                    },
                },
                "required": ["target"],
            },
        )

    def _build_typed_tool(
        self,
        target_name: str,
        handoff: Handoff,
        handoff_request: dict[str, Any],
    ) -> Tool:
        tool_name = handoff.tool_name or f"transfer_to_{target_name}"
        agent_desc = (handoff.agent.instructions or "").strip()
        if len(agent_desc) > 160:
            agent_desc = agent_desc[:157] + "..."
        description = handoff.description or (
            f"Hand off the conversation to peer {target_name!r}. "
            f"{agent_desc}"
        )

        if handoff.input_type is None:
            # Untyped peer in a typed-mode swarm: still emits a
            # per-target tool, but with a single optional `message`
            # field to keep the schema consistent.
            schema: dict[str, Any] = {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": (
                            "Optional context to carry to "
                            f"{target_name}."
                        ),
                    },
                },
                "required": [],
            }
        else:
            schema = handoff.input_type.model_json_schema()
            # Strip pydantic's $defs / title metadata that confuse
            # some model tool-call APIs.
            schema = _strip_pydantic_metadata(schema)

        input_type = handoff.input_type

        async def _typed_handoff(**kwargs: Any) -> str:
            if input_type is not None:
                try:
                    validated = input_type.model_validate(kwargs)
                    payload = validated.model_dump()
                except Exception as exc:  # noqa: BLE001 — surface to model
                    return (
                        f"Error: invalid handoff payload for "
                        f"{target_name}: {exc}"
                    )
            else:
                payload = dict(kwargs)
            handoff_request["target"] = target_name
            handoff_request["payload"] = payload
            # Synthesize a `message` for the transition marker so
            # legacy event consumers still see something readable.
            handoff_request["message"] = payload.get("message") or ""
            return f"[handoff requested → {target_name}]"

        return Tool(
            name=tool_name,
            description=description,
            fn=_typed_handoff,
            input_schema=schema,
        )


def _strip_pydantic_metadata(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove ``$defs``, ``title``, and other pydantic-isms that
    aren't required (or accepted) by every tool-calling model API."""
    out = {k: v for k, v in schema.items() if k not in {"$defs", "title"}}
    props = out.get("properties")
    if isinstance(props, dict):
        out["properties"] = {
            name: {k: v for k, v in spec.items() if k != "title"}
            if isinstance(spec, dict)
            else spec
            for name, spec in props.items()
        }
    return out


def _is_cycling(recent: deque[tuple[str, str]]) -> bool:
    """Detect ``A→B→A→B`` repetition in the most recent 4 handoffs.

    Conservative: only fires on exact 2-cycle repetition.
    Triple-cycles (``A→B→C→A→B→C``) and longer aren't caught here;
    the ``max_handoffs`` cap is the backstop.
    """
    if len(recent) < 4:
        return False
    return recent[0] == recent[2] and recent[1] == recent[3]
