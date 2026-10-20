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

Matches the reference swarms (OpenAI Swarm, OpenAI Agents SDK
handoffs, LangGraph Swarm):

1. **Setup.** N peer :class:`Agent` instances; one designated as
   the entry agent (receives the first user message). ``handoffs``
   can limit who may hand off to whom; by default every peer may
   hand off to every *other* peer (never to itself).
2. **Active turn.** The active agent runs with its handoff tool(s)
   injected and may call them freely.
3. **Handoff.** A handoff tool is ``ends_turn``: the moment it
   succeeds the agent's turn ends and the target takes over. The
   target sees the shared thread: the request, each earlier peer's
   tool calls and their results, what it said, and a
   ``[Handoff: A → B]`` marker. If no handoff is called, the agent's
   output is the final answer.
4. **Next message.** Within the same ``session_id``, the user's next
   message goes to the peer that answered last
   (``resume_last_agent``; in-process, so a restart begins again at
   the entry agent).
5. **Loop protection.** ``detect_cycles`` stops two peers bouncing a
   request back and forth three times; ``max_handoffs`` caps the
   total. Either way the peer just handed the conversation answers
   once more with no handoff tools, and the run is marked
   ``interrupted`` (``swarm:cycle_detected`` / ``swarm:max_handoffs``).

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

import json
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..core.context import inherit_ambient_memory
from ..core.types import Event
from ..tools.registry import Tool
from .base import AgentSession, Dependencies
from .helpers import (
    SubagentInvocation,
    budget_gate,
    consume_worker_usage,
    usage_from_result_dict,
)

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


_CYCLE_WINDOW = 6
"""Handoffs the cycle check looks back over: three A↔B round trips."""

_RESUME_CAPACITY = 10_000
"""Conversations whose last active agent a Swarm remembers (LRU)."""

_TOOL_ARGS_CHARS = 300
_TOOL_OUTPUT_CHARS = 1500


@dataclass
class _PeerTurn:
    """What one peer's turn produced, read after its events drain."""

    output: str = ""
    interrupted: bool = False
    interruption_reason: str | None = None
    tool_lines: list[str] = field(default_factory=list)
    rejected: bool = False  # cross-user persistent worker; run already stopped


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
        handoffs: Mapping[str, Sequence[str]] | None = None,
        resume_last_agent: bool = True,
        worker_registry: dict[str, Any] | None = None,
        role_to_worker_id: dict[str, str] | None = None,
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
        self._targets = _resolve_targets(list(self._agents), handoffs)
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
        # Which peer last answered each conversation, so the user's
        # next message goes to it (OpenAI Agents SDK ``last_agent``,
        # LangGraph Swarm's ``active_agent``). In-process and bounded:
        # after a restart a conversation starts again at entry_agent.
        self._resume = resume_last_agent
        self._last_active: OrderedDict[tuple[str | None, str], str] = OrderedDict()
        # Persistent-subagent wiring — set when Team.swarm built us
        # with ``persistent_subagents=True``. The registry maps
        # ``worker_<role>_<ULID>`` → ``_WorkerHandle``; the
        # role-to-id map lets us translate the peer key
        # (e.g. ``"researcher"``) used in this architecture to the
        # handle. When unset, peers run with per-handoff session_ids
        # (legacy stateless behavior).
        self._worker_registry = worker_registry
        self._role_to_worker_id = role_to_worker_id

    def declared_workers(self) -> dict[str, Agent]:
        return dict(self._agents)

    async def run(
        self,
        session: AgentSession,
        deps: Dependencies,
        prompt: str,
    ) -> AsyncIterator[Event]:
        conversation = (deps.context.user_id, session.id)
        active_name = self._entry_agent
        resumed = False
        if self._resume:
            last = self._last_active.get(conversation)
            if last is not None and last in self._agents:
                active_name, resumed = last, True
        handoff_count = 0
        recent_handoffs: deque[tuple[str, str]] = deque(maxlen=_CYCLE_WINDOW)
        # history[0] is the request (or an input_filter's rewrite of
        # it); every handoff adds one segment: the handing-off agent's
        # tool calls, its text, and the transition marker.
        history: list[str] = [prompt]
        # Latest non-empty peer text: the best answer available if the
        # budget or a cross-user check stops the swarm mid-way.
        last_output = ""

        yield Event.architecture_event(
            session.id,
            "swarm.started",
            entry_agent=active_name,
            resumed=resumed,
            num_peers=len(self._agents),
            mode="typed" if self._typed_mode else "legacy",
        )

        while True:
            # Budget gate — one check per handoff iteration (mirrors
            # Blackboard's per-round gate), so an exhausted budget
            # can't keep spawning peers until max_handoffs.
            blocked, gate_events = await budget_gate(deps, session)
            for gate_event in gate_events:
                yield gate_event
            if blocked:
                session.output = last_output
                return

            yield Event.architecture_event(
                session.id,
                "swarm.active",
                agent=active_name,
                handoff_count=handoff_count,
            )

            # The handoff tool(s) record the request via a closure
            # variable; we read it after the agent's turn ends. They
            # are ``ends_turn`` tools, so the agent stops the moment
            # it hands off instead of making another model call.
            handoff_request: dict[str, Any] = {}
            handoff_tools = self._build_handoff_tools(active_name, handoff_request)
            turn = _PeerTurn()
            async for ev in self._run_peer(
                session,
                deps,
                active_name,
                self._prompt_for(history),
                handoff_tools,
                session_suffix=str(handoff_count),
                turn=turn,
            ):
                yield ev
            if turn.rejected:
                session.output = last_output
                return
            last_output = turn.output or last_output

            if turn.interrupted:
                session.interrupted = True
                session.interruption_reason = (
                    f"swarm:{active_name}:"
                    f"{turn.interruption_reason or 'unknown'}"
                )
                session.output = turn.output
                yield Event.architecture_event(
                    session.id,
                    "swarm.peer_interrupted",
                    agent=active_name,
                )
                return

            # Did the model call handoff?
            if not handoff_request:
                # No handoff → agent produced the final answer.
                session.output = turn.output
                self._remember(conversation, active_name)
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

            # The receiving peer gets what the user would see in a
            # shared transcript: the tools the previous peer ran and
            # their results, what it said, and why it handed off.
            marker = f"[Handoff: {active_name} → {target}]"
            history.append(
                "\n".join(
                    [
                        *turn.tool_lines,
                        *([turn.output] if turn.output else []),
                        f"{marker} {message}" if message else marker,
                    ]
                )
            )
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

            recent_handoffs.append((active_name, target))
            active_name = target
            handoff_count += 1

            stop: str | None = None
            if self._detect_cycles and _is_cycling(recent_handoffs):
                stop = "cycle_detected"
                yield Event.architecture_event(
                    session.id,
                    "swarm.cycle_detected",
                    recent=list(recent_handoffs),
                )
            elif handoff_count >= self._max_handoffs:
                stop = "max_handoffs"
                yield Event.architecture_event(
                    session.id,
                    "swarm.max_handoffs",
                    handoffs=handoff_count,
                )
            if stop is not None:
                # Cut short, but still answer: the peer that was just
                # handed the conversation gets one last turn with no
                # handoff tools. The run is flagged so callers can tell.
                final = _PeerTurn()
                async for ev in self._run_peer(
                    session,
                    deps,
                    active_name,
                    self._prompt_for(history)
                    + "\n\n[No more handoffs are possible: answer the request yourself.]",
                    [],
                    session_suffix=f"{handoff_count}_final",
                    turn=final,
                ):
                    yield ev
                if final.rejected:
                    session.output = last_output
                    return
                session.output = final.output
                session.interrupted = True
                session.interruption_reason = f"swarm:{stop}"
                self._remember(conversation, active_name)
                return

    # -----------------------------------------------------------------
    # One peer turn
    # -----------------------------------------------------------------

    async def _run_peer(
        self,
        session: AgentSession,
        deps: Dependencies,
        name: str,
        prompt: str,
        tools: list[Tool],
        *,
        session_suffix: str,
        turn: _PeerTurn,
    ) -> AsyncIterator[Event]:
        """Run peer ``name`` once, forwarding its events into ``turn``."""
        from ..agent.worker_registry import (
            CrossUserWorkerError,
            acquire_worker_session,
            resolve_persistent_session,
        )

        agent = self._agents[name]
        # Persistent path: the peer has a registered handle, so use the
        # handle's stable session_id (every handoff to the same peer
        # reuses the same memory partition); the lock serialises
        # concurrent calls. Otherwise a per-turn deterministic id, so
        # replays of the parent journal hit the cache.
        sub_session_id, handle = resolve_persistent_session(
            name,
            fallback=f"{session.id}__swarm_{name}_{session_suffix}",
            registry=self._worker_registry,
            role_to_id=self._role_to_worker_id,
        )
        invocation = SubagentInvocation(
            agent,
            prompt,
            session_id=sub_session_id,
            extra_tools=tools,
            rollup_into=session,
        )
        skip = {t.name for t in tools}
        calls: dict[str, tuple[str, Any]] = {}

        def note(ev: Event) -> None:
            # Keep the peer's own tool calls and their results for the
            # next peer; the handoff call itself becomes the marker.
            if ev.kind.value == "tool_call":
                call = ev.payload.get("call") or {}
                if call.get("tool") not in skip:
                    calls[str(call.get("id"))] = (str(call.get("tool")), call.get("args"))
            elif ev.kind.value == "tool_result":
                result = ev.payload.get("result") or {}
                found = calls.pop(str(result.get("call_id")), None)
                if found is not None:
                    turn.tool_lines.append(_render_tool_call(name, *found, result))

        # Memory propagation: install the coordinator's memory as
        # ambient so a worker constructed without explicit ``memory=``
        # inherits it.
        with inherit_ambient_memory(deps.memory):
            if handle is not None:
                # Cross-user check + lock + touch, held for the peer's
                # whole turn so concurrent invocations of the same
                # persistent worker serialise.
                try:
                    async with acquire_worker_session(handle, deps.context.user_id):
                        async for ev in invocation.events():
                            note(ev)
                            yield ev
                except CrossUserWorkerError as exc:
                    session.interrupted = True
                    session.interruption_reason = f"swarm:{name}:cross_user_worker"
                    turn.rejected = True
                    yield Event.architecture_event(
                        session.id,
                        "swarm.cross_user_rejected",
                        agent=name,
                        error=str(exc),
                    )
                    return
            else:
                async for ev in invocation.events():
                    note(ev)
                    yield ev

        session.turns += int(invocation.result.get("turns", 0) or 0)
        # Charge the peer's spend against the parent budget
        # (cumulative_usage was already rolled up by rollup_into).
        await consume_worker_usage(deps, agent, usage_from_result_dict(invocation.result))
        turn.output = str(invocation.result.get("output", "") or "")
        turn.interrupted = bool(invocation.result.get("interrupted", False))
        turn.interruption_reason = invocation.result.get("interruption_reason")

    def _prompt_for(self, history: list[str]) -> str:
        if self._pass_full_history or len(history) == 1:
            return "\n\n".join(history)
        # The request plus the latest handoff segment.
        return f"{history[0]}\n\n{history[-1]}"

    def _remember(self, conversation: tuple[str | None, str], name: str) -> None:
        if not self._resume:
            return
        self._last_active[conversation] = name
        self._last_active.move_to_end(conversation)
        while len(self._last_active) > _RESUME_CAPACITY:
            self._last_active.popitem(last=False)

    # -----------------------------------------------------------------
    # Tool factories — legacy single-tool vs typed per-target
    # -----------------------------------------------------------------

    def _build_handoff_tools(
        self, active_name: str, handoff_request: dict[str, Any]
    ) -> list[Tool]:
        targets = self._targets[active_name]
        if not targets:
            return []
        if self._typed_mode:
            return [
                self._build_typed_tool(name, self._handoffs[name], handoff_request)
                for name in targets
            ]
        return [self._build_legacy_tool(targets, handoff_request)]

    def _build_legacy_tool(
        self, targets: tuple[str, ...], handoff_request: dict[str, Any]
    ) -> Tool:
        agents = self._agents

        async def _handoff(target: str, message: str = "") -> str:
            # Raising makes the call fail, so the turn continues and the
            # model sees the error instead of the run switching peers.
            if target not in targets:
                raise ValueError(
                    f"cannot hand off to {target!r}; choose one of: "
                    f"{', '.join(targets)}"
                )
            _claim(handoff_request, target)
            handoff_request["message"] = message
            return f"[handoff requested → {target}]"

        # Build a per-peer description list so the model sees not
        # just the names but each peer's role at a glance.
        peer_descriptions = "\n".join(
            f"  - {name}: {(agents[name].instructions or '').strip()[:120]}"
            for name in targets
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
            ends_turn=True,
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
                        "enum": list(targets),
                        "description": (
                            "Name of the peer to hand off to. "
                            "Must be one of: "
                            f"{', '.join(targets)}."
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
                    payload = input_type.model_validate(kwargs).model_dump()
                except Exception as exc:  # noqa: BLE001 — surface to model
                    # Fail the call so the model can retry with valid args.
                    raise ValueError(
                        f"invalid handoff payload for {target_name}: {exc}"
                    ) from exc
            else:
                payload = dict(kwargs)
            _claim(handoff_request, target_name)
            handoff_request["payload"] = payload
            # Synthesize a `message` for the transition marker so
            # legacy event consumers still see something readable.
            handoff_request["message"] = payload.get("message") or ""
            return f"[handoff requested → {target_name}]"

        return Tool(
            name=tool_name,
            description=description,
            fn=_typed_handoff,
            ends_turn=True,
            input_schema=schema,
        )


def _resolve_targets(
    names: list[str], handoffs: Mapping[str, Sequence[str]] | None
) -> dict[str, tuple[str, ...]]:
    """Who each peer may hand off to. Never itself.

    Default: every other peer. With an explicit ``handoffs`` map, only
    the listed targets; a peer missing from the map can't hand off.
    """
    if handoffs is None:
        return {n: tuple(m for m in names if m != n) for n in names}
    unknown = [k for k in handoffs if k not in names]
    if unknown:
        raise ValueError(f"handoffs names unknown peers: {', '.join(unknown)}")
    resolved: dict[str, tuple[str, ...]] = {}
    for n in names:
        targets = tuple(dict.fromkeys(handoffs.get(n, ())))
        bad = [t for t in targets if t not in names]
        if bad:
            raise ValueError(f"handoffs[{n!r}] names unknown peers: {', '.join(bad)}")
        if n in targets:
            raise ValueError(f"handoffs[{n!r}] lists {n!r} itself")
        resolved[n] = targets
    return resolved


def _claim(handoff_request: dict[str, Any], target: str) -> None:
    # Parallel tool calls can request two handoffs in one turn; the
    # first wins and the others fail back to the model.
    if "target" in handoff_request:
        raise ValueError(f"already handing off to {handoff_request['target']!r}")
    handoff_request["target"] = target


def _render_tool_call(agent: str, tool: str, args: Any, result: dict[str, Any]) -> str:
    args_text = _clip(json.dumps(args or {}, ensure_ascii=False, default=str), _TOOL_ARGS_CHARS)
    if result.get("ok"):
        output = result.get("output")
        out_text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False, default=str)
    else:
        out_text = f"failed: {result.get('error') or result.get('reason') or 'unknown error'}"
    return f"[{agent} called {tool}({args_text}) → {_clip(out_text, _TOOL_OUTPUT_CHARS)}]"


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


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
    """Detect two peers bouncing a request back and forth.

    True once the last :data:`_CYCLE_WINDOW` handoffs (three round
    trips, ``A→B→A→B→A→B``) alternate between the same pair. A single
    hand-back, or two, is normal; longer loops through three or more
    peers are left to the ``max_handoffs`` cap.
    """
    if len(recent) < _CYCLE_WINDOW:
        return False
    a, b = recent[0]
    return a != b and all(
        pair == ((a, b) if i % 2 == 0 else (b, a)) for i, pair in enumerate(recent)
    )
