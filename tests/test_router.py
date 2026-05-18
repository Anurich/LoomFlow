"""Router architecture tests.

Covers:

* Protocol satisfaction.
* Constructor validation: empty routes, duplicate names, invalid
  fallback, confidence range.
* :func:`_parse_classification` regex behaviour: route + confidence,
  missing confidence (defaults to 1.0), missing route (returns
  empty + 0.0), confidence clamping.
* Successful classification + dispatch (specialist's output becomes
  Router's output).
* Confidence below threshold + fallback → fallback specialist runs.
* Confidence below threshold + no fallback → unresolved (graceful
  failure).
* Unknown route name + fallback → fallback runs.
* Unknown route name + no fallback → unresolved.
* :meth:`Router.declared_workers` exposes routes by name.
* Specialist's interruption surfaces in parent session.
* Deterministic specialist session_id for replay correctness.
* Architecture progress events visible via ``Agent.stream``.
"""

from __future__ import annotations

from typing import Any

import pytest

from loomflow import Agent, Architecture, ScriptedModel, ScriptedTurn
from loomflow.architecture import Router, RouterRoute
from loomflow.architecture.router import _parse_classification

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Protocol surface
# ---------------------------------------------------------------------------


def _make_specialist(text: str) -> Agent:
    """Build a specialist Agent that produces a fixed text response."""
    model = ScriptedModel([ScriptedTurn(text=text)])
    return Agent("specialist", model=model)


def test_router_satisfies_architecture_protocol() -> None:
    routes = [
        RouterRoute(name="a", agent=_make_specialist("alpha"))
    ]
    assert isinstance(Router(routes=routes), Architecture)


def test_router_name_is_router() -> None:
    routes = [
        RouterRoute(name="a", agent=_make_specialist("alpha"))
    ]
    assert Router(routes=routes).name == "router"


def test_router_declared_workers_exposes_routes_by_name() -> None:
    a, b = _make_specialist("a"), _make_specialist("b")
    router = Router(
        routes=[
            RouterRoute(name="alpha", agent=a),
            RouterRoute(name="beta", agent=b),
        ]
    )
    workers = router.declared_workers()
    assert workers == {"alpha": a, "beta": b}


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_router_rejects_empty_routes() -> None:
    with pytest.raises(ValueError, match="at least one route"):
        Router(routes=[])


def test_router_rejects_duplicate_route_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        Router(
            routes=[
                RouterRoute(name="x", agent=_make_specialist("1")),
                RouterRoute(name="x", agent=_make_specialist("2")),
            ]
        )


def test_router_rejects_unknown_fallback_route() -> None:
    with pytest.raises(ValueError, match="fallback_route"):
        Router(
            routes=[
                RouterRoute(name="a", agent=_make_specialist("1")),
            ],
            fallback_route="nonexistent",
        )


def test_router_rejects_confidence_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="confidence"):
        Router(
            routes=[
                RouterRoute(name="a", agent=_make_specialist("1")),
            ],
            require_confidence_above=1.5,
        )
    with pytest.raises(ValueError, match="confidence"):
        Router(
            routes=[
                RouterRoute(name="a", agent=_make_specialist("1")),
            ],
            require_confidence_above=-0.1,
        )


# ---------------------------------------------------------------------------
# Classification parser
# ---------------------------------------------------------------------------


def test_parse_classification_two_line_format() -> None:
    text = "route: billing\nconfidence: 0.92\nClear billing query."
    route, conf = _parse_classification(text)
    assert route == "billing"
    assert conf == 0.92


def test_parse_classification_handles_equals_separator() -> None:
    text = "route=tech\nconfidence=0.7"
    route, conf = _parse_classification(text)
    assert route == "tech"
    assert conf == 0.7


def test_parse_classification_missing_confidence_defaults_to_one() -> None:
    """If the model doesn't emit confidence, treat it as fully
    confident (the safer default — otherwise we'd reject everything
    when ``require_confidence_above > 0``)."""
    text = "route: sales"
    route, conf = _parse_classification(text)
    assert route == "sales"
    assert conf == 1.0


def test_parse_classification_missing_route_returns_empty() -> None:
    text = "confidence: 0.9 but no route specified"
    route, conf = _parse_classification(text)
    assert route == ""
    assert conf == 0.0


def test_parse_classification_clamps_confidence_to_unit_interval() -> None:
    text = "route: x\nconfidence: 1.5"
    _, conf = _parse_classification(text)
    assert conf == 1.0


# ---------------------------------------------------------------------------
# Successful dispatch
# ---------------------------------------------------------------------------


async def test_router_classifies_and_dispatches_to_specialist() -> None:
    """Classifier picks 'tech', dispatcher invokes tech specialist,
    specialist's output is the Router's output."""
    billing = _make_specialist("billing answer")
    tech = _make_specialist("tech answer")

    # Parent model is the classifier; specialists have their own.
    parent_model = ScriptedModel(
        [ScriptedTurn(text="route: tech\nconfidence: 0.95")]
    )
    agent = Agent(
        "router-host",
        model=parent_model,
        architecture=Router(
            routes=[
                RouterRoute(name="billing", agent=billing),
                RouterRoute(name="tech", agent=tech),
            ]
        ),
    )
    result = await agent.run("my server is down")
    assert result.output == "tech answer"


# ---------------------------------------------------------------------------
# Confidence threshold + fallback
# ---------------------------------------------------------------------------


async def test_router_uses_fallback_when_confidence_below_threshold() -> None:
    billing = _make_specialist("billing")
    general = _make_specialist("general handles edge cases")

    parent_model = ScriptedModel(
        [ScriptedTurn(text="route: billing\nconfidence: 0.4")]
    )
    agent = Agent(
        "test",
        model=parent_model,
        architecture=Router(
            routes=[
                RouterRoute(name="billing", agent=billing),
                RouterRoute(name="general", agent=general),
            ],
            fallback_route="general",
            require_confidence_above=0.7,
        ),
    )
    result = await agent.run("ambiguous query")
    assert result.output == "general handles edge cases"


async def test_router_unresolved_when_low_confidence_and_no_fallback() -> None:
    billing = _make_specialist("billing")

    parent_model = ScriptedModel(
        [ScriptedTurn(text="route: billing\nconfidence: 0.3")]
    )
    agent = Agent(
        "test",
        model=parent_model,
        architecture=Router(
            routes=[
                RouterRoute(name="billing", agent=billing),
            ],
            require_confidence_above=0.7,
        ),
    )
    result = await agent.run("anything")
    # Specialist did NOT run; output reports the routing failure.
    assert "Could not route" in result.output


# ---------------------------------------------------------------------------
# Unknown route name handling
# ---------------------------------------------------------------------------


async def test_router_falls_back_when_classifier_emits_unknown_route() -> None:
    """Classifier returns a route name we don't know; fallback runs."""
    billing = _make_specialist("billing")
    general = _make_specialist("general response")

    parent_model = ScriptedModel(
        [ScriptedTurn(text="route: nonexistent\nconfidence: 0.95")]
    )
    agent = Agent(
        "test",
        model=parent_model,
        architecture=Router(
            routes=[
                RouterRoute(name="billing", agent=billing),
                RouterRoute(name="general", agent=general),
            ],
            fallback_route="general",
        ),
    )
    result = await agent.run("query")
    assert result.output == "general response"


async def test_router_unresolved_when_route_unknown_and_no_fallback() -> None:
    billing = _make_specialist("billing")

    parent_model = ScriptedModel(
        [ScriptedTurn(text="route: nope\nconfidence: 0.95")]
    )
    agent = Agent(
        "test",
        model=parent_model,
        architecture=Router(
            routes=[RouterRoute(name="billing", agent=billing)],
        ),
    )
    result = await agent.run("query")
    assert "Could not route" in result.output


# ---------------------------------------------------------------------------
# Specialist interruption propagation
# ---------------------------------------------------------------------------


async def test_router_propagates_specialist_interruption() -> None:
    """If the specialist Agent is interrupted (max_turns / budget),
    the parent Router's session_id reflects that — the parent's
    interruption_reason is namespaced with the route name."""
    # Specialist with max_turns=0 will interrupt immediately on its
    # first turn-cap check.
    specialist = Agent(
        "specialist",
        model=ScriptedModel([ScriptedTurn(text="never run")]),
        max_turns=0,
    )
    parent_model = ScriptedModel(
        [ScriptedTurn(text="route: x\nconfidence: 0.95")]
    )
    agent = Agent(
        "test",
        model=parent_model,
        architecture=Router(
            routes=[RouterRoute(name="x", agent=specialist)],
        ),
    )
    result = await agent.run("anything")
    assert result.interrupted
    assert result.interruption_reason is not None
    assert "specialist:x:" in result.interruption_reason


# ---------------------------------------------------------------------------
# Deterministic specialist session_id
# ---------------------------------------------------------------------------


async def test_router_rejects_invalid_conversation_scope() -> None:
    """Only ``per_route`` and ``shared`` are valid — anything else is
    a wiring mistake that should fail loud at construction."""
    with pytest.raises(ValueError, match="conversation_scope"):
        Router(
            routes=[
                RouterRoute(name="a", agent=Agent("x", model="echo"))
            ],
            conversation_scope="something_else",  # type: ignore[arg-type]
        )


async def test_router_shared_scope_uses_parent_session_id() -> None:
    """``conversation_scope='shared'`` makes the specialist run under
    the PARENT session_id verbatim (no ``__route_<name>`` suffix, no
    persistent-worker stable id override). That's the contract chat
    frontends rely on for cross-route history rehydration."""
    captured: list[str] = []

    class _CaptureModel:
        name = "capture"

        async def stream(self, messages, *, tools=None, **kwargs):  # type: ignore[no-untyped-def]
            from loomflow.core.types import ModelChunk, Usage
            yield ModelChunk(kind="text", text="ok")
            yield ModelChunk(kind="finish", usage=Usage())

    class _SnoopAgent(Agent):
        async def run(  # type: ignore[override]
            self, prompt: str, **kwargs: Any
        ):
            captured.append(kwargs["session_id"])
            return await super().run(prompt, **kwargs)

    specialist = _SnoopAgent("snooper", model=_CaptureModel())  # type: ignore[arg-type]
    parent_model = ScriptedModel(
        [ScriptedTurn(text="route: snoop\nconfidence: 0.95")]
    )
    agent = Agent(
        "test",
        model=parent_model,
        architecture=Router(
            routes=[RouterRoute(name="snoop", agent=specialist)],
            conversation_scope="shared",
        ),
    )
    result = await agent.run("hi")
    assert len(captured) == 1
    # Equals — not endswith. Shared scope means EXACTLY the parent's
    # session_id; no derivation.
    assert captured[0] == result.session_id
    assert "__route_" not in captured[0]


async def test_router_shared_scope_bypasses_persistent_subagent_id() -> None:
    """When persistent_subagents are wired (which Team.router does
    by default) AND scope is 'shared', the parent session_id wins
    over the persistent-worker handle's stable session_id. Without
    this guarantee, the shared-conversation feature breaks the
    moment persistent_subagents is on (which is the default)."""
    captured: list[str] = []

    class _CaptureModel:
        name = "capture"

        async def stream(self, messages, *, tools=None, **kwargs):  # type: ignore[no-untyped-def]
            from loomflow.core.types import ModelChunk, Usage
            yield ModelChunk(kind="text", text="ok")
            yield ModelChunk(kind="finish", usage=Usage())

    class _SnoopAgent(Agent):
        async def run(  # type: ignore[override]
            self, prompt: str, **kwargs: Any
        ):
            captured.append(kwargs["session_id"])
            return await super().run(prompt, **kwargs)

    specialist = _SnoopAgent("snooper", model=_CaptureModel())  # type: ignore[arg-type]
    parent_model = ScriptedModel(
        [ScriptedTurn(text="route: snoop\nconfidence: 0.95")]
    )
    # Construct a fake worker registry that WOULD return a stable
    # session_id if not bypassed — mirrors what Team.router does
    # when persistent_subagents=True.
    from loomflow.agent.worker_registry import build_worker_registry
    registry, role_to_id = build_worker_registry({"snoop": specialist})
    agent = Agent(
        "test",
        model=parent_model,
        architecture=Router(
            routes=[RouterRoute(name="snoop", agent=specialist)],
            worker_registry=registry,
            role_to_worker_id=role_to_id,
            conversation_scope="shared",
        ),
    )
    result = await agent.run("hi")
    assert len(captured) == 1
    # The persistent-worker handle's session_id starts with
    # ``persistent_worker_``. Shared scope must NOT use it.
    assert not captured[0].startswith("persistent_worker_")
    assert captured[0] == result.session_id


async def test_router_uses_deterministic_specialist_session_id() -> None:
    """The specialist's session_id should follow the
    ``{parent}__route_{name}`` pattern so replay finds it."""
    captured_session_ids: list[str] = []

    class _CaptureModel:
        name = "capture"

        async def stream(self, messages, *, tools=None, **kwargs):  # type: ignore[no-untyped-def]
            from loomflow.core.types import ModelChunk, Usage

            yield ModelChunk(
                kind="text", text="ok"
            )
            yield ModelChunk(kind="finish", usage=Usage())

    class _SnoopAgent(Agent):
        async def run(  # type: ignore[override]
            self, prompt: str, **kwargs: Any
        ):
            sid = kwargs.get("session_id")
            assert sid is not None
            captured_session_ids.append(sid)
            return await super().run(prompt, **kwargs)

    specialist = _SnoopAgent("snooper", model=_CaptureModel())  # type: ignore[arg-type]

    parent_model = ScriptedModel(
        [ScriptedTurn(text="route: snoop\nconfidence: 0.95")]
    )
    agent = Agent(
        "test",
        model=parent_model,
        architecture=Router(
            routes=[RouterRoute(name="snoop", agent=specialist)],
        ),
    )
    result = await agent.run("hi")
    assert len(captured_session_ids) == 1
    assert captured_session_ids[0].endswith("__route_snoop")
    # And it should start with the parent's session id.
    assert captured_session_ids[0].startswith(result.session_id + "__")


# ---------------------------------------------------------------------------
# Architecture events surface in stream
# ---------------------------------------------------------------------------


async def test_router_emits_classified_and_dispatched_and_completed_events() -> None:
    specialist = _make_specialist("answer")
    parent_model = ScriptedModel(
        [ScriptedTurn(text="route: only\nconfidence: 0.95")]
    )
    agent = Agent(
        "test",
        model=parent_model,
        architecture=Router(
            routes=[RouterRoute(name="only", agent=specialist)],
        ),
    )
    events = [e async for e in agent.stream("go")]
    arch_names = [
        e.payload["name"]
        for e in events
        if e.kind == "architecture_event"
    ]
    assert "router.classified" in arch_names
    assert "router.dispatched" in arch_names
    assert "router.completed" in arch_names


async def test_router_emits_unresolved_event_on_routing_failure() -> None:
    parent_model = ScriptedModel(
        [ScriptedTurn(text="route: missing\nconfidence: 0.1")]
    )
    agent = Agent(
        "test",
        model=parent_model,
        architecture=Router(
            routes=[
                RouterRoute(name="known", agent=_make_specialist("x")),
            ],
            require_confidence_above=0.5,
        ),
    )
    events = [e async for e in agent.stream("go")]
    arch_names = [
        e.payload["name"]
        for e in events
        if e.kind == "architecture_event"
    ]
    assert "router.unresolved" in arch_names
    # Should NOT have dispatched or completed.
    assert "router.dispatched" not in arch_names
    assert "router.completed" not in arch_names
