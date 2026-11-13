"""A classifier reply that is just a route name is routed."""

from __future__ import annotations

import pytest

from loomflow import Agent, ScriptedModel, ScriptedTurn
from loomflow.architecture import RouterRoute
from loomflow.team import Team

pytestmark = pytest.mark.anyio


def _router(reply: str, **kwargs: object) -> Agent:
    billing = Agent("billing", model=ScriptedModel([ScriptedTurn(text="refund issued")]))
    tech = Agent("tech", model=ScriptedModel([ScriptedTurn(text="rebooted")]))
    return Team.router(
        routes=[
            RouterRoute(name="billing", agent=billing, description="payments"),
            RouterRoute(name="tech_support", agent=tech, description="outages"),
        ],
        model=ScriptedModel([ScriptedTurn(text=reply)]),
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("reply", ["billing", "Billing", " `billing`\n", "**billing**", "billing."])
async def test_bare_route_name_is_routed(reply: str) -> None:
    assert (await _router(reply).run("I was charged twice")).output == "refund issued"


async def test_bare_multi_word_route_name_is_routed() -> None:
    assert (await _router("tech support").run("it is down")).output == "rebooted"


async def test_a_sentence_mentioning_a_route_is_still_not_routed() -> None:
    result = await _router("I think billing should take this").run("hm")
    assert result.output.startswith("Could not route this request")


async def test_bare_name_has_no_confidence_so_a_threshold_sends_it_to_the_fallback() -> None:
    router = _router("billing", require_confidence_above=0.5, fallback_route="tech_support")
    assert (await router.run("hm")).output == "rebooted"
