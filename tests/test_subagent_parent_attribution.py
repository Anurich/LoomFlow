"""Tests for subagent parent-attribution metadata (0.10.18).

When ``SubagentInvocation`` spawns a child agent inside an active
parent run, the child's :class:`RunContext.metadata` now carries
two reserved keys recording who spawned it:

* ``_loomflow_parent_session_id``
* ``_loomflow_parent_run_id``

Useful for observability (telemetry/audit can attribute child
work back to its parent), for cache/memory partitioning by
parent, and for any custom tool that wants "am I running as a
subagent or directly?" — the keys are absent on direct
``Agent.run()`` calls.

Coverage:

* Reserved keys appear on a child's context.metadata when
  spawned via SubagentInvocation inside a parent run.
* User-supplied metadata isn't clobbered; reserved keys are
  added alongside.
* When called OUTSIDE an active parent run (no contextvar
  installed), no attribution keys are injected — the child runs
  anonymously, same as before.
* Explicit context= override still gets parent attribution
  layered on (additive metadata).
"""

from __future__ import annotations

import pytest

from loomflow import Agent, EchoModel
from loomflow.architecture.helpers import SubagentInvocation
from loomflow.core.context import RunContext, get_run_context, set_run_context

pytestmark = pytest.mark.anyio


async def test_subagent_inherits_parent_session_metadata() -> None:
    """Inside an active parent run, the child's context.metadata
    has ``_loomflow_parent_session_id`` set to the parent's
    session_id."""
    parent_ctx = RunContext(
        user_id="alice",
        session_id="parent-session-001",
        run_id="parent-run-001",
        metadata={"user_custom": "preserved"},
    )
    async with set_run_context(parent_ctx):
        # Inside the parent run scope — SubagentInvocation reads
        # the live contextvar and adds attribution.
        worker = Agent(instructions="", model=EchoModel())
        inv = SubagentInvocation(worker, "hello")
        # The invocation's snapshotted context is what the child
        # will see when its loop installs the new RunContext.
        ctx = inv._context  # internal — direct introspection
    assert (
        ctx.metadata.get("_loomflow_parent_session_id")
        == "parent-session-001"
    )
    assert (
        ctx.metadata.get("_loomflow_parent_run_id")
        == "parent-run-001"
    )
    # User metadata still there — attribution is additive.
    assert ctx.metadata.get("user_custom") == "preserved"


async def test_subagent_outside_parent_run_has_no_attribution() -> None:
    """When SubagentInvocation is constructed OUTSIDE an active
    parent run, ``get_run_context()`` returns the empty default
    (session_id=None, run_id=""). With no parent identity to
    record, the reserved keys are NOT injected — child runs
    anonymously, identical to pre-0.10.18 behaviour."""
    # No ``async with set_run_context`` wrapping this call → the
    # contextvar default applies.
    default_ctx = get_run_context()
    assert default_ctx.session_id is None or default_ctx.session_id == ""

    worker = Agent(instructions="", model=EchoModel())
    inv = SubagentInvocation(worker, "hello")
    ctx = inv._context
    # No parent identity → no attribution keys.
    assert "_loomflow_parent_session_id" not in ctx.metadata
    assert "_loomflow_parent_run_id" not in ctx.metadata


async def test_subagent_explicit_context_still_gets_attribution() -> None:
    """When the caller passes ``context=`` explicitly,
    SubagentInvocation still layers the parent-attribution
    metadata on top — additive, not destructive. The explicit
    context wins on user_id / session_id (overrides parent), but
    attribution is purely informational."""
    explicit = RunContext(
        user_id="bob",
        session_id="explicit-session",
        run_id="explicit-run",
        metadata={"custom": "value"},
    )
    worker = Agent(instructions="", model=EchoModel())
    inv = SubagentInvocation(worker, "hello", context=explicit)
    ctx = inv._context
    # The explicit context's identity wins.
    assert ctx.user_id == "bob"
    assert ctx.session_id == "explicit-session"
    # Attribution layered from the EXPLICIT context (it had its
    # own session_id + run_id — those become the child's view of
    # its "parent" since it's the closest ancestor identity).
    assert (
        ctx.metadata.get("_loomflow_parent_session_id")
        == "explicit-session"
    )
    assert (
        ctx.metadata.get("_loomflow_parent_run_id") == "explicit-run"
    )
    # User's own metadata preserved.
    assert ctx.metadata.get("custom") == "value"


async def test_attribution_does_not_clobber_existing_keys() -> None:
    """If the parent already has ``_loomflow_parent_session_id``
    in its own metadata (nested subagents — grandparent's
    attribution was already laid down), the new layer uses
    ``setdefault`` so the OUTER-most ancestor wins. Otherwise
    every depth would overwrite and you'd lose the original
    spawner identity."""
    grandparent_ctx = RunContext(
        user_id="alice",
        session_id="grandparent-session",
        run_id="grandparent-run",
        metadata={
            "_loomflow_parent_session_id": "very-original-session",
            "_loomflow_parent_run_id": "very-original-run",
        },
    )
    async with set_run_context(grandparent_ctx):
        worker = Agent(instructions="", model=EchoModel())
        inv = SubagentInvocation(worker, "hi")
        ctx = inv._context
    # ``setdefault`` preserved the original ancestor identity.
    assert (
        ctx.metadata["_loomflow_parent_session_id"]
        == "very-original-session"
    )
    assert (
        ctx.metadata["_loomflow_parent_run_id"]
        == "very-original-run"
    )
