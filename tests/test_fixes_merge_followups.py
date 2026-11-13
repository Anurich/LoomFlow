"""Follow-ups to the review fixes: things that sit between two areas."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from loomflow import Agent
from loomflow.architecture.supervisor import Supervisor
from loomflow.serve import create_app
from loomflow.team import Team
from loomflow.workspace.disk import LocalDiskWorkspace

from .test_serve import _BoomModel, _parse_sse, _request

pytestmark = pytest.mark.anyio


async def test_stream_error_event_hides_exception_text() -> None:
    agent = Agent("x", model=_BoomModel())  # type: ignore[arg-type]
    resp = await _request(create_app(agent), "POST", "/stream", json_body={"prompt": "hi"})
    assert b"kaboom" not in resp.body.lower()
    errors = [json.loads(data) for name, data in _parse_sse(resp.body) if name == "error"]
    assert all(e["payload"]["type"] == "internal_error" for e in errors)


async def test_stream_error_event_keeps_text_in_debug() -> None:
    agent = Agent("x", model=_BoomModel())  # type: ignore[arg-type]
    resp = await _request(
        create_app(agent, debug=True), "POST", "/stream", json_body={"prompt": "hi"}
    )
    assert b"kaboom" in resp.body.lower()


def test_team_supervisor_passes_delegation_settings() -> None:
    worker = Agent("w", model="echo")
    team = Team.supervisor(
        {"w": worker}, model="echo", delegate_timeout=30.0, max_delegation_depth=2
    )
    arch = team._architecture
    assert isinstance(arch, Supervisor)
    assert arch._delegate_timeout == 30.0
    assert arch._max_delegation_depth == 2


def test_workspace_warns_once_about_a_folder_under_the_old_name(tmp_path: Path) -> None:
    (tmp_path / "alice_example.com").mkdir()
    ws = LocalDiskWorkspace(tmp_path)
    with pytest.warns(UserWarning, match="alice%40example.com"):
        root = ws._user_root("alice@example.com")
    assert root.name == "alice%40example.com"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ws._user_root("alice@example.com")  # once per id
        ws._user_root("bob")  # an id that never moved
