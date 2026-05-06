"""Audit log tests — in-memory + file backends, signing, Agent wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jeevesagent import (
    Agent,
    FileAuditLog,
    InMemoryAuditLog,
    tool,
)
from jeevesagent.core.types import ToolCall
from jeevesagent.model.scripted import ScriptedModel, ScriptedTurn
from jeevesagent.security.audit import verify_signature

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# InMemoryAuditLog
# ---------------------------------------------------------------------------


async def test_in_memory_audit_log_appends_with_monotonic_seq() -> None:
    log = InMemoryAuditLog(secret="s3cr3t")
    e1 = await log.append(
        session_id="s1", actor="user", action="run_started", payload={}
    )
    e2 = await log.append(
        session_id="s1", actor="user", action="run_completed", payload={}
    )
    assert e1.seq == 1
    assert e2.seq == 2
    assert e1.timestamp <= e2.timestamp


async def test_in_memory_audit_log_signature_verifies() -> None:
    log = InMemoryAuditLog(secret="s3cr3t")
    entry = await log.append(
        session_id="s1",
        actor="user",
        action="run_started",
        payload={"prompt": "hi"},
    )
    assert verify_signature(entry, "s3cr3t")
    assert not verify_signature(entry, "wrong-secret")


async def test_in_memory_audit_log_query_filters_by_session() -> None:
    log = InMemoryAuditLog()
    await log.append(session_id="s1", actor="u", action="x", payload={})
    await log.append(session_id="s2", actor="u", action="x", payload={})
    await log.append(session_id="s1", actor="u", action="y", payload={})

    s1 = await log.query(session_id="s1")
    assert {e.action for e in s1} == {"x", "y"}
    s2 = await log.query(session_id="s2")
    assert len(s2) == 1


async def test_in_memory_audit_log_query_filters_by_action() -> None:
    log = InMemoryAuditLog()
    await log.append(session_id="s1", actor="u", action="run_started", payload={})
    await log.append(session_id="s1", actor="u", action="tool_call", payload={})
    await log.append(session_id="s2", actor="u", action="tool_call", payload={})

    tool_calls = await log.query(action="tool_call")
    assert len(tool_calls) == 2


# ---------------------------------------------------------------------------
# FileAuditLog
# ---------------------------------------------------------------------------


async def test_file_audit_log_writes_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = FileAuditLog(path, secret="s3cr3t")
    await log.append(session_id="s1", actor="u", action="x", payload={"k": 1})
    await log.append(session_id="s1", actor="u", action="y", payload={"k": 2})

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["seq"] == 1
    assert parsed[1]["seq"] == 2
    assert parsed[0]["payload"] == {"k": 1}


async def test_file_audit_log_recovers_seq_on_restart(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log1 = FileAuditLog(path, secret="x")
    await log1.append(session_id="s1", actor="u", action="x", payload={})
    await log1.append(session_id="s1", actor="u", action="x", payload={})

    # New instance reading the same file should pick up at seq=3.
    log2 = FileAuditLog(path, secret="x")
    e3 = await log2.append(session_id="s1", actor="u", action="x", payload={})
    assert e3.seq == 3


async def test_file_audit_log_query_reads_back(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = FileAuditLog(path, secret="x")
    await log.append(
        session_id="s1", actor="user", action="run_started", payload={}
    )
    await log.append(
        session_id="s1", actor="model", action="tool_call", payload={"t": "ping"}
    )

    entries = await log.query(session_id="s1")
    assert len(entries) == 2
    assert entries[0].seq == 1
    assert entries[1].action == "tool_call"


async def test_file_audit_log_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "deep" / "audit.jsonl"
    FileAuditLog(path)  # should mkdir -p
    assert path.parent.exists()


# ---------------------------------------------------------------------------
# Agent wiring
# ---------------------------------------------------------------------------


async def test_agent_run_writes_run_started_and_completed() -> None:
    log = InMemoryAuditLog()
    agent = Agent("hi", model="echo", audit_log=log)
    result = await agent.run("hello")

    actions = [e.action for e in await log.query(session_id=result.session_id)]
    assert "run_started" in actions
    assert "run_completed" in actions
    assert actions.index("run_started") < actions.index("run_completed")


async def test_agent_writes_tool_call_and_result_audit_entries() -> None:
    @tool
    async def ping() -> str:
        """Return pong."""
        return "pong"

    model = ScriptedModel(
        [
            ScriptedTurn(
                tool_calls=[ToolCall(id="c1", tool="ping", args={})]
            ),
            ScriptedTurn(text="ok"),
        ]
    )
    log = InMemoryAuditLog()
    agent = Agent("hi", model=model, tools=[ping], audit_log=log)
    result = await agent.run("ping?")

    entries = await log.query(session_id=result.session_id)
    actions = [e.action for e in entries]
    assert "tool_call" in actions
    assert "tool_result" in actions

    # Payloads include call_id and tool name.
    tool_call_entries = [e for e in entries if e.action == "tool_call"]
    assert tool_call_entries[0].payload["tool"] == "ping"
    assert tool_call_entries[0].payload["call_id"] == "c1"

    tool_result_entries = [e for e in entries if e.action == "tool_result"]
    assert tool_result_entries[0].payload["call_id"] == "c1"
    assert tool_result_entries[0].payload["ok"] is True


async def test_no_audit_log_means_no_audit_overhead() -> None:
    """Default Agent without audit_log must work and produce no audit entries."""
    agent = Agent("hi", model="echo")
    result = await agent.run("hello")
    # Just verifying nothing crashed and a result came back.
    assert result.output


async def test_run_completed_payload_carries_run_summary() -> None:
    log = InMemoryAuditLog()
    agent = Agent("hi", model="echo", audit_log=log)
    await agent.run("hello")

    completed = await log.query(action="run_completed")
    assert len(completed) == 1
    payload = completed[0].payload
    assert "turns" in payload
    assert "tokens_in" in payload
    assert "tokens_out" in payload
    assert "elapsed_ms" in payload


async def test_audit_entries_are_signed_and_verifiable() -> None:
    log = InMemoryAuditLog(secret="prod-secret")
    agent = Agent("hi", model="echo", audit_log=log)
    result = await agent.run("hello")

    entries = await log.query(session_id=result.session_id)
    assert all(verify_signature(e, "prod-secret") for e in entries)
    # Tampering with payload invalidates the signature.
    tampered = entries[0].model_copy(update={"payload": {"bogus": True}})
    assert not verify_signature(tampered, "prod-secret")
