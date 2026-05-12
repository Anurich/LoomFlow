"""``FullTranscriptAuditLog`` — opt-in verbatim capture of prompts,
outputs, and tool result bodies."""

from __future__ import annotations

from pathlib import Path

import pytest

from loomflow import (
    Agent,
    FullTranscriptAuditLog,
    InMemoryAuditLog,
    tool,
)
from loomflow.security.audit import (
    FileAuditLog,
    resolve_audit_log,
    verify_signature,
    wants_full_transcripts,
)

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Marker contract
# ---------------------------------------------------------------------------


class TestMarkerContract:
    def test_plain_inmemory_log_does_not_opt_in(self) -> None:
        assert wants_full_transcripts(InMemoryAuditLog()) is False

    def test_none_is_false(self) -> None:
        assert wants_full_transcripts(None) is False

    def test_wrapper_opts_in(self) -> None:
        wrapped = FullTranscriptAuditLog(InMemoryAuditLog())
        assert wants_full_transcripts(wrapped) is True
        assert wrapped.full_transcripts is True


# ---------------------------------------------------------------------------
# Forwarding semantics — wrapper does not lose entries or break sigs
# ---------------------------------------------------------------------------


class TestForwarding:
    async def test_append_writes_to_inner_log(self) -> None:
        inner = InMemoryAuditLog(secret="hmac-key")
        wrapped = FullTranscriptAuditLog(inner)

        entry = await wrapped.append(
            session_id="s1",
            actor="user",
            action="run_started",
            payload={"prompt": "hi"},
            user_id="alice",
        )

        all_inner = await inner.all_entries()
        assert all_inner == [entry]

    async def test_signatures_still_verify_through_wrapper(self) -> None:
        inner = InMemoryAuditLog(secret="hmac-key")
        wrapped = FullTranscriptAuditLog(inner)

        entry = await wrapped.append(
            session_id="s1",
            actor="user",
            action="run_started",
            payload={"prompt": "x"},
        )
        assert verify_signature(entry, "hmac-key") is True
        assert verify_signature(entry, "wrong-key") is False

    async def test_query_forwards_to_inner(self) -> None:
        inner = InMemoryAuditLog(secret="hmac-key")
        wrapped = FullTranscriptAuditLog(inner)
        await wrapped.append(
            session_id="s1", actor="x", action="a", payload={}, user_id="alice"
        )
        await wrapped.append(
            session_id="s1", actor="x", action="a", payload={}, user_id="bob"
        )

        alice_only = await wrapped.query(user_id="alice")
        assert len(alice_only) == 1
        assert alice_only[0].user_id == "alice"

    async def test_inner_property_exposes_wrapped_log(self) -> None:
        inner = InMemoryAuditLog()
        wrapped = FullTranscriptAuditLog(inner)
        assert wrapped.inner is inner


# ---------------------------------------------------------------------------
# Agent integration — prompts, outputs, tool results all captured verbatim
# ---------------------------------------------------------------------------


# A prompt longer than 500 chars to prove truncation no longer happens
# when the wrapper is in place.
_LONG_PROMPT = "x" * 800 + "  END_MARKER"


class TestAgentRunStartedCaptureFullPrompt:
    async def test_default_log_truncates_at_500(self) -> None:
        log = InMemoryAuditLog()
        agent = Agent(
            "Be helpful.", model="echo", audit_log=log,
        )
        await agent.run(_LONG_PROMPT)

        started = await log.query(action="run_started")
        assert len(started) == 1
        captured = started[0].payload["prompt"]
        assert len(captured) == 500
        assert "END_MARKER" not in captured

    async def test_wrapped_log_records_full_prompt(self) -> None:
        log = FullTranscriptAuditLog(InMemoryAuditLog())
        agent = Agent(
            "Be helpful.", model="echo", audit_log=log,
        )
        await agent.run(_LONG_PROMPT)

        started = await log.query(action="run_started")
        assert len(started) == 1
        captured = started[0].payload["prompt"]
        assert captured == _LONG_PROMPT
        assert "END_MARKER" in captured


class TestRunCompletedCarriesOutput:
    async def test_default_log_omits_output(self) -> None:
        log = InMemoryAuditLog()
        agent = Agent("Be helpful.", model="echo", audit_log=log)
        await agent.run("hello")

        completed = await log.query(action="run_completed")
        assert len(completed) == 1
        assert "output" not in completed[0].payload

    async def test_wrapped_log_includes_output(self) -> None:
        log = FullTranscriptAuditLog(InMemoryAuditLog())
        agent = Agent("Be helpful.", model="echo", audit_log=log)
        result = await agent.run("hello")

        completed = await log.query(action="run_completed")
        assert len(completed) == 1
        assert completed[0].payload["output"] == result.output
        # EchoModel always echoes the user's input back.
        assert "hello" in completed[0].payload["output"]


# ---------------------------------------------------------------------------
# Tool result bodies are captured when the wrapper is wired
# ---------------------------------------------------------------------------


@tool
async def secret_box(x: int) -> dict:
    """Return a dict so we can prove the full value lands in audit."""
    return {"echoed": x, "secret_marker": "FULL_AUDIT_TEST_42"}


class TestToolResultBodies:
    """Use a scripted model to force one tool call, then check the
    `tool_result` audit payload."""

    async def test_default_log_omits_tool_output(self) -> None:
        from loomflow.core.types import ToolCall
        from loomflow.model.scripted import ScriptedModel, ScriptedTurn

        model = ScriptedModel(
            turns=[
                ScriptedTurn(
                    text="",
                    tool_calls=[
                        ToolCall(id="c1", tool="secret_box", args={"x": 7})
                    ],
                ),
                ScriptedTurn(text="ok"),
            ],
        )
        log = InMemoryAuditLog()
        agent = Agent(
            "Use the tool.",
            model=model,
            tools=[secret_box],
            audit_log=log,
        )
        await agent.run("call the tool")

        tool_results = await log.query(action="tool_result")
        assert len(tool_results) == 1
        assert "output" not in tool_results[0].payload

    async def test_wrapped_log_records_full_tool_output(self) -> None:
        from loomflow.core.types import ToolCall
        from loomflow.model.scripted import ScriptedModel, ScriptedTurn

        model = ScriptedModel(
            turns=[
                ScriptedTurn(
                    text="",
                    tool_calls=[
                        ToolCall(id="c1", tool="secret_box", args={"x": 7})
                    ],
                ),
                ScriptedTurn(text="ok"),
            ],
        )
        log = FullTranscriptAuditLog(InMemoryAuditLog())
        agent = Agent(
            "Use the tool.",
            model=model,
            tools=[secret_box],
            audit_log=log,
        )
        await agent.run("call the tool")

        tool_results = await log.query(action="tool_result")
        assert len(tool_results) == 1
        payload = tool_results[0].payload
        # ``output`` is the verbatim ToolResult.output value.
        assert payload["output"] == {
            "echoed": 7,
            "secret_marker": "FULL_AUDIT_TEST_42",
        }
        # duration_ms also lands when full transcripts are on.
        assert "duration_ms" in payload


# ---------------------------------------------------------------------------
# Tool args were already captured fully in tool_call — keep regression cover
# ---------------------------------------------------------------------------


class TestToolCallArgsAlwaysFull:
    async def test_tool_call_args_captured_for_both_logs(self) -> None:
        from loomflow.core.types import ToolCall
        from loomflow.model.scripted import ScriptedModel, ScriptedTurn

        for log in (
            InMemoryAuditLog(),
            FullTranscriptAuditLog(InMemoryAuditLog()),
        ):
            model = ScriptedModel(
                turns=[
                    ScriptedTurn(
                        text="",
                        tool_calls=[
                            ToolCall(
                                id="c1", tool="secret_box", args={"x": 99}
                            )
                        ],
                    ),
                    ScriptedTurn(text="ok"),
                ],
            )
            agent = Agent(
                "Use the tool.",
                model=model,
                tools=[secret_box],
                audit_log=log,
            )
            await agent.run("call it")
            calls = await log.query(action="tool_call")
            assert calls[0].payload["args"] == {"x": 99}


# ---------------------------------------------------------------------------
# resolve_audit_log — the unified resolver (None / instance / path / dict)
# ---------------------------------------------------------------------------


class TestResolveAuditLog:
    def test_none_returns_none(self) -> None:
        assert resolve_audit_log(None) is None

    def test_instance_passes_through(self) -> None:
        log = InMemoryAuditLog()
        assert resolve_audit_log(log) is log

    def test_full_transcript_instance_passes_through(self) -> None:
        log = FullTranscriptAuditLog(InMemoryAuditLog())
        assert resolve_audit_log(log) is log

    def test_str_path_builds_file_log(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        resolved = resolve_audit_log(str(log_path))
        assert isinstance(resolved, FileAuditLog)

    def test_pathlib_path_builds_file_log(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        resolved = resolve_audit_log(log_path)
        assert isinstance(resolved, FileAuditLog)

    def test_dict_without_name_builds_in_memory(self) -> None:
        resolved = resolve_audit_log({"scope_full": False})
        assert isinstance(resolved, InMemoryAuditLog)

    def test_dict_with_name_builds_file_log(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        resolved = resolve_audit_log({"name": str(log_path)})
        assert isinstance(resolved, FileAuditLog)

    def test_dict_with_scope_full_wraps_in_full_transcript(self) -> None:
        resolved = resolve_audit_log({"scope_full": True})
        assert isinstance(resolved, FullTranscriptAuditLog)
        assert isinstance(resolved.inner, InMemoryAuditLog)

    def test_dict_with_name_and_scope_full(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        resolved = resolve_audit_log(
            {"name": str(log_path), "scope_full": True}
        )
        assert isinstance(resolved, FullTranscriptAuditLog)
        assert isinstance(resolved.inner, FileAuditLog)

    async def test_dict_secret_flows_through_to_signature(self) -> None:
        resolved = resolve_audit_log({"secret": "abc", "scope_full": True})
        assert resolved is not None
        entry = await resolved.append(
            session_id="s", actor="u", action="a", payload={}
        )
        assert verify_signature(entry, "abc") is True
        assert verify_signature(entry, "wrong") is False

    def test_unknown_dict_key_raises(self) -> None:
        with pytest.raises(TypeError, match="unknown key"):
            resolve_audit_log({"bogus": True})

    def test_non_string_name_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a str"):
            resolve_audit_log({"name": 42})

    def test_unsupported_type_raises_with_helpful_message(self) -> None:
        with pytest.raises(TypeError, match="audit_log="):
            resolve_audit_log(42)  # type: ignore[arg-type]


class TestAgentAcceptsDictForm:
    async def test_agent_accepts_dict_with_scope_full(self) -> None:
        agent = Agent(
            "Be helpful.",
            model="echo",
            audit_log={"scope_full": True, "secret": "k"},
        )
        await agent.run("hi")
        # The dict resolved into a FullTranscriptAuditLog wrapping
        # an InMemoryAuditLog — full output should be captured.
        log = agent._audit_log
        assert isinstance(log, FullTranscriptAuditLog)
        completed = await log.query(action="run_completed")
        assert "output" in completed[0].payload

    async def test_agent_accepts_plain_string_path(
        self, tmp_path: Path
    ) -> None:
        log_path = tmp_path / "audit.jsonl"
        agent = Agent(
            "Be helpful.", model="echo", audit_log=str(log_path),
        )
        await agent.run("hi")
        assert isinstance(agent._audit_log, FileAuditLog)
        assert log_path.exists()
