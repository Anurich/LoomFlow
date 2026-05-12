"""Example 15 — Build an Agent from a TOML / dict config.

Use case: you've outgrown inline ``Agent(...)`` construction. Ops
wants to flip backends per environment (sqlite locally, postgres in
prod) without rebuilding the image. SRE wants budget caps and
permission policies in a reviewed config file, not buried in code.
Compliance wants the audit-log path declared declaratively so
auditors can see it without reading Python.

Loom's :meth:`Agent.from_config` (TOML file) and
:meth:`Agent.from_dict` (parsed dict — useful with YAML / env-var
overrides / settings libraries) accept the same shape:

  * Scalars: ``instructions``, ``model``, ``max_turns``,
    ``architecture``, ``effort``, ``response_tone``,
    ``auto_consolidate``, ``auto_extract``, ``strict_effort``.
  * Backend tables: ``[memory]`` / ``[runtime]`` / ``[telemetry]``
    / ``[audit_log]`` / ``[permissions]`` / ``[budget]``.
  * Arrays of tables: ``[[skills]]`` (per-skill ``path`` + optional
    ``label``) and ``[[mcp]]`` (per-server ``transport`` + transport-
    specific connection fields).

Each backend block goes through the same resolver Agent uses for
its kwargs, so anything you can build via ``Agent(memory="sqlite:./
mem.db", ...)`` you can also declare in TOML and vice-versa. Pre-
built instances (real callable tools, hooks, custom secret stores)
still come through Python kwargs since TOML can't express them.

Run with::

    python examples/15_config_file.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import textwrap
from pathlib import Path

# Prefer the in-repo source over any site-packages install so the
# example exercises the same code as the local tests.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loomflow import Agent  # noqa: E402
from loomflow.architecture import ReAct  # noqa: E402
from loomflow.governance.budget import StandardBudget  # noqa: E402
from loomflow.memory import SqliteMemory  # noqa: E402
from loomflow.observability import InMemoryTelemetry  # noqa: E402
from loomflow.runtime import SqliteRuntime  # noqa: E402
from loomflow.security import (  # noqa: E402
    FileAuditLog,
    FullTranscriptAuditLog,
    Mode,
    StandardPermissions,
)

# ---------------------------------------------------------------------------
# 1. The TOML config — one file, every backend wired
# ---------------------------------------------------------------------------


def _write_config(workdir: Path) -> Path:
    """Write an agent.toml that touches every supported block."""
    # A trivial skill so the [[skills]] entry actually loads.
    skill_dir = workdir / "skills" / "concise"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent(
            """\
            ---
            name: concise
            description: Reply in one sentence; no preamble, no caveats.
            ---

            Always answer in a single sentence. No "Sure!" / "Of course!"
            / "I'd be happy to help" preambles. No caveats, no disclaimers.
            """
        )
    )

    cfg_path = workdir / "agent.toml"
    cfg_path.write_text(
        textwrap.dedent(
            f"""\
            # ---------- top-level scalars ---------------------------
            instructions = "You are a concise research assistant."
            model = "echo"               # offline-friendly stand-in
            architecture = "react"
            max_turns = 5
            auto_consolidate = false
            response_tone = "concise"

            # ---------- backend tables ------------------------------
            [memory]
            backend = "sqlite"
            path = "{workdir / 'memory.db'}"

            [runtime]
            backend = "sqlite"
            path = "{workdir / 'journal.db'}"

            [telemetry]
            backend = "memory"           # introspectable in tests

            [audit_log]
            name = "{workdir / 'audit.jsonl'}"
            scope_full = true            # capture prompts + outputs

            [permissions]
            backend = "standard"
            mode = "default"
            denied_tools = ["bash"]

            [budget]
            max_tokens = 50_000
            max_cost_usd = 0.10
            max_wall_clock_minutes = 1
            soft_warning_at = 0.8

            # ---------- arrays of tables ----------------------------
            [[skills]]
            path = "{skill_dir}"
            label = "Bundled"

            # MCP server example — commented because it would try to
            # spawn a real subprocess. Uncomment + adjust if you have
            # an mcp server installed:
            #
            # [[mcp]]
            # name = "git"
            # transport = "stdio"
            # command = "uvx"
            # args = ["mcp-server-git", "--repo", "."]
            #
            # [[mcp]]
            # name = "remote"
            # transport = "http"
            # url = "https://example.com/mcp"
            # headers = {{ Authorization = "Bearer ..." }}
            """
        )
    )
    return cfg_path


# ---------------------------------------------------------------------------
# 2. Run the agent loaded from the TOML config
# ---------------------------------------------------------------------------


async def run_from_toml() -> None:
    with tempfile.TemporaryDirectory() as raw:
        workdir = Path(raw)
        cfg_path = _write_config(workdir)
        agent = Agent.from_config(cfg_path)

        # Sanity-check the resolved stack — each backend is a concrete
        # instance built from its config block.
        assert isinstance(agent.memory, SqliteMemory)
        assert isinstance(agent.runtime, SqliteRuntime)
        assert isinstance(agent._telemetry, InMemoryTelemetry)  # noqa: SLF001
        # The audit log is wrapped in FullTranscriptAuditLog because
        # scope_full = true in the config.
        inner = getattr(agent._audit_log, "inner", agent._audit_log)  # noqa: SLF001
        assert isinstance(inner, FileAuditLog)
        assert isinstance(
            agent._audit_log, FullTranscriptAuditLog  # noqa: SLF001
        )
        assert isinstance(agent._permissions, StandardPermissions)  # noqa: SLF001
        assert agent._permissions._mode == Mode.DEFAULT  # noqa: SLF001
        assert isinstance(agent.budget, StandardBudget)
        assert isinstance(agent.architecture, ReAct)
        # The skill catalog is in the agent prompt.
        assert agent.skills is not None and "concise" in agent.skills

        print("Agent assembled from", cfg_path)
        print("  memory     :", type(agent.memory).__name__)
        print("  runtime    :", type(agent.runtime).__name__)
        print("  telemetry  :", type(agent._telemetry).__name__)  # noqa: SLF001
        print("  audit_log  :", type(agent._audit_log).__name__)  # noqa: SLF001
        print("  permissions:", type(agent._permissions).__name__)  # noqa: SLF001
        print("  arch       :", agent.architecture.name)
        print("  skills     :", list(agent.skills.names()))

        result = await agent.run("What is the capital of France?", user_id="alice")
        print("\nAgent output:\n  ", result.output)


# ---------------------------------------------------------------------------
# 3. The same config as a Python dict — useful when you already have
#    settings flowing through pydantic / env vars / a service config
# ---------------------------------------------------------------------------


async def run_from_dict() -> None:
    """``Agent.from_dict`` accepts the same shape parsed in-memory.

    Handy when your settings come from Pydantic ``BaseSettings``, a
    Helm chart's env vars, or a config service. Kwargs override
    the dict — use them for things TOML can't express (real callable
    tools, custom secret stores, retry policies)."""
    with tempfile.TemporaryDirectory() as raw:
        workdir = Path(raw)
        cfg = {
            "instructions": "You're a helpful assistant.",
            "model": "echo",
            "max_turns": 3,
            "memory": {"backend": "sqlite", "path": str(workdir / "m.db")},
            "telemetry": "memory",  # string form is fine too
            "permissions": "strict",
            "budget": {"max_tokens": 10_000},
        }
        agent = Agent.from_dict(cfg)
        assert isinstance(agent.memory, SqliteMemory)
        assert isinstance(agent._telemetry, InMemoryTelemetry)  # noqa: SLF001
        assert isinstance(agent._permissions, StandardPermissions)  # noqa: SLF001
        result = await agent.run("hello")
        print("from_dict run output:", result.output)


async def main() -> None:
    print("=== from_config (TOML file) ===\n")
    await run_from_toml()
    print("\n=== from_dict (in-memory) ===\n")
    await run_from_dict()


if __name__ == "__main__":
    asyncio.run(main())
