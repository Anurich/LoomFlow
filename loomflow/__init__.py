"""Loom — production workflow + agent framework.

Public API tiers:

* **Top-level** (this module) — the daily-use surface. Importing
  ``Agent``, ``Workflow``, ``step``, ``tool``, common types, the
  default in-memory backends, and the most-caught errors. This is
  what 90% of code touches in the first hour. Stable; will not
  break in 0.x without a migration note.

* **Submodule imports** — for backend-specific or specialized
  classes. Examples::

      from loomflow.memory.postgres import PostgresMemory
      from loomflow.memory.chroma import ChromaMemory
      from loomflow.vectorstore.chroma import ChromaVectorStore
      from loomflow.model.openai import OpenAIModel
      from loomflow.model.anthropic import AnthropicModel
      from loomflow.architecture import (
          SelfRefine, Reflexion, TreeOfThoughts, MultiAgentDebate,
          Supervisor, Swarm, Router, ActorCritic, BlackboardArchitecture,
          PlanAndExecute, ReWOO,
      )
      from loomflow.observability import OTelTelemetry
      from loomflow.runtime import SqliteRuntime, PostgresRuntime
      from loomflow.security import (
          PerUserPermissions, FilesystemSandbox, SubprocessSandbox,
          FileAuditLog, EnvSecrets, DictSecrets,
      )
      from loomflow.tools import (
          read_tool, write_tool, edit_tool, bash_tool, filesystem_tools,
      )
      from loomflow.team import Team
      from loomflow.skills import Skill, SkillRegistry
      from loomflow.mcp import MCPClient, MCPRegistry, MCPServerSpec

  Demoting these keeps the top-level autocomplete focused on
  things users actually pick first, and avoids loading optional
  SDK dependencies at ``import loomflow`` time.

* **Internal** — anything beginning with ``_``. No stability
  promise; subject to change without notice.
"""

# ---------------------------------------------------------------------------
# Top-level (Tier 1) — the daily-use API surface.
# ---------------------------------------------------------------------------

from .agent import Agent
from .architecture import Architecture, ReAct
from .core import (
    Budget,
    BudgetExceeded,
    BudgetStatus,
    ConfigError,
    Embedder,
    Episode,
    Event,
    EventKind,
    Fact,
    HookHost,
    IsolationWarning,
    LoomDeprecationWarning,
    LoomError,
    Memory,
    MemoryBlock,
    MemoryExport,
    MemoryProfile,
    Message,
    Model,
    OutputValidationError,
    PermissionDecision,
    Permissions,
    Role,
    RunContext,
    RunResult,
    Runtime,
    Sandbox,
    Secrets,
    Telemetry,
    ToolCall,
    ToolDef,
    ToolHost,
    ToolResult,
    Usage,
    get_run_context,
    new_id,
    set_run_context,
)
from .governance import BudgetConfig, NoBudget, StandardBudget
from .memory import HashEmbedder, InMemoryMemory, resolve_memory
from .model import EchoModel, ScriptedModel, ScriptedTurn
from .observability import NoTelemetry
from .runtime import InProcRuntime
from .security import (
    AllowAll,
    AuditLog,
    HookRegistry,
    InMemoryAuditLog,
    Mode,
    NoSandbox,
    StandardPermissions,
)
from .tools import Tool, tool
from .workflow import END, START, Workflow, WorkflowResult, step

__version__ = "0.9.19"

__all__ = [
    "__version__",
    # ----- Daily-use building blocks -----
    "Agent",
    "Workflow",
    "WorkflowResult",
    "step",
    "START",
    "END",
    "tool",
    "Tool",
    # ----- Run context (always-relevant) -----
    "RunContext",
    "RunResult",
    "get_run_context",
    "set_run_context",
    # ----- Common types you'll annotate / construct -----
    "Message",
    "Role",
    "Episode",
    "Fact",
    "ToolCall",
    "ToolDef",
    "ToolResult",
    "Event",
    "EventKind",
    "Usage",
    "MemoryBlock",
    "MemoryProfile",
    "MemoryExport",
    "BudgetStatus",
    "PermissionDecision",
    # ----- Core protocols (for type annotations) -----
    "Memory",
    "Model",
    "Permissions",
    "Budget",
    "HookHost",
    "Sandbox",
    "Telemetry",
    "ToolHost",
    "Runtime",
    "Embedder",
    "Secrets",
    "AuditLog",
    "Architecture",
    # ----- Default in-memory / no-op backends + always-on architecture -----
    "InMemoryMemory",
    "InMemoryAuditLog",
    "NoBudget",
    "NoTelemetry",
    "NoSandbox",
    "AllowAll",
    "Mode",
    "StandardPermissions",
    "HookRegistry",
    "StandardBudget",
    "BudgetConfig",
    "InProcRuntime",
    "ReAct",
    "HashEmbedder",
    # ----- Resolvers (string spec → instance) -----
    "resolve_memory",
    # ----- Test / dev fakes (no API key required) -----
    "EchoModel",
    "ScriptedModel",
    "ScriptedTurn",
    # ----- Common errors -----
    "LoomError",
    "OutputValidationError",
    "IsolationWarning",
    "BudgetExceeded",
    "ConfigError",
    "LoomDeprecationWarning",
    # ----- ID utilities -----
    "new_id",
]
