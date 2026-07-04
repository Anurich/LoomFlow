"""Tool registry and decorators + built-in filesystem / shell tools.

Users typically construct tools via :func:`tool` (decorator) and pass
the resulting :class:`Tool` objects to :class:`Agent`. The agent wraps
them in an :class:`InProcessToolHost`.

For the canonical "Claude-Code-shaped" tool set (read / write / edit
/ bash), import the four factory functions from
:mod:`loomflow.tools.builtin` (also re-exported at the top level).
"""

from .builtin import (
    PathEscapeError,
    bash_tool,
    default_workdir,
    edit_tool,
    filesystem_tools,
    find_tool,
    grep_tool,
    ls_tool,
    read_tool,
    write_tool,
)
from .code_mode import make_code_mode_tools
from .executor import CodeExecutor, ExecResult, SubprocessExecutor
from .plan import (
    LivingPlan,
    LivingPlanStep,
    get_active_plan,
    living_plan_prompt_section,
    make_plan_tools,
    make_recall_past_plans_tool,
)
from .plan_resolver import ResolvedLivingPlan, resolve_living_plan
from .registry import InProcessToolHost, Tool, tool
from .web import web_tool

__all__ = [
    "CodeExecutor",
    "ExecResult",
    "InProcessToolHost",
    "LivingPlan",
    "LivingPlanStep",
    "PathEscapeError",
    "ResolvedLivingPlan",
    "SubprocessExecutor",
    "Tool",
    "bash_tool",
    "default_workdir",
    "edit_tool",
    "filesystem_tools",
    "find_tool",
    "get_active_plan",
    "grep_tool",
    "living_plan_prompt_section",
    "ls_tool",
    "make_code_mode_tools",
    "make_plan_tools",
    "make_recall_past_plans_tool",
    "read_tool",
    "resolve_living_plan",
    "tool",
    "web_tool",
    "write_tool",
]
