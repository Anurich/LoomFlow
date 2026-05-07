"""Skills — packaged, on-demand instructions for agent tasks.

Anthropic Agent Skills (Oct 2025) plus the LangChain DeepAgents
extensions, implemented over our existing primitives:

* A **Skill** is a directory containing ``SKILL.md`` plus optional
  supporting files (additional markdown docs, scripts, templates).
* The agent sees every registered skill's NAME + DESCRIPTION at
  startup (~50 tokens per skill in the system prompt).
* When the user's request matches a skill's description, the model
  calls a ``load_skill(name)`` tool to read the full body — the
  recipe — into context.
* Bundled supporting files are read via the standard ``read_tool``
  / ``bash_tool`` the agent already has; we don't need a new
  filesystem abstraction for skills.

Multi-source layering with last-source-wins override::

    agent = Agent(
        "...",
        skills=[
            "~/.jeeves/skills/system/",           # base
            "~/.jeeves/skills/user/",             # user override
            ("./.jeeves-skills/", "Project"),      # project, labelled
        ],
    )

Inline skills (no folder needed)::

    agent = Agent(
        "...",
        skills=[
            Skill.from_text('''---
            name: standup-format
            description: Format a daily standup update.
            ---
            # Standup
            Always 3 sections: Yesterday, Today, Blockers.
            '''),
        ],
    )

Public surface:

* :class:`Skill` — one loadable skill
* :class:`SkillSource` — a directory of skills with optional label
* :class:`SkillRegistry` — collection with override semantics
* :class:`SkillMetadata` — startup-loaded descriptor
* :class:`SkillError` — raised on bad SKILL.md or unknown skill name
* :func:`make_load_skill_tool` — internal: builds the ``load_skill``
  tool the framework injects into agents that have skills configured
"""

from .registry import SkillRegistry, SkillSpec
from .skill import Skill, SkillError, SkillMetadata
from .source import SkillSource
from .tools import make_load_skill_tool

__all__ = [
    "Skill",
    "SkillError",
    "SkillMetadata",
    "SkillRegistry",
    "SkillSource",
    "SkillSpec",
    "make_load_skill_tool",
]
