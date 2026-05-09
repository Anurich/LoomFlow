"""SkillRegistry — manages a collection of available skills.

Built once when an :class:`Agent` is constructed. Holds every
:class:`Skill` discovered from the user's sources, applies
last-source-wins override semantics by name, and provides the two
hooks the framework needs to surface skills to the model:

* :meth:`catalog_section` — the markdown bullet list injected into
  the system prompt at startup (the cheap "metadata" tier of
  progressive disclosure)
* :meth:`load` — return a skill's full body when the model calls
  the ``load_skill`` tool

Override semantics matches LangChain DeepAgents: when two sources
ship a skill with the same ``name``, the LATER source wins. This
lets users layer system → user → project skills and override at
any level::

    skills=[
        "~/.jeeves/skills/system/",      # base
        "~/.jeeves/skills/user/",        # user override
        "./.jeeves-skills/",             # project override
    ]
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

from ..tools.registry import Tool
from .skill import Skill, SkillError, SkillMetadata
from .source import SkillSource

SkillSpec = Skill | SkillSource | str | Path | tuple[str | Path, str]
"""Anything an :class:`Agent`'s ``skills=`` argument accepts."""


class SkillRegistry:
    """A keyed collection of :class:`Skill` instances."""

    def __init__(
        self, items: Iterable[SkillSpec] | None = None
    ) -> None:
        self._skills: dict[str, Skill] = {}
        # Track which skills' pending tools we've already pushed
        # into the agent's tool host; load_skill becomes idempotent.
        self._loaded: set[str] = set()
        if items is not None:
            for item in items:
                self._ingest(item)

    # ---- ingestion -----------------------------------------------------

    def _ingest(self, item: SkillSpec) -> None:
        """Add one user-supplied spec to the registry, applying
        last-wins override semantics."""
        if isinstance(item, Skill):
            self._add_one(item)
            return
        source = SkillSource.coerce(item)
        for skill in source.discover():
            self._add_one(skill)

    def _add_one(self, skill: Skill) -> None:
        # Last source wins by name — that's the documented behaviour.
        # We keep override silent because it's load-bearing for the
        # layered-sources pattern; users WILL override base skills
        # by design.
        self._skills[skill.name] = skill

    def add(self, skill: Skill) -> None:
        """Append (or override) a single skill after construction."""
        self._add_one(skill)

    def remove(self, name: str) -> Skill | None:
        """Drop a skill by name. Returns the removed instance or
        ``None`` if no such skill was registered."""
        return self._skills.pop(name, None)

    # ---- lookup --------------------------------------------------------

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._skills

    def __len__(self) -> int:
        return len(self._skills)

    def __iter__(self) -> Iterator[Skill]:
        return iter(self._skills.values())

    def names(self) -> list[str]:
        return list(self._skills.keys())

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    # ---- agent-facing helpers ------------------------------------------

    def metadata_map(self) -> Mapping[str, SkillMetadata]:
        """All currently-registered skills' metadata, keyed by name.
        Cheap to compute — used to build the catalog section."""
        return {name: s.metadata for name, s in self._skills.items()}

    def catalog_section(self) -> str:
        """The markdown bullet list that gets appended to the
        agent's system prompt.

        Empty registry → empty string (so the constructor can
        unconditionally call this without polluting the system
        prompt with a blank "Available skills" header)."""
        if not self._skills:
            return ""
        bullets = "\n".join(
            s.metadata.to_catalog_line()
            for s in sorted(self._skills.values(), key=lambda s: s.name)
        )
        return (
            "## Available skills\n\n"
            "Each is a packaged playbook for a specific task. Call "
            "`load_skill(name)` to read the full instructions for "
            "any of these when the user's request matches its "
            "description; otherwise just answer normally.\n\n"
            f"{bullets}\n"
        )

    def load(self, name: str) -> str:
        """Return the full body of a skill (the load_skill tool's
        result). Raises :class:`SkillError` for unknown names so
        the model gets a clear error in the tool result.

        Does NOT register pending Tools. For the full load-and-
        register flow, see :meth:`load_with_tools`."""
        skill = self._skills.get(name)
        if skill is None:
            available = ", ".join(sorted(self._skills)) or "(none)"
            raise SkillError(
                f"Unknown skill {name!r}. Available: {available}"
            )
        return skill.load_body()

    def load_with_tools(
        self, name: str
    ) -> tuple[str, list[Tool]]:
        """Return ``(body, newly_pending_tools)`` — the body of the
        skill plus the Tool instances the framework should register
        with the agent's tool host on this load.

        Idempotent: subsequent calls for the same skill return the
        body and an empty tool list, since registration only needs
        to happen once."""
        body = self.load(name)
        if name in self._loaded:
            return body, []
        skill = self._skills[name]
        self._loaded.add(name)
        return body, list(skill.pending_tools)

    def is_loaded(self, name: str) -> bool:
        """Whether the skill's pending tools have been registered."""
        return name in self._loaded
