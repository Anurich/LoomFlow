"""SkillSource — a directory of skills with an optional label.

A user-facing input. Wraps a path to a skills folder; we scan it
recursively at construction time, building one :class:`Skill` per
discovered ``SKILL.md`` file.

The optional ``label`` shows up in the catalog the agent sees, e.g.
``"  - my-skill [Project]: ..."``. Useful when multiple sources
are mounted and you want to see at a glance which one a skill
came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .skill import Skill, SkillError


@dataclass(frozen=True)
class SkillSource:
    """A folder of skills + an optional label."""

    path: Path
    label: str | None = None

    @classmethod
    def coerce(
        cls,
        item: SkillSource | str | Path | tuple[str | Path, str],
    ) -> SkillSource:
        """Normalize one user-supplied source spec.

        Accepts:
        * ``SkillSource(...)`` — used as-is
        * ``str`` / ``Path`` — bare path, no label
        * ``(path, label)`` — path with explicit label
        """
        if isinstance(item, SkillSource):
            return item
        if isinstance(item, str | Path):
            return cls(Path(item).expanduser(), None)
        if isinstance(item, tuple) and len(item) == 2:
            path, label = item
            return cls(Path(path).expanduser(), str(label))
        raise SkillError(
            f"Cannot coerce {item!r} to SkillSource. Pass a path "
            "string/Path, a (path, label) tuple, or a SkillSource "
            "instance."
        )

    def discover(self) -> list[Skill]:
        """Find every SKILL.md under this source directory.

        Recurses one level (most common layout: ``skills/<name>/SKILL.md``)
        but also handles deeper nesting. Each SKILL.md becomes one
        :class:`Skill` instance with this source's label attached.
        """
        if not self.path.exists():
            raise SkillError(
                f"Skill source path does not exist: {self.path}"
            )
        if self.path.is_file():
            # User pointed directly at a SKILL.md.
            return [Skill(self.path, source_label=self.label)]
        if not self.path.is_dir():
            raise SkillError(
                f"Skill source path is not a directory: {self.path}"
            )
        skills: list[Skill] = []
        for skill_md in sorted(self.path.rglob("SKILL.md")):
            skills.append(
                Skill(skill_md.parent, source_label=self.label)
            )
        return skills
