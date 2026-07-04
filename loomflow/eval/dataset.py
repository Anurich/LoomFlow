"""Eval datasets: :class:`Case` and :class:`Dataset` with JSONL I/O.

A :class:`Case` is one labelled example — an input prompt plus optional
ground truth (an ``expected`` output string and/or the ``expected_tools``
the agent should call). A :class:`Dataset` is an ordered collection of
cases with ``from_jsonl`` / ``to_jsonl`` round-trip persistence so eval
suites live as flat files next to the code they gate.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..core.ids import new_id

__all__ = ["Case", "Dataset"]


class Case(BaseModel):
    """One eval example: an input plus optional ground truth.

    ``expected`` is the reference output for :class:`ExactMatch` /
    :class:`Contains`; ``expected_tools`` is the set of tool names the
    agent is expected to call, for :class:`ToolSelectionAccuracy`.
    Either (or both) may be ``None`` — metrics that need ground truth
    skip cases that don't carry it (see ``Metric.applies``). ``id`` is
    auto-generated when not supplied, so hand-written datasets don't
    need to invent identifiers.
    """

    id: str = Field(default_factory=lambda: new_id("case"))
    input: str
    expected: str | None = None
    expected_tools: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Dataset:
    """An ordered list of :class:`Case`\\ s with JSONL persistence.

    JSONL format: one JSON object per line, each the ``model_dump`` of
    a :class:`Case`. Lines that are blank are skipped on read.
    """

    def __init__(self, cases: Iterable[Case] = ()) -> None:
        self.cases: list[Case] = list(cases)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[Case]:
        return iter(self.cases)

    def __getitem__(self, index: int) -> Case:
        return self.cases[index]

    def add(self, case: Case) -> None:
        self.cases.append(case)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Dataset:
        """Load a dataset from a JSONL file (one Case object per line)."""
        cases: list[Case] = []
        text = Path(path).read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            cases.append(Case.model_validate(data))
        return cls(cases)

    def to_jsonl(self, path: str | Path) -> None:
        """Write the dataset to ``path`` as JSONL (overwrites)."""
        lines = [case.model_dump_json() for case in self.cases]
        Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
