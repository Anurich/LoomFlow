"""LLM-driven fact extraction from episodes.

Given a list of episodes and a :class:`FactStore`, the consolidator
asks a :class:`Model` to emit a JSON array of ``{subject, predicate,
object, confidence}`` objects per episode, parses the response, and
appends each extracted :class:`Fact` to the store. The store is then
responsible for any supersession / temporal-window bookkeeping.

The default prompt is a no-frills extractor; users with strong
opinions about ontology / taxonomy can pass a custom
``system_prompt=`` at construction.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from ..core.protocols import Model
from ..core.types import Episode, Fact, Message, Role
from .facts import FactStore

DEFAULT_SYSTEM_PROMPT = """\
You extract semantic facts from conversation episodes.

A fact is a stable claim about an entity, expressed as a triple of
(subject, predicate, object). Only extract claims that are likely to
remain true beyond this episode. Skip greetings, transient state,
small talk, and acknowledgements.

You will be shown a single episode. Return a JSON array of facts.
Each fact must have exactly these fields:

* "subject": the entity (e.g. "user", "Alice", "Project Atlas")
* "predicate": the relation (e.g. "name_is", "lives_in", "prefers")
* "object": the value (e.g. "Alice", "Tokyo", "dark mode")
* "confidence": a float between 0.0 and 1.0

Example output:
[{"subject": "user", "predicate": "name_is", "object": "Alice", "confidence": 0.95}]

Return ONLY the JSON array. No prose, no markdown, no code fences.
If there's nothing worth extracting, return an empty array: [].
"""


class Consolidator:
    """Wraps a :class:`Model` to extract :class:`Fact` rows from episodes."""

    def __init__(
        self,
        *,
        model: Model,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_facts_per_episode: int = 20,
    ) -> None:
        self._model = model
        self._system_prompt = system_prompt
        self._max_facts_per_episode = max_facts_per_episode

    async def consolidate(
        self,
        episodes: Iterable[Episode],
        *,
        store: FactStore,
    ) -> list[Fact]:
        """Process ``episodes``; append extracted facts to ``store``;
        return the new :class:`Fact` instances in extraction order.

        Uses ``store.append_many`` when available so the underlying
        store can batch the embedder calls (one ``embed_batch`` API
        round-trip instead of N individual ``embed`` calls). Falls
        back to per-fact ``append`` for stores that haven't
        implemented ``append_many``.
        """
        new_facts: list[Fact] = []
        for episode in episodes:
            extracted = await self._extract(episode)
            new_facts.extend(extracted)

        if not new_facts:
            return []

        # Prefer the bulk path when the store supports it.
        append_many = getattr(store, "append_many", None)
        if callable(append_many):
            await append_many(new_facts)
        else:
            for fact in new_facts:
                await store.append(fact)
        return new_facts

    # ---- extraction -----------------------------------------------------

    async def _extract(self, episode: Episode) -> list[Fact]:
        messages = self._build_messages(episode)
        response_text = await self._collect_response(messages)
        parsed = self._parse_response(response_text)
        return [self._build_fact(episode, item) for item in parsed]

    def _build_messages(self, episode: Episode) -> list[Message]:
        body = (
            f"Episode (occurred at {episode.occurred_at.isoformat()}):\n\n"
            f"USER: {episode.input}\n"
            f"ASSISTANT: {episode.output}\n"
        )
        return [
            Message(role=Role.SYSTEM, content=self._system_prompt),
            Message(role=Role.USER, content=body),
        ]

    async def _collect_response(self, messages: list[Message]) -> str:
        text_parts: list[str] = []
        async for chunk in self._model.stream(messages):
            if chunk.kind == "text" and chunk.text is not None:
                text_parts.append(chunk.text)
        return "".join(text_parts).strip()

    def _parse_response(self, raw: str) -> list[dict[str, Any]]:
        if not raw:
            return []
        # Some models wrap output in ``` fences despite the prompt.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[len("json"):]
            cleaned = cleaned.strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        usable: list[dict[str, Any]] = []
        for item in data[: self._max_facts_per_episode]:
            if not isinstance(item, dict):
                continue
            if not all(k in item for k in ("subject", "predicate", "object")):
                continue
            usable.append(item)
        return usable

    def _build_fact(
        self, episode: Episode, item: dict[str, Any]
    ) -> Fact:
        confidence = item.get("confidence", 1.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 1.0
        return Fact(
            user_id=episode.user_id,
            subject=str(item["subject"]),
            predicate=str(item["predicate"]),
            object=str(item["object"]),
            confidence=max(0.0, min(1.0, confidence)),
            valid_from=episode.occurred_at,
            recorded_at=datetime.now(UTC),
            sources=[episode.id],
        )
