"""Jev adapter — TypeSafe AI's System One model over the wire.

Lazy import of the vendor SDK; install with
``pip install 'loomflow[typesafe]'`` and set ``TYPESAFE_API_KEY``
(explicit ``api_key=`` and the Secrets backend also work, in the
usual precedence). Early-access API as of September 2026.

The adapter is deliberately thin: our question types mirror the SDK's
(:class:`Choice` / :class:`Score` / :class:`Noul` / :class:`Box` with
the same constructor fields), so mapping is field-for-field, and
responses are read duck-typed so SDK point releases don't break us.

Images: every :class:`~loomflow.Image` found in the state or the
questions becomes a ``typesafe_sdk.Image`` (uploaded as multipart by
the SDK), and the request gets the ``TypeSafe-Danger-Zone-Images``
opt-in header automatically — as does any request with a Box.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..core.types import Image, Usage, sniff_image_type
from ..model._pricing import estimate_cost
from .base import DecisionError, DecisionState
from .types import (
    Box,
    BoxDecision,
    Choice,
    ChoiceDecision,
    DecisionAnswer,
    Decisions,
    Noul,
    NoulDecision,
    Question,
    Score,
    ScoreDecision,
    describe_content,
)

__all__ = ["JevModel"]

_IMAGES_HEADER = "TypeSafe-Danger-Zone-Images"
_LONG_CONTEXT_HEADER = "TypeSafe-Danger-Zone-Long-Context"
_JEV_ACCEPTS = ("image/png", "image/jpeg")
_VERSION = re.compile(r"jev-(\d+)\.(\d+)")


class JevModel:
    """System One decisions served by TypeSafe's ``/v1/systemone``.

    * ``model`` — ``"jev-latest"`` (stable alias), ``"jev-preview"``
      (newest build; needed for :class:`Box`), or a pinned version
      like ``"jev-1.14.0"``. Defaults to ``TYPESAFE_DEFAULT_MODEL``,
      else ``"jev-latest"``.
    * ``base_url`` — another TypeSafe-compatible endpoint, e.g. a
      gateway (OpenRouter ``https://openrouter.ai/api`` with model
      ``"~typesafe/jev-latest"``). Also read from ``TYPESAFE_BASE_URL``.
    * ``timeout`` / ``max_retries`` — per-HTTP-call seconds, and
      retries on 429/5xx (SDK default 2; ``0`` disables).
    * ``long_context=True`` — 262k-token requests via the
      ``TypeSafe-Danger-Zone-Long-Context`` header (same price,
      slower, less accurate).
    * ``client=`` — inject a ready client (tests, custom transports).

    Use ``async with JevModel() as jev:`` (or ``await jev.aclose()``)
    to release the HTTP connection pool when done.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        headers: Mapping[str, str] | None = None,
        long_context: bool = False,
        client: Any | None = None,
        secrets: Any | None = None,
    ) -> None:
        self.name = (
            model
            or (os.environ.get("TYPESAFE_DEFAULT_MODEL") or "").strip()
            or "jev-latest"
        )
        self.long_context = long_context
        self._owns_client = client is None
        if client is not None:
            self._client = client
            return
        try:
            from typesafe_sdk import (  # type: ignore[import-not-found, import-untyped]
                AsyncTypeSafeClient,
                RetryPolicy,
            )
        except ImportError as exc:  # pragma: no cover — depends on env
            raise ImportError(
                "TypeSafe SDK not installed. "
                "Install with: pip install 'loomflow[typesafe]'"
            ) from exc
        # Key precedence: explicit arg → Secrets backend → env. The
        # SDK also self-reads TYPESAFE_API_KEY, but resolving here
        # keeps vault-backed Secrets working and errors early.
        resolved_key = api_key
        if resolved_key is None and secrets is not None:
            resolved_key = secrets.lookup_sync("TYPESAFE_API_KEY")
        if resolved_key is None:
            resolved_key = os.environ.get("TYPESAFE_API_KEY")
        kwargs: dict[str, Any] = {"api_key": resolved_key}
        # Only forward what was set, so the SDK's own env defaults
        # (TYPESAFE_BASE_URL, …) and built-in policies still apply.
        if base_url is not None:
            kwargs["base_url"] = base_url
        if timeout is not None:
            kwargs["timeout"] = timeout
        if headers:
            kwargs["headers"] = dict(headers)
        if max_retries is not None:
            kwargs["retry"] = RetryPolicy(max_retries=max_retries)
        self._client = AsyncTypeSafeClient(**kwargs)

    async def __aenter__(self) -> JevModel:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the HTTP client this model created (an injected
        ``client=`` belongs to the caller and is left open)."""
        if self._owns_client and hasattr(self._client, "aclose"):
            await self._client.aclose()

    async def list_models(self) -> list[str]:
        """Model names this account can use (aliases; pinned
        versions like ``jev-1.14.0`` are accepted even if unlisted)."""
        try:
            response = await self._client.models.list()
        except Exception as exc:
            if _is_sdk_error(exc):
                raise DecisionError(f"Jev model listing failed: {exc}") from exc
            raise
        return [str(m.name) for m in getattr(response, "models", ())]

    async def decide(
        self,
        state: DecisionState,
        *,
        questions: Mapping[str, Question],
    ) -> Decisions:
        if not questions:
            raise DecisionError("decide() requires at least one question")
        has_box = any(isinstance(q, Box) for q in questions.values())
        if has_box and _predates_box(self.name):
            raise DecisionError(
                f"Box questions need Jev 1.14+, but this JevModel is "
                f"pinned to {self.name!r} — use JevModel('jev-preview') "
                "or JevModel('jev-1.14.0')"
            )
        wire = _Wire(_load_sdk())
        call_kwargs: dict[str, Any] = {
            "state": wire.content(state),
            "questions": {
                name: wire.question(q) for name, q in questions.items()
            },
            # Forward the requested model version — ``JevModel(
            # "jev-1.13.0")`` must pin that version on the wire, not
            # silently use the client's configured default.
            "model": self.name,
        }
        headers: dict[str, str] = {}
        if wire.saw_image or has_box:
            headers[_IMAGES_HEADER] = "true"
        if self.long_context:
            headers[_LONG_CONTEXT_HEADER] = "true"
        if headers:
            call_kwargs["extra_headers"] = headers
        try:
            response = await self._call(call_kwargs)
        except Exception as exc:
            if _is_sdk_error(exc):
                raise DecisionError(
                    _failure_message(exc, has_box=has_box)
                ) from exc
            raise

        raw_answers = getattr(response, "answers", None) or {}
        answers: dict[str, DecisionAnswer] = {}
        for name, question in questions.items():
            raw = (
                raw_answers.get(name)
                if isinstance(raw_answers, Mapping)
                else getattr(raw_answers, name, None)
            )
            if raw is None:
                raise DecisionError(
                    f"Jev response missing answer for question {name!r}"
                )
            answers[name] = _parse_answer(name, question, raw)

        u = getattr(response, "usage", None)
        in_tok = int(getattr(u, "input_tokens", 0) or 0)
        out_tok = int(getattr(u, "output_tokens", 0) or 0)
        usage = Usage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=estimate_cost(_pricing_name(self.name), in_tok, out_tok),
        )
        answered_by = getattr(response, "model", None)
        return Decisions(
            answers=answers,
            usage=usage,
            model=str(answered_by) if answered_by else None,
            request_id=_request_id(response),
        )

    async def _call(self, call_kwargs: dict[str, Any]) -> Any:
        """One ``system_one`` call. Clients/SDKs predating per-call
        model selection get one precise retry without the kwarg."""
        try:
            return await self._client.system_one(**call_kwargs)
        except TypeError as exc:
            if "model" not in str(exc):
                raise
            call_kwargs = {k: v for k, v in call_kwargs.items() if k != "model"}
            return await self._client.system_one(**call_kwargs)


# ---------------------------------------------------------------------------
# Request mapping
# ---------------------------------------------------------------------------


def _load_sdk() -> Any | None:
    try:
        import typesafe_sdk  # type: ignore[import-not-found, import-untyped]
    except ImportError:
        return None
    return typesafe_sdk


class _Wire:
    """Translates one request's state + questions into SDK shapes.

    With the SDK installed, questions become SDK question objects and
    each :class:`Image` a ``typesafe_sdk.Image`` (one per distinct
    image object, so an image used twice uploads once). Without it (an
    injected fake ``client=`` in tests), our frozen dataclasses pass
    through with the same field names, so a fake sees identical shapes.
    """

    def __init__(self, sdk: Any | None) -> None:
        self._sdk = sdk
        self._images: dict[int, Any] = {}
        self.saw_image = False

    def content(self, value: Any) -> Any:
        if isinstance(value, Image):
            self.saw_image = True
            return self._image(value)
        if value is None or isinstance(value, str | int | float | bool):
            return value
        if isinstance(value, Mapping):
            return {str(k): self.content(v) for k, v in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
            return [self.content(v) for v in value]
        return value

    def _image(self, image: Image) -> Any:
        cached = self._images.get(id(image))
        if cached is not None:
            return cached
        try:
            raw = base64.b64decode(image.data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise DecisionError("Image.data is not valid base64") from exc
        kind = sniff_image_type(raw)
        if kind not in _JEV_ACCEPTS:
            raise DecisionError(
                f"Jev accepts PNG or JPEG images only (got "
                f"{kind or image.media_type}); convert GIF/WebP/HEIC "
                "to PNG or JPEG first"
            )
        converted = image if self._sdk is None else self._sdk.Image(raw)
        self._images[id(image)] = converted
        return converted

    def question(self, q: Question) -> Any:
        instructions = self.content(q.instructions)
        if isinstance(q, Choice):
            criteria: Any = {k: self.content(v) for k, v in q.criteria.items()}
            if self._sdk is None:
                return Choice(instructions, criteria)
            return self._sdk.Choice(instructions=instructions, criteria=criteria)
        if isinstance(q, Score):
            levels = [self.content(v) for v in q.criteria]
            if self._sdk is None:
                return Score(instructions, levels)
            return self._sdk.Score(instructions=instructions, criteria=levels)
        if isinstance(q, Noul):
            wire = q.wire_criteria()
            outcomes = (
                None if wire is None
                else {k: self.content(v) for k, v in wire.items()}
            )
            if self._sdk is None:
                return dataclasses.replace(
                    q, instructions=instructions, criteria=outcomes
                )
            return self._sdk.Noul(instructions=instructions, criteria=outcomes)
        if isinstance(q, Box):
            if self._sdk is None:
                return Box(instructions)
            sdk_box = getattr(self._sdk, "Box", None)
            if sdk_box is None:
                raise DecisionError(
                    "Box questions need typesafe-sdk>=0.7: "
                    "pip install -U 'loomflow[typesafe]'"
                )
            return sdk_box(instructions=instructions)
        raise DecisionError(f"unknown question type: {type(q).__name__}")


def _predates_box(model: str) -> bool:
    """True only for an explicitly pinned pre-1.14 version — aliases
    (``jev-latest``) move, so those are left to the server."""
    m = _VERSION.search(model)
    return m is not None and (int(m[1]), int(m[2])) < (1, 14)


def _pricing_name(model: str) -> str:
    """Gateway ids (``~typesafe/jev-latest``, ``typesafe-ai/jev``)
    price as the underlying Jev model."""
    return model.rsplit("/", 1)[-1].lstrip("~")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def _is_sdk_error(exc: BaseException) -> bool:
    return type(exc).__module__.split(".", 1)[0] == "typesafe_sdk"


def _failure_message(exc: BaseException, *, has_box: bool) -> str:
    # str(TypeSafeAPIError) already carries status, endpoint, and
    # request_id; add the fix for the failures we can predict.
    msg = f"Jev request failed: {exc}"
    status = getattr(exc, "status", None)
    if status in (400, 422) and has_box:
        msg += (
            " — Box questions need Jev 1.14+ (JevModel('jev-preview') "
            "or 'jev-1.14.0')"
        )
    elif status == 413:
        msg += (
            " — too large: 16 MiB per image, 32 MiB per request; "
            "images must be 280-1568 px per side"
        )
    return msg


def _request_id(response: Any) -> str | None:
    # The SDK's ``request_id`` property RAISES when the header is
    # absent, so getattr's default alone doesn't cover it.
    try:
        value = getattr(response, "request_id", None)
    except Exception:  # noqa: BLE001 — metadata only, never fail a decision
        return None
    return str(value) if value else None


# ---------------------------------------------------------------------------
# Response mapping
# ---------------------------------------------------------------------------


def _parse_answer(
    name: str, question: Question, raw: Any
) -> DecisionAnswer:
    """Duck-typed read of one SDK answer into our frozen types."""
    if isinstance(question, Choice):
        return ChoiceDecision(
            choice=str(getattr(raw, "choice", "")),
            probabilities=dict(getattr(raw, "probabilities", {}) or {}),
            confidence=float(getattr(raw, "confidence", 0.0) or 0.0),
        )
    if isinstance(question, Score):
        # The SDK returns Score ``legend`` / ``probabilities`` as
        # MAPPINGS ("levels repeated by number"). ``list(mapping)``
        # yields the KEYS — the bug that turned [0.1, 0.3, 0.6] into
        # [0, 1, 2] — so both fields go through order-aware coercion
        # that accepts mapping and sequence forms alike.
        legend = _level_texts(
            getattr(raw, "legend", None), question.criteria
        )
        return ScoreDecision(
            score=float(getattr(raw, "score", 0.0) or 0.0),
            legend=legend,
            probabilities=_level_probs(
                getattr(raw, "probabilities", None), legend
            ),
            confidence=float(getattr(raw, "confidence", 0.0) or 0.0),
        )
    if isinstance(question, Noul):
        noul = getattr(raw, "noul", None)
        if noul is None:
            raise DecisionError(
                f"Jev answer for Noul question {name!r} has no "
                "'noul' probability"
            )
        return NoulDecision(noul=float(noul))
    if isinstance(question, Box):
        return _parse_box(name, raw)
    raise DecisionError(f"unknown question type: {type(question).__name__}")


def _parse_box(name: str, raw: Any) -> BoxDecision:
    box = getattr(raw, "box", None)
    if box is None:
        raise DecisionError(f"Jev answer for Box question {name!r} has no 'box'")

    def edge(key: str) -> float:
        value = box.get(key) if isinstance(box, Mapping) else getattr(box, key, None)
        if value is None:
            raise DecisionError(f"Jev box answer {name!r} is missing {key!r}")
        return float(value)

    confidence = getattr(raw, "confidence", None)
    return BoxDecision(
        top=edge("top"),
        left=edge("left"),
        bottom=edge("bottom"),
        right=edge("right"),
        confidence=None if confidence is None else float(confidence),
    )


def _level_texts(
    value: Any, fallback: Any
) -> list[str]:
    """Coerce a Score ``legend`` into an ordered list of level texts.

    Accepts the SDK's number-keyed mapping form (``{0: "Calm", ...}``,
    int or numeric-string keys), a plain sequence, or ``None`` (falls
    back to the question's own criteria).
    """
    if value is None:
        return [describe_content(level) for level in fallback]
    if isinstance(value, Mapping):
        if not value:
            return [describe_content(level) for level in fallback]
        try:
            items = sorted(value.items(), key=lambda kv: int(kv[0]))
        except (TypeError, ValueError):
            items = list(value.items())
        return [describe_content(text) for _, text in items]
    texts = [describe_content(level) for level in value]
    return texts if texts else [describe_content(level) for level in fallback]


def _level_probs(value: Any, legend: list[str]) -> list[float]:
    """Coerce Score ``probabilities`` into level order.

    Mapping forms may be keyed by level NUMBER (int / numeric string
    → numeric order) or by level TEXT (→ aligned to ``legend``);
    sequences pass through. Missing → empty list.
    """
    if value is None:
        return []
    if isinstance(value, Mapping):
        if not value:
            return []
        try:
            items = sorted(value.items(), key=lambda kv: int(kv[0]))
            return [float(p) for _, p in items]
        except (TypeError, ValueError):
            return [float(value.get(text, 0.0)) for text in legend]
    return [float(p) for p in value]
