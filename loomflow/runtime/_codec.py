"""JSON wire format for journal values.

Journal values are step results: :class:`~loomflow.core.types.ToolResult`,
model replies (tuples of text, tool calls and usage), stream chunks, and
whatever a ``runtime.step`` callable returns. They were stored with
:mod:`pickle`, so anyone able to write the journal could run code in
every process that replayed from it. This codec stores them as tagged
JSON instead: JSON types as themselves, and tuples, sets, bytes,
datetimes, enums, Pydantic models and dataclasses as ``{"$t": ...}``
objects.

Decoding never imports a module or runs a constructor: a model,
dataclass or enum is rebuilt only from a class whose module is already
loaded, models through ``model_construct`` (no validators), dataclasses
by setting their fields on a bare instance.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import sys
import uuid
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import Any

from pydantic import BaseModel

#: Prefix of every JSON-encoded value (pickles start with ``0x80``).
MAGIC = b"LJ1"

_TAG = "$t"


class UndecodableValue(Exception):
    """A stored value this codec can't (or won't) rebuild."""


def _class_path(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _resolve(path: str) -> type:
    module_name, _, qualname = path.partition(":")
    module = sys.modules.get(module_name)
    if module is None:
        raise UndecodableValue(f"{path}: module {module_name!r} is not loaded")
    obj: Any = module
    for part in qualname.split("."):
        obj = getattr(obj, part, None)
    if not isinstance(obj, type):
        raise UndecodableValue(f"{path} is not a class")
    return obj


def _encode(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value) and _TAG not in value:
            return {k: _encode(v) for k, v in value.items()}
        return {_TAG: "map", "v": [[_encode(k), _encode(v)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {_TAG: "tuple", "v": [_encode(v) for v in value]}
    if isinstance(value, frozenset):
        return {_TAG: "frozenset", "v": [_encode(v) for v in value]}
    if isinstance(value, set):
        return {_TAG: "set", "v": [_encode(v) for v in value]}
    if isinstance(value, bytes):
        return {_TAG: "bytes", "v": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Enum):
        return {_TAG: "enum", "cls": _class_path(type(value)), "v": _encode(value.value)}
    if isinstance(value, BaseModel):
        return {
            _TAG: "model",
            "cls": _class_path(type(value)),
            "v": {k: _encode(v) for k, v in value.__dict__.items()},
            "x": _encode(value.__pydantic_extra__ or {}),
            "set": sorted(value.model_fields_set),
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            _TAG: "dataclass",
            "cls": _class_path(type(value)),
            "v": {f.name: _encode(getattr(value, f.name)) for f in dataclasses.fields(value)},
        }
    if isinstance(value, datetime):
        return {_TAG: "datetime", "v": value.isoformat()}
    if isinstance(value, date):
        return {_TAG: "date", "v": value.isoformat()}
    if isinstance(value, time):
        return {_TAG: "time", "v": value.isoformat()}
    if isinstance(value, timedelta):
        return {_TAG: "timedelta", "v": [value.days, value.seconds, value.microseconds]}
    if isinstance(value, Decimal):
        return {_TAG: "decimal", "v": str(value)}
    if isinstance(value, uuid.UUID):
        return {_TAG: "uuid", "v": str(value)}
    if isinstance(value, PurePath):
        return {_TAG: "path", "cls": _class_path(type(value)), "v": str(value)}
    raise TypeError(f"can't journal a {type(value).__module__}.{type(value).__qualname__} as JSON")


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    tag = value.get(_TAG)
    if tag is None:
        return {k: _decode(v) for k, v in value.items()}
    v: Any = value.get("v")
    if tag == "map":
        return {_decode(k): _decode(item) for k, item in v}
    if tag == "tuple":
        return tuple(_decode(i) for i in v)
    if tag == "set":
        return {_decode(i) for i in v}
    if tag == "frozenset":
        return frozenset(_decode(i) for i in v)
    if tag == "bytes":
        return base64.b64decode(v)
    if tag == "datetime":
        return datetime.fromisoformat(v)
    if tag == "date":
        return date.fromisoformat(v)
    if tag == "time":
        return time.fromisoformat(v)
    if tag == "timedelta":
        return timedelta(days=v[0], seconds=v[1], microseconds=v[2])
    if tag == "decimal":
        return Decimal(v)
    if tag == "uuid":
        return uuid.UUID(v)
    cls = _resolve(value["cls"])
    if tag == "path" and issubclass(cls, PurePath):
        return cls(v)
    if tag == "enum" and issubclass(cls, Enum):
        return cls(_decode(v))
    if tag == "model" and issubclass(cls, BaseModel):
        fields = {k: _decode(item) for k, item in v.items()}
        fields.update(_decode(value.get("x") or {}))
        return cls.model_construct(set(value.get("set") or ()), **fields)
    if tag == "dataclass" and dataclasses.is_dataclass(cls):
        obj = object.__new__(cls)
        for name, item in v.items():
            object.__setattr__(obj, name, _decode(item))
        return obj
    raise UndecodableValue(f"unknown or mismatched tag {tag!r} for {value.get('cls')}")


def dumps(value: Any) -> bytes:
    """Encode ``value``; ``TypeError`` when it holds a type this codec
    doesn't store."""
    return MAGIC + json.dumps(_encode(value), separators=(",", ":")).encode("utf-8")


def loads(blob: bytes) -> Any:
    """Decode a value written by :func:`dumps`; ``UndecodableValue``
    for anything else (a pickle, an unknown class)."""
    if not blob.startswith(MAGIC):
        raise UndecodableValue("not a JSON journal value")
    try:
        return _decode(json.loads(blob[len(MAGIC) :]))
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise UndecodableValue(str(exc)) from exc
