"""Read MCP protocol fields on any supported ``mcp`` SDK.

mcp 1.x exposes protocol fields under their wire names (``inputSchema``,
``isError``, ``mimeType``, ...); mcp 2.x renamed the Python attributes
to snake_case (``input_schema``, ``is_error``, ``mime_type``). Reading
only one spelling silently returns ``None`` on the other SDK, which
lost tool schemas, ignored ``isError`` and dropped ``destructiveHint``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_MISSING = object()
_CAMEL = re.compile(r"(?<!^)(?=[A-Z])")


def mcp_field(obj: Any, wire_name: str, default: Any = None) -> Any:
    """``obj``'s field named ``wire_name`` (camelCase), on mcp 1.x or 2.x.

    Dicts are read by either spelling too, for servers that hand back
    raw JSON.
    """
    if obj is None:
        return default
    snake = _CAMEL.sub("_", wire_name).lower()
    if isinstance(obj, Mapping):
        if wire_name in obj:
            return obj[wire_name]
        return obj.get(snake, default)
    for attr in (wire_name, snake):
        value = getattr(obj, attr, _MISSING)
        if value is not _MISSING:
            return value
    return default
