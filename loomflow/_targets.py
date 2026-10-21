"""Loading an object named by ``module:attribute``, as the CLI's
``--agent`` / ``loom serve`` arguments and Studio's eval agents do."""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any


def load_target(spec: str) -> Any:
    """Import ``module:attribute`` and return the attribute. The
    current folder is importable, as it is for ``python -m``, so
    ``app:agent`` finds ``app.py`` where you run the command."""
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"expected 'module:attribute', got {spec!r}")
    cwd = os.getcwd()
    if cwd not in sys.path and "" not in sys.path:
        sys.path.insert(0, cwd)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ValueError(f"module {module_name!r} has no attribute {attr!r}") from exc
