"""First-party Jeeves Gateway integration.

Sugar over :mod:`loomflow.mcp` for the public Jeeves MCP gateway.
``JeevesGateway`` is itself a ``ToolHost`` so it drops straight into
``Agent(tools=...)``.

Quick start::

    from loomflow import Agent
    from loomflow.jeeves import JeevesGateway

    agent = Agent(
        "You are a productivity assistant",
        tools=JeevesGateway.from_env(),
    )
"""

from .client import (
    JEEVES_API_KEY_ENV,
    JEEVES_DEFAULT_BASE_URL,
    JEEVES_TOKEN_PREFIX,
    JeevesConfig,
    JeevesGateway,
    looks_like_jeeves_key,
)

__all__ = [
    "JEEVES_API_KEY_ENV",
    "JEEVES_DEFAULT_BASE_URL",
    "JEEVES_TOKEN_PREFIX",
    "JeevesConfig",
    "JeevesGateway",
    "looks_like_jeeves_key",
]
