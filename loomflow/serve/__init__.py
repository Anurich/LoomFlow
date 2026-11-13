"""loomflow.serve — framework-free ASGI deployment surface (G12).

Tier-2 submodule: import from ``loomflow.serve`` explicitly (nothing is
re-exported from the top-level ``loomflow`` package)::

    from loomflow.serve import create_app

    app = create_app(agent)          # plain ASGI-3 callable
    # uvicorn my_service:app  — or mount inside FastAPI/Starlette.

The app is open unless you pass ``auth=`` — and without it the
``user_id`` is whatever the client sends. Anything reachable beyond
loopback should authenticate::

    from loomflow.serve import bearer_token_auth, create_app

    app = create_app(agent, auth=bearer_token_auth({"tok-a": "alice"}))

Run it with any ASGI server (``pip install 'loomflow[serve]'`` for
uvicorn) or via the CLI: ``loom serve my_service:agent``.
"""

from .app import ASGIApp, ServableAgent, create_app
from .auth import (
    TRUSTED_CALLER,
    AuthError,
    AuthHook,
    AuthRequest,
    bearer_token_auth,
)

__all__ = [
    "TRUSTED_CALLER",
    "ASGIApp",
    "AuthError",
    "AuthHook",
    "AuthRequest",
    "ServableAgent",
    "bearer_token_auth",
    "create_app",
]
