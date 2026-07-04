"""loomflow.serve — framework-free ASGI deployment surface (G12).

Tier-2 submodule: import from ``loomflow.serve`` explicitly (nothing is
re-exported from the top-level ``loomflow`` package)::

    from loomflow.serve import create_app

    app = create_app(agent)          # plain ASGI-3 callable
    # uvicorn my_service:app  — or mount inside FastAPI/Starlette.

Run it with any ASGI server (``pip install 'loomflow[serve]'`` for
uvicorn) or via the CLI: ``loom serve my_service:agent``.
"""

from .app import ASGIApp, ServableAgent, create_app

__all__ = ["ASGIApp", "ServableAgent", "create_app"]
