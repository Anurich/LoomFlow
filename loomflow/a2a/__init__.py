"""A2A (Agent-to-Agent) protocol support (G10).

Server side — expose a loomflow Agent as an A2A v1.0 endpoint::

    from loomflow.a2a import serve_a2a
    app = serve_a2a(agent)            # pure-ASGI, mounts anywhere

Client side — call a remote A2A agent, optionally as a tool::

    from loomflow.a2a import A2AClient
    remote = A2AClient("https://bots.example/a2a")
    reply = await remote.send("hello")
    agent = Agent("...", tools=[remote.as_tool(name="remote_bot")])

Wire types live in :mod:`loomflow.a2a.types` — self-contained,
implemented from the spec (no ``a2a-sdk`` dependency); the only
optional dependency is httpx, for the client (``loomflow[a2a]``).
"""

from .client import A2AClient
from .server import serve_a2a
from .types import A2AError, AgentCard

__all__ = ["A2AClient", "A2AError", "AgentCard", "serve_a2a"]
