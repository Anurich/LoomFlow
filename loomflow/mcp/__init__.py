"""Model Context Protocol (MCP) integration.

* :class:`MCPServerSpec` — declarative description of an MCP server
  (transport + connection details).
* :class:`MCPClient` — wraps a single ``mcp.ClientSession``. The
  ``mcp`` SDK is imported lazily inside :meth:`MCPClient.connect` so
  the module loads without the ``mcp`` extra installed; the import
  fires only when actually connecting to a real server.
* :class:`MCPRegistry` — implements
  :class:`~loomflow.core.protocols.ToolHost` over N MCP servers.
  Connects all servers in parallel through an
  :func:`anyio.create_task_group`, builds a tool name index with
  auto-disambiguation, and routes ``call(tool, args)`` to the right
  session.
"""

from .client import MCPClient
from .registry import MCPRegistry
from .spec import MCPServerSpec, SamplingHandler

__all__ = ["MCPClient", "MCPRegistry", "MCPServerSpec", "SamplingHandler"]
