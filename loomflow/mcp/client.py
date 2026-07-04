"""Per-server MCP client wrapping ``mcp.ClientSession`` lifetime.

The ``mcp`` SDK is imported lazily inside the connection task.
Tests can bypass the real connection entirely by passing a
``session=`` kwarg whose object exposes the methods we use:
``initialize()``, ``list_tools()``, ``call_tool(name, args)``.

Task-affinity design
--------------------
The MCP SDK's transport context managers (``stdio_client``,
``streamablehttp_client``, ``ClientSession``) contain anyio cancel
scopes, which **must** be entered and exited by the same task —
otherwise anyio raises ``RuntimeError: Attempted to exit cancel
scope in a different task than it was entered in`` (or hangs).
Callers of :meth:`connect` and :meth:`aclose` are frequently
*different* tasks: :class:`~loomflow.mcp.registry.MCPRegistry`
connects each client inside a task-group child, the child exits,
and ``aclose`` later runs in whatever task tears the registry down.

So the client never holds the context managers across calls.
Instead each connected client owns a private portal thread running
ONE dedicated background task (:meth:`_lifecycle`) that performs
the full connect → serve → close sequence: it enters the context
managers, signals readiness (session + shutdown event) via
``task_status.started()``, then parks on the shutdown event.
:meth:`aclose` sets that event, so the *owning* task unwinds the
exit stack, satisfying anyio's task-affinity rules no matter which
task calls :meth:`connect` / :meth:`aclose`. Session calls are
marshalled into the portal's loop via ``BlockingPortal.call`` from
a worker thread.
"""

from __future__ import annotations

import inspect
import logging
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import AbstractContextManager, AsyncExitStack
from functools import partial
from typing import TYPE_CHECKING, Any

import anyio
import anyio.to_thread
from anyio.from_thread import BlockingPortal, start_blocking_portal

from ..core.errors import MCPError
from .spec import MCPServerSpec, SamplingHandler

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from anyio.abc import TaskStatus

logger = logging.getLogger(__name__)

#: How long ``aclose`` waits for the lifecycle task to unwind its
#: transport stack before abandoning it to portal cancellation.
_CLOSE_GRACE_S = 30.0

#: JSON-RPC method of the MCP "tool list changed" notification.
_TOOLS_LIST_CHANGED = "notifications/tools/list_changed"


class MCPClient:
    """One client per MCP server. Holds the live ``ClientSession``."""

    def __init__(
        self,
        spec: MCPServerSpec,
        *,
        session: Any | None = None,
        sampling_handler: SamplingHandler | None = None,
    ) -> None:
        self._spec = spec
        self._session: Any | None = session
        self._sampling_handler = (
            sampling_handler
            if sampling_handler is not None
            else spec.sampling_handler
        )
        self._connect_lock = anyio.Lock()
        # Populated only for real (non-injected) sessions:
        self._portal_cm: AbstractContextManager[BlockingPortal] | None = None
        self._portal: BlockingPortal | None = None
        self._lifecycle_future: Future[Any] | None = None
        self._shutdown_event: anyio.Event | None = None
        # Set by MCPRegistry: fires when the server sends
        # ``notifications/tools/list_changed``. IMPORTANT: the callback
        # is invoked from the session's message handler, which runs in
        # this client's *portal thread* event loop — it must be
        # synchronous, non-blocking, and thread-safe (e.g. flip a
        # lock-guarded flag). Awaiting/blocking here would stall the
        # session's read loop; calling back into this client's portal
        # would deadlock it.
        self.on_tools_changed: Callable[[str], None] | None = None

    # ---- properties -----------------------------------------------------

    @property
    def spec(self) -> MCPServerSpec:
        return self._spec

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    # ---- lifecycle ------------------------------------------------------

    async def connect(self) -> None:
        """Start the lifecycle task and wait for the session.

        No-op if already connected (or a fake session was injected at
        construction time). Safe to call from any task — the transport
        context managers live in a dedicated background task, not in
        the caller's task.
        """
        if self._session is not None:
            return
        async with self._connect_lock:
            # Re-check under the lock: another task may have finished
            # connecting while we awaited it. mypy's narrowing can't
            # see cross-task mutation across the await point.
            if self._session is not None:
                return  # type: ignore[unreachable]

            def _start_portal() -> tuple[
                AbstractContextManager[BlockingPortal], BlockingPortal
            ]:
                cm = start_blocking_portal()
                portal = cm.__enter__()
                return cm, portal

            cm, portal = await anyio.to_thread.run_sync(_start_portal)
            try:
                # ``start_task`` blocks until the lifecycle task calls
                # ``task_status.started(...)`` (session ready) or dies,
                # in which case its exception is re-raised here.
                future, ready = await anyio.to_thread.run_sync(
                    partial(portal.start_task, self._lifecycle)
                )
            except BaseException:
                with anyio.CancelScope(shield=True):
                    await anyio.to_thread.run_sync(
                        partial(cm.__exit__, None, None, None)
                    )
                raise
            session, shutdown_event = ready
            self._portal_cm = cm
            self._portal = portal
            self._lifecycle_future = future
            self._shutdown_event = shutdown_event
            self._session = session

    async def _lifecycle(
        self,
        *,
        task_status: TaskStatus[tuple[Any, anyio.Event]],
    ) -> None:
        """Own the whole transport lifetime inside ONE task.

        Runs in the client's portal thread. Enters every context
        manager, signals readiness, then parks on the shutdown event;
        when :meth:`aclose` sets the event, the ``async with`` block
        unwinds the stack in this same task.
        """
        shutdown = anyio.Event()
        async with AsyncExitStack() as stack:
            session = await self._open_session(stack)
            task_status.started((session, shutdown))
            await shutdown.wait()

    async def _open_session(self, stack: AsyncExitStack) -> Any:
        """Enter transport + session context managers on ``stack``.

        Split out from :meth:`_lifecycle` so tests can substitute a
        fake transport/session without a real MCP server.
        """
        read, write = await self._open_transport(stack)
        try:
            from mcp.client.session import (  # type: ignore[import-not-found, import-untyped]
                ClientSession,
            )
        except ImportError as exc:  # pragma: no cover — depends on user env
            raise MCPError(
                "MCP SDK not installed. "
                "Install with: pip install 'loomflow[mcp]'"
            ) from exc
        session = await stack.enter_async_context(
            ClientSession(read, write, **self._session_kwargs(ClientSession))
        )
        await session.initialize()
        return session

    def _session_kwargs(self, session_cls: Any) -> dict[str, Any]:
        """Extra ``ClientSession(...)`` kwargs, feature-detected.

        Older SDK versions may lack ``message_handler`` /
        ``sampling_callback`` — inspect the constructor signature and
        only pass what it accepts, degrading gracefully (with a
        warning when a user-supplied sampling handler can't be wired).
        """
        params: Mapping[str, Any]
        try:
            params = inspect.signature(session_cls.__init__).parameters
        except (TypeError, ValueError):  # pragma: no cover — exotic SDKs
            params = {}
        kwargs: dict[str, Any] = {}
        if "message_handler" in params:
            kwargs["message_handler"] = self._handle_incoming_message
        else:  # pragma: no cover — depends on installed SDK
            logger.debug(
                "MCP SDK ClientSession lacks message_handler; "
                "listChanged notifications from %r will be ignored",
                self._spec.name,
            )
        if self._sampling_handler is not None:
            if "sampling_callback" in params:
                kwargs["sampling_callback"] = self._make_sampling_callback()
            else:  # pragma: no cover — depends on installed SDK
                logger.warning(
                    "MCP SDK ClientSession lacks sampling_callback; "
                    "sampling handler for %r will not be wired",
                    self._spec.name,
                )
        return kwargs

    # ---- server-initiated traffic -----------------------------------------

    async def _handle_incoming_message(self, message: Any) -> None:
        """Session message handler (runs in the portal thread's loop).

        Watches for ``notifications/tools/list_changed`` and fires
        :attr:`on_tools_changed`. Everything else is ignored — the SDK
        already routes requests (sampling, roots, ...) to their
        dedicated callbacks before this handler sees them.
        """
        root = getattr(message, "root", message)
        if getattr(root, "method", None) == _TOOLS_LIST_CHANGED:
            self._fire_tools_changed()

    def _fire_tools_changed(self) -> None:
        """Invoke :attr:`on_tools_changed` (isolated; never raises).

        May run on the portal thread — the registry's callback only
        flips a lock-guarded flag, so this cannot block the session's
        read loop or deadlock the portal.
        """
        callback = self.on_tools_changed
        if callback is None:
            return
        try:
            callback(self._spec.name)
        except Exception:  # noqa: BLE001 — a bad callback must not kill the session
            logger.exception(
                "on_tools_changed callback for MCP server %r raised",
                self._spec.name,
            )

    def _make_sampling_callback(self) -> Any:
        """Adapt the user's ``(messages, model_preferences) -> str``
        handler to the SDK's ``SamplingFnT`` shape.

        The returned coroutine function runs inside the portal
        thread's loop when the server requests a completion. Handler
        errors are returned to the server as JSON-RPC ``ErrorData``
        rather than crashing the session.
        """
        handler = self._sampling_handler
        if handler is None:  # pragma: no cover — gated by caller
            raise MCPError(
                f"MCP client {self._spec.name!r}: no sampling handler configured"
            )

        async def _sampling(context: Any, params: Any) -> Any:
            import mcp.types as types  # type: ignore[import-not-found, import-untyped]

            try:
                result = handler(
                    getattr(params, "messages", None),
                    getattr(params, "modelPreferences", None),
                )
                if inspect.isawaitable(result):
                    result = await result
            except Exception as exc:  # noqa: BLE001 — surface to the server
                return types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"sampling handler failed: {exc}",
                )
            return types.CreateMessageResult(
                role="assistant",
                content=types.TextContent(type="text", text=str(result)),
                model="loomflow-sampling-handler",
            )

        return _sampling

    async def aclose(self) -> None:
        """Tear down the session and underlying transport.

        Signals the lifecycle task (which owns the context managers)
        to unwind, waits for it, then stops the portal thread. Safe
        to call from any task.
        """
        portal_cm = self._portal_cm
        portal = self._portal
        future = self._lifecycle_future
        shutdown = self._shutdown_event
        self._session = None
        self._portal_cm = None
        self._portal = None
        self._lifecycle_future = None
        self._shutdown_event = None
        if portal is None or portal_cm is None:
            return

        def _teardown() -> None:
            try:
                if shutdown is not None:

                    async def _signal() -> None:
                        shutdown.set()

                    try:
                        portal.call(_signal)
                    except RuntimeError:
                        pass  # portal no longer running
                if future is not None:
                    try:
                        future.result(timeout=_CLOSE_GRACE_S)
                    except FutureTimeoutError:
                        # Lifecycle didn't unwind in time; portal exit
                        # cancels it inside its own task (still safe).
                        pass
                    except Exception:  # noqa: BLE001 — session already broken
                        pass
            finally:
                portal_cm.__exit__(None, None, None)

        with anyio.CancelScope(shield=True):
            await anyio.to_thread.run_sync(_teardown)

    async def __aenter__(self) -> MCPClient:
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    # ---- protocol surface -----------------------------------------------

    async def list_tools(self) -> list[Any]:
        """Return whatever the SDK gave us — a list of tool descriptors.

        Each descriptor has ``name``, ``description``, ``inputSchema``.
        We don't translate to :class:`ToolDef` here — the registry does
        that, since it also assigns names with disambiguation.
        """
        await self.connect()
        session = self._session
        if session is None:
            raise MCPError(f"MCP client {self._spec.name!r}: session not initialised")
        result = await self._run_session_call(session.list_tools)
        return list(getattr(result, "tools", result) or [])

    async def call_tool(
        self, name: str, args: dict[str, Any]
    ) -> Any:
        """Invoke ``name`` with ``args``. Returns the SDK's CallToolResult."""
        await self.connect()
        session = self._session
        if session is None:
            raise MCPError(f"MCP client {self._spec.name!r}: session not initialised")
        return await self._run_session_call(partial(session.call_tool, name, args))

    async def list_resources(self) -> list[Any]:
        """Return the SDK's resource descriptors (``uri``, ``name``, ...)."""
        session = await self._require_session()
        result = await self._run_session_call(session.list_resources)
        return list(getattr(result, "resources", result) or [])

    async def read_resource(self, uri: str) -> Any:
        """Read ``uri``. Returns the SDK's ReadResourceResult.

        The SDK validates/coerces the string into an ``AnyUrl`` via
        its pydantic request params, so plain strings are fine here.
        """
        session = await self._require_session()
        return await self._run_session_call(partial(session.read_resource, uri))

    async def list_prompts(self) -> list[Any]:
        """Return the SDK's prompt descriptors (``name``, ``arguments``, ...)."""
        session = await self._require_session()
        result = await self._run_session_call(session.list_prompts)
        return list(getattr(result, "prompts", result) or [])

    async def get_prompt(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> Any:
        """Fetch prompt ``name``. Returns the SDK's GetPromptResult."""
        session = await self._require_session()
        return await self._run_session_call(
            partial(session.get_prompt, name, arguments)
        )

    async def _require_session(self) -> Any:
        """Connect if needed and return the live session (or raise)."""
        await self.connect()
        session = self._session
        if session is None:
            raise MCPError(f"MCP client {self._spec.name!r}: session not initialised")
        return session

    async def _run_session_call(self, fn: Any) -> Any:
        """Run one session coroutine in the loop that owns the session.

        Real sessions live in the portal thread's loop, so the call is
        marshalled there; injected fake sessions have no portal and are
        awaited inline.
        """
        portal = self._portal
        if portal is None:
            return await fn()
        return await anyio.to_thread.run_sync(
            partial(portal.call, fn), abandon_on_cancel=True
        )

    # ---- transport plumbing ---------------------------------------------

    async def _open_transport(
        self, stack: AsyncExitStack
    ) -> tuple[Any, Any]:
        """Open the right transport for the spec; return ``(read, write)``."""
        if self._spec.transport == "stdio":
            try:
                from mcp.client.stdio import (  # type: ignore[import-not-found, import-untyped]
                    StdioServerParameters,
                    stdio_client,
                )
            except ImportError as exc:  # pragma: no cover
                raise MCPError(
                    "MCP SDK not installed. "
                    "Install with: pip install 'loomflow[mcp]'"
                ) from exc
            if not self._spec.command:
                raise MCPError(
                    f"stdio MCP spec {self._spec.name!r} has no command set"
                )
            params = StdioServerParameters(
                command=self._spec.command,
                args=list(self._spec.args),
                env=dict(self._spec.env) if self._spec.env else None,
            )
            read, write = await stack.enter_async_context(stdio_client(params))
            return read, write

        if self._spec.transport == "http":
            try:
                from mcp.client.streamable_http import (  # type: ignore[import-not-found, import-untyped]
                    streamablehttp_client,
                )
            except ImportError as exc:  # pragma: no cover
                raise MCPError(
                    "MCP SDK not installed. "
                    "Install with: pip install 'loomflow[mcp]'"
                ) from exc
            if not self._spec.url:
                raise MCPError(
                    f"http MCP spec {self._spec.name!r} has no url set"
                )
            ctx = streamablehttp_client(
                self._spec.url,
                headers=dict(self._spec.headers) if self._spec.headers else None,
            )
            triplet = await stack.enter_async_context(ctx)
            # streamablehttp_client returns (read, write, get_session_id)
            read, write = triplet[0], triplet[1]
            return read, write

        raise MCPError(f"unsupported transport: {self._spec.transport!r}")
