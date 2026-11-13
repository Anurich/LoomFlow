"""Authentication hook for the HTTP surfaces (``serve`` and ``a2a``).

Both :func:`loomflow.serve.create_app` and
:func:`loomflow.a2a.serve_a2a` accept ``auth=``: a sync or async
callable that receives an :class:`AuthRequest` (method, path,
headers, raw ASGI scope) and returns who the caller is::

    async def auth(request: AuthRequest) -> str | None:
        claims = verify_jwt(request.bearer_token)   # your code
        return claims["sub"] if claims else None

    app = create_app(agent, auth=auth)

The hook's return value decides the request:

* a non-empty ``str`` — the authenticated identity. It **is** the
  run's ``user_id``; a ``user_id`` in the request body that differs
  is rejected with 403.
* :data:`TRUSTED_CALLER` — the caller is authenticated but is not
  itself an end user (typically your own backend holding a shared
  secret). The ``user_id`` in the request body is honoured.
* ``None``, any other value, or any exception (raise
  :class:`AuthError` to say why in the server log) — 401.

:func:`bearer_token_auth` is the built-in option: a constant-time
``Authorization: Bearer <token>`` check.

Without ``auth`` the servers are open and ``user_id`` is whatever
the client put in the body — fine on loopback for local development,
not for anything reachable by other people.
"""

from __future__ import annotations

import hmac
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Final

__all__ = [
    "TRUSTED_CALLER",
    "AuthError",
    "AuthHook",
    "AuthRequest",
    "Identity",
    "bearer_token_auth",
    "is_loopback_host",
]

_log = logging.getLogger("loomflow.serve")


class AuthError(Exception):
    """Raise from an ``auth`` hook to reject the request (401).

    The message is logged server-side and never sent to the client.
    """


class _TrustedCaller:
    """Type of the :data:`TRUSTED_CALLER` sentinel."""

    _instance: _TrustedCaller | None = None

    def __new__(cls) -> _TrustedCaller:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "TRUSTED_CALLER"


TRUSTED_CALLER: Final = _TrustedCaller()
"""Return this from an ``auth`` hook for an authenticated caller that
may act for any user (e.g. your backend): the request body's
``user_id`` is then honoured as-is."""

Identity = str | _TrustedCaller
"""What a successful ``auth`` hook returns."""


@dataclass(frozen=True)
class AuthRequest:
    """What an ``auth`` hook sees of the incoming request.

    ``headers`` maps lower-cased header names to their (latin-1
    decoded) values; when a header repeats, the last one wins.
    ``scope`` is the raw ASGI scope for anything else (client
    address, TLS extension, query string).
    """

    method: str
    path: str
    headers: Mapping[str, str]
    scope: Mapping[str, Any] = field(repr=False, default_factory=dict)

    def header(self, name: str) -> str | None:
        """Case-insensitive header lookup."""
        return self.headers.get(name.lower())

    @property
    def bearer_token(self) -> str | None:
        """The token from ``Authorization: Bearer <token>``, if any."""
        value = self.headers.get("authorization")
        if not value:
            return None
        scheme, _, token = value.partition(" ")
        token = token.strip()
        if scheme.lower() != "bearer" or not token:
            return None
        return token


AuthHook = Callable[[AuthRequest], "Identity | None | Awaitable[Identity | None]"]
"""Signature of the ``auth=`` hook (sync or async)."""


def bearer_token_auth(
    tokens: str | Mapping[str, str],
    *,
    user_id: str | None = None,
) -> AuthHook:
    """Built-in ``auth`` hook: ``Authorization: Bearer <token>``.

    * ``bearer_token_auth("s3cret")`` — one shared secret. Whoever
      presents it is a :data:`TRUSTED_CALLER` (your backend, a
      gateway): the body ``user_id`` is honoured. Pass
      ``user_id="alice"`` to pin every request to one identity
      instead.
    * ``bearer_token_auth({"tok-a": "alice", "tok-b": "bob"})`` —
      one token per user; the mapped value is the identity, and a
      body ``user_id`` naming someone else is a 403.

    Tokens are compared in constant time (every configured token is
    checked on every request, so timing reveals neither a prefix
    match nor which entry matched).
    """
    table: list[tuple[bytes, Identity]] = []
    if isinstance(tokens, str):
        table.append(
            (tokens.encode("utf-8"), user_id if user_id else TRUSTED_CALLER)
        )
    else:
        if user_id is not None:
            raise ValueError(
                "user_id= only applies to a single shared token; a "
                "token mapping already names each token's user"
            )
        for token, who in tokens.items():
            if not isinstance(who, str) or not who:
                raise ValueError("token mapping values must be non-empty user ids")
            table.append((token.encode("utf-8"), who))
    if not table or any(not token for token, _ in table):
        raise ValueError("bearer_token_auth needs at least one non-empty token")

    def _auth(request: AuthRequest) -> Identity | None:
        presented = request.bearer_token
        candidate = (presented or "").encode("utf-8")
        matched: Identity | None = None
        for token, who in table:
            if hmac.compare_digest(candidate, token):
                matched = who
        return matched if presented else None

    return _auth


def _auth_request(scope: MutableMapping[str, Any]) -> AuthRequest:
    headers: dict[str, str] = {}
    for key, value in scope.get("headers") or ():
        headers[bytes(key).decode("latin-1").lower()] = bytes(value).decode("latin-1")
    return AuthRequest(
        method=str(scope.get("method", "GET")).upper(),
        path=str(scope.get("path", "")),
        headers=headers,
        scope=scope,
    )


async def authenticate(
    auth: AuthHook, scope: MutableMapping[str, Any]
) -> Identity | None:
    """Run ``auth`` for this request. ``None`` means "respond 401".

    Fails closed: a hook that raises (anything) or returns something
    that isn't an identity is treated as unauthenticated.
    """
    try:
        outcome = auth(_auth_request(scope))
        if inspect.isawaitable(outcome):
            outcome = await outcome
    except AuthError as exc:
        _log.info("request rejected by auth hook: %s", exc)
        return None
    except Exception:  # noqa: BLE001 — a broken hook must not let requests through
        _log.exception("auth hook raised; treating the request as unauthenticated")
        return None
    if isinstance(outcome, _TrustedCaller):
        return outcome
    if isinstance(outcome, str) and outcome:
        return outcome
    return None


def resolve_user_id(
    identity: Identity | None, claimed: str | None
) -> tuple[bool, str | None]:
    """Reconcile the authenticated identity with the body's user id.

    Returns ``(allowed, user_id)``. With no auth configured
    (``identity is None``) or a :data:`TRUSTED_CALLER`, the claimed
    id is used. With a concrete identity, the identity wins and a
    different claimed id is refused (``allowed`` is ``False``).
    """
    if identity is None or isinstance(identity, _TrustedCaller):
        return True, claimed
    if claimed is not None and claimed != identity:
        return False, None
    return True, identity


_LOOPBACK_NAMES = frozenset({"localhost", "ip6-localhost", "ip6-loopback"})


def is_loopback_host(host: str | None) -> bool:
    """``True`` when ``host`` is a loopback bind address / name.

    ``0.0.0.0``, ``::`` and the empty string (all interfaces) are
    NOT loopback. Unix-socket servers (``None``) count as local.
    """
    if host is None:
        return True
    name = host.strip().strip("[]").lower()
    if name in _LOOPBACK_NAMES or name.endswith(".localhost"):
        return True
    import ipaddress  # noqa: PLC0415

    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


OPEN_SERVER_WARNING = (
    "\n"
    "  ************************************************************\n"
    "  WARNING: serving WITHOUT authentication on {host}.\n"
    "  Anyone who can reach this address can run the agent, and the\n"
    "  user_id is whatever the client puts in the request body, so\n"
    "  callers can act as any user. Pass auth= (see\n"
    "  loomflow.serve.bearer_token_auth), use `loom serve\n"
    "  --auth-token` / LOOM_SERVE_TOKEN, or bind to 127.0.0.1.\n"
    "  ************************************************************\n"
)


def warn_if_open(scope: Mapping[str, Any], warned: list[bool]) -> None:
    """Log :data:`OPEN_SERVER_WARNING` once when an auth-less app sees
    a request on a non-loopback listening address."""
    if warned:
        return
    server = scope.get("server")
    if not server:
        return
    host = server[0]
    if isinstance(host, str) and not is_loopback_host(host):
        warned.append(True)
        _log.warning(OPEN_SERVER_WARNING.format(host=host))
