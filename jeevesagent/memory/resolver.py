"""String / dict resolver for the ``memory=`` :class:`Agent` kwarg.

Mirrors the design of :func:`jeevesagent.agent.api._resolve_model`:
the user passes a string ("``sqlite:./bot.db``") or a dict
({"backend": "chroma", "path": "./mem", ...}) and the framework
returns a fully-constructed :class:`Memory` (or a
:class:`LazyMemory` proxy for async-connect backends).

Recognised string schemes:

* ``inmemory`` — in-process dict, lost on restart (the default)
* ``sqlite:<path>`` — single-file SQLite, persistent
* ``sqlite`` (no path) — alias for ``sqlite::memory:`` (ephemeral)
* ``chroma`` — ephemeral Chroma client
* ``chroma:<path>`` — persistent Chroma at ``<path>``
* ``postgres://<dsn>`` / ``postgresql://<dsn>`` — Postgres+pgvector
* ``redis://<dsn>`` / ``rediss://<dsn>`` — Redis (with optional
  RediSearch vector index)

Recognised dict keys:

* ``backend`` (required) — same scheme set as above
* ``path`` / ``url`` / ``dsn`` — backend-specific connection target
* ``namespace`` — partition / collection / key-prefix (consistent
  name across backends; each backend maps it to its native kwarg)
* ``embedder`` — ``"openai"`` / ``"hash"`` / explicit
  :class:`Embedder` instance. ``None`` triggers auto-pick (OpenAI
  if ``OPENAI_API_KEY`` set, else Hash)
* ``with_facts`` — default ``True`` for the resolver path; pass
  ``False`` to skip the fact-store wiring
* ``collection_name`` (Chroma), ``key_prefix`` (Redis), etc. —
  any backend-native kwarg passes through unchanged

The resolver is **non-async** even when the underlying backend
needs an async connect. Postgres / Redis URLs return a
:class:`LazyMemory` proxy that opens the connection on first use,
so the ``Agent.__init__`` call site stays synchronous.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from ..core.errors import ConfigError
from ..core.protocols import Embedder, Memory
from .embedder import HashEmbedder

__all__ = ["resolve_memory"]


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------


def resolve_memory(spec: Any) -> Memory:
    """Resolve a ``memory=`` argument into a concrete :class:`Memory`.

    ``None`` returns the default in-memory backend; a string parses
    by URL scheme; a dict parses by ``backend`` key; anything else
    is assumed to already be a :class:`Memory` and passed through
    unchanged.
    """
    if spec is None:
        return _build_inmemory()
    if isinstance(spec, str):
        return _resolve_string(spec)
    if isinstance(spec, Mapping):
        return _resolve_dict(spec)
    # Already-constructed Memory — pass through. Duck-typed: anything
    # with the protocol methods is acceptable so users can supply
    # their own custom backends without inheriting from a base class.
    return spec  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# String form
# ---------------------------------------------------------------------------


def _resolve_string(spec: str) -> Memory:
    spec = spec.strip()
    if not spec:
        raise ConfigError(
            "memory= empty string. Use 'inmemory' for the default, "
            "or one of: 'sqlite:./path.db', 'chroma:./mem_dir', "
            "'postgres://...', 'redis://...'."
        )

    # Bare aliases first — no scheme separator.
    if spec == "inmemory":
        return _build_inmemory()
    if spec == "sqlite":
        return _build_sqlite(":memory:", embedder=_default_embedder())
    if spec == "chroma":
        return _build_chroma_ephemeral(embedder=_default_embedder())

    # Schemes carrying a target.
    if spec.startswith("sqlite:") and not spec.startswith("sqlite://"):
        # ``sqlite:./path.db`` — colon is the scheme separator, not
        # a network URL. Trim the prefix and pass the rest as a path.
        path = spec[len("sqlite:"):]
        return _build_sqlite(path or ":memory:", embedder=_default_embedder())
    if spec.startswith("chroma:") and not spec.startswith("chroma://"):
        path = spec[len("chroma:"):]
        return _build_chroma_local(path, embedder=_default_embedder())
    if spec.startswith(("postgres://", "postgresql://")):
        return _build_postgres_lazy(spec, embedder=_default_embedder())
    if spec.startswith(("redis://", "rediss://")):
        return _build_redis_lazy(spec, embedder=_default_embedder())

    raise ConfigError(
        f"memory= unrecognised string spec {spec!r}. Recognised:\n"
        "  inmemory                           — in-process default\n"
        "  sqlite:./path.db                   — persistent single-file\n"
        "  chroma                             — ephemeral Chroma\n"
        "  chroma:./mem_dir                   — persistent Chroma\n"
        "  postgres://user:pw@host/db         — Postgres + pgvector\n"
        "  redis://localhost:6379/0           — Redis\n"
        "Or pass a Memory-protocol instance directly."
    )


# ---------------------------------------------------------------------------
# Dict form
# ---------------------------------------------------------------------------


def _resolve_dict(spec: Mapping[str, Any]) -> Memory:
    # Tolerate either ``backend`` or ``type`` for the discriminator;
    # dict-form configs come from YAML/TOML/env where users already
    # use both conventions.
    backend = spec.get("backend") or spec.get("type")
    if not isinstance(backend, str):
        raise ConfigError(
            "memory= dict must include 'backend' (or 'type'). Recognised "
            "values: 'inmemory', 'sqlite', 'chroma', 'postgres', 'redis'."
        )
    backend = backend.lower()

    embedder = _resolve_embedder(spec.get("embedder"))
    with_facts = bool(spec.get("with_facts", True))

    if backend == "inmemory":
        return _build_inmemory(with_facts=with_facts)

    if backend == "sqlite":
        path = spec.get("path") or spec.get("url") or ":memory:"
        return _build_sqlite(
            str(path), embedder=embedder, with_facts=with_facts
        )

    if backend == "chroma":
        path = spec.get("path") or spec.get("persist_directory")
        collection = (
            spec.get("namespace")
            or spec.get("collection_name")
            or "jeeves_episodes"
        )
        if path:
            return _build_chroma_local(
                str(path),
                embedder=embedder,
                with_facts=with_facts,
                collection_name=str(collection),
            )
        return _build_chroma_ephemeral(
            embedder=embedder,
            with_facts=with_facts,
            collection_name=str(collection),
        )

    if backend in ("postgres", "postgresql"):
        url = spec.get("url") or spec.get("dsn")
        if not isinstance(url, str):
            raise ConfigError(
                "memory= postgres backend requires 'url' (or 'dsn')."
            )
        namespace = spec.get("namespace") or "default"
        return _build_postgres_lazy(
            url,
            embedder=embedder,
            namespace=str(namespace),
            with_facts=with_facts,
        )

    if backend == "redis":
        url = spec.get("url") or "redis://localhost:6379/0"
        if not isinstance(url, str):
            raise ConfigError("memory= redis backend 'url' must be a string.")
        # Redis namespacing comes through as either ``namespace`` (the
        # consistent resolver-level name) or its native ``key_prefix``.
        key_prefix = (
            spec.get("key_prefix")
            or (
                f"{spec.get('namespace')}:"
                if spec.get("namespace") is not None
                else "jeeves:episode:"
            )
        )
        use_vector_index = bool(spec.get("use_vector_index", True))
        return _build_redis_lazy(
            url,
            embedder=embedder,
            key_prefix=str(key_prefix),
            use_vector_index=use_vector_index,
            with_facts=with_facts,
        )

    raise ConfigError(
        f"memory= dict 'backend' = {backend!r} not recognised. "
        "Use 'inmemory', 'sqlite', 'chroma', 'postgres', or 'redis'."
    )


# ---------------------------------------------------------------------------
# Backend builders
# ---------------------------------------------------------------------------


def _build_inmemory(*, with_facts: bool = True) -> Memory:
    """Default backend — dict-backed, ephemeral. ``with_facts`` is
    accepted for consistency with the dict resolver but the in-memory
    backend always provides a fact store regardless (it's free)."""
    from .inmemory import InMemoryMemory

    _ = with_facts  # InMemoryMemory always carries InMemoryFactStore
    return InMemoryMemory()


def _build_sqlite(
    path: str, *, embedder: Embedder, with_facts: bool = True
) -> Memory:
    from .sqlite import SqliteMemory

    return SqliteMemory(path, embedder=embedder, with_facts=with_facts)


def _build_chroma_ephemeral(
    *,
    embedder: Embedder,
    with_facts: bool = True,
    collection_name: str = "jeeves_episodes",
) -> Memory:
    from .chroma import ChromaMemory

    return ChromaMemory.ephemeral(
        embedder=embedder,
        collection_name=collection_name,
        with_facts=with_facts,
    )


def _build_chroma_local(
    path: str,
    *,
    embedder: Embedder,
    with_facts: bool = True,
    collection_name: str = "jeeves_episodes",
) -> Memory:
    from .chroma import ChromaMemory

    return ChromaMemory.local(
        persist_directory=path,
        embedder=embedder,
        collection_name=collection_name,
        with_facts=with_facts,
    )


def _build_postgres_lazy(
    dsn: str,
    *,
    embedder: Embedder,
    namespace: str = "default",
    with_facts: bool = True,
) -> Memory:
    """Wrap ``await PostgresMemory.connect(dsn)`` in a
    :class:`LazyMemory` so the sync :class:`Agent` constructor
    doesn't have to ``await``. Connection opens on first call into
    the resolved memory."""
    from .lazy import LazyMemory
    from .postgres import PostgresMemory

    async def _build() -> Memory:
        result: Memory = await PostgresMemory.connect(
            dsn,
            embedder=embedder,
            namespace=namespace,
            with_facts=with_facts,
        )
        return result

    return LazyMemory(_build, description=dsn)


def _build_redis_lazy(
    url: str,
    *,
    embedder: Embedder,
    key_prefix: str = "jeeves:episode:",
    use_vector_index: bool = True,
    with_facts: bool = True,
) -> Memory:
    from .lazy import LazyMemory
    from .redis import RedisMemory

    async def _build() -> Memory:
        result: Memory = await RedisMemory.connect(
            url,
            embedder=embedder,
            key_prefix=key_prefix,
            use_vector_index=use_vector_index,
            with_facts=with_facts,
        )
        return result

    return LazyMemory(_build, description=url)


# ---------------------------------------------------------------------------
# Embedder auto-pick
# ---------------------------------------------------------------------------


def _resolve_embedder(spec: Any) -> Embedder:
    """Convert a string / instance / None into a concrete
    :class:`Embedder`.

    Strings:

    * ``"hash"`` — :class:`HashEmbedder` (zero-key, deterministic)
    * ``"openai"`` / ``"openai-small"`` —
      :class:`OpenAIEmbedder` with ``text-embedding-3-small``
    * ``"openai-large"`` —
      :class:`OpenAIEmbedder` with ``text-embedding-3-large``
    * ``"voyage"`` — :class:`VoyageEmbedder`
    * ``"cohere"`` — :class:`CohereEmbedder`

    ``None`` → :func:`_default_embedder` auto-picks based on what
    API keys are present in the environment.
    """
    if spec is None:
        return _default_embedder()
    if isinstance(spec, str):
        s = spec.lower()
        if s == "hash":
            return HashEmbedder()
        if s in ("openai", "openai-small"):
            from .embedder import OpenAIEmbedder
            return OpenAIEmbedder("text-embedding-3-small")
        if s == "openai-large":
            from .embedder import OpenAIEmbedder
            return OpenAIEmbedder("text-embedding-3-large")
        if s == "voyage":
            from .embedder import VoyageEmbedder
            return VoyageEmbedder()
        if s == "cohere":
            from .embedder import CohereEmbedder
            return CohereEmbedder()
        raise ConfigError(
            f"memory= embedder spec {spec!r} not recognised. Use one of: "
            "'hash', 'openai', 'openai-large', 'voyage', 'cohere', or "
            "pass an Embedder instance."
        )
    # Duck-typed: anything with ``embed``/``embed_batch`` is fine.
    return spec  # type: ignore[no-any-return]


def _default_embedder() -> Embedder:
    """Pick a sensible embedder based on the environment.

    OpenAI by default when ``OPENAI_API_KEY`` is set (production-
    quality semantic search); falls back to :class:`HashEmbedder`
    otherwise (deterministic, zero-key — fine for dev / tests, not
    for production retrieval quality).
    """
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from .embedder import OpenAIEmbedder
            return OpenAIEmbedder("text-embedding-3-small")
        except ImportError:
            pass
    return HashEmbedder()
