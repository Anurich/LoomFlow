"""Exception hierarchy.

All harness-raised exceptions inherit from :class:`JeevesAgentError` so
callers can catch the family without binding to specific subtypes.
"""

from __future__ import annotations


class JeevesAgentError(Exception):
    """Base class for all harness errors."""


class ConfigError(JeevesAgentError):
    """Invalid or unresolvable configuration passed to ``Agent``."""


class BudgetExceeded(JeevesAgentError):
    """A run was halted because a budget limit was hit."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class PermissionDenied(JeevesAgentError):
    """A tool call was denied by the permission layer or a user hook."""

    def __init__(self, tool: str, reason: str) -> None:
        super().__init__(f"{tool}: {reason}")
        self.tool = tool
        self.reason = reason


class ToolError(JeevesAgentError):
    """A tool invocation failed at the tool's own boundary."""


class SandboxError(JeevesAgentError):
    """The sandbox refused or failed to execute a tool."""


class RuntimeJournalError(JeevesAgentError):
    """The durable runtime journal is unreadable or inconsistent."""


class MemoryStoreError(JeevesAgentError):
    """The memory backend failed an operation."""


class MCPError(JeevesAgentError):
    """An MCP transport, handshake, or protocol error."""


class FreshnessError(JeevesAgentError):
    """A certified value failed its freshness policy."""


class LineageError(JeevesAgentError):
    """A certified value failed its lineage policy."""


class CancelledByUser(JeevesAgentError):
    """A user-driven interruption (signal, timeout) ended the run."""


class OutputValidationError(JeevesAgentError):
    """The model's final answer did not validate against the supplied
    ``output_schema``.

    Raised by :meth:`Agent.run` when the caller passed
    ``output_schema=`` and the model's final assistant text could
    not be parsed/validated as the requested Pydantic model — even
    after the optional one-shot "retry with the validation error"
    turn.

    Carries the raw model output (``raw``), the underlying Pydantic
    :class:`pydantic.ValidationError` (``cause``, also exposed via
    ``__cause__``), and the schema that was being targeted
    (``schema``) so callers can build whatever recovery strategy
    they need (re-prompt with extra examples, fall back to
    free-text, etc.).
    """

    def __init__(
        self,
        message: str,
        *,
        raw: str,
        schema: type,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.raw = raw
        self.schema = schema
        self.cause = cause


# ---------------------------------------------------------------------------
# Model-call error taxonomy
# ---------------------------------------------------------------------------
#
# Every model SDK has its own exception types (``openai.RateLimitError``,
# ``anthropic.APIStatusError``, etc.) — the framework normalises them
# into the hierarchy below so callers + the retry layer can react
# uniformly.
#
#     ModelError                           — base (catch-all model failure)
#     ├── TransientModelError              — retry-able
#     │   ├── RateLimitError               — 429 / token budget hit; carries retry_after
#     │   └── (anything else transient)    — 5xx, network blips, timeouts
#     └── PermanentModelError              — don't retry
#         ├── AuthenticationError          — 401 / bad API key
#         ├── InvalidRequestError          — 400 / malformed prompt or args
#         └── ContentFilterError           — safety system rejected request/response
#
# All inherit from :class:`JeevesAgentError` so existing
# ``except JeevesAgentError`` catches keep working.


class ModelError(JeevesAgentError):
    """A call to the underlying model adapter failed.

    Base of the model-error taxonomy: catch this to handle every
    model failure regardless of whether it is transient or
    permanent. The SDK exception that triggered the classification
    is attached via ``__cause__`` (and ``cause``) so debug code
    can still inspect the raw error.
    """

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class TransientModelError(ModelError):
    """A model call failed in a way that may succeed on retry.

    Covers HTTP 5xx responses, network errors, timeouts, and
    provider-side rate limits. The retry layer treats this family
    as retryable and applies backoff.

    ``retry_after`` (in seconds) carries a provider-supplied hint
    when one is available — e.g. an ``Retry-After`` HTTP header on
    a 429 response. The retry layer respects the larger of the
    policy's computed backoff and ``retry_after`` so we never wait
    less than the provider asked for.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, cause=cause)
        self.retry_after = retry_after


class RateLimitError(TransientModelError):
    """The provider returned a 429 / quota-exhausted response.

    Carries ``retry_after`` when the provider supplied one. Subclass
    of :class:`TransientModelError` so generic transient handlers
    cover it; catch ``RateLimitError`` specifically when you need
    to surface "slow down" to the caller (e.g. propagate a 429 to
    your own clients)."""


class PermanentModelError(ModelError):
    """A model call failed in a way that retrying will not fix.

    Wrong API key, malformed request, content-filter rejection,
    deprecated model name, etc. The retry layer raises these
    immediately without backoff so callers can fail fast and
    surface the real problem.
    """


class AuthenticationError(PermanentModelError):
    """Invalid, missing, or revoked API credentials."""


class InvalidRequestError(PermanentModelError):
    """The request was malformed or violated the provider's API
    contract — bad parameters, oversized prompt, unknown model
    name, etc. Fix the request, don't retry."""


class ContentFilterError(PermanentModelError):
    """The provider's safety system blocked the request or response.

    Typically a permanent failure for the same prompt; users may
    rephrase but the framework should not silently retry."""
