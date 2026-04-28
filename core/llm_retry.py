"""Retry-with-backoff wrapper around ``litellm.completion``.

Distinguishes retryable errors (network / rate-limit / 5xx) from terminal
ones (auth / bad-request / content-policy) so we don't waste tokens
reattempting an unrecoverable failure. Non-retryable exceptions propagate
unchanged on the first attempt.
"""
from __future__ import annotations

import random
import time
from typing import Any

import litellm

# Provider-side transient errors. Everything else (Authentication,
# BadRequest, ContentPolicyViolation, malformed messages, ...) is terminal:
# retrying just reproduces the failure and burns the API budget.
_RETRYABLE: tuple[type[BaseException], ...] = (
    litellm.RateLimitError,
    litellm.APIConnectionError,
    litellm.Timeout,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
    litellm.APIError,
)


def completion_with_retry(
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    log_prefix: str = "",
    **kwargs: Any,
):
    """Drop-in replacement for ``litellm.completion(**kwargs)`` with retries.

    Args:
        max_attempts: Total attempts including the initial one (3 = 1 initial +
            up to 2 retries).
        base_delay: Seconds for exponential backoff
            (``delay = base_delay * 2**(attempt-1) + jitter``).
        log_prefix: Prepended to retry-log lines so they align with the
            surrounding session-loop / summariser indentation.
        **kwargs: Forwarded verbatim to ``litellm.completion``.

    Re-raises the last retryable exception once the budget is exhausted, or
    propagates non-retryable exceptions on the first occurrence.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return litellm.completion(**kwargs)
        except _RETRYABLE as e:
            last_exc = e
            if attempt >= max_attempts:
                break
            # Exponential backoff with small jitter so concurrent callers
            # (we don't have any yet, cheap belt-and-braces) don't sync up.
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            print(
                f"{log_prefix}[retry {attempt}/{max_attempts - 1} after "
                f"{type(e).__name__}; sleeping {delay:.1f}s]"
            )
            time.sleep(delay)
        # Non-retryable exceptions intentionally fall through (uncaught) so
        # AuthenticationError / ContentPolicyViolationError / BadRequestError
        # surface immediately to the caller's existing except block.
    assert last_exc is not None  # only reached if all attempts were retryable
    raise last_exc
