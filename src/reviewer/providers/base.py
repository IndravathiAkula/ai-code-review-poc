"""Provider abstraction — one Protocol, many backends.

The rest of the pipeline (`review_patch`, `_review_chunk`, the retry layer)
talks to a ``Provider`` instead of a particular SDK. Every backend converts
its native response shape to ``ChatResponse`` and its native exceptions to
``ProviderTransientError`` / ``ProviderPermanentError`` so the retry logic
doesn't need to know who served the call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class Usage:
    """Token accounting for one model call.

    ``prompt_tokens`` and ``completion_tokens`` are the universal fields.
    ``cache_read_tokens`` / ``cache_creation_tokens`` are populated by
    providers that support prompt caching (e.g. Anthropic); cost-aware
    pricing uses them to compute the effective spend.
    """
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        # All input tokens (cache hits, cache writes, regular) plus output.
        return (
            self.prompt_tokens
            + self.completion_tokens
            + self.cache_read_tokens
            + self.cache_creation_tokens
        )


@dataclass
class ChatResponse:
    """Provider-agnostic response shape."""
    content: str
    usage: Usage | None


@runtime_checkable
class Provider(Protocol):
    """Anything that can answer ``complete(messages) -> ChatResponse``.

    Attributes:
        name: short identifier used in usage_log and routing decisions
            (e.g. ``"github-models"``, ``"groq"``, ``"openrouter"``).
    """
    name: str

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        response_format: dict | None = None,
    ) -> ChatResponse: ...


class ProviderError(Exception):
    """Base for all provider-layer errors."""


class ProviderTransientError(ProviderError):
    """Retryable: 429, 5xx, connection drop, timeout. ``_call_with_retry``
    will back off and try again."""
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class ProviderPermanentError(ProviderError):
    """Non-retryable: auth (401/403), bad request (400/422), not found.
    The retry layer will surface these to the user with a clear message."""
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code
