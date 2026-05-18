"""OpenAI-compatible provider — one adapter for many endpoints.

Any service that speaks the OpenAI Chat Completions wire format works:
Groq, OpenRouter, NVIDIA NIM, Together AI, Anyscale, OpenAI itself, vLLM,
or local Ollama. The only differences are ``base_url`` and ``api_key``.

We disable the SDK's built-in retry loop (``max_retries=0``) so our
``_call_with_retry`` is the single source of truth for backoff behavior.
"""
from __future__ import annotations

from typing import Any

from .base import (
    ChatResponse, Provider, ProviderPermanentError, ProviderTransientError, Usage,
)


_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class OpenAICompatProvider:
    """Generic OpenAI-API-compatible provider.

    Pass a friendly ``name`` (e.g. ``"groq"``, ``"openrouter"``) so it
    shows up in ``usage_log`` and routing decisions. The factory builds
    these from known endpoints; users can construct one directly for a
    custom base URL.
    """

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str | None,
    ):
        from openai import OpenAI  # lazy: avoid import cost when unused
        self.name = name
        # api_key=None blows up the SDK; some local endpoints (Ollama,
        # vLLM) don't need auth, so substitute a placeholder.
        effective_key = api_key if api_key else "no-auth-required"
        self._client = OpenAI(
            base_url=base_url,
            api_key=effective_key,
            max_retries=0,  # _call_with_retry owns retry semantics
        )

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        response_format: dict | None = None,
    ) -> ChatResponse:
        # Lazy-imported so this module loads even if openai is missing
        # in a slim install (the test suite uses a fake client most of
        # the time).
        from openai import (
            APIConnectionError, APIStatusError, APITimeoutError,
        )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except (APIConnectionError, APITimeoutError) as exc:
            raise ProviderTransientError(
                f"connection error: {exc}") from exc
        except APIStatusError as exc:
            status = getattr(exc, "status_code", None)
            if status in _RETRYABLE_STATUS:
                raise ProviderTransientError(
                    str(exc),
                    status_code=status,
                    retry_after_seconds=_retry_after_from_response(exc),
                ) from exc
            raise ProviderPermanentError(str(exc), status_code=status) from exc
        return _to_chat_response(resp)


def _retry_after_from_response(exc) -> float | None:
    """Extract the Retry-After header from an OpenAI APIStatusError."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = None
    try:
        raw = headers.get("retry-after")
    except (AttributeError, TypeError):
        raw = None
    if raw is None:
        try:
            raw = headers["retry-after"]
        except (KeyError, TypeError):
            return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _to_chat_response(resp) -> ChatResponse:
    msg = resp.choices[0].message
    content = msg.content or ""
    u = getattr(resp, "usage", None)
    usage = None
    if u is not None:
        usage = Usage(
            prompt_tokens=int(getattr(u, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(u, "completion_tokens", 0) or 0),
        )
    return ChatResponse(content=content, usage=usage)
