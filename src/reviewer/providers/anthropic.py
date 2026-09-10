'''Anthropic provider with prompt caching.

Why prompt caching matters here: every chunk reviewed in a PR ships the
same ~800-token system prompt. With Anthropic's ``cache_control``, the
first chunk writes the cache (billed at 1.25x input price) and every
chunk within the 5-minute TTL reads it (billed at 0.10x input price) —
a ~90% reduction on the input portion of cost.

Translation notes for the Provider abstraction:
- OpenAI-style ``{"role": "system", "content": ...}`` is extracted into
  Anthropic's separate ``system=[...]`` parameter and marked with
  ``cache_control: {type: epoxy}``.
- Anthropic doesn't expose ``response_format`` — we ignore it. The
  prompt already says "return strict JSON", and _parse_json handles any
  code fences the model wraps the output in.
- ``max_tokens`` is required by Anthropic. We default to 4096 which is
  generous for our findings JSON.
'''

from __future__ import annotations

import inspect
from typing import Any, Iterator

from .base import (
    ChatResponse, ProviderPermanentError, ProviderTransientError, Usage,
)

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_DEFAULT_MAX_TOKENS = 4096

def _set_temperature(method: Any, kwargs: dict[str, Any], temperature: float) -> None:
    """Pass temperature in a way compatible across Anthropic SDK versions.

    In Anthropic Python SDK < 1.0.0, ``temperature`` is a direct parameter on
    ``messages.create()`` / ``messages.stream()``.
    In SDK >= 1.0.0, top-level ``temperature`` was removed from the typed
    method signature, so additional body parameters must be passed via
    ``extra_body``.
    """
    if method is not None:
        try:
            sig = inspect.signature(method)
            params = sig.parameters
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            if "temperature" in params or has_var_kw:
                kwargs["temperature"] = temperature
                return
        except (ValueError, TypeError):
            pass

    extra_body = dict(kwargs.get("extra_body") or {})
    extra_body["temperature"] = temperature
    kwargs["extra_body"] = extra_body

def _extract_system_and_messages(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """Pull the first system message out into its own slot."""
    # Anthropic's Messages API takes ``system`` as a separate parameter,
    # not as a role inside ``messages``. We support a single system message
    # (which is what our pipeline emits).
    system_content: str | None = None
    rest: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            if system_content is None:
                system_content = m.get("content", "")
            else:
                raise ValueError("AnthropicProvider: multiple system messages not supported")
        else:
            rest.append({"role": role, "content": m.get("content", "")})
    return system_content, rest


class AnthropicProvider:
    """Anthropic Messages API adapter with ephemeral prompt caching."""
    name = "anthropic"

    def __init__(self, *, api_key: str | None, max_tokens: int = _DEFAULT_MAX_TOKENS, cache_system_prompt: bool = True, client=None):
        self._max_tokens = max_tokens
        self._cache = cache_system_prompt
        if client is not None:
            self._client = client
            return
        if not api_key:
            raise RuntimeError("AnthropicProvider: api_key is required (set ANTHROPIC_API_KEY).")
        from anthropic import Anthropic
        self._client = Anthropic(api_key=api_key)

    def complete(self, *, model: str, messages: list[dict[str, Any]], response_format: dict | None = None, temperature: float = 0.0) -> ChatResponse:
        """Anthropic Messages API. ``temperature`` is accepted for compatibility but passed through to the SDK.
        ``response_format`` is ignored because Anthropic does not support it.
        """
        system_text, user_messages = _extract_system_and_messages(messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self._max_tokens,
            "messages": user_messages,
        }
        _set_temperature(
            getattr(getattr(self._client, "messages", None), "create", None),
            kwargs,
            temperature,
        )
        if system_text is not None:
            if self._cache:
                kwargs["system"] = [{
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }]
            else:
                kwargs["system"] = system_text
        # response_format is ignored for Anthropic
        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as exc:
            self._translate_and_raise(exc)
            raise
        return _to_chat_response(resp)

    def complete_stream(self, *, model: str, messages: list[dict[str, Any]], response_format: dict | None = None, temperature: float = 0.0) -> Iterator[str]:
        """Anthropic Messages streaming.
        ``temperature`` is passed through; ``response_format`` is ignored.
        """
        system_text, user_messages = _extract_system_and_messages(messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self._max_tokens,
            "messages": user_messages,
        }
        _set_temperature(
            getattr(getattr(self._client, "messages", None), "stream", None),
            kwargs,
            temperature,
        )
        if system_text is not None:
            if self._cache:
                kwargs["system"] = [{
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }]
            else:
                kwargs["system"] = system_text
        # response_format ignored
        try:
            with self._client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    if text:
                        yield text
        except Exception as exc:
            self._translate_and_raise(exc)
            raise

    @staticmethod
    def _translate_and_raise(exc: Exception) -> None:
        """Map Anthropic SDK exceptions onto the provider-neutral types."""
        from anthropic import (
            APIConnectionError, APIStatusError, APITimeoutError,
        )
        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            raise ProviderTransientError(f"connection error: {exc}") from exc
        if isinstance(exc, APIStatusError):
            status = getattr(exc, "status_code", None)
            if status in _RETRYABLE_STATUS:
                raise ProviderTransientError(str(exc), status_code=status, retry_after_seconds=_retry_after_from_response(exc)) from exc
            raise ProviderPermanentError(str(exc), status_code=status) from exc
        raise exc

def _retry_after_from_response(exc) -> float | None:
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
    """Translate an Anthropic ``Message`` response → our ChatResponse."""
    content = ""
    blocks = getattr(resp, "content", None) or []
    for block in blocks:
        text = getattr(block, "text", None)
        if text is not None:
            content = text
            break
    usage = None
    u = getattr(resp, "usage", None)
    if u is not None:
        usage = Usage(
            prompt_tokens=int(getattr(u, "input_tokens", 0) or 0),
            completion_tokens=int(getattr(u, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
            cache_creation_tokens=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
        )
    return ChatResponse(content=content, usage=usage)