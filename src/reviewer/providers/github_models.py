"""GitHub Models provider — wraps the ``azure-ai-inference`` SDK.

Endpoint: ``https://models.github.ai/inference``. Auth: GITHUB_TOKEN with
``models: read`` permission.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Iterator

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ServiceRequestError

from .base import (
    ChatResponse, Provider, ProviderPermanentError, ProviderTransientError, Usage,
)


_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

_GH_TOKEN_PREFIXES = (
    "ghp_", "github_pat_", "ghs_", "gho_", "ghu_", "ghr_",
)


def _looks_like_github_token(token: str) -> bool:
    return token.startswith(_GH_TOKEN_PREFIXES)


def _to_sdk_messages(messages: list[dict[str, Any]]):
    """Translate role/content dicts → Azure SDK message objects."""
    out = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            out.append(SystemMessage(content))
        elif role == "user":
            out.append(UserMessage(content))
        else:
            raise ValueError(f"GitHubModelsProvider: unsupported role {role!r}")
    return out


class GitHubModelsProvider:
    """Talks to GitHub Models via the Azure AI Inference SDK."""

    name = "github-models"

    def __init__(
        self,
        endpoint: str | None = None,
        token: str | None = None,
        client: ChatCompletionsClient | None = None,
    ):
        if client is not None:
            self._client = client
            return
        endpoint = endpoint or os.environ.get(
            "MODEL_ENDPOINT", "https://models.github.ai/inference")
        # Look up tokens in priority order:
        #   1. ``token`` arg (explicit caller override)
        #   2. ``MODEL_API_TOKEN`` env (workflow-scoped, has models:read)
        #   3. ``GITHUB_TOKEN`` env (app token or whatever the runtime sets)
        #   4. ``GH_TOKEN`` env (rarely used)
        # The split between MODEL_API_TOKEN and GITHUB_TOKEN lets consumers
        # use a GitHub App installation token for posting comments while
        # keeping the workflow's GITHUB_TOKEN (with models:read via
        # ``permissions:`` in the yaml) for the inference call.
        token = (
            token
            or os.environ.get("MODEL_API_TOKEN")
            or os.environ.get("GITHUB_TOKEN")
            or os.environ.get("GH_TOKEN")
        )
        if not token:
            raise RuntimeError(
                "No token found for GitHub Models — set MODEL_API_TOKEN, "
                "GITHUB_TOKEN, or GH_TOKEN.")
        if not _looks_like_github_token(token):
            print(
                "[warn] GITHUB_TOKEN does not have a recognised GitHub prefix "
                "(ghp_/github_pat_/ghs_/etc.). A wrong token will fail the "
                "model call with HTTP 401.",
                file=sys.stderr,
            )
        self._client = ChatCompletionsClient(
            endpoint=endpoint, credential=AzureKeyCredential(token))

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        response_format: dict | None = None,
    ) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": _to_sdk_messages(messages),
            "temperature": temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        try:
            resp = self._client.complete(**kwargs)
        except HttpResponseError as exc:
            status = getattr(exc, "status_code", None)
            if status in _RETRYABLE_STATUS:
                raise ProviderTransientError(
                    str(exc),
                    status_code=status,
                    retry_after_seconds=_retry_after(exc),
                ) from exc
            raise ProviderPermanentError(str(exc), status_code=status) from exc
        except ServiceRequestError as exc:
            raise ProviderTransientError(
                f"connection error: {exc}", status_code=None) from exc
        return _to_chat_response(resp)

    def complete_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        response_format: dict | None = None,
    ) -> Iterator[str]:
        """Fallback streaming for GitHub Models.

        The Azure AI Inference SDK's streaming surface has been
        unstable across versions; rather than couple this provider to
        a specific SDK API shape, we fall back to a single-chunk yield
        from the non-streaming ``complete()`` call. The caller's
        progress callback fires once with the full content. For real
        token-level streaming, route through ``openai`` or
        ``anthropic`` providers instead.
        """
        resp = self.complete(
            model=model, messages=messages,
            temperature=temperature, response_format=response_format,
        )
        if resp.content:
            yield resp.content


def _retry_after(exc: HttpResponseError) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response else None
    if not headers or not hasattr(headers, "get"):
        return None
    raw = headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _to_chat_response(resp) -> ChatResponse:
    content = resp.choices[0].message.content or ""
    u = getattr(resp, "usage", None)
    usage = None
    if u is not None:
        usage = Usage(
            prompt_tokens=int(getattr(u, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(u, "completion_tokens", 0) or 0),
        )
    return ChatResponse(content=content, usage=usage)
