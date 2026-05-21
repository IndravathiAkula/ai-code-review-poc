"""Backwards-compatible factory for the legacy ``client`` object.

New code should use :func:`reviewer.providers.build_provider` instead;
``build_client`` is preserved so old tests and external callers keep
working unchanged.
"""
from __future__ import annotations

import os
import sys

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential


_GH_TOKEN_PREFIXES = (
    "ghp_", "github_pat_", "ghs_", "gho_", "ghu_", "ghr_",
)


def _looks_like_github_token(token: str) -> bool:
    return token.startswith(_GH_TOKEN_PREFIXES)


def build_client(endpoint: str | None = None, token: str | None = None) -> ChatCompletionsClient:
    """Build a GitHub Models ``ChatCompletionsClient`` (legacy interface).

    Kept for backward compatibility with existing callers; prefer
    :func:`reviewer.providers.build_provider` for new code, which
    supports Groq, OpenRouter, NVIDIA, OpenAI direct, Anthropic, and
    custom OpenAI-compatible endpoints.
    """
    endpoint = endpoint or os.environ.get(
        "MODEL_ENDPOINT", "https://models.github.ai/inference")
    # Same lookup order as GitHubModelsProvider — MODEL_API_TOKEN wins so
    # consumers can split the model-call token from the posting token.
    token = (
        token
        or os.environ.get("MODEL_API_TOKEN")
        or os.environ.get("GITHUB_TOKEN")
        or os.environ.get("GH_TOKEN")
    )
    if not token:
        raise RuntimeError(
            "No token found — set MODEL_API_TOKEN, GITHUB_TOKEN, or GH_TOKEN.")
    if not _looks_like_github_token(token):
        print(
            "[warn] GITHUB_TOKEN does not have a recognised GitHub prefix "
            "(ghp_/github_pat_/ghs_/etc.). Check that you copied the right "
            "value — a wrong token will fail the model call with HTTP 401.",
            file=sys.stderr,
        )
    return ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(token))
