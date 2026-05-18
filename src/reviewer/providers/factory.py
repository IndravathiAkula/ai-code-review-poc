"""Factory for building Provider instances by name.

Known providers are pre-configured with their base URLs and the
environment variable that holds the API key. Users can override
``base_url`` or ``auth_env`` per-provider, or add a brand-new provider
via the ``REVIEWER_PROVIDER`` + ``REVIEWER_BASE_URL`` + ``REVIEWER_API_KEY_ENV``
env trio (no code change needed for custom OpenAI-compatible endpoints).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .anthropic import AnthropicProvider
from .base import Provider
from .github_models import GitHubModelsProvider
from .openai_compat import OpenAICompatProvider


@dataclass(frozen=True)
class ProviderSpec:
    """Static metadata for a known provider."""
    name: str
    kind: str              # "github-models" | "openai-compat"
    base_url: str | None   # None for github-models (it has its own SDK)
    auth_env: str | None   # env var that holds the API key; None for unauthed (Ollama)


PROVIDER_REGISTRY: dict[str, ProviderSpec] = {
    "github-models": ProviderSpec(
        name="github-models",
        kind="github-models",
        base_url="https://models.github.ai/inference",
        auth_env="GITHUB_TOKEN",
    ),
    "openai": ProviderSpec(
        name="openai",
        kind="openai-compat",
        base_url="https://api.openai.com/v1",
        auth_env="OPENAI_API_KEY",
    ),
    "anthropic": ProviderSpec(
        name="anthropic",
        kind="anthropic",
        base_url=None,  # SDK manages the endpoint
        auth_env="ANTHROPIC_API_KEY",
    ),
    "groq": ProviderSpec(
        name="groq",
        kind="openai-compat",
        base_url="https://api.groq.com/openai/v1",
        auth_env="GROQ_API_KEY",
    ),
    "openrouter": ProviderSpec(
        name="openrouter",
        kind="openai-compat",
        base_url="https://openrouter.ai/api/v1",
        auth_env="OPENROUTER_API_KEY",
    ),
    "nvidia": ProviderSpec(
        name="nvidia",
        kind="openai-compat",
        base_url="https://integrate.api.nvidia.com/v1",
        auth_env="NVIDIA_API_KEY",
    ),
    "together": ProviderSpec(
        name="together",
        kind="openai-compat",
        base_url="https://api.together.xyz/v1",
        auth_env="TOGETHER_API_KEY",
    ),
    "anyscale": ProviderSpec(
        name="anyscale",
        kind="openai-compat",
        base_url="https://api.endpoints.anyscale.com/v1",
        auth_env="ANYSCALE_API_KEY",
    ),
    "cerebras": ProviderSpec(
        name="cerebras",
        kind="openai-compat",
        base_url="https://api.cerebras.ai/v1",
        auth_env="CEREBRAS_API_KEY",
    ),
    "ollama": ProviderSpec(
        name="ollama",
        kind="openai-compat",
        base_url="http://localhost:11434/v1",
        auth_env=None,  # local, no auth
    ),
}


def known_providers() -> list[str]:
    return sorted(PROVIDER_REGISTRY)


def build_provider(
    name: str | None = None,
    *,
    env: dict[str, str] | None = None,
) -> Provider:
    """Build a Provider instance by name.

    Resolution:
      1. ``name`` argument (highest)
      2. ``REVIEWER_PROVIDER`` env var
      3. ``github-models`` (default)

    For custom endpoints not in the registry, set
    ``REVIEWER_PROVIDER=custom``, ``REVIEWER_BASE_URL=<url>``,
    ``REVIEWER_API_KEY_ENV=<env-var-name>`` and it'll be treated as
    OpenAI-compatible.
    """
    env = env if env is not None else os.environ
    name = name or env.get("REVIEWER_PROVIDER", "github-models")
    name = name.strip().lower()

    spec = PROVIDER_REGISTRY.get(name)
    if spec is None:
        # Allow a fully env-driven custom provider without touching the registry.
        if name == "custom":
            base_url = env.get("REVIEWER_BASE_URL")
            if not base_url:
                raise RuntimeError(
                    "REVIEWER_PROVIDER=custom requires REVIEWER_BASE_URL to be set.")
            auth_env = env.get("REVIEWER_API_KEY_ENV")
            api_key = env.get(auth_env) if auth_env else None
            return OpenAICompatProvider(
                name="custom", base_url=base_url, api_key=api_key)
        raise ValueError(
            f"unknown provider {name!r}; known: {known_providers()} "
            f"(or set REVIEWER_PROVIDER=custom + REVIEWER_BASE_URL for a "
            f"new OpenAI-compatible endpoint).")

    if spec.kind == "github-models":
        return GitHubModelsProvider(
            endpoint=spec.base_url,
            token=env.get(spec.auth_env) if spec.auth_env else None,
        )
    if spec.kind == "openai-compat":
        api_key = env.get(spec.auth_env) if spec.auth_env else None
        if spec.auth_env and not api_key:
            raise RuntimeError(
                f"provider {name!r} requires {spec.auth_env} to be set.")
        return OpenAICompatProvider(
            name=spec.name, base_url=spec.base_url, api_key=api_key)
    if spec.kind == "anthropic":
        api_key = env.get(spec.auth_env) if spec.auth_env else None
        if spec.auth_env and not api_key:
            raise RuntimeError(
                f"provider {name!r} requires {spec.auth_env} to be set.")
        return AnthropicProvider(api_key=api_key)
    raise RuntimeError(f"unhandled provider kind {spec.kind!r} for {name!r}")
