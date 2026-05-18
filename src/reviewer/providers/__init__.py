"""Provider abstraction — multi-backend support for chat-completions LLMs.

Public API:
    Provider, ChatResponse, Usage,
    ProviderTransientError, ProviderPermanentError,
    build_provider, known_providers,
    GitHubModelsProvider, OpenAICompatProvider
"""
from .anthropic import AnthropicProvider
from .base import (
    ChatResponse, Provider, ProviderError,
    ProviderPermanentError, ProviderTransientError, Usage,
)
from .factory import build_provider, known_providers, PROVIDER_REGISTRY
from .github_models import GitHubModelsProvider
from .openai_compat import OpenAICompatProvider

__all__ = [
    "Provider", "ChatResponse", "Usage",
    "ProviderError", "ProviderTransientError", "ProviderPermanentError",
    "build_provider", "known_providers", "PROVIDER_REGISTRY",
    "AnthropicProvider", "GitHubModelsProvider", "OpenAICompatProvider",
]
