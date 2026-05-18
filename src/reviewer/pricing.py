"""Approximate per-model prices, USD per 1M tokens (input / output).

These are *list prices* for the underlying providers as of early 2026 and are
used only to estimate $/PR in the eval harness. GitHub Models itself is free
within its rate limits; the numbers here reflect what the same model would
cost on the provider's paid API (OpenAI, Mistral La Plateforme, etc.) so the
leaderboard gives a realistic signal for scale-out.

Override any row at runtime by editing this dict or by setting the
REVIEWER_PRICES env var to a JSON object of the same shape, e.g.:

    REVIEWER_PRICES='{"openai/gpt-4o-mini": {"input": 0.15, "output": 0.60}}'

Unknown models return None from cost_usd() and show as "-" in the leaderboard.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input_per_1m: float   # USD per 1,000,000 prompt tokens
    output_per_1m: float  # USD per 1,000,000 completion tokens


PRICES: dict[str, Price] = {
    # Treated as GitHub Models pricing when no provider is specified.
    "openai/gpt-4o-mini":             Price(0.15, 0.60),
    "openai/gpt-4o":                  Price(2.50, 10.00),
    "openai/o3-mini":                 Price(1.10, 4.40),
    "openai/o4-mini":                 Price(1.10, 4.40),
    "meta/llama-3.3-70b-instruct":    Price(0.60, 0.80),
    "meta/llama-3.1-405b-instruct":   Price(2.70, 2.70),
    "mistral-ai/mistral-large-2411":  Price(2.00, 6.00),
    "mistral-ai/mistral-small":       Price(0.20, 0.60),
    "microsoft/phi-4":                Price(0.30, 0.50),
    "microsoft/phi-3.5-mini-instruct": Price(0.15, 0.30),
    "cohere/cohere-command-r-plus":   Price(2.50, 10.00),
}


# Per-provider pricing — wins when the caller passes ``provider=...`` to
# ``cost_usd``. Same Price units (USD per 1M tokens, input / output).
PROVIDER_PRICES: dict[str, dict[str, Price]] = {
    "anthropic": {
        # Cache write costs 1.25x input; cache read costs 0.10x input.
        # Those multipliers are applied in cost_usd().
        "claude-sonnet-4-5":            Price(3.00, 15.00),
        "claude-sonnet-4-6":            Price(3.00, 15.00),
        "claude-haiku-4-5":             Price(0.80, 4.00),
        "claude-opus-4-7":              Price(15.00, 75.00),
        "claude-3-5-sonnet-20241022":   Price(3.00, 15.00),
        "claude-3-5-haiku-20241022":    Price(0.80, 4.00),
    },
    "groq": {
        "llama-3.3-70b-versatile":      Price(0.59, 0.79),
        "llama-3.1-8b-instant":         Price(0.05, 0.08),
        "deepseek-r1-distill-llama-70b": Price(0.75, 0.99),
        "mixtral-8x7b-32768":           Price(0.24, 0.24),
    },
    "openrouter": {
        # Native list prices; OpenRouter adds a small markup at billing time.
        "anthropic/claude-3.5-sonnet":  Price(3.00, 15.00),
        "anthropic/claude-3.5-haiku":   Price(0.80, 4.00),
        "google/gemini-2.5-flash":      Price(0.15, 0.60),
        "meta-llama/llama-3.3-70b-instruct": Price(0.60, 0.80),
    },
    "openai": {
        "gpt-4o":                       Price(2.50, 10.00),
        "gpt-4o-mini":                  Price(0.15, 0.60),
        "o3-mini":                      Price(1.10, 4.40),
    },
    "nvidia": {
        # NIM has free credits; list prices shown for parity at scale.
        "meta/llama-3.3-70b-instruct":  Price(0.60, 0.80),
        "nvidia/llama-3.3-nemotron-super-49b-v1": Price(0.90, 1.20),
    },
    "cerebras": {
        "llama3.3-70b":                 Price(0.85, 1.20),
    },
    "ollama": {
        # Local inference — energy cost only.
    },
}


def _apply_overrides() -> None:
    raw = os.environ.get("REVIEWER_PRICES")
    if not raw:
        return
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return
    for model, row in data.items():
        try:
            PRICES[model] = Price(float(row["input"]), float(row["output"]))
        except (KeyError, TypeError, ValueError):
            continue


_apply_overrides()


# Cache pricing multipliers (relative to input_per_1m). Anthropic charges
# 1.25x to write the cache and 0.10x to read from it.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


def cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    provider: str | None = None,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float | None:
    """Estimated USD cost for a single call.

    Resolution order for prices:
      1. ``PROVIDER_PRICES[provider][model]`` (when provider is given)
      2. ``PRICES[model]`` (case-sensitive)
      3. ``PRICES[model.lower()]``
      4. ``None`` (model not in any pricing table)

    When ``cache_read_tokens`` or ``cache_creation_tokens`` are non-zero
    (Anthropic prompt caching), they're billed at 0.10x and 1.25x of the
    input rate respectively — significantly cheaper on repeat prompts.
    """
    p = None
    if provider:
        p = PROVIDER_PRICES.get(provider, {}).get(model)
        if p is None:
            p = PROVIDER_PRICES.get(provider, {}).get(model.lower())
    if p is None:
        p = PRICES.get(model) or PRICES.get(model.lower())
    if p is None:
        return None
    cost = (
        (prompt_tokens / 1_000_000) * p.input_per_1m
        + (completion_tokens / 1_000_000) * p.output_per_1m
        + (cache_creation_tokens / 1_000_000) * p.input_per_1m * CACHE_WRITE_MULTIPLIER
        + (cache_read_tokens / 1_000_000) * p.input_per_1m * CACHE_READ_MULTIPLIER
    )
    return cost
