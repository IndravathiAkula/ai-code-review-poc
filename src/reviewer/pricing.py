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


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    p = PRICES.get(model) or PRICES.get(model.lower())
    if p is None:
        return None
    return (prompt_tokens / 1_000_000) * p.input_per_1m + \
           (completion_tokens / 1_000_000) * p.output_per_1m
