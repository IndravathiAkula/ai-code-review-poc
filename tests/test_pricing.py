"""Tests for the pricing table and cache-aware cost calculation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewer.pricing import (
    CACHE_READ_MULTIPLIER, CACHE_WRITE_MULTIPLIER, cost_usd,
)


def test_cost_usd_basic_input_output():
    # gpt-4o-mini: $0.15/1M input, $0.60/1M output
    c = cost_usd("openai/gpt-4o-mini", prompt_tokens=1_000_000,
                 completion_tokens=1_000_000)
    assert abs(c - (0.15 + 0.60)) < 1e-9


def test_cost_usd_unknown_model_returns_none():
    assert cost_usd("not-a-real/model", 100, 50) is None


def test_cost_usd_provider_prices_win_over_flat_table():
    """When provider=groq is passed and the model is in PROVIDER_PRICES,
    those prices win over the flat top-level table."""
    # llama-3.3-70b-versatile is in groq's table at $0.59/$0.79
    c = cost_usd("llama-3.3-70b-versatile", 1_000_000, 1_000_000,
                 provider="groq")
    assert abs(c - (0.59 + 0.79)) < 1e-9


def test_cost_usd_anthropic_with_cache_creation():
    """Cache creation tokens cost 1.25x the input rate."""
    # claude-sonnet-4-6 input is $3/1M
    c = cost_usd("claude-sonnet-4-6", prompt_tokens=0, completion_tokens=0,
                 provider="anthropic", cache_creation_tokens=1_000_000)
    expected = 3.00 * CACHE_WRITE_MULTIPLIER  # 3.75
    assert abs(c - expected) < 1e-9


def test_cost_usd_anthropic_with_cache_read():
    """Cache read tokens cost 0.10x the input rate — the big savings."""
    c = cost_usd("claude-sonnet-4-6", prompt_tokens=0, completion_tokens=0,
                 provider="anthropic", cache_read_tokens=1_000_000)
    expected = 3.00 * CACHE_READ_MULTIPLIER  # 0.30
    assert abs(c - expected) < 1e-9


def test_cost_usd_anthropic_realistic_pr_scenario():
    """Simulate a 5-chunk PR with system-prompt caching:
    Chunk 1: writes the 800-token system cache + 200 user + 100 output.
    Chunks 2-5: read 800 cached + 200 user + 100 output each.
    """
    sonnet_in = 3.00 / 1_000_000   # $/token input
    sonnet_out = 15.00 / 1_000_000

    # Chunk 1
    c1 = cost_usd("claude-sonnet-4-6",
                   prompt_tokens=200, completion_tokens=100,
                   provider="anthropic",
                   cache_creation_tokens=800)
    expected_c1 = (200 * sonnet_in
                   + 100 * sonnet_out
                   + 800 * sonnet_in * CACHE_WRITE_MULTIPLIER)
    assert abs(c1 - expected_c1) < 1e-9

    # Chunks 2-5 (cache hits)
    c_next = cost_usd("claude-sonnet-4-6",
                       prompt_tokens=200, completion_tokens=100,
                       provider="anthropic",
                       cache_read_tokens=800)
    expected_next = (200 * sonnet_in
                     + 100 * sonnet_out
                     + 800 * sonnet_in * CACHE_READ_MULTIPLIER)
    assert abs(c_next - expected_next) < 1e-9

    # Sanity check: cache hits are much cheaper than cache writes.
    assert c_next < c1


def test_cost_usd_cache_read_is_cheaper_than_uncached_call():
    """A cache-hit chunk should cost less than the same call without
    caching — that's the whole point of prompt caching."""
    uncached = cost_usd("claude-sonnet-4-6",
                         prompt_tokens=1000,  # 800 system + 200 user
                         completion_tokens=100,
                         provider="anthropic")
    cached = cost_usd("claude-sonnet-4-6",
                       prompt_tokens=200,
                       completion_tokens=100,
                       provider="anthropic",
                       cache_read_tokens=800)
    # Single-call savings are diluted by output cost; aggregate savings
    # across many chunks are what matter. Still, a single cache hit
    # should shave a meaningful chunk off.
    assert cached < uncached * 0.6


def test_cost_usd_cache_savings_grow_with_more_chunks():
    """Across N chunks of a PR, the per-chunk amortized cost converges
    toward (200 user + 100 output + 800 cache_read) — much cheaper than
    (1000 input + 100 output) every time."""
    # Cache-write cost on chunk 1, plus N-1 cache-hit chunks.
    n_chunks = 10
    sonnet_in = 3.00 / 1_000_000
    sonnet_out = 15.00 / 1_000_000

    cached_total = cost_usd(
        "claude-sonnet-4-6", prompt_tokens=200, completion_tokens=100,
        provider="anthropic", cache_creation_tokens=800,
    )
    for _ in range(n_chunks - 1):
        cached_total += cost_usd(
            "claude-sonnet-4-6", prompt_tokens=200, completion_tokens=100,
            provider="anthropic", cache_read_tokens=800,
        )

    uncached_total = n_chunks * cost_usd(
        "claude-sonnet-4-6", prompt_tokens=1000, completion_tokens=100,
        provider="anthropic",
    )

    # Cached aggregate should be materially cheaper at 10 chunks.
    assert cached_total < uncached_total * 0.7


def test_cost_usd_falls_back_to_flat_table_when_provider_missing_model():
    """If provider table doesn't have the model, fall back to flat PRICES."""
    # openai/gpt-4o-mini is in the flat table. The 'groq' provider table
    # doesn't have it. Asking for it under provider=groq should fall back.
    c = cost_usd("openai/gpt-4o-mini", 1_000_000, 0, provider="groq")
    assert c == 0.15
