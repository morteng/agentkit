from decimal import Decimal

from agentkit._messages import Usage
from agentkit.providers.anthropic.pricing import MODEL_PRICING, estimate_cost_usd


def test_known_model_costs_input_and_output():
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    cost = estimate_cost_usd("claude-sonnet-4-6", usage)
    expected = MODEL_PRICING["claude-sonnet-4-6"].input + MODEL_PRICING["claude-sonnet-4-6"].output
    assert cost == expected


def test_cached_input_priced_at_cache_read_rate():
    usage = Usage(cached_input_tokens=1_000_000)
    cost = estimate_cost_usd("claude-sonnet-4-6", usage)
    assert cost == MODEL_PRICING["claude-sonnet-4-6"].cache_read


def test_unknown_model_returns_zero():
    assert estimate_cost_usd("totally-fictional-model", Usage(input_tokens=1)) == Decimal("0")
