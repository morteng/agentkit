"""Pricing table for OpenRouter models. Conservative defaults when unknown.

Cost numbers are USD per 1M tokens. Update from https://openrouter.ai/models.
"""

from dataclasses import dataclass
from decimal import Decimal

from agentkit._messages import Usage


@dataclass(frozen=True)
class ModelPricing:
    input: Decimal
    output: Decimal
    cache_read: Decimal = Decimal("0")


# Curated subset; extend as needed.
MODEL_PRICING: dict[str, ModelPricing] = {
    "anthropic/claude-opus-4-7": ModelPricing(Decimal("15.00"), Decimal("75.00"), Decimal("1.50")),
    "anthropic/claude-sonnet-4-6": ModelPricing(Decimal("3.00"), Decimal("15.00"), Decimal("0.30")),
    "openai/gpt-5": ModelPricing(Decimal("5.00"), Decimal("15.00")),
    "openai/gpt-4o-mini": ModelPricing(Decimal("0.15"), Decimal("0.60")),
    "google/gemini-2.5-pro": ModelPricing(Decimal("1.25"), Decimal("5.00"), Decimal("0.31")),
    "deepseek/deepseek-chat": ModelPricing(Decimal("0.14"), Decimal("0.28")),
}


def estimate_cost_usd(model: str, usage: Usage) -> Decimal:
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return Decimal("0")
    million = Decimal("1000000")
    return (
        Decimal(usage.input_tokens) * pricing.input / million
        + Decimal(usage.output_tokens) * pricing.output / million
        + Decimal(usage.cached_input_tokens) * pricing.cache_read / million
    )
