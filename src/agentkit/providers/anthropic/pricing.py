"""Per-model token pricing for cost estimation.

Numbers are USD per 1M tokens. Update when Anthropic publishes new prices.
"""

from dataclasses import dataclass
from decimal import Decimal

from agentkit._messages import Usage


@dataclass(frozen=True)
class ModelPricing:
    input: Decimal  # per 1M input tokens
    output: Decimal  # per 1M output tokens
    cache_read: Decimal  # per 1M cached input tokens
    cache_write: Decimal  # per 1M cache-creation tokens


MODEL_PRICING: dict[str, ModelPricing] = {
    # Update these from https://www.anthropic.com/pricing when versions roll.
    "claude-opus-4-7": ModelPricing(
        input=Decimal("15.00"),
        output=Decimal("75.00"),
        cache_read=Decimal("1.50"),
        cache_write=Decimal("18.75"),
    ),
    "claude-sonnet-4-6": ModelPricing(
        input=Decimal("3.00"),
        output=Decimal("15.00"),
        cache_read=Decimal("0.30"),
        cache_write=Decimal("3.75"),
    ),
    "claude-haiku-4-5-20251001": ModelPricing(
        input=Decimal("0.80"),
        output=Decimal("4.00"),
        cache_read=Decimal("0.08"),
        cache_write=Decimal("1.00"),
    ),
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
        + Decimal(usage.cache_creation_tokens) * pricing.cache_write / million
    )
