import os

import pytest

from agentkit.providers.anthropic import AnthropicProvider
from agentkit.providers.openrouter import OpenRouterProvider


@pytest.fixture(
    params=["anthropic", "openrouter"],
    ids=["anthropic", "openrouter"],
)
def provider(request: pytest.FixtureRequest) -> AnthropicProvider | OpenRouterProvider:
    """Yields a Provider instance for each registered provider.

    Tests are parameterised so adding a provider runs the same suite.
    """
    if request.param == "anthropic":
        if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("VCR_REPLAY")):
            pytest.skip("ANTHROPIC_API_KEY not set; set VCR_REPLAY=1 to use cassettes")
        return AnthropicProvider(api_key=os.getenv("ANTHROPIC_API_KEY", "test"))
    if request.param == "openrouter":
        if not (os.getenv("OPENROUTER_API_KEY") or os.getenv("VCR_REPLAY")):
            pytest.skip("OPENROUTER_API_KEY not set; set VCR_REPLAY=1 to use cassettes")
        return OpenRouterProvider(api_key=os.getenv("OPENROUTER_API_KEY", "test"))
    raise RuntimeError(f"unknown provider: {request.param}")


@pytest.fixture
def model(request: pytest.FixtureRequest) -> str:
    return {
        "anthropic": "claude-haiku-4-5-20251001",
        "openrouter": "openai/gpt-4o-mini",
    }[request.node.callspec.params["provider"]]
