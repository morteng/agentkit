"""F3: OpenRouterProvider exposes per-model capabilities."""

from agentkit.providers.base import ProviderCapabilities
from agentkit.providers.openrouter.adapter import OpenRouterProvider


def test_capabilities_for_known_deepseek_v4_flash_is_1m_context():
    """DeepSeek V4 Flash actually has a 1M context — not the 128k conservative default."""
    p = OpenRouterProvider(api_key="x")
    caps = p.capabilities_for("deepseek/deepseek-v4-flash")
    assert caps.max_context_tokens == 1_048_576


def test_capabilities_for_unknown_model_falls_back_to_default():
    p = OpenRouterProvider(api_key="x")
    caps = p.capabilities_for("vendor/never-seen-model")
    assert caps == p.capabilities  # the class-level fallback


def test_caller_can_register_custom_model_capabilities():
    custom = ProviderCapabilities(
        supports_tool_use=True,
        supports_parallel_tools=False,
        supports_prompt_caching=False,
        supports_vision=False,
        supports_thinking=False,
        max_context_tokens=4_000_000,
        max_output_tokens=16_384,
    )
    p = OpenRouterProvider(api_key="x", model_capabilities={"vendor/exotic": custom})
    assert p.capabilities_for("vendor/exotic") == custom
    # Built-in entries still resolve.
    assert p.capabilities_for("deepseek/deepseek-v4-flash").max_context_tokens == 1_048_576


def test_caller_overrides_can_replace_built_in_entry():
    over = ProviderCapabilities(
        supports_tool_use=True,
        supports_parallel_tools=True,
        supports_prompt_caching=True,
        supports_vision=False,
        supports_thinking=False,
        max_context_tokens=999,
        max_output_tokens=42,
    )
    p = OpenRouterProvider(api_key="x", model_capabilities={"deepseek/deepseek-v4-flash": over})
    assert p.capabilities_for("deepseek/deepseek-v4-flash") == over
