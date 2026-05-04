from agentkit.providers.base import (
    ProviderCapabilities,
    ProviderEvent,
    ProviderRequest,
    SystemBlock,
)


def test_capabilities_construct_with_explicit_flags():
    caps = ProviderCapabilities(
        supports_tool_use=True,
        supports_parallel_tools=True,
        supports_prompt_caching=True,
        supports_vision=True,
        supports_thinking=True,
        max_context_tokens=200_000,
        max_output_tokens=8192,
    )
    assert caps.max_context_tokens == 200_000


def test_provider_request_has_required_fields():
    req = ProviderRequest(
        model="claude-sonnet-4-6",
        system=[SystemBlock(text="You are helpful.")],
        messages=[],
        tools=[],
        max_tokens=4096,
    )
    assert req.model == "claude-sonnet-4-6"
    assert req.temperature is None
    assert req.thinking is None


def test_provider_event_discriminator_round_trips():
    from pydantic import TypeAdapter

    ev = {"type": "text_delta", "delta": "hello"}
    adapter = TypeAdapter(ProviderEvent)
    parsed = adapter.validate_python(ev)
    assert parsed.type == "text_delta"
