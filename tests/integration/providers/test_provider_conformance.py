from datetime import UTC, datetime

import pytest
import vcr

from agentkit._content import TextBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.providers.base import (
    MessageComplete,
    MessageStart,
    ProviderRequest,
    SystemBlock,
    TextDelta,
    ToolCallComplete,
    ToolDefinition,
    UsageEvent,
)

pytestmark = [pytest.mark.integration]


_VCR = vcr.VCR(
    cassette_library_dir="tests/integration/providers/cassettes",
    record_mode="once",  # type: ignore[arg-type]
    filter_headers=["authorization", "x-api-key", "openai-organization"],
    match_on=["method", "scheme", "host", "port", "path", "query", "body"],
)


def _user(text: str) -> Message:
    return Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=MessageRole.USER,
        content=[TextBlock(text=text)],
        created_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_text_stream_yields_message_start_then_deltas_then_complete(provider, model, request):
    cassette_name = f"{request.node.callspec.params['provider']}_text.yaml"
    with _VCR.use_cassette(cassette_name):  # type: ignore[attr-defined]
        req = ProviderRequest(
            model=model,
            system=[SystemBlock(text="Reply with exactly: hi")],
            messages=[_user("say hi")],
            max_tokens=128,
        )
        events = [ev async for ev in provider.stream(req)]

    assert isinstance(events[0], MessageStart)
    assert any(isinstance(e, TextDelta) for e in events)
    assert isinstance(events[-1], MessageComplete)


@pytest.mark.asyncio
async def test_usage_event_has_nonzero_tokens(provider, model, request):
    cassette_name = f"{request.node.callspec.params['provider']}_usage.yaml"
    with _VCR.use_cassette(cassette_name):  # type: ignore[attr-defined]
        req = ProviderRequest(
            model=model,
            system=[SystemBlock(text="reply with one word")],
            messages=[_user("hi")],
            max_tokens=64,
        )
        events = [ev async for ev in provider.stream(req)]

    usage_events = [e for e in events if isinstance(e, UsageEvent)]
    assert usage_events
    u = usage_events[-1].usage
    assert u.input_tokens > 0
    assert u.output_tokens > 0


@pytest.mark.asyncio
async def test_tool_call_round_trips_arguments(provider, model, request):
    cassette_name = f"{request.node.callspec.params['provider']}_tool.yaml"
    with _VCR.use_cassette(cassette_name):  # type: ignore[attr-defined]
        req = ProviderRequest(
            model=model,
            system=[SystemBlock(text="Use the greet tool with name='world'.")],
            messages=[_user("greet world")],
            tools=[
                ToolDefinition(
                    name="greet",
                    description="Greet someone by name.",
                    parameters={
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                )
            ],
            max_tokens=256,
        )
        events = [ev async for ev in provider.stream(req)]

    completes = [e for e in events if isinstance(e, ToolCallComplete)]
    assert completes
    assert completes[0].tool_name == "greet"
    assert completes[0].arguments == {"name": "world"}
