from decimal import Decimal

import pytest

from agentkit._messages import Usage
from agentkit.providers.base import ProviderRequest
from agentkit.providers.fakes import FakeProvider


@pytest.mark.asyncio
async def test_fake_provider_streams_text():
    p = FakeProvider().script(FakeProvider.text("hello world"))
    events = [ev async for ev in p.stream(ProviderRequest(model="m"))]
    types = [ev.type for ev in events]
    assert types[0] == "message_start"
    assert "text_delta" in types
    assert types[-1] == "message_complete"


@pytest.mark.asyncio
async def test_fake_provider_streams_tool_call():
    p = FakeProvider().script(FakeProvider.tool_call("greet", {"name": "world"}))
    events = [ev async for ev in p.stream(ProviderRequest(model="m"))]
    tcc = [ev for ev in events if ev.type == "tool_call_complete"]
    assert tcc and tcc[0].tool_name == "greet"
    mc = [ev for ev in events if ev.type == "message_complete"]
    assert mc[0].finish_reason == "tool_use"


@pytest.mark.asyncio
async def test_fake_provider_consumes_responses_in_order():
    p = FakeProvider().script(
        FakeProvider.tool_call("a", {}),
        FakeProvider.text("done"),
    )
    first = [ev.type async for ev in p.stream(ProviderRequest(model="m"))]
    second = [ev.type async for ev in p.stream(ProviderRequest(model="m"))]
    assert "tool_call_complete" in first
    assert "text_delta" in second


def test_fake_provider_estimate_cost_is_zero():
    p = FakeProvider()
    assert p.estimate_cost(Usage(input_tokens=100, output_tokens=50)) == Decimal("0")
