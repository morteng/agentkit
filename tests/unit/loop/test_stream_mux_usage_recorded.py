"""StreamMux yields UsageRecorded events alongside the internal
ctx.metadata['usages'] append. The internal capture stays for any
consumer reading ctx.metadata directly (backward compatibility)."""

from datetime import UTC, datetime

import pytest

from agentkit._messages import Usage
from agentkit.events import MessageCompleted, UsageRecorded
from agentkit.loop.context import FixedClock, TurnContext
from agentkit.loop.stream_mux import StreamMux
from agentkit.providers.base import (
    MessageComplete,
    MessageStart,
    UsageEvent,
)
from agentkit.tools.registry import ToolRegistry


async def _provider_stream():
    yield MessageStart()
    yield UsageEvent(
        usage=Usage(input_tokens=10, output_tokens=20),
        model="openai/gpt-5",
        provider_name="openrouter",
    )
    yield MessageComplete(finish_reason="end_turn")


@pytest.fixture
def ctx():
    return TurnContext.empty(clock=FixedClock(datetime.now(UTC)))


@pytest.mark.asyncio
async def test_mux_yields_usage_recorded(ctx):
    """The mux yields UsageRecorded with all the upstream fields preserved."""
    mux = StreamMux(ctx, registry=ToolRegistry())
    events = [e async for e in mux.translate(_provider_stream())]
    usage_records = [e for e in events if isinstance(e, UsageRecorded)]
    completes = [e for e in events if isinstance(e, MessageCompleted)]
    assert len(usage_records) == 1, f"expected 1 UsageRecorded, got {len(usage_records)}"
    assert usage_records[0].model == "openai/gpt-5"
    assert usage_records[0].usage.input_tokens == 10
    assert usage_records[0].provider_name == "openrouter"
    assert len(completes) == 1


@pytest.mark.asyncio
async def test_mux_still_appends_to_ctx_metadata_usages(ctx):
    """Backward compatibility: existing consumers reading ctx.metadata['usages']
    must keep working."""
    mux = StreamMux(ctx, registry=ToolRegistry())
    _ = [e async for e in mux.translate(_provider_stream())]
    assert ctx.metadata.get("usages")
    assert ctx.metadata["usages"][0].input_tokens == 10


@pytest.mark.asyncio
async def test_mux_emits_usage_before_message_completed(ctx):
    """Ordering: consumers using MessageCompleted as a per-call flush boundary
    must see UsageRecorded before MessageCompleted."""
    mux = StreamMux(ctx, registry=ToolRegistry())
    events = [e async for e in mux.translate(_provider_stream())]
    usage_idx = next(i for i, e in enumerate(events) if isinstance(e, UsageRecorded))
    complete_idx = next(i for i, e in enumerate(events) if isinstance(e, MessageCompleted))
    assert usage_idx < complete_idx
