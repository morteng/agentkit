from datetime import UTC, datetime

import pytest

from agentkit._messages import Usage
from agentkit.events import (
    TextDelta as PubTextDelta,
)
from agentkit.events import (
    ToolCallStarted as PubToolCallStarted,
)
from agentkit.loop.context import FixedClock, TurnContext
from agentkit.loop.stream_mux import StreamMux
from agentkit.providers.base import (
    MessageComplete,
    MessageStart,
    TextDelta,
    ToolCallComplete,
    ToolCallStart,
    UsageEvent,
)


@pytest.mark.asyncio
async def test_stream_mux_translates_text_delta():
    ctx = TurnContext.empty(clock=FixedClock(datetime.now(UTC)))
    mux = StreamMux(ctx, sequence_start=10)

    async def src():
        yield MessageStart()
        yield TextDelta(delta="he")
        yield TextDelta(delta="llo")
        yield UsageEvent(usage=Usage(input_tokens=5, output_tokens=2))
        yield MessageComplete(finish_reason="end_turn")

    out = []
    async for ev in mux.translate(src()):
        out.append(ev)

    types = [type(e).__name__ for e in out]
    assert types[0] == "MessageStarted"
    assert types[-1] == "MessageCompleted"
    deltas = [e.delta for e in out if isinstance(e, PubTextDelta)]
    assert deltas == ["he", "llo"]


@pytest.mark.asyncio
async def test_stream_mux_translates_tool_call():
    ctx = TurnContext.empty(clock=FixedClock(datetime.now(UTC)))
    mux = StreamMux(ctx, sequence_start=0)

    async def src():
        yield MessageStart()
        yield ToolCallStart(call_id="call_1", tool_name="x")
        yield ToolCallComplete(call_id="call_1", tool_name="x", arguments={"a": 1})
        yield MessageComplete(finish_reason="tool_use")

    started = [e async for e in mux.translate(src()) if isinstance(e, PubToolCallStarted)]
    assert started and started[0].tool_name == "x"
    assert started[0].arguments == {"a": 1}  # arguments are populated when complete arrives
