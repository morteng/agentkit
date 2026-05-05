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
    # Sequence numbers come from ctx.next_sequence(); pre-advance to verify the
    # mux honors the centralized counter rather than starting at zero.
    for _ in range(10):
        ctx.next_sequence()
    mux = StreamMux(ctx)

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
    # First emitted event should have sequence=10 (the next free slot).
    assert out[0].sequence == 10
    # Sequence numbers are strictly increasing.
    seqs = [e.sequence for e in out]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)


@pytest.mark.asyncio
async def test_stream_mux_translates_tool_call():
    ctx = TurnContext.empty(clock=FixedClock(datetime.now(UTC)))
    mux = StreamMux(ctx)

    async def src():
        yield MessageStart()
        yield ToolCallStart(call_id="call_1", tool_name="x")
        yield ToolCallComplete(call_id="call_1", tool_name="x", arguments={"a": 1})
        yield MessageComplete(finish_reason="tool_use")

    started = [e async for e in mux.translate(src()) if isinstance(e, PubToolCallStarted)]
    assert started and started[0].tool_name == "x"
    assert started[0].arguments == {"a": 1}  # arguments are populated when complete arrives
