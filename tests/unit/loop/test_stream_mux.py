from datetime import UTC, datetime

import pytest
import structlog

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
    ToolCallDelta,
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
        yield UsageEvent(
            usage=Usage(input_tokens=5, output_tokens=2), model="fake/test", provider_name="fake"
        )
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


@pytest.mark.asyncio
async def test_start_without_complete_logs_residue():
    """A provider that starts a tool call and never completes it produces NO
    user-facing event at all — starts and deltas are parked here by design. This
    is the provider-agnostic seam where that symptom becomes greppable, whatever
    the parser upstream did or failed to do."""
    ctx = TurnContext.empty(clock=FixedClock(datetime.now(UTC)))
    mux = StreamMux(ctx)

    async def src():
        yield MessageStart()
        yield ToolCallStart(call_id="call_1", tool_name="rm_file")
        yield ToolCallDelta(call_id="call_1", arguments_delta='{"path":')
        yield MessageComplete(finish_reason="max_tokens")

    with structlog.testing.capture_logs() as logs:
        out = [e async for e in mux.translate(src())]

    assert not [e for e in out if isinstance(e, PubToolCallStarted)], (
        "an incomplete call must not reach the consumer"
    )
    residue = [entry for entry in logs if entry["event"] == "tool_call_start_without_complete"]
    assert len(residue) == 1, f"expected one residue log, got {[e['event'] for e in logs]}"
    assert residue[0]["log_level"] == "warning"
    assert residue[0]["count"] == 1
    assert residue[0]["tools"] == ["rm_file"]
    assert residue[0]["session_id"] == str(ctx.session_id)
    assert residue[0]["turn_id"] == str(ctx.turn_id)


@pytest.mark.asyncio
async def test_completed_tool_call_leaves_no_residue():
    """The residue log runs on every stream for every provider — it must stay
    silent when every start was matched, or it is noise nobody will grep for."""
    ctx = TurnContext.empty(clock=FixedClock(datetime.now(UTC)))
    mux = StreamMux(ctx)

    async def src():
        yield MessageStart()
        yield ToolCallStart(call_id="call_1", tool_name="x")
        yield ToolCallComplete(call_id="call_1", tool_name="x", arguments={"a": 1})
        yield MessageComplete(finish_reason="tool_use")

    with structlog.testing.capture_logs() as logs:
        _ = [e async for e in mux.translate(src())]

    assert [entry["event"] for entry in logs] == []
