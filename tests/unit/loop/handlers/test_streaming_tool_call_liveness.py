"""A tool call that takes longer to generate than the chunk timeout is not a stall.

The timeout wraps ``anext()`` on the *muxed* stream, so it measures the gap
between muxed events — not the gap between provider chunks. ``StreamMux`` used
to yield nothing at all for ``tool_call_start`` and nothing for each
``tool_call_delta``, so a model composing a long argument buffer (a batch of
calls, or one call carrying a long URL) produced a busy wire and a silent
iterator. The turn died at exactly the chunk timeout and reported it as the
provider going quiet.

The incident: a household assistant asked to queue a large batch of downloads
emitted its preamble sentence, began generating the calls, and was killed at
60s — twice in a row, identically, because at that batch size it is
deterministic rather than flaky.

``ProviderActivity`` restores the invariant the timeout was written against:
every provider chunk produces a muxed event. These tests pin both halves —
that the deltas keep the turn alive, and that no consumer ever sees the tick.
"""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from agentkit._content import TextBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole, Usage
from agentkit.events import Errored, ProviderActivity, ToolCallStarted
from agentkit.loop.context import TurnContext
from agentkit.loop.handlers.streaming import handle_streaming
from agentkit.loop.message_builder import MessageBuilder
from agentkit.loop.phase import Phase
from agentkit.providers.base import (
    MessageComplete,
    MessageStart,
    ProviderCapabilities,
    ProviderEvent,
    ProviderRequest,
    ToolCallComplete,
    ToolCallDelta,
    ToolCallStart,
)
from agentkit.providers.base import TextDelta as ProviderTextDelta
from agentkit.tools.registry import ToolRegistry

CHUNK_TIMEOUT = 0.05

#: Enough deltas that the total generation time exceeds CHUNK_TIMEOUT several
#: times over, while every individual gap stays comfortably under it. That is
#: precisely the shape the old code could not tell apart from a dead socket.
DELTA_COUNT = 12
DELTA_GAP = CHUNK_TIMEOUT / 3


class _ToolCallProvider:
    """Streams a preamble, then a tool call whose arguments arrive slowly."""

    name = "stub"
    capabilities = ProviderCapabilities(
        supports_tool_use=True,
        supports_parallel_tools=False,
        supports_prompt_caching=False,
        supports_vision=False,
        supports_thinking=False,
        max_context_tokens=1000,
        max_output_tokens=100,
    )

    def __init__(self, *, preamble: str = "On it, queuing those now.") -> None:
        self._preamble = preamble
        self.closed = False
        self.exhausted = False

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderEvent]:
        try:
            yield MessageStart()
            if self._preamble:
                yield ProviderTextDelta(delta=self._preamble, block_index=0)
            yield ToolCallStart(call_id="c1", tool_name="torrent_add")
            for i in range(DELTA_COUNT):
                await asyncio.sleep(DELTA_GAP)
                yield ToolCallDelta(call_id="c1", arguments_delta=f'"part{i}",')
            yield ToolCallComplete(
                call_id="c1",
                tool_name="torrent_add",
                arguments={"url_or_magnet": "magnet:?xt=urn:btih:abc", "category": "music"},
            )
            yield MessageComplete(finish_reason="tool_use")
            self.exhausted = True
        finally:
            self.closed = True

    def estimate_tokens(self, messages: list[Message]) -> int:
        return 0

    def estimate_cost(self, usage: Usage) -> Decimal:
        return Decimal("0")


def _user(text: str) -> Message:
    return Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=MessageRole.USER,
        content=[TextBlock(text=text)],
        created_at=datetime.now(UTC),
    )


def _deps(provider, **overrides):
    deps = {
        "provider": provider,
        "message_builder": MessageBuilder(model="m", max_tokens=128),
        "registry": ToolRegistry(),
        "system_blocks": [],
        "success_claim": None,
        "streaming_chunk_timeout_seconds": CHUNK_TIMEOUT,
        "max_stream_retries": 0,
        "stream_retry_base_delay_seconds": 0.0,
    }
    deps.update(overrides)
    return deps


def _drain(queue: asyncio.Queue) -> list:
    out = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


@pytest.mark.asyncio
async def test_a_slowly_generated_tool_call_does_not_read_as_a_stall():
    """The regression. Goes red without ProviderActivity: the deltas produce no
    muxed event, so ``anext`` waits past the deadline and the turn errors."""
    provider = _ToolCallProvider()
    ctx = TurnContext.empty()
    ctx.add_message(_user("get all you can"))
    queue: asyncio.Queue = asyncio.Queue()
    ctx.event_queue = queue

    next_ = await asyncio.wait_for(handle_streaming(ctx, _deps(provider)), timeout=10)

    events = _drain(queue)
    assert not [e for e in events if isinstance(e, Errored)], (
        "a busy provider was reported as a stalled one"
    )
    assert next_ is Phase.TOOL_PHASE
    assert provider.exhausted is True


@pytest.mark.asyncio
async def test_the_tool_call_still_reaches_the_consumer():
    """Control for the test above.

    Without this, 'no Errored event' would also pass in a build that silently
    dropped the tool call, which is a worse failure than the one being fixed:
    the turn would look healthy and do nothing.
    """
    provider = _ToolCallProvider()
    ctx = TurnContext.empty()
    ctx.add_message(_user("get all you can"))
    queue: asyncio.Queue = asyncio.Queue()
    ctx.event_queue = queue

    await asyncio.wait_for(handle_streaming(ctx, _deps(provider)), timeout=10)

    started = [e for e in _drain(queue) if isinstance(e, ToolCallStarted)]
    assert len(started) == 1
    assert started[0].tool_name == "torrent_add"
    assert started[0].arguments["category"] == "music"


@pytest.mark.asyncio
async def test_the_liveness_tick_never_reaches_a_consumer():
    """ProviderActivity is internal. It must not appear on the event queue.

    This is the property that lets the other agentkit consumer take this change
    without touching its dispatcher: it cannot receive a frame it does not know.
    """
    provider = _ToolCallProvider()
    ctx = TurnContext.empty()
    ctx.add_message(_user("get all you can"))
    queue: asyncio.Queue = asyncio.Queue()
    ctx.event_queue = queue

    await asyncio.wait_for(handle_streaming(ctx, _deps(provider)), timeout=10)

    events = _drain(queue)
    assert not [e for e in events if isinstance(e, ProviderActivity)]
    # Control: the queue is not simply empty — real events did flow through it,
    # so "no ProviderActivity" is a filter working and not a dead queue.
    assert events, "no events reached the consumer at all"
    assert any(isinstance(e, ToolCallStarted) for e in events)


@pytest.mark.asyncio
async def test_a_genuinely_dead_provider_still_stalls():
    """The timeout must still fire for real silence.

    Emitting a tick per chunk widens what counts as alive; if it also
    suppressed the deadline, the fix would have replaced a false stall with a
    hang, which is what the timeout exists to prevent.
    """

    class _DeadAfterToolStart(_ToolCallProvider):
        async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderEvent]:
            try:
                yield MessageStart()
                yield ToolCallStart(call_id="c1", tool_name="torrent_add")
                yield ToolCallDelta(call_id="c1", arguments_delta='{"url":')
                await asyncio.sleep(3600)  # the half-open socket
            finally:
                self.closed = True

    provider = _DeadAfterToolStart()
    ctx = TurnContext.empty()
    ctx.add_message(_user("get all you can"))
    queue: asyncio.Queue = asyncio.Queue()
    ctx.event_queue = queue

    next_ = await asyncio.wait_for(handle_streaming(ctx, _deps(provider)), timeout=10)

    assert next_ is Phase.ERRORED
    errs = [e for e in _drain(queue) if isinstance(e, Errored)]
    assert len(errs) == 1
    assert "stalled" in errs[0].message
    assert provider.closed is True
