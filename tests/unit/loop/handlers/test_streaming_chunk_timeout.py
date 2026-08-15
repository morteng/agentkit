"""A provider that goes quiet mid-stream must not park the turn forever.

``AgentConfig.loop.streaming_chunk_timeout_seconds`` used to be a published knob
that nothing read: ``async for event in mux.translate(...)`` waited without a
deadline, so a half-open socket held the consumer in a streaming state until
something outside the library gave up. These tests pin the deadline, the
recoverable classification, and the stream teardown.
"""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal

import pytest
import structlog

from agentkit._content import TextBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole, Usage
from agentkit.events import ErrorCode, Errored
from agentkit.guards.success_claim import ClaimVerdict
from agentkit.loop.context import TurnContext
from agentkit.loop.handlers.streaming import _chunk_timeout, handle_streaming
from agentkit.loop.message_builder import MessageBuilder
from agentkit.loop.phase import Phase
from agentkit.providers.base import (
    MessageComplete,
    MessageStart,
    ProviderCapabilities,
    ProviderEvent,
    ProviderRequest,
)
from agentkit.providers.base import TextDelta as ProviderTextDelta
from agentkit.tools.registry import ToolRegistry


class _ScriptedProvider:
    """Provider stub that records whether its stream was closed before exhaustion."""

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

    def __init__(self, *, stall: bool = False, text: str = "hello") -> None:
        self._stall = stall
        self._text = text
        self.closed = False
        self.exhausted = False

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderEvent]:
        try:
            yield MessageStart()
            for chunk in self._text.split():
                yield ProviderTextDelta(delta=chunk, block_index=0)
            if self._stall:
                await asyncio.sleep(3600)  # the half-open socket
            yield MessageComplete(finish_reason="end_turn")
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
        "streaming_chunk_timeout_seconds": 0.05,
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


def test_chunk_timeout_parsing():
    assert _chunk_timeout({}) == 60.0
    assert _chunk_timeout({"streaming_chunk_timeout_seconds": 5}) == 5.0
    assert _chunk_timeout({"streaming_chunk_timeout_seconds": 0}) is None
    assert _chunk_timeout({"streaming_chunk_timeout_seconds": None}) is None


@pytest.mark.asyncio
async def test_stalled_stream_errors_instead_of_hanging():
    provider = _ScriptedProvider(stall=True)
    ctx = TurnContext.empty()
    ctx.add_message(_user("hi"))
    queue: asyncio.Queue = asyncio.Queue()
    ctx.event_queue = queue

    with structlog.testing.capture_logs() as logs:
        next_ = await asyncio.wait_for(handle_streaming(ctx, _deps(provider)), timeout=5)

    assert next_ is Phase.ERRORED
    errs = [e for e in _drain(queue) if isinstance(e, Errored)]
    assert len(errs) == 1
    assert errs[0].code is ErrorCode.PROVIDER_FAULT
    assert errs[0].recoverable is True
    assert "stalled" in errs[0].message
    assert [e for e in logs if e["event"] == "stream_chunk_timeout"]
    # The partial text the consumer already saw is kept in history.
    assert any(m.role is MessageRole.ASSISTANT for m in ctx.history)


@pytest.mark.asyncio
async def test_stalled_stream_is_closed_not_abandoned():
    provider = _ScriptedProvider(stall=True)
    ctx = TurnContext.empty()
    ctx.add_message(_user("hi"))
    ctx.event_queue = asyncio.Queue()

    await asyncio.wait_for(handle_streaming(ctx, _deps(provider)), timeout=5)

    assert provider.closed is True
    assert provider.exhausted is False


@pytest.mark.asyncio
async def test_stall_before_any_output_is_retried():
    """No text emitted yet -> the stall is a clean blip the retry budget absorbs."""
    provider = _ScriptedProvider(stall=True, text="")
    ctx = TurnContext.empty()
    ctx.add_message(_user("hi"))
    ctx.event_queue = asyncio.Queue()

    next_ = await asyncio.wait_for(
        handle_streaming(ctx, _deps(provider, max_stream_retries=2)), timeout=5
    )

    assert next_ is Phase.CONTEXT_BUILD
    assert ctx.metadata["stream_retry_count"] == 1


@pytest.mark.asyncio
async def test_healthy_stream_is_unaffected_by_the_deadline():
    provider = _ScriptedProvider(text="all good here")
    ctx = TurnContext.empty()
    ctx.add_message(_user("hi"))
    ctx.event_queue = asyncio.Queue()

    next_ = await asyncio.wait_for(handle_streaming(ctx, _deps(provider)), timeout=5)

    assert next_ is Phase.FINALIZE_CHECK
    assert provider.exhausted is True
    assert provider.closed is True


@pytest.mark.asyncio
async def test_claim_trip_closes_the_provider_stream():
    """The success-claim early return must not leak the provider generator."""

    class _Flag:
        async def check(self, text_so_far: str, ctx: TurnContext) -> ClaimVerdict:
            return ClaimVerdict(flag=True, suggested_correction="call the tool first")

    provider = _ScriptedProvider(text="I have created it")
    ctx = TurnContext.empty()
    ctx.add_message(_user("hi"))
    ctx.event_queue = asyncio.Queue()

    next_ = await asyncio.wait_for(
        handle_streaming(ctx, _deps(provider, success_claim=_Flag(), max_claim_corrections=1)),
        timeout=5,
    )

    assert next_ is Phase.CONTEXT_BUILD
    assert provider.closed is True
    assert provider.exhausted is False
