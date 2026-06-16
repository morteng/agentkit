import asyncio
from datetime import UTC, datetime

import pytest

from agentkit._content import TextBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.events import Errored, MessageCompleted, TurnEnded, TurnEndReason
from agentkit.events import TextDelta as PubTextDelta
from agentkit.events import ToolCallStarted as PubToolCallStarted
from agentkit.loop.context import TurnContext
from agentkit.loop.handlers.streaming import handle_streaming
from agentkit.loop.message_builder import MessageBuilder
from agentkit.loop.orchestrator import Loop
from agentkit.loop.phase import Phase
from agentkit.providers.fakes import FakeProvider
from agentkit.tools.registry import ToolRegistry


def _drain(queue: asyncio.Queue) -> list:
    out = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


def _retry_deps(provider: FakeProvider, **overrides):
    """Streaming deps with the recoverable-stream retry knobs. base delay 0 so
    tests don't actually sleep."""
    deps = {
        "provider": provider,
        "message_builder": MessageBuilder(model="m", max_tokens=128),
        "registry": ToolRegistry(),
        "system_blocks": [],
        "success_claim": None,
        "max_stream_retries": 2,
        "stream_retry_base_delay_seconds": 0.0,
    }
    deps.update(overrides)
    return deps


def _user(text: str) -> Message:
    return Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=MessageRole.USER,
        content=[TextBlock(text=text)],
        created_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_streaming_with_text_response_returns_finalize_check():
    """Plain text response transitions FINALIZE_CHECK because the stream ended without a tool call.

    NB: in v0.1 we treat finish_reason == 'end_turn' WITH no kit.finalize call as
    "let finalize_check decide" — the validator may reject and force another
    iteration via CONTEXT_BUILD.
    """
    provider = FakeProvider().script(FakeProvider.text("hello"))
    queue: asyncio.Queue = asyncio.Queue()
    ctx = TurnContext.empty()
    ctx.add_message(_user("hi"))
    ctx.event_queue = queue

    builder = MessageBuilder(model="m", max_tokens=128)
    deps = {
        "provider": provider,
        "message_builder": builder,
        "registry": ToolRegistry(),
        "system_blocks": [],
        "success_claim": None,
    }
    next_ = await handle_streaming(ctx, deps)
    assert next_ is Phase.FINALIZE_CHECK

    # Drain emitted events.
    emitted = []
    while not queue.empty():
        emitted.append(queue.get_nowait())
    assert any(isinstance(e, PubTextDelta) for e in emitted)
    assert any(isinstance(e, MessageCompleted) for e in emitted)


@pytest.mark.asyncio
async def test_streaming_with_tool_call_returns_tool_phase():
    provider = FakeProvider().script(FakeProvider.tool_call("kit.x", {"a": 1}))
    queue: asyncio.Queue = asyncio.Queue()
    ctx = TurnContext.empty()
    ctx.add_message(_user("do x"))
    ctx.event_queue = queue

    deps = {
        "provider": provider,
        "message_builder": MessageBuilder(model="m", max_tokens=128),
        "registry": ToolRegistry(),
        "system_blocks": [],
        "success_claim": None,
    }
    next_ = await handle_streaming(ctx, deps)
    assert next_ is Phase.TOOL_PHASE

    emitted = []
    while not queue.empty():
        emitted.append(queue.get_nowait())
    starts = [e for e in emitted if isinstance(e, PubToolCallStarted)]
    assert starts and starts[0].tool_name == "kit.x"


@pytest.mark.asyncio
async def test_recoverable_error_with_budget_retries_via_context_build():
    """A recoverable error before any output is held (not surfaced) and the
    handler re-enters CONTEXT_BUILD for another attempt."""
    provider = FakeProvider().script(
        FakeProvider.error("rate_limited", "429 slow down", recoverable=True)
    )
    queue: asyncio.Queue = asyncio.Queue()
    ctx = TurnContext.empty()
    ctx.add_message(_user("hi"))
    ctx.event_queue = queue

    next_ = await handle_streaming(ctx, _retry_deps(provider))

    assert next_ is Phase.CONTEXT_BUILD
    assert ctx.metadata["stream_retry_count"] == 1
    # The error was held back, not forwarded to the consumer.
    assert not any(isinstance(e, Errored) for e in _drain(queue))


@pytest.mark.asyncio
async def test_recoverable_error_out_of_budget_surfaces_and_errors():
    """Once the per-attempt budget is spent, the held error is surfaced and the
    turn ends in ERRORED."""
    provider = FakeProvider().script(
        FakeProvider.error("rate_limited", "429 slow down", recoverable=True)
    )
    queue: asyncio.Queue = asyncio.Queue()
    ctx = TurnContext.empty()
    ctx.add_message(_user("hi"))
    ctx.event_queue = queue
    ctx.metadata["stream_retry_count"] = 2  # already at max_stream_retries

    next_ = await handle_streaming(ctx, _retry_deps(provider))

    assert next_ is Phase.ERRORED
    errs = [e for e in _drain(queue) if isinstance(e, Errored)]
    assert errs and errs[0].recoverable is True


@pytest.mark.asyncio
async def test_nonrecoverable_error_never_retries():
    """A non-recoverable error errors out immediately, regardless of budget, and
    leaves the retry counter untouched."""
    provider = FakeProvider().script(FakeProvider.error("auth_failed", "bad key"))
    queue: asyncio.Queue = asyncio.Queue()
    ctx = TurnContext.empty()
    ctx.add_message(_user("hi"))
    ctx.event_queue = queue

    next_ = await handle_streaming(ctx, _retry_deps(provider, max_stream_retries=5))

    assert next_ is Phase.ERRORED
    assert "stream_retry_count" not in ctx.metadata
    errs = [e for e in _drain(queue) if isinstance(e, Errored)]
    assert errs and errs[0].recoverable is False


@pytest.mark.asyncio
async def test_zero_budget_disables_retry_for_recoverable_error():
    """max_stream_retries=0 surfaces every error (opt-out)."""
    provider = FakeProvider().script(
        FakeProvider.error("timeout", "took too long", recoverable=True)
    )
    queue: asyncio.Queue = asyncio.Queue()
    ctx = TurnContext.empty()
    ctx.add_message(_user("hi"))
    ctx.event_queue = queue

    next_ = await handle_streaming(ctx, _retry_deps(provider, max_stream_retries=0))

    assert next_ is Phase.ERRORED
    assert any(isinstance(e, Errored) for e in _drain(queue))


@pytest.mark.asyncio
async def test_successful_stream_resets_retry_counter():
    """A clean stream clears the counter so the next iteration gets a fresh
    allowance against transient blips."""
    provider = FakeProvider().script(FakeProvider.text("hello"))
    queue: asyncio.Queue = asyncio.Queue()
    ctx = TurnContext.empty()
    ctx.add_message(_user("hi"))
    ctx.event_queue = queue
    ctx.metadata["stream_retry_count"] = 1  # a prior iteration retried once

    next_ = await handle_streaming(ctx, _retry_deps(provider))

    assert next_ is Phase.FINALIZE_CHECK
    assert ctx.metadata["stream_retry_count"] == 0


@pytest.mark.asyncio
async def test_end_to_end_recovers_from_recoverable_error_and_completes():
    """Full loop: a recoverable error on the first stream re-enters CONTEXT_BUILD,
    the retry succeeds, and the turn completes — the consumer never sees the blip.
    This is the bulk-resilience guarantee end to end."""
    provider = FakeProvider().script(
        FakeProvider.error("connection", "reset by peer", recoverable=True),
        FakeProvider.text("done"),
    )
    queue: asyncio.Queue = asyncio.Queue()
    ctx = TurnContext.empty()
    ctx.add_message(_user("do the bulk thing"))
    ctx.event_queue = queue

    async def _to_context_build(ctx, deps):
        return Phase.CONTEXT_BUILD

    async def _to_streaming(ctx, deps):
        return Phase.STREAMING

    async def _to_memory_extract(ctx, deps):
        return Phase.MEMORY_EXTRACT

    async def _to_turn_ended(ctx, deps):
        return Phase.TURN_ENDED

    handlers = {
        Phase.INTENT_GATE: _to_context_build,
        Phase.CONTEXT_BUILD: _to_streaming,
        Phase.STREAMING: handle_streaming,
        Phase.FINALIZE_CHECK: _to_memory_extract,
        Phase.MEMORY_EXTRACT: _to_turn_ended,
    }
    loop = Loop(
        ctx=ctx,
        handlers=handlers,
        deps=_retry_deps(provider),
        end_reason=TurnEndReason.COMPLETED,
    )

    events = [ev async for ev in loop.run()]

    assert isinstance(events[-1], TurnEnded)
    assert events[-1].reason is TurnEndReason.COMPLETED
    # Two streaming attempts: first errored recoverably, the retry succeeded.
    streaming_visits = [frm for (frm, _to, _ms) in ctx.phase_log if frm == Phase.STREAMING.value]
    assert len(streaming_visits) == 2
    # The consumer saw the recovered output and never the transient error.
    consumer_events = _drain(queue)
    assert not any(isinstance(e, Errored) for e in consumer_events)
    assert any(isinstance(e, PubTextDelta) for e in consumer_events)
