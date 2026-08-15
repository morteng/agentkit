"""A turn that fails must say so — log, metadata, and an event on the queue.

The pre-0.22 orchestrator swallowed every handler exception with a bare
``except Exception``: no log, no Errored event, no stored message. A turn simply
stopped. These tests pin the three artefacts that make a failure auditable.
"""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

import pytest
import structlog

from agentkit._content import TextBlock, ToolUseBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole, Usage
from agentkit.events import ErrorCode, Errored, PhaseChanged, TurnEnded, TurnEndReason
from agentkit.loop.context import TurnContext
from agentkit.loop.orchestrator import Loop, PhaseHandler
from agentkit.loop.phase import Phase


def _drain(queue: asyncio.Queue) -> list:
    out = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


async def _boom(ctx, deps):
    raise RuntimeError("handler exploded")


def _walk() -> dict[Phase, PhaseHandler]:
    async def passthrough(ctx, deps):
        return {
            Phase.INTENT_GATE: Phase.CONTEXT_BUILD,
            Phase.CONTEXT_BUILD: Phase.STREAMING,
            Phase.STREAMING: Phase.FINALIZE_CHECK,
            Phase.FINALIZE_CHECK: Phase.MEMORY_EXTRACT,
            Phase.MEMORY_EXTRACT: Phase.TURN_ENDED,
        }[deps["current_phase"]]

    return dict.fromkeys(
        (
            Phase.INTENT_GATE,
            Phase.CONTEXT_BUILD,
            Phase.STREAMING,
            Phase.FINALIZE_CHECK,
            Phase.MEMORY_EXTRACT,
        ),
        passthrough,
    )


@pytest.mark.asyncio
async def test_handler_exception_is_logged_with_context():
    ctx = TurnContext.empty()
    ctx.event_queue = asyncio.Queue()
    loop = Loop(ctx=ctx, handlers={Phase.INTENT_GATE: _boom})

    with structlog.testing.capture_logs() as logs:
        [ev async for ev in loop.run()]

    entries = [e for e in logs if e["event"] == "loop_handler_exception"]
    assert len(entries) == 1, f"expected one exception log, got {[e['event'] for e in logs]}"
    assert entries[0]["log_level"] == "error"
    assert entries[0]["phase"] == Phase.INTENT_GATE.value
    assert entries[0]["error_type"] == "RuntimeError"
    assert entries[0]["session_id"] == str(ctx.session_id)
    assert entries[0]["turn_id"] == str(ctx.turn_id)


@pytest.mark.asyncio
async def test_handler_exception_emits_errored_event_and_stores_the_error():
    ctx = TurnContext.empty()
    queue: asyncio.Queue = asyncio.Queue()
    ctx.event_queue = queue
    loop = Loop(ctx=ctx, handlers={Phase.INTENT_GATE: _boom})

    events = [ev async for ev in loop.run()]

    errored = [e for e in _drain(queue) if isinstance(e, Errored)]
    assert len(errored) == 1
    assert errored[0].code is ErrorCode.INTERNAL
    assert errored[0].recoverable is False
    assert "handler exploded" in errored[0].message

    stored = ctx.metadata["turn_error"]
    assert stored["phase"] == Phase.INTENT_GATE.value
    assert stored["type"] == "RuntimeError"
    assert stored["message"] == "handler exploded"

    assert isinstance(events[-1], TurnEnded)
    assert events[-1].reason is TurnEndReason.ERROR


@pytest.mark.asyncio
async def test_handler_exception_without_queue_still_records_and_ends():
    """A context with no event queue (subagent-internal) must not crash the run."""
    ctx = TurnContext.empty()
    loop = Loop(ctx=ctx, handlers={Phase.INTENT_GATE: _boom})

    events = [ev async for ev in loop.run()]

    assert ctx.metadata["turn_error"]["type"] == "RuntimeError"
    assert isinstance(events[-1], TurnEnded)


@pytest.mark.asyncio
async def test_missing_handler_is_reported():
    ctx = TurnContext.empty()
    ctx.event_queue = asyncio.Queue()
    loop = Loop(ctx=ctx, handlers={})

    with structlog.testing.capture_logs() as logs:
        [ev async for ev in loop.run()]

    assert [e for e in logs if e["event"] == "loop_turn_error"]
    assert ctx.metadata["turn_error"]["type"] == "NoHandler"
    assert [e for e in _drain(ctx.event_queue) if isinstance(e, Errored)]


@pytest.mark.asyncio
async def test_illegal_transition_is_reported():
    async def jump(ctx, deps):
        return Phase.TOOL_EXECUTING  # not reachable from INTENT_GATE

    ctx = TurnContext.empty()
    ctx.event_queue = asyncio.Queue()
    loop = Loop(ctx=ctx, handlers={Phase.INTENT_GATE: jump})

    with structlog.testing.capture_logs() as logs:
        [ev async for ev in loop.run()]

    assert [e for e in logs if e["event"] == "loop_turn_error"]
    assert ctx.metadata["turn_error"]["type"] == "InvalidPhaseTransition"
    assert [e for e in _drain(ctx.event_queue) if isinstance(e, Errored)]


@pytest.mark.asyncio
async def test_publish_phase_changed_false_keeps_phase_events_off_the_stream():
    """events.publish_phase_changed=False suppresses the event, not the log."""
    ctx = TurnContext.empty()
    loop = Loop(ctx=ctx, handlers=_walk(), publish_phase_changed=False)

    events = [ev async for ev in loop.run()]

    assert not any(isinstance(e, PhaseChanged) for e in events)
    assert isinstance(events[-1], TurnEnded)
    # The observability record survives — only the wire traffic is dropped.
    assert [to for (_frm, to, _ms) in ctx.phase_log] == [
        Phase.CONTEXT_BUILD.value,
        Phase.STREAMING.value,
        Phase.FINALIZE_CHECK.value,
        Phase.MEMORY_EXTRACT.value,
        Phase.TURN_ENDED.value,
    ]


@pytest.mark.asyncio
async def test_publish_phase_changed_defaults_to_emitting():
    ctx = TurnContext.empty()
    loop = Loop(ctx=ctx, handlers=_walk())
    events = [ev async for ev in loop.run()]
    assert len([e for e in events if isinstance(e, PhaseChanged)]) == 5


class _PricedProvider:
    """Minimal provider stub: only ``estimate_cost`` is exercised here."""

    name = "priced"

    def estimate_cost(self, usage: Usage) -> Decimal:
        return Decimal("0.001") * usage.output_tokens


@pytest.mark.asyncio
async def test_turn_ended_metrics_are_populated():
    """TurnEnded.metrics used to be an empty TurnMetrics(), which made
    Provider.estimate_cost a knob nothing ever turned."""

    async def stream_handler(ctx, deps):
        ctx.metadata.setdefault("usages", []).append(
            Usage(input_tokens=100, output_tokens=20, cached_input_tokens=7, thinking_tokens=3)
        )
        ctx.add_message(
            Message(
                id=new_id(MessageId),
                session_id=ctx.session_id,
                role=MessageRole.ASSISTANT,
                content=[
                    TextBlock(text="working"),
                    ToolUseBlock(id="c1", name="srv.do", arguments={}),
                ],
                created_at=datetime.now(UTC),
            )
        )
        return Phase.FINALIZE_CHECK

    handlers = _walk()
    handlers[Phase.STREAMING] = stream_handler

    ctx = TurnContext.empty()
    ctx.add_message(
        Message(
            id=new_id(MessageId),
            session_id=new_id(SessionId),
            role=MessageRole.USER,
            content=[TextBlock(text="hi")],
            created_at=datetime.now(UTC),
        )
    )
    loop = Loop(ctx=ctx, handlers=handlers, deps={"provider": _PricedProvider()})

    events = [ev async for ev in loop.run()]
    ended = events[-1]
    assert isinstance(ended, TurnEnded)
    m = ended.metrics
    assert m.input_tokens == 100
    assert m.output_tokens == 20
    assert m.cached_input_tokens == 7
    assert m.thinking_tokens == 3
    assert m.cost_usd == Decimal("0.020")
    assert m.tool_calls == 1
    assert m.iterations == 1
    assert m.duration_ms >= 0


@pytest.mark.asyncio
async def test_turn_metrics_survive_a_provider_that_cannot_price():
    class _Broken:
        name = "broken"

        def estimate_cost(self, usage):
            raise ValueError("no pricing table")

    async def stream_handler(ctx, deps):
        ctx.metadata.setdefault("usages", []).append(Usage(input_tokens=5, output_tokens=5))
        return Phase.FINALIZE_CHECK

    handlers = _walk()
    handlers[Phase.STREAMING] = stream_handler
    ctx = TurnContext.empty()
    loop = Loop(ctx=ctx, handlers=handlers, deps={"provider": _Broken()})

    events = [ev async for ev in loop.run()]
    ended = events[-1]
    assert isinstance(ended, TurnEnded)
    assert ended.reason is TurnEndReason.COMPLETED
    assert ended.metrics.cost_usd == Decimal("0")
    assert ended.metrics.input_tokens == 5
