"""Subagent event surfacing into the parent stream (slice 3 of F8).

Exercises the ``_surface_subagent_events`` helper directly with synthetic
event streams so we can validate the debounce and tool-call-translation
behavior without standing up a full nested Loop.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from agentkit._ids import EventId, MessageId, SessionId, TurnId, new_id
from agentkit.events import (
    ApprovalNeeded,
    PhaseChanged,
    TextDelta,
    ToolCallProgress,
    ToolCallStarted,
)
from agentkit.loop.context import FixedClock, TurnContext
from agentkit.loop.phase import Phase
from agentkit.subagents.dispatcher import (
    _child_event_stream,
    _pump_loop_into_queue,
    _surface_subagent_events,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from agentkit.events.base import BaseEvent


def _parent_with_queue() -> tuple[TurnContext, asyncio.Queue[Any]]:
    queue: asyncio.Queue[Any] = asyncio.Queue()
    ctx = TurnContext.empty(clock=FixedClock(datetime.now(UTC)), call_id="parent-call")
    ctx.event_queue = queue
    return ctx, queue


def _text_delta(delta: str) -> TextDelta:
    return TextDelta(
        event_id=new_id(EventId),
        session_id=SessionId("session"),
        turn_id=TurnId("turn"),
        ts=datetime.now(UTC),
        sequence=0,
        message_id=new_id(MessageId),
        delta=delta,
        block_index=0,
    )


def _tool_started(tool_name: str) -> ToolCallStarted:
    return ToolCallStarted(
        event_id=new_id(EventId),
        session_id=SessionId("session"),
        turn_id=TurnId("turn"),
        ts=datetime.now(UTC),
        sequence=0,
        call_id="child-call",
        tool_name=tool_name,
        arguments={},
        risk="read",
    )


async def _stream(events: list[BaseEvent]) -> AsyncIterator[BaseEvent]:
    for ev in events:
        yield ev


async def _drain_progress(queue: asyncio.Queue[Any]) -> list[ToolCallProgress]:
    out: list[ToolCallProgress] = []
    while not queue.empty():
        item = queue.get_nowait()
        assert isinstance(item, ToolCallProgress)
        out.append(item)
    return out


@pytest.mark.asyncio
async def test_text_below_threshold_flushes_at_end() -> None:
    parent, queue = _parent_with_queue()
    await _surface_subagent_events(parent, _stream([_text_delta("hi"), _text_delta(" there")]))

    progress = await _drain_progress(queue)
    assert len(progress) == 1
    assert progress[0].message == "subagent: hi there"


@pytest.mark.asyncio
async def test_newline_triggers_flush() -> None:
    parent, queue = _parent_with_queue()
    await _surface_subagent_events(
        parent,
        _stream(
            [
                _text_delta("first sentence.\n"),
                _text_delta("second part"),
            ]
        ),
    )

    progress = await _drain_progress(queue)
    # Newline flushes the first chunk; tail flush handles "second part".
    assert [p.message for p in progress] == [
        "subagent: first sentence.",
        "subagent: second part",
    ]


@pytest.mark.asyncio
async def test_long_run_flushes_at_threshold() -> None:
    parent, queue = _parent_with_queue()
    long_chunk = "a" * 100  # exceeds 80-char threshold in one delta
    await _surface_subagent_events(parent, _stream([_text_delta(long_chunk)]))

    progress = await _drain_progress(queue)
    # One flush from the threshold; nothing left over for the tail flush.
    assert len(progress) == 1
    assert progress[0].message == f"subagent: {long_chunk}"


@pytest.mark.asyncio
async def test_tool_call_started_surfaces_with_pending_text_first() -> None:
    parent, queue = _parent_with_queue()
    await _surface_subagent_events(
        parent,
        _stream(
            [
                _text_delta("about to call "),
                _tool_started("kit.current_time"),
                _text_delta("done."),
            ]
        ),
    )

    progress = await _drain_progress(queue)
    # Text-before-tool should flush before the tool-call announcement so the
    # parent UI reads in causal order.
    assert [p.message for p in progress] == [
        "subagent: about to call",
        "subagent calling kit.current_time",
        "subagent: done.",
    ]


@pytest.mark.asyncio
async def test_other_event_types_are_ignored() -> None:
    """Approval/phase events are internal to the child; only text + tool
    starts should reach the parent."""
    parent, queue = _parent_with_queue()
    phase_event = PhaseChanged(
        event_id=new_id(EventId),
        session_id=SessionId("s"),
        turn_id=TurnId("t"),
        ts=datetime.now(UTC),
        sequence=0,
        from_=Phase.INTENT_GATE,
        to=Phase.CONTEXT_BUILD,
        duration_ms=0,
    )
    approval_event = ApprovalNeeded(
        event_id=new_id(EventId),
        session_id=SessionId("s"),
        turn_id=TurnId("t"),
        ts=datetime.now(UTC),
        sequence=0,
        call_id="c",
        tool_name="kit.x",
        arguments={},
        risk="read",
        timeout_at=datetime.now(UTC),
    )
    await _surface_subagent_events(
        parent,
        _stream([phase_event, approval_event, _text_delta("hello")]),
    )

    progress = await _drain_progress(queue)
    assert len(progress) == 1
    assert progress[0].message == "subagent: hello"


@pytest.mark.asyncio
async def test_progress_events_attach_to_parent_call_id() -> None:
    parent, queue = _parent_with_queue()
    parent.call_id = "parent-call-xyz"

    await _surface_subagent_events(parent, _stream([_text_delta("ok")]))

    progress = await _drain_progress(queue)
    assert progress[0].call_id == "parent-call-xyz"


@pytest.mark.asyncio
async def test_no_op_when_only_whitespace_buffered() -> None:
    """Whitespace-only output shouldn't generate noisy parent events."""
    parent, queue = _parent_with_queue()
    await _surface_subagent_events(parent, _stream([_text_delta("   \n  ")]))

    progress = await _drain_progress(queue)
    assert progress == []


# ---- the merged child stream -------------------------------------------------
#
# ``Loop.run()`` yields only lifecycle events (TurnStarted / PhaseChanged /
# TurnEnded); TextDelta and ToolCallStarted are pushed onto ``ctx.event_queue``
# by the streaming handler. Filtering ``loop.run()`` alone — which the
# dispatcher used to do — could therefore never surface anything at all. The
# dispatcher now pumps the loop's events onto the *same* queue the handlers
# write to, exactly as ``AgentSession._drain_loop_into_queue`` does.


class _LifecycleOnlyLoop:
    """Stands in for ``Loop``: emits lifecycle events, and pushes handler-style
    events onto the shared queue the way a real phase handler would."""

    def __init__(self, queue: asyncio.Queue[Any], handler_events: list[BaseEvent]) -> None:
        self._queue = queue
        self._handler_events = handler_events

    async def run(self) -> AsyncIterator[BaseEvent]:
        yield _phase_changed()
        for ev in self._handler_events:
            await self._queue.put(ev)
        yield _phase_changed()


def _phase_changed() -> PhaseChanged:
    return PhaseChanged(
        event_id=new_id(EventId),
        session_id=SessionId("s"),
        turn_id=TurnId("t"),
        ts=datetime.now(UTC),
        sequence=0,
        from_=Phase.STREAMING,
        to=Phase.FINALIZE_CHECK,
        duration_ms=0,
    )


@pytest.mark.asyncio
async def test_merged_stream_carries_handler_events_not_just_lifecycle() -> None:
    parent, parent_queue = _parent_with_queue()
    child_queue: asyncio.Queue[Any] = asyncio.Queue()
    loop = _LifecycleOnlyLoop(child_queue, [_text_delta("hello from the child")])

    pump = asyncio.create_task(_pump_loop_into_queue(loop, child_queue))  # type: ignore[arg-type]
    await _surface_subagent_events(parent, _child_event_stream(child_queue))
    await pump

    progress = await _drain_progress(parent_queue)
    assert [p.message for p in progress] == ["subagent: hello from the child"]


@pytest.mark.asyncio
async def test_merged_stream_terminates_when_the_loop_raises() -> None:
    """A loop that dies before TurnEnded must not hang the consumer forever."""
    parent, _parent_queue = _parent_with_queue()
    child_queue: asyncio.Queue[Any] = asyncio.Queue()

    class _ExplodingLoop:
        async def run(self) -> AsyncIterator[BaseEvent]:
            yield _phase_changed()
            raise RuntimeError("child loop died")

    pump = asyncio.create_task(_pump_loop_into_queue(_ExplodingLoop(), child_queue))  # type: ignore[arg-type]
    await asyncio.wait_for(
        _surface_subagent_events(parent, _child_event_stream(child_queue)), timeout=2.0
    )
    with pytest.raises(RuntimeError, match="child loop died"):
        await pump
