"""Tests for the centralized sequence counter and tool-progress emit API on
:class:`TurnContext` (added for F8 — tool progress events)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from agentkit.events import ToolCallProgress
from agentkit.loop.context import (
    FixedClock,
    TurnContext,
    from_checkpoint_payload,
    to_checkpoint_payload,
)


def test_next_sequence_is_monotonic_and_unique() -> None:
    ctx = TurnContext.empty(clock=FixedClock(datetime.now(UTC)))
    seqs = [ctx.next_sequence() for _ in range(5)]
    assert seqs == [0, 1, 2, 3, 4]


def test_next_sequence_starts_at_zero() -> None:
    ctx = TurnContext.empty(clock=FixedClock(datetime.now(UTC)))
    assert ctx.event_sequence == 0
    first = ctx.next_sequence()
    assert first == 0
    assert ctx.event_sequence == 1


@pytest.mark.asyncio
async def test_report_tool_progress_emits_event_when_call_id_set() -> None:
    queue: asyncio.Queue = asyncio.Queue()
    ctx = TurnContext.empty(clock=FixedClock(datetime.now(UTC)), call_id="call_42")
    ctx.event_queue = queue

    await ctx.report_tool_progress("starting query", progress=0.0, total=100.0)
    await ctx.report_tool_progress("got 50 rows", progress=50.0, total=100.0)
    await ctx.report_tool_progress("done")

    evts: list[ToolCallProgress] = []
    while not queue.empty():
        item = queue.get_nowait()
        assert isinstance(item, ToolCallProgress)
        evts.append(item)

    assert len(evts) == 3
    assert all(e.call_id == "call_42" for e in evts)
    assert [e.message for e in evts] == ["starting query", "got 50 rows", "done"]
    assert evts[0].progress == 0.0
    assert evts[0].total == 100.0
    assert evts[1].progress == 50.0
    assert evts[2].progress is None
    assert evts[2].total is None
    # Sequences are allocated from the centralized counter.
    assert [e.sequence for e in evts] == [0, 1, 2]


@pytest.mark.asyncio
async def test_report_tool_progress_overrides_call_id() -> None:
    queue: asyncio.Queue = asyncio.Queue()
    ctx = TurnContext.empty(clock=FixedClock(datetime.now(UTC)), call_id="default_call")
    ctx.event_queue = queue

    await ctx.report_tool_progress("explicit", call_id="custom_call")

    evt = queue.get_nowait()
    assert isinstance(evt, ToolCallProgress)
    assert evt.call_id == "custom_call"


@pytest.mark.asyncio
async def test_report_tool_progress_no_op_without_queue() -> None:
    ctx = TurnContext.empty(clock=FixedClock(datetime.now(UTC)), call_id="call_1")
    # event_queue is None by default.
    assert ctx.event_queue is None
    # Should silently no-op rather than raise.
    await ctx.report_tool_progress("won't be heard")
    # Sequence counter must NOT advance for a dropped emit, otherwise consumers
    # that later attach a queue would see non-contiguous numbering.
    assert ctx.event_sequence == 0


@pytest.mark.asyncio
async def test_report_tool_progress_no_op_without_call_id() -> None:
    queue: asyncio.Queue = asyncio.Queue()
    ctx = TurnContext.empty(clock=FixedClock(datetime.now(UTC)))
    # call_id defaults to "" via TurnContext.empty.
    ctx.event_queue = queue

    await ctx.report_tool_progress("orphan progress")
    assert queue.empty()
    assert ctx.event_sequence == 0


def test_event_sequence_round_trips_through_checkpoint() -> None:
    ctx = TurnContext.empty(clock=FixedClock(datetime.now(UTC)))
    for _ in range(7):
        ctx.next_sequence()
    assert ctx.event_sequence == 7

    payload = to_checkpoint_payload(ctx)
    data = from_checkpoint_payload(payload)
    assert int(data["event_sequence"]) == 7

    # The session resume path applies it to a new ctx; emulate that.
    restored = TurnContext.empty(clock=FixedClock(datetime.now(UTC)))
    restored.event_sequence = int(data["event_sequence"])
    assert restored.next_sequence() == 7
    assert restored.next_sequence() == 8
