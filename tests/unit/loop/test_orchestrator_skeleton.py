import pytest

from agentkit.events import PhaseChanged, TurnEnded, TurnEndReason
from agentkit.loop.context import TurnContext
from agentkit.loop.orchestrator import Loop, PhaseHandler
from agentkit.loop.phase import Phase


@pytest.mark.asyncio
async def test_loop_runs_through_to_terminal_phase():
    """Wire fake handlers that walk a known path:
    INTENT_GATE -> CONTEXT_BUILD -> STREAMING -> FINALIZE_CHECK -> MEMORY_EXTRACT -> TURN_ENDED.
    """
    visited: list[Phase] = []

    def make_handler(next_phase: Phase) -> PhaseHandler:
        async def h(ctx, deps):
            visited.append(deps["current_phase"])
            return next_phase

        return h

    handlers: dict[Phase, PhaseHandler] = {
        Phase.INTENT_GATE: make_handler(Phase.CONTEXT_BUILD),
        Phase.CONTEXT_BUILD: make_handler(Phase.STREAMING),
        Phase.STREAMING: make_handler(Phase.FINALIZE_CHECK),
        Phase.FINALIZE_CHECK: make_handler(Phase.MEMORY_EXTRACT),
        Phase.MEMORY_EXTRACT: make_handler(Phase.TURN_ENDED),
    }

    ctx = TurnContext.empty()
    loop = Loop(ctx=ctx, handlers=handlers, end_reason=TurnEndReason.COMPLETED)

    events = [ev async for ev in loop.run()]
    assert visited == [
        Phase.INTENT_GATE,
        Phase.CONTEXT_BUILD,
        Phase.STREAMING,
        Phase.FINALIZE_CHECK,
        Phase.MEMORY_EXTRACT,
    ]

    types = [type(e).__name__ for e in events]
    assert types[0] == "TurnStarted"
    assert types.count("PhaseChanged") >= 5
    assert isinstance(events[-1], TurnEnded)


@pytest.mark.asyncio
async def test_loop_emits_phase_changed_for_every_transition():
    handlers = {
        Phase.INTENT_GATE: _terminal_handler(Phase.ERRORED),
    }
    ctx = TurnContext.empty()
    loop = Loop(ctx=ctx, handlers=handlers, end_reason=TurnEndReason.ERROR)
    events = [ev async for ev in loop.run()]
    pc = [e for e in events if isinstance(e, PhaseChanged)]
    assert len(pc) >= 1
    assert pc[0].from_ == Phase.INTENT_GATE


def _terminal_handler(target):
    async def h(ctx, deps):
        return target

    return h


def _walk_through(visited_summary: str | None) -> dict[Phase, PhaseHandler]:
    """Build handlers that walk a valid INTENT_GATE → ... → TURN_ENDED path,
    setting ``ctx.finalize_reason`` to ``visited_summary`` along the way (or
    leaving it None when the argument is None)."""

    def make_handler(next_phase: Phase, *, set_finalize: bool = False) -> PhaseHandler:
        async def h(ctx, deps):
            if set_finalize and visited_summary is not None:
                ctx.finalize_called = True
                ctx.finalize_reason = visited_summary
            return next_phase

        return h

    return {
        Phase.INTENT_GATE: make_handler(Phase.CONTEXT_BUILD),
        Phase.CONTEXT_BUILD: make_handler(Phase.STREAMING),
        Phase.STREAMING: make_handler(Phase.FINALIZE_CHECK, set_finalize=True),
        Phase.FINALIZE_CHECK: make_handler(Phase.MEMORY_EXTRACT),
        Phase.MEMORY_EXTRACT: make_handler(Phase.TURN_ENDED),
    }


@pytest.mark.asyncio
async def test_loop_surfaces_finalize_reason_as_turn_ended_summary():
    """F6: kit.finalize's freeform `reason` argument is dead weight unless it
    flows through to TurnEnded. This test proves the orchestrator does the
    wiring on a valid INTENT_GATE → … → TURN_ENDED path."""
    summary = "Reversed the string and wrote it to disk."
    ctx = TurnContext.empty()
    loop = Loop(ctx=ctx, handlers=_walk_through(summary), end_reason=TurnEndReason.COMPLETED)

    events = [ev async for ev in loop.run()]
    ended = events[-1]
    assert isinstance(ended, TurnEnded)
    assert ended.reason is TurnEndReason.COMPLETED
    assert ended.summary == summary


@pytest.mark.asyncio
async def test_loop_turn_ended_summary_is_none_when_finalize_not_called():
    """No kit.finalize means no summary — explicit absence, not stale state."""
    ctx = TurnContext.empty()
    loop = Loop(ctx=ctx, handlers=_walk_through(None), end_reason=TurnEndReason.COMPLETED)

    events = [ev async for ev in loop.run()]
    ended = events[-1]
    assert isinstance(ended, TurnEnded)
    assert ended.reason is TurnEndReason.COMPLETED
    assert ended.summary is None


@pytest.mark.asyncio
async def test_loop_emits_max_iterations_when_metadata_signals_it():
    """The iteration-cap signal flows through the orchestrator: when
    ``ctx.metadata["suspend_reason"]`` is set to MAX_ITERATIONS, the final
    ``TurnEnded`` event reflects that, not the default COMPLETED. Mirrors
    how the AWAITING_APPROVAL suspend path already works.
    """

    async def streaming_handler(ctx, deps):
        # Simulate: tool_results decided we hit max-iter, stamped the reason,
        # and routed to FINALIZE_CHECK. We compress that into the legal path
        # the phase machine allows from STREAMING.
        ctx.metadata["suspend_reason"] = TurnEndReason.MAX_ITERATIONS.value
        return Phase.FINALIZE_CHECK

    async def passthrough(ctx, deps):
        return {
            Phase.INTENT_GATE: Phase.CONTEXT_BUILD,
            Phase.CONTEXT_BUILD: Phase.STREAMING,
            Phase.FINALIZE_CHECK: Phase.MEMORY_EXTRACT,
            Phase.MEMORY_EXTRACT: Phase.TURN_ENDED,
        }[deps["current_phase"]]

    handlers: dict[Phase, PhaseHandler] = {
        Phase.INTENT_GATE: passthrough,
        Phase.CONTEXT_BUILD: passthrough,
        Phase.STREAMING: streaming_handler,
        Phase.FINALIZE_CHECK: passthrough,
        Phase.MEMORY_EXTRACT: passthrough,
    }

    ctx = TurnContext.empty()
    loop = Loop(ctx=ctx, handlers=handlers)
    events = [ev async for ev in loop.run()]
    ended = events[-1]
    assert isinstance(ended, TurnEnded)
    assert ended.reason is TurnEndReason.MAX_ITERATIONS
