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
