"""Phase enum and transition table for the agent loop.

This is the formal state machine that replaces Pikkolo's nested control flow.
Every legal move is in TRANSITIONS; everything else raises.
"""

from enum import StrEnum

from agentkit.errors import InvalidPhaseTransition


class Phase(StrEnum):
    IDLE = "idle"
    INTENT_GATE = "intent_gate"
    CONTEXT_BUILD = "context_build"
    STREAMING = "streaming"
    TOOL_PHASE = "tool_phase"
    APPROVAL_WAIT = "approval_wait"
    TOOL_EXECUTING = "tool_executing"
    TOOL_RESULTS = "tool_results"
    FINALIZE_CHECK = "finalize_check"
    MEMORY_EXTRACT = "memory_extract"
    TURN_ENDED = "turn_ended"
    ERRORED = "errored"


TERMINAL: frozenset[Phase] = frozenset({Phase.TURN_ENDED, Phase.ERRORED})


def is_terminal(phase: Phase) -> bool:
    return phase in TERMINAL


# The transition table — single source of truth for the state machine.
# Every entry: phase -> set of phases reachable in one step.
TRANSITIONS: dict[Phase, frozenset[Phase]] = {
    Phase.IDLE: frozenset({Phase.INTENT_GATE}),
    Phase.INTENT_GATE: frozenset({Phase.CONTEXT_BUILD, Phase.ERRORED}),
    Phase.CONTEXT_BUILD: frozenset({Phase.STREAMING, Phase.ERRORED}),
    Phase.STREAMING: frozenset({Phase.TOOL_PHASE, Phase.FINALIZE_CHECK, Phase.ERRORED}),
    Phase.TOOL_PHASE: frozenset(
        {
            Phase.APPROVAL_WAIT,
            Phase.TOOL_EXECUTING,
            Phase.TOOL_RESULTS,  # all calls auto-denied
            Phase.ERRORED,
        }
    ),
    Phase.APPROVAL_WAIT: frozenset({Phase.TURN_ENDED}),  # suspend; resume creates new turn
    Phase.TOOL_EXECUTING: frozenset({Phase.TOOL_RESULTS, Phase.ERRORED}),
    Phase.TOOL_RESULTS: frozenset(
        {
            Phase.CONTEXT_BUILD,  # next provider iteration
            Phase.FINALIZE_CHECK,  # agent called finalize
            Phase.ERRORED,
        }
    ),
    Phase.FINALIZE_CHECK: frozenset(
        {
            Phase.MEMORY_EXTRACT,  # accepted
            Phase.CONTEXT_BUILD,  # rejected; retry
            Phase.ERRORED,
        }
    ),
    Phase.MEMORY_EXTRACT: frozenset({Phase.TURN_ENDED}),
}


def validate_transition(from_: Phase, to: Phase) -> None:
    """Raise InvalidPhaseTransition if `to` is not reachable from `from_`."""
    allowed = TRANSITIONS.get(from_)
    if allowed is None or to not in allowed:
        raise InvalidPhaseTransition(from_.value, to.value)
