import contextlib

import pytest
from hypothesis import given
from hypothesis import strategies as st

from agentkit.errors import InvalidPhaseTransition
from agentkit.loop.phase import TRANSITIONS, Phase, is_terminal, validate_transition


def test_phase_enum_has_all_eleven():
    assert {p.value for p in Phase} == {
        "idle",
        "intent_gate",
        "context_build",
        "streaming",
        "tool_phase",
        "approval_wait",
        "tool_executing",
        "tool_results",
        "finalize_check",
        "memory_extract",
        "turn_ended",
        "errored",
    }


def test_terminal_phases():
    assert is_terminal(Phase.TURN_ENDED)
    assert is_terminal(Phase.ERRORED)
    assert not is_terminal(Phase.IDLE)
    assert not is_terminal(Phase.STREAMING)


def test_transition_table_covers_every_non_terminal_phase():
    for p in Phase:
        if not is_terminal(p):
            assert p in TRANSITIONS, f"{p} has no outgoing transitions"
            assert TRANSITIONS[p], f"{p} has empty transition set"


def test_validate_transition_accepts_legal_move():
    validate_transition(Phase.INTENT_GATE, Phase.CONTEXT_BUILD)  # no raise


def test_validate_transition_rejects_illegal_move():
    with pytest.raises(InvalidPhaseTransition):
        validate_transition(Phase.IDLE, Phase.TOOL_EXECUTING)


def test_intent_gate_can_transition_to_errored():
    assert Phase.ERRORED in TRANSITIONS[Phase.INTENT_GATE]


def test_streaming_can_transition_to_tool_phase_or_finalize():
    assert Phase.TOOL_PHASE in TRANSITIONS[Phase.STREAMING]
    assert Phase.FINALIZE_CHECK in TRANSITIONS[Phase.STREAMING]


def test_approval_wait_terminates_to_turn_ended():
    # Suspend semantics: approval-wait yields TurnEnded(reason=AWAITING_APPROVAL).
    # The phase transition is APPROVAL_WAIT -> TURN_ENDED for the suspend; resume
    # creates a new turn that re-enters at TOOL_EXECUTING via load-from-checkpoint.
    assert Phase.TURN_ENDED in TRANSITIONS[Phase.APPROVAL_WAIT]


@given(sequence=st.lists(st.sampled_from(list(Phase)), min_size=1, max_size=20))
def test_random_sequences_only_raise_invalid_phase_transition(sequence):
    """No matter what sequence we throw at validate_transition, we only see
    InvalidPhaseTransition or success — never an unexpected exception."""
    for i in range(len(sequence) - 1):
        with contextlib.suppress(InvalidPhaseTransition):
            validate_transition(sequence[i], sequence[i + 1])
