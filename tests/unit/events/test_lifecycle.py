from datetime import UTC, datetime

from agentkit._ids import EventId, SessionId, TurnId, new_id
from agentkit.events.lifecycle import (
    ErrorCode,
    Errored,
    TurnEnded,
    TurnEndReason,
    TurnMetrics,
)


def test_turn_ended_carries_metrics():
    ev = TurnEnded(
        event_id=new_id(EventId),
        session_id=new_id(SessionId),
        turn_id=new_id(TurnId),
        ts=datetime.now(UTC),
        sequence=99,
        reason=TurnEndReason.COMPLETED,
        metrics=TurnMetrics(input_tokens=100, output_tokens=50, duration_ms=2300, tool_calls=0),
    )
    assert ev.metrics.input_tokens == 100
    # summary defaults to None — the model didn't call kit.finalize.
    assert ev.summary is None


def test_turn_ended_carries_summary_from_finalize():
    ev = TurnEnded(
        event_id=new_id(EventId),
        session_id=new_id(SessionId),
        turn_id=new_id(TurnId),
        ts=datetime.now(UTC),
        sequence=99,
        reason=TurnEndReason.COMPLETED,
        metrics=TurnMetrics(),
        summary="Reversed the input string and wrote it to /tmp/out.txt.",
    )
    assert ev.summary == "Reversed the input string and wrote it to /tmp/out.txt."


def test_errored_event_marks_recoverability():
    ev = Errored(
        event_id=new_id(EventId),
        session_id=new_id(SessionId),
        turn_id=new_id(TurnId),
        ts=datetime.now(UTC),
        sequence=5,
        code=ErrorCode.RATE_LIMITED,
        message="too many turns",
        recoverable=True,
    )
    assert ev.recoverable
