from datetime import UTC, datetime
from typing import cast

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
        event_id=cast(EventId, new_id(EventId)),
        session_id=cast(SessionId, new_id(SessionId)),
        turn_id=cast(TurnId, new_id(TurnId)),
        ts=datetime.now(UTC),
        sequence=99,
        reason=TurnEndReason.COMPLETED,
        metrics=TurnMetrics(input_tokens=100, output_tokens=50, duration_ms=2300, tool_calls=0),
    )
    assert ev.metrics.input_tokens == 100


def test_errored_event_marks_recoverability():
    ev = Errored(
        event_id=cast(EventId, new_id(EventId)),
        session_id=cast(SessionId, new_id(SessionId)),
        turn_id=cast(TurnId, new_id(TurnId)),
        ts=datetime.now(UTC),
        sequence=5,
        code=ErrorCode.RATE_LIMITED,
        message="too many turns",
        recoverable=True,
    )
    assert ev.recoverable
