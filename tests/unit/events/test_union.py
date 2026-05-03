"""Test discriminated union round-trip serialization for all event types."""

from datetime import UTC, datetime

from agentkit._ids import EventId, MessageId, SessionId, TurnId, new_id
from agentkit.events import (
    EVENT_ADAPTER,
    ApprovalDenied,
    ApprovalGranted,
    ApprovalNeeded,
    ErrorCode,
    Errored,
    MessageCompleted,
    MessageStarted,
    PhaseChanged,
    SubagentEnded,
    SubagentEvent,
    SubagentStarted,
    TextDelta,
    ThinkingDelta,
    ToolCallProgress,
    ToolCallResult,
    ToolCallStarted,
    TurnEnded,
    TurnEndReason,
    TurnMetrics,
    TurnStarted,
)
from agentkit.loop.phase import Phase


def _common(seq: int = 0) -> dict:
    return {
        "event_id": new_id(EventId),
        "session_id": new_id(SessionId),
        "turn_id": new_id(TurnId),
        "ts": datetime.now(UTC),
        "sequence": seq,
    }


def test_all_event_types_round_trip():
    """Test all event types serialize/deserialize through the discriminated union."""
    msg_id = new_id(MessageId)
    samples = [
        PhaseChanged(**_common(0), from_=Phase.IDLE, to=Phase.INTENT_GATE, duration_ms=1),
        TurnStarted(**_common(1), user_message_id=msg_id),
        MessageStarted(**_common(2), message_id=msg_id),
        TextDelta(**_common(3), message_id=msg_id, delta="hi"),
        ThinkingDelta(**_common(4), message_id=msg_id, delta="thinking..."),
        MessageCompleted(**_common(5), message_id=msg_id, finish_reason="end_turn"),
        ToolCallStarted(**_common(6), call_id="c1", tool_name="x", arguments={}, risk="read"),
        ToolCallProgress(**_common(7), call_id="c1", message="working"),
        ToolCallResult(
            **_common(8),
            call_id="c1",
            status="ok",
            content_summary="ok",
            duration_ms=10,
            cached=False,
        ),
        ApprovalNeeded(
            **_common(9),
            call_id="c1",
            tool_name="x",
            arguments={},
            risk="high_write",
            timeout_at=datetime.now(UTC),
        ),
        ApprovalGranted(**_common(10), call_id="c1"),
        ApprovalDenied(**_common(11), call_id="c1"),
        Errored(**_common(12), code=ErrorCode.RATE_LIMITED, message="x", recoverable=True),
        SubagentStarted(**_common(13), subagent_id="s1", parent_call_id="c1", purpose="research"),
        SubagentEvent(**_common(14), subagent_id="s1", inner={"type": "text_delta", "delta": "x"}),
        SubagentEnded(
            **_common(15), subagent_id="s1", reason=TurnEndReason.COMPLETED, summary="done"
        ),
        TurnEnded(**_common(16), reason=TurnEndReason.COMPLETED, metrics=TurnMetrics()),
    ]
    for ev in samples:
        dumped = ev.model_dump(mode="json")
        parsed = EVENT_ADAPTER.validate_python(dumped)
        assert parsed.type == ev.type
        assert parsed.sequence == ev.sequence
