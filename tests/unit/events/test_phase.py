from datetime import UTC, datetime
from typing import cast

from agentkit._ids import EventId, SessionId, TurnId, new_id
from agentkit.events.phase import PhaseChanged
from agentkit.loop.phase import Phase


def test_phase_changed_carries_durations():
    ev = PhaseChanged(
        event_id=cast(EventId, new_id(EventId)),
        session_id=cast(SessionId, new_id(SessionId)),
        turn_id=cast(TurnId, new_id(TurnId)),
        ts=datetime.now(UTC),
        sequence=1,
        from_=Phase.IDLE,
        to=Phase.INTENT_GATE,
        duration_ms=12,
    )
    assert ev.type == "phase_changed"
