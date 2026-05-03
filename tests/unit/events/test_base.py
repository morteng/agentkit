from datetime import UTC, datetime

from agentkit._ids import EventId, SessionId, TurnId, new_id
from agentkit.events.base import BaseEvent


def test_base_event_construction():
    ev = BaseEvent(
        type="test",
        event_id=new_id(EventId),
        session_id=new_id(SessionId),
        turn_id=new_id(TurnId),
        ts=datetime.now(UTC),
        sequence=0,
    )
    assert ev.sequence == 0
