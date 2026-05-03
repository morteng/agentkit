from datetime import UTC, datetime
from typing import cast

from agentkit._ids import EventId, SessionId, TurnId, new_id
from agentkit.events.base import BaseEvent


def test_base_event_construction():
    ev = BaseEvent(
        type="test",
        event_id=cast(EventId, new_id(EventId)),
        session_id=cast(SessionId, new_id(SessionId)),
        turn_id=cast(TurnId, new_id(TurnId)),
        ts=datetime.now(UTC),
        sequence=0,
    )
    assert ev.sequence == 0
