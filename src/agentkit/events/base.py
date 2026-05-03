"""Base event type. All agentkit events inherit from this."""

from datetime import datetime

from pydantic import BaseModel

from agentkit._ids import EventId, SessionId, TurnId


class BaseEvent(BaseModel):
    type: str
    event_id: EventId
    session_id: SessionId
    turn_id: TurnId
    ts: datetime
    sequence: int  # monotonic per turn

    model_config = {"frozen": True}
