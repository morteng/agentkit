"""Public event union and re-exports."""

from typing import Annotated

from pydantic import Field, TypeAdapter

from agentkit.events.approval import ApprovalDenied, ApprovalGranted, ApprovalNeeded
from agentkit.events.base import BaseEvent
from agentkit.events.lifecycle import (
    ErrorCode,
    Errored,
    TurnEnded,
    TurnEndReason,
    TurnMetrics,
    TurnStarted,
)
from agentkit.events.phase import PhaseChanged
from agentkit.events.streaming import (
    MessageCompleted,
    MessageStarted,
    TextDelta,
    ThinkingDelta,
)
from agentkit.events.subagent import SubagentEnded, SubagentEvent, SubagentStarted
from agentkit.events.tool import ToolCallProgress, ToolCallResult, ToolCallStarted

Event = Annotated[
    PhaseChanged
    | MessageStarted
    | TextDelta
    | ThinkingDelta
    | MessageCompleted
    | ToolCallStarted
    | ToolCallProgress
    | ToolCallResult
    | ApprovalNeeded
    | ApprovalGranted
    | ApprovalDenied
    | TurnStarted
    | TurnEnded
    | Errored
    | SubagentStarted
    | SubagentEvent
    | SubagentEnded,
    Field(discriminator="type"),
]


EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)


__all__ = [
    "BaseEvent",
    "Event",
    "EVENT_ADAPTER",
    "PhaseChanged",
    "TurnStarted",
    "TurnEnded",
    "Errored",
    "TurnEndReason",
    "ErrorCode",
    "TurnMetrics",
    "MessageStarted",
    "TextDelta",
    "ThinkingDelta",
    "MessageCompleted",
    "ToolCallStarted",
    "ToolCallProgress",
    "ToolCallResult",
    "ApprovalNeeded",
    "ApprovalGranted",
    "ApprovalDenied",
    "SubagentStarted",
    "SubagentEvent",
    "SubagentEnded",
]
