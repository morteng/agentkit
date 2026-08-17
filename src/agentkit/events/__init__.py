"""Public event union and re-exports."""

from typing import Annotated

from pydantic import Field, TypeAdapter

from agentkit.events.approval import (
    ApprovalDenied,
    ApprovalGranted,
    ApprovalNeeded,
    ApprovalResolved,
)
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
    ProviderActivity,
    TextDelta,
    ThinkingDelta,
    UsageRecorded,
)
from agentkit.events.subagent import SubagentEnded, SubagentEvent, SubagentStarted
from agentkit.events.tool import ToolCallProgress, ToolCallResult, ToolCallStarted

Event = Annotated[
    PhaseChanged
    | MessageStarted
    | TextDelta
    | ThinkingDelta
    | MessageCompleted
    | UsageRecorded
    | ToolCallStarted
    | ToolCallProgress
    | ToolCallResult
    | ApprovalNeeded
    | ApprovalGranted
    | ApprovalDenied
    | ApprovalResolved
    | TurnStarted
    | TurnEnded
    | Errored
    | SubagentStarted
    | SubagentEvent
    | SubagentEnded,
    Field(discriminator="type"),
]


EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)

# ``ProviderActivity`` is deliberately absent from ``Event``. It is an internal
# liveness tick that ``_consume_stream`` drops before the consumer queue, so it
# is never serialised and no consumer ever has to know it exists. Adding it to
# the union would put it on the wire contract and oblige every consumer to
# handle a frame that, by construction, they cannot receive.


__all__ = [
    "EVENT_ADAPTER",
    "ApprovalDenied",
    "ApprovalGranted",
    "ApprovalNeeded",
    "ApprovalResolved",
    "BaseEvent",
    "ErrorCode",
    "Errored",
    "Event",
    "MessageCompleted",
    "MessageStarted",
    "PhaseChanged",
    "ProviderActivity",
    "SubagentEnded",
    "SubagentEvent",
    "SubagentStarted",
    "TextDelta",
    "ThinkingDelta",
    "ToolCallProgress",
    "ToolCallResult",
    "ToolCallStarted",
    "TurnEndReason",
    "TurnEnded",
    "TurnMetrics",
    "TurnStarted",
    "UsageRecorded",
]
