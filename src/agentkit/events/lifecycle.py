"""Lifecycle events (turn start/end, errors)."""

from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from agentkit._ids import MessageId
from agentkit.events.base import BaseEvent


class TurnEndReason(StrEnum):
    COMPLETED = "completed"
    AWAITING_APPROVAL = "awaiting_approval"
    ERROR = "error"
    CANCELLED = "cancelled"
    MAX_ITERATIONS = "max_iterations"


class ErrorCode(StrEnum):
    PROVIDER_FAULT = "provider_fault"
    TOOL_FAULT = "tool_fault"
    RATE_LIMITED = "rate_limited"
    INTENT_REJECTED = "intent_rejected"
    APPROVAL_TIMEOUT = "approval_timeout"
    INTERNAL = "internal"


class TurnMetrics(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    thinking_tokens: int = 0
    cost_usd: Decimal = Decimal("0")
    duration_ms: int = 0
    tool_calls: int = 0
    iterations: int = 0


class TurnStarted(BaseEvent):
    type: Literal["turn_started"] = Field(default="turn_started")  # type: ignore[reportIncompatibleVariableOverride]
    user_message_id: MessageId


class TurnEnded(BaseEvent):
    """Terminal event for a turn.

    ``reason`` is the structured outcome (the enum). ``summary`` is the
    optional freeform string the model passed to :func:`kit.finalize` —
    populated only when the model actually called the tool with a non-empty
    ``reason`` argument. Audit logs and UIs that want to render
    "Completed: <one-line summary>" instead of just "Completed" should read
    this field; the structured ``reason`` enum remains the source of truth
    for control flow.
    """

    type: Literal["turn_ended"] = Field(default="turn_ended")  # type: ignore[reportIncompatibleVariableOverride]
    reason: TurnEndReason
    metrics: TurnMetrics
    summary: str | None = None


class Errored(BaseEvent):
    type: Literal["errored"] = Field(default="errored")  # type: ignore[reportIncompatibleVariableOverride]
    code: ErrorCode
    message: str
    recoverable: bool
