"""Lifecycle events (turn start/end, errors)."""

from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from agentkit._codes import ErrorCode
from agentkit._ids import MessageId
from agentkit.events.base import BaseEvent


class TurnEndReason(StrEnum):
    COMPLETED = "completed"
    AWAITING_APPROVAL = "awaiting_approval"
    ERROR = "error"
    CANCELLED = "cancelled"
    MAX_ITERATIONS = "max_iterations"
    #: The turn ended without the model producing a final answer — it never
    #: called ``finalize_response``, and the re-prompt budget for asking it to
    #: is spent. Distinct from COMPLETED, which asserts the model finished.
    #:
    #: Before this existed the two were indistinguishable on the wire, so a
    #: consumer had no way to tell a finished answer from a turn that stopped
    #: mid-thought — and the observed failure is that it runs its normal
    #: completion path, leaving the reader looking at a conversation that ends
    #: on a row of tool calls with no reply and no sign anything went wrong.
    NO_RESPONSE = "no_response"
    #: The model produced content and *did* call ``finalize_response``, but the
    #: envelope was rejected on every attempt within ``max_finalize_retries``.
    #: Distinct from NO_RESPONSE, which asserts the model never finalized at
    #: all — saying that here would be wrong, because the user has content in
    #: front of them and only the envelope failed.
    #:
    #: A separate member rather than COMPLETED for the same reason NO_RESPONSE
    #: is one: COMPLETED asserts the model finished, and a turn whose every
    #: envelope was rejected demonstrably has not. The degraded state was
    #: recorded only in ``ctx.metadata["finalize_exhausted"]``, which never
    #: leaves the runtime and has no reader anywhere in ``src``, so on the wire
    #: this was byte-identical to success and a consumer branching on this enum
    #: could not reach the case at all.
    FINALIZE_REJECTED = "finalize_rejected"


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
