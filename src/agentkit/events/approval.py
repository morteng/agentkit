"""Approval-gate events."""

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from agentkit.events.base import BaseEvent


class ApprovalNeeded(BaseEvent):
    type: Literal["approval_needed"] = Field(default="approval_needed")  # type: ignore[reportIncompatibleVariableOverride]
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    rationale: str | None = None
    risk: str  # RiskLevel value as str
    timeout_at: datetime


class ApprovalGranted(BaseEvent):
    type: Literal["approval_granted"] = Field(default="approval_granted")  # type: ignore[reportIncompatibleVariableOverride]
    call_id: str
    edited_args: dict[str, Any] | None = None


class ApprovalDenied(BaseEvent):
    type: Literal["approval_denied"] = Field(default="approval_denied")  # type: ignore[reportIncompatibleVariableOverride]
    call_id: str
    reason: str | None = None
