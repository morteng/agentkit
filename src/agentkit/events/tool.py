"""Tool execution events."""

from typing import Any, Literal

from pydantic import Field

from agentkit.events.base import BaseEvent


class ToolCallStarted(BaseEvent):
    type: Literal["tool_call_started"] = Field(default="tool_call_started")  # type: ignore[reportIncompatibleVariableOverride]
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    risk: str  # RiskLevel value, kept as str to avoid circular import


class ToolCallProgress(BaseEvent):
    type: Literal["tool_call_progress"] = Field(default="tool_call_progress")  # type: ignore[reportIncompatibleVariableOverride]
    call_id: str
    message: str


class ToolCallResult(BaseEvent):
    type: Literal["tool_call_result"] = Field(default="tool_call_result")  # type: ignore[reportIncompatibleVariableOverride]
    call_id: str
    status: Literal["ok", "error", "denied", "timeout", "cancelled"]
    content_summary: str  # short, human-readable; full content in storage
    duration_ms: int
    cached: bool
