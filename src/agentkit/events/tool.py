"""Tool execution events."""

from typing import Any, Literal

from pydantic import Field

from agentkit.events.base import BaseEvent
from agentkit.tools.spec import ContentBlockOut, ToolError


class ToolCallStarted(BaseEvent):
    """Fired when the model proposes a tool call.

    .. note::

       Despite the name, this event fires when the model *decides* to call a
       tool — before the approval gate runs and before actual execution. UIs
       wanting "the tool is now running" should pair this with a subsequent
       ``ApprovalNeeded`` (gating) and the eventual ``ToolCallResult``. A
       follow-up event with execution-started semantics may be added in a
       future release.
    """

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
    error: ToolError | None = None
    content: list[ContentBlockOut] = Field(default_factory=list)  # type: ignore[reportUnknownVariableType]
