"""Subagent events. SubagentEvent wraps inner events from a nested loop."""

from typing import Any, Literal

from pydantic import Field

from agentkit.events.base import BaseEvent
from agentkit.events.lifecycle import TurnEndReason


class SubagentStarted(BaseEvent):
    type: Literal["subagent_started"] = Field(default="subagent_started")  # type: ignore[reportIncompatibleVariableOverride]
    subagent_id: str
    parent_call_id: str
    purpose: str


class SubagentEvent(BaseEvent):
    """Wrapper carrying one event from a nested subagent.

    The `inner` field is a JSON-serialised event (not the typed union itself —
    that would create a circular type definition). Consumers reconstruct via
    TypeAdapter(Event).validate_python(ev.inner).
    """

    type: Literal["subagent_event"] = Field(default="subagent_event")  # type: ignore[reportIncompatibleVariableOverride]
    subagent_id: str
    inner: dict[str, Any] = Field(default_factory=dict)


class SubagentEnded(BaseEvent):
    type: Literal["subagent_ended"] = Field(default="subagent_ended")  # type: ignore[reportIncompatibleVariableOverride]
    subagent_id: str
    reason: TurnEndReason
    summary: str
