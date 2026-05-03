"""Phase transition events."""

from typing import Literal

from pydantic import Field

from agentkit.events.base import BaseEvent
from agentkit.loop.phase import Phase


class PhaseChanged(BaseEvent):
    type: Literal["phase_changed"] = Field(default="phase_changed")  # type: ignore[reportIncompatibleVariableOverride]
    from_: Phase
    to: Phase
    duration_ms: int
