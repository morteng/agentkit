"""Canonical Message + Usage types. Provider-agnostic."""

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from agentkit._content import ContentBlock
from agentkit._ids import MessageId, SessionId


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_tokens: int = 0  # Anthropic-specific; 0 elsewhere
    thinking_tokens: int = 0  # Anthropic extended thinking
    cost_usd: Decimal = Decimal("0")


class PhaseTransition(BaseModel):
    """One step in the phase log — captured for observability."""

    from_: str = Field(alias="from")
    to: str
    duration_ms: int
    ts: datetime

    model_config = {"populate_by_name": True}


class MessageMetadata(BaseModel):
    provider: str | None = None
    model: str | None = None
    usage: Usage | None = None
    phase_log: list[PhaseTransition] = Field(default_factory=list)  # type: ignore[reportUnknownVariableType]
    # tool_calls/tool_results are encoded inside content as ToolUseBlock/ToolResultBlock;
    # this dict is for free-form consumer annotations.
    annotations: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    id: MessageId
    session_id: SessionId
    role: MessageRole
    content: list[ContentBlock]
    metadata: MessageMetadata = Field(default_factory=MessageMetadata)
    created_at: datetime
