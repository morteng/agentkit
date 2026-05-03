"""Streaming events for assistant message construction."""

from typing import Literal

from pydantic import Field

from agentkit._ids import MessageId
from agentkit.events.base import BaseEvent


class MessageStarted(BaseEvent):
    type: Literal["message_started"] = Field(default="message_started")  # type: ignore[reportIncompatibleVariableOverride]
    message_id: MessageId
    role: Literal["assistant"] = "assistant"


class TextDelta(BaseEvent):
    type: Literal["text_delta"] = Field(default="text_delta")  # type: ignore[reportIncompatibleVariableOverride]
    message_id: MessageId
    delta: str
    block_index: int = 0


class ThinkingDelta(BaseEvent):
    type: Literal["thinking_delta"] = Field(default="thinking_delta")  # type: ignore[reportIncompatibleVariableOverride]
    message_id: MessageId
    delta: str


class MessageCompleted(BaseEvent):
    type: Literal["message_completed"] = Field(default="message_completed")  # type: ignore[reportIncompatibleVariableOverride]
    message_id: MessageId
    finish_reason: Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"]
