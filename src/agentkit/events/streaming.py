"""Streaming events for assistant message construction."""

from typing import Literal

from pydantic import Field

from agentkit._ids import MessageId
from agentkit._messages import Usage
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


class UsageRecorded(BaseEvent):
    """Per-LLM-call usage event yielded after MessageCompleted.

    Consumers can persist per-call cost rows by listening for this event.
    The internal ctx.metadata['usages'] capture stays for any consumer
    still inspecting it directly.
    """

    type: Literal["usage_recorded"] = Field(default="usage_recorded")  # type: ignore[reportIncompatibleVariableOverride]
    message_id: MessageId
    # ``str | None``, mirroring providers.base.UsageEvent: a provider that
    # cannot confirm the resolved model or the serving upstream provider must
    # say so with ``None`` rather than echo the request. This widens the wire
    # schema (JSON field becomes nullable) — a non-Python consumer with a
    # strict non-null string type for these fields needs a matching update.
    model: str | None
    usage: Usage
    provider_name: str | None
