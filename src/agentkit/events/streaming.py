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


class ProviderActivity(BaseEvent):
    """Internal liveness tick: the provider sent a chunk that produces no
    user-facing event.

    **Never reaches a consumer.** ``_consume_stream`` treats it as a timer
    reset and drops it before the queue put; it exists only to travel the
    same iterator the stall timeout is measuring.

    The timeout wraps ``anext()`` on the *muxed* stream, so it measures the
    gap between muxed events and not the gap between provider chunks. Those
    are the same number right up until the model emits a tool call:
    ``StreamMux`` yields nothing for ``tool_call_start`` and nothing for each
    ``tool_call_delta``, so a long argument buffer — a batch of calls, or one
    call carrying a long URL — is silence on the iterator while the wire is
    busy. A turn that asks for enough work then dies at exactly the chunk
    timeout, deterministically, and reports it as the provider going quiet.

    Emitting a tick from those arms restores the invariant the timeout was
    written against: every provider chunk produces a muxed event.
    """

    type: Literal["provider_activity"] = Field(default="provider_activity")  # type: ignore[reportIncompatibleVariableOverride]


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
