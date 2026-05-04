"""Translate Anthropic SDK streaming events into ProviderEvents.

The Anthropic SDK exposes a structured event stream via ``messages.stream``;
this parser maps each event variant onto agentkit's normalised type.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

from agentkit._messages import Usage
from agentkit.providers.base import (
    MessageComplete,
    MessageStart,
    ProviderEvent,
    TextDelta,
    ThinkingDelta,
    ToolCallComplete,
    ToolCallDelta,
    ToolCallStart,
    UsageEvent,
)

_FINISH_REASON_MAP: dict[str, str] = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
}


def _parse_message_start_usage(ev: Any) -> Usage | None:
    """Extract usage from a message_start event, if present."""
    usage = getattr(ev.message, "usage", None)
    if usage is None:
        return None
    return Usage(
        input_tokens=getattr(usage, "input_tokens", 0),
        cached_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )


def _parse_message_delta_usage(ev: Any, prior: Usage | None) -> Usage:
    """Extract usage from a message_delta event."""
    usage = getattr(ev, "usage", None)
    prior_input = prior.input_tokens if prior else 0
    prior_cached = prior.cached_input_tokens if prior else 0
    prior_creation = prior.cache_creation_tokens if prior else 0
    if usage is None:
        return Usage(
            input_tokens=prior_input,
            cached_input_tokens=prior_cached,
            cache_creation_tokens=prior_creation,
        )
    return Usage(
        input_tokens=getattr(usage, "input_tokens", prior_input),
        output_tokens=getattr(usage, "output_tokens", 0),
        cached_input_tokens=(getattr(usage, "cache_read_input_tokens", prior_cached) or 0),
        cache_creation_tokens=(getattr(usage, "cache_creation_input_tokens", prior_creation) or 0),
    )


async def parse_anthropic_stream(events: AsyncIterator[Any]) -> AsyncIterator[ProviderEvent]:
    """Map Anthropic SDK events to ProviderEvents.

    The Anthropic Python SDK yields:
      - MessageStartEvent
      - ContentBlockStartEvent (with ContentBlock — text or tool_use)
      - ContentBlockDeltaEvent (TextDelta / InputJSONDelta / ThinkingDelta)
      - ContentBlockStopEvent
      - MessageDeltaEvent (carries stop_reason + final usage)
      - MessageStopEvent
    """
    pending_tool_args: dict[int, str] = {}
    pending_tool_meta: dict[int, dict[str, Any]] = {}
    finish_reason: str = "end_turn"
    final_usage: Usage | None = None

    async for ev in events:
        ev_type = getattr(ev, "type", None)

        if ev_type == "message_start":
            yield MessageStart()
            final_usage = _parse_message_start_usage(ev)

        elif ev_type == "content_block_start":
            block = ev.content_block
            if getattr(block, "type", None) == "tool_use":
                pending_tool_args[ev.index] = ""
                pending_tool_meta[ev.index] = {"call_id": block.id, "name": block.name}
                yield ToolCallStart(call_id=block.id, tool_name=block.name)

        elif ev_type == "content_block_delta":
            async for yielded in _handle_content_delta(ev, pending_tool_args, pending_tool_meta):
                yield yielded

        elif ev_type == "content_block_stop":
            result = _handle_block_stop(ev.index, pending_tool_args, pending_tool_meta)
            if result is not None:
                yield result

        elif ev_type == "message_delta":
            stop_reason = getattr(ev.delta, "stop_reason", None)
            if stop_reason:
                finish_reason = _FINISH_REASON_MAP.get(stop_reason, "end_turn")
            final_usage = _parse_message_delta_usage(ev, final_usage)

        elif ev_type == "message_stop":
            if final_usage is not None:
                yield UsageEvent(usage=final_usage)
            yield MessageComplete(finish_reason=finish_reason)  # type: ignore[arg-type]


def _handle_block_stop(
    index: int,
    pending_tool_args: dict[int, str],
    pending_tool_meta: dict[int, dict[str, Any]],
) -> ToolCallComplete | None:
    """Build a ToolCallComplete event when a tool-use block closes, or None."""
    if index not in pending_tool_meta:
        return None
    meta = pending_tool_meta[index]
    args_raw = pending_tool_args.get(index, "") or "{}"
    try:
        args: dict[str, Any] = json.loads(args_raw)
    except json.JSONDecodeError:
        args = {}
    return ToolCallComplete(
        call_id=meta["call_id"],
        tool_name=meta["name"],
        arguments=args,
    )


async def _handle_content_delta(
    ev: Any,
    pending_tool_args: dict[int, str],
    pending_tool_meta: dict[int, dict[str, Any]],
) -> AsyncIterator[ProviderEvent]:
    """Yield events for a content_block_delta SDK event."""
    delta = ev.delta
    delta_type = getattr(delta, "type", None)
    if delta_type == "text_delta":
        yield TextDelta(delta=delta.text, block_index=ev.index)
    elif delta_type == "thinking_delta":
        yield ThinkingDelta(delta=delta.thinking)
    elif delta_type == "input_json_delta":
        pending_tool_args[ev.index] = pending_tool_args.get(ev.index, "") + delta.partial_json
        meta = pending_tool_meta.get(ev.index)
        if meta:
            yield ToolCallDelta(call_id=meta["call_id"], arguments_delta=delta.partial_json)
