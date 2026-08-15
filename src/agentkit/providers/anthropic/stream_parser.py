"""Translate Anthropic SDK streaming events into ProviderEvents.

The Anthropic SDK exposes a structured event stream via ``messages.stream``;
this parser maps each event variant onto agentkit's normalised type.
"""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any, cast

from agentkit._messages import Usage
from agentkit.providers.base import (
    ErrorEvent,
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
from agentkit.providers.tool_call_errors import (
    INVALID_TOOL_ARGUMENTS_CODE,
    invalid_arguments_message,
)

logger = logging.getLogger(__name__)

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


async def parse_anthropic_stream(
    events: AsyncIterator[Any], *, model: str
) -> AsyncIterator[ProviderEvent]:
    """Map Anthropic SDK events to ProviderEvents.

    The Anthropic Python SDK yields:
      - MessageStartEvent
      - ContentBlockStartEvent (with ContentBlock — text or tool_use)
      - ContentBlockDeltaEvent (TextDelta / InputJSONDelta / ThinkingDelta)
      - ContentBlockStopEvent
      - MessageDeltaEvent (carries stop_reason + final usage)
      - MessageStopEvent

    Args:
        events: Async iterator of Anthropic SDK stream events.
        model: The model identifier used for this request (e.g. ``"anthropic/claude-opus-4-7"``).
            Stamped onto the emitted :class:`UsageEvent` so cost-ledger consumers
            can attribute usage without inspecting the originating request.
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
                yield UsageEvent(usage=final_usage, model=model, provider_name="anthropic")
            yield MessageComplete(finish_reason=finish_reason)  # type: ignore[arg-type]


def _handle_block_stop(
    index: int,
    pending_tool_args: dict[int, str],
    pending_tool_meta: dict[int, dict[str, Any]],
) -> ProviderEvent | None:
    """Resolve a closing tool-use block into exactly one terminal event.

    Either a dispatchable :class:`ToolCallComplete` or an :class:`ErrorEvent`
    explaining why the call could not be dispatched — never a guess. This used
    to emit the call with ``arguments={}`` on unparseable JSON, which is the
    same defect the OpenRouter parser had; see
    :mod:`agentkit.providers.tool_call_errors` for why an empty argument set is
    the most dangerous possible reading of a corrupted call.

    Unlike the OpenRouter path there is no ``json_repair`` pass. The failure
    mode here is truncation (the SDK closes open blocks on ``max_tokens``), and
    repairing a *truncated* argument object invents a complete-looking call out
    of a half-streamed one — the guess this change exists to stop.
    """
    if index not in pending_tool_meta:
        return None
    meta = pending_tool_meta[index]
    args_raw = pending_tool_args.get(index, "") or "{}"
    try:
        # `object`, not `dict[str, Any]`. The model controls this string, so the
        # annotation must describe what json.loads can actually return — and the
        # isinstance check below is the thing that makes it a dict. Annotating it
        # as a dict up front makes that check look redundant to a type checker
        # while leaving the runtime hazard exactly where it was.
        args: object = json.loads(args_raw)
    except json.JSONDecodeError:
        # Never log ``args_raw`` itself: it can carry user text. Only its length.
        logger.warning(
            "anthropic.tool_args_unparseable_rejected",
            extra={"tool_name": meta["name"], "args_buf_len": len(args_raw)},
        )
        return ErrorEvent(
            code=INVALID_TOOL_ARGUMENTS_CODE,
            message=invalid_arguments_message(tool_name=meta["name"], call_id=meta["call_id"]),
            recoverable=True,
        )
    if not isinstance(args, dict):
        # Valid JSON that is not an argument object (``[1,2]``, ``null``, ``4``).
        # ``json.loads`` accepts these; a tool handler cannot use them.
        logger.warning(
            "anthropic.tool_args_not_an_object",
            extra={"tool_name": meta["name"], "args_buf_len": len(args_raw)},
        )
        return ErrorEvent(
            code=INVALID_TOOL_ARGUMENTS_CODE,
            message=invalid_arguments_message(tool_name=meta["name"], call_id=meta["call_id"]),
            recoverable=True,
        )
    return ToolCallComplete(
        call_id=meta["call_id"],
        tool_name=meta["name"],
        # The isinstance above narrows to `dict`, but JSON gives no key/value
        # types, so this is where the claim "the keys are strings" is made. It
        # holds because JSON object keys are strings by grammar — not because
        # anything here checked.
        arguments=cast("dict[str, Any]", args),
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
