"""Translate OpenAI-protocol streaming responses into ProviderEvents."""

import json
from collections.abc import AsyncIterator, Generator
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
from agentkit.providers.openrouter.model_quirks import parse_finish_reason
from agentkit.providers.openrouter.tool_translator import parse_tool_args_with_repair


def parse_tool_call_arguments(args_str: str) -> dict[str, Any] | None:
    """Parse tool-call argument JSON, using json_repair as a fallback.

    Wraps ``parse_tool_args_with_repair`` for the stream parser's tool-call
    argument path. Returns a dict on success (including repaired JSON).
    Raises ``json.JSONDecodeError`` on unrecoverable parse failure so callers
    that previously caught ``json.JSONDecodeError`` from ``json.loads`` continue
    to work unchanged.
    """
    parsed, err = parse_tool_args_with_repair(args_str)
    if err is not None:
        raise json.JSONDecodeError(err, args_str, 0)
    return parsed


async def parse_openrouter_stream(chunks: AsyncIterator[Any]) -> AsyncIterator[ProviderEvent]:
    """Map OpenAI ChatCompletionChunk events to ProviderEvents.

    OpenAI streams ``ChatCompletionChunk`` objects with ``choices[0].delta``
    containing one of: text content, tool_calls (partial), or finish_reason.
    """
    started = False
    finish_reason_raw: str | None = None
    final_usage: Usage | None = None
    pending_tools: dict[int, dict[str, Any]] = {}  # index -> {"id", "name", "args_buf"}

    async for chunk in chunks:
        if not started:
            yield MessageStart()
            started = True

        choice = chunk.choices[0] if chunk.choices else None
        if choice is None:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                final_usage = _usage_from_openai(usage)
            continue

        delta = choice.delta

        # F1: Reasoning content (DeepSeek and other reasoning models). OpenRouter
        # surfaces chain-of-thought via ``reasoning_content`` (or ``reasoning``);
        # forward as ThinkingDelta so consumers can render a "thinking..." UI
        # affordance during the latency before the first visible TextDelta.
        if reasoning := getattr(delta, "reasoning_content", None) or getattr(
            delta, "reasoning", None
        ):
            yield ThinkingDelta(delta=reasoning)

        # Text content.
        if text := getattr(delta, "content", None):
            yield TextDelta(delta=text, block_index=0)

        # Tool calls — streamed as partial JSON across multiple chunks.
        if tool_calls := getattr(delta, "tool_calls", None):
            for ev in _process_tool_call_deltas(tool_calls, pending_tools):
                yield ev

        if choice.finish_reason:
            finish_reason_raw = choice.finish_reason

    # End of stream — flush completed tool calls only if finish_reason confirms completion.
    # Tool calls only "complete" when the stream's finish_reason is "tool_calls".
    # If a stream terminates abnormally (None/"length"/"stop" with pending tools),
    # we drop the partial tool calls rather than risk executing with empty args.
    if finish_reason_raw == "tool_calls":
        for ev in _flush_pending_tools(pending_tools):
            yield ev

    if final_usage is not None:
        yield UsageEvent(usage=final_usage)
    yield MessageComplete(finish_reason=parse_finish_reason(finish_reason_raw))


def _process_tool_call_deltas(
    tool_calls: Any,
    pending_tools: dict[int, dict[str, Any]],
) -> Generator[ProviderEvent, None, None]:
    """Process streaming tool-call delta chunks and yield events."""
    for tc in tool_calls:
        idx = tc.index
        slot = pending_tools.setdefault(idx, {"id": None, "name": None, "args_buf": ""})
        if tc.id and slot["id"] is None:
            slot["id"] = tc.id
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        if fn.name and slot["name"] is None:
            slot["name"] = fn.name
            yield ToolCallStart(call_id=slot["id"] or f"call_{idx}", tool_name=fn.name)
        if fn.arguments:
            slot["args_buf"] += fn.arguments
            yield ToolCallDelta(
                call_id=slot["id"] or f"call_{idx}",
                arguments_delta=fn.arguments,
            )


def _flush_pending_tools(
    pending_tools: dict[int, dict[str, Any]],
) -> Generator[ProviderEvent, None, None]:
    """Yield ToolCallComplete events for all accumulated tool calls."""
    for slot in pending_tools.values():
        if slot["name"] is None:
            continue
        try:
            parsed: Any = parse_tool_call_arguments(slot["args_buf"] or "")
            args: dict[str, Any] = dict(parsed) if isinstance(parsed, dict) else {}  # type: ignore[reportUnknownArgumentType]
        except json.JSONDecodeError:
            args = {}
        yield ToolCallComplete(
            call_id=slot["id"] or "",
            tool_name=slot["name"],
            arguments=args,
        )


def _usage_from_openai(u: Any) -> Usage:
    cached = 0
    details = getattr(u, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    return Usage(
        input_tokens=getattr(u, "prompt_tokens", 0),
        output_tokens=getattr(u, "completion_tokens", 0),
        cached_input_tokens=cached,
    )
