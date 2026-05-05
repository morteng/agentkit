"""Streaming handler — drive the provider stream and route to TOOL_PHASE/FINALIZE_CHECK.

Responsibilities:
  - Build the ProviderRequest via MessageBuilder.
  - Run StreamMux to translate provider events into user-facing events.
  - Push events into ctx.event_queue (consumer drains).
  - Track pending tool calls in ctx.metadata['pending_tool_calls'] (filled when
    ToolCallStarted is emitted).
  - Optionally invoke SuccessClaimGuard on text-so-far; on flag, cancel the
    stream and transition CONTEXT_BUILD with an injected correction.
  - Decide next phase: TOOL_PHASE if tool calls were emitted, else FINALIZE_CHECK.
"""

import asyncio
from typing import TYPE_CHECKING, Any

from agentkit._content import TextBlock, ToolUseBlock
from agentkit._messages import Message, MessageRole
from agentkit.events import Errored, TextDelta, ToolCallStarted
from agentkit.loop.context import TurnContext
from agentkit.loop.phase import Phase
from agentkit.loop.stream_mux import StreamMux

if TYPE_CHECKING:
    from agentkit.guards.success_claim import SuccessClaimGuard
    from agentkit.loop.message_builder import MessageBuilder
    from agentkit.providers.base import Provider
    from agentkit.tools.registry import ToolRegistry


async def handle_streaming(ctx: TurnContext, deps: dict[str, Any]) -> Phase:  # noqa: PLR0912 — turn-level dispatch necessarily branches per event type
    provider: Provider = deps["provider"]
    builder: MessageBuilder = deps["message_builder"]
    registry: ToolRegistry = deps["registry"]
    queue: asyncio.Queue[Any] = ctx.event_queue if ctx.event_queue is not None else asyncio.Queue()
    if ctx.event_queue is None:
        ctx.event_queue = queue
    success_claim: SuccessClaimGuard | None = deps.get("success_claim")

    request = builder.build(
        system_blocks=deps.get("system_blocks", []),
        history=ctx.history,
        tool_specs=registry.list_specs(),
    )
    mux = StreamMux(ctx, sequence_start=0, registry=registry)

    text_so_far: list[str] = []
    tool_calls_seen: list[dict[str, Any]] = []
    text_blocks: list[TextBlock] = []
    saw_error = False

    async for event in mux.translate(provider.stream(request)):
        await queue.put(event)
        if isinstance(event, Errored):
            saw_error = True
            ctx.metadata["last_error_message"] = event.message
            ctx.metadata["last_error_code"] = event.code.value
            continue
        if isinstance(event, TextDelta):
            text_so_far.append(event.delta)
            # Build up text blocks for history insertion.
            if not text_blocks:
                text_blocks.append(TextBlock(text=event.delta))
            else:
                text_blocks[-1] = TextBlock(text=text_blocks[-1].text + event.delta)
            if success_claim is not None:
                verdict = await success_claim.check("".join(text_so_far), ctx)
                if verdict.flag:
                    ctx.metadata["claim_correction"] = verdict.suggested_correction
                    return Phase.CONTEXT_BUILD
        elif isinstance(event, ToolCallStarted):
            tool_calls_seen.append(
                {
                    "id": event.call_id,
                    "name": event.tool_name,
                    "arguments": event.arguments,
                }
            )

    # Append assistant message to history with whatever was streamed.
    content: list[Any] = list(text_blocks)
    for tc in tool_calls_seen:
        content.append(ToolUseBlock(id=tc["id"], name=tc["name"], arguments=tc["arguments"]))
    if content:
        ctx.add_message(
            Message(
                id=mux.message_id,
                session_id=ctx.session_id,
                role=MessageRole.ASSISTANT,
                content=content,
                created_at=ctx.clock.now(),
            )
        )

    ctx.metadata["pending_tool_calls"] = tool_calls_seen

    if saw_error:
        return Phase.ERRORED
    if tool_calls_seen:
        return Phase.TOOL_PHASE
    return Phase.FINALIZE_CHECK
