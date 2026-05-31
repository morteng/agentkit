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
from agentkit.providers.base import NamedToolChoice

if TYPE_CHECKING:
    from agentkit.guards.success_claim import SuccessClaimGuard
    from agentkit.loop.message_builder import MessageBuilder
    from agentkit.providers.base import Provider
    from agentkit.tools.registry import ToolRegistry


# Bare tool-name suffixes recognized as the finalize tool — the same
# convention the finalize validator uses (see finalize_validator.py).
_FINALIZE_BARE_NAMES = ("finalize_response", "finalize")


def _finalize_tool_name(registry: "ToolRegistry") -> str | None:
    """Resolve the finalize tool's fully-qualified name from the registry.

    Returns None when no finalize tool is registered (consumer opted out), so
    the caller falls back to an unconstrained re-prompt rather than forcing a
    tool that does not exist.
    """
    for spec in registry.list_specs():
        if spec.name.split(".", 1)[-1] in _FINALIZE_BARE_NAMES:
            return spec.name
    return None


async def handle_streaming(ctx: TurnContext, deps: dict[str, Any]) -> Phase:  # noqa: PLR0912 — turn-level dispatch necessarily branches per event type
    selector = deps.get("provider_selector")
    provider: Provider = selector(ctx) if selector is not None else deps["provider"]
    model_selector = deps.get("model_selector")
    model_override: str | None = model_selector(ctx) if model_selector is not None else None
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
        model_override=model_override,
    )

    # A missing-finalize re-prompt can be constrained to the finalize tool so
    # the model emits the envelope immediately instead of spending another
    # free-form turn (thinking / re-narrating) that holds the consumer in a
    # streaming state. The flag is one-shot: pop it here so only the re-prompt
    # turn is forced. Falls back to an unconstrained turn if no finalize tool
    # is registered.
    if ctx.metadata.pop("force_finalize_tool_choice", False):
        finalize_name = _finalize_tool_name(registry)
        if finalize_name is not None:
            request.tool_choice = NamedToolChoice(name=finalize_name)

    mux = StreamMux(ctx, registry=registry)

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
