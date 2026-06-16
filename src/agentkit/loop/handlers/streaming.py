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
from agentkit._logging import get_logger
from agentkit._messages import Message, MessageRole
from agentkit.events import Errored, TextDelta, ToolCallStarted
from agentkit.loop.context import TurnContext
from agentkit.loop.phase import Phase
from agentkit.loop.stream_mux import StreamMux
from agentkit.providers.base import NamedToolChoice, ProviderRequest

if TYPE_CHECKING:
    from agentkit.guards.success_claim import SuccessClaimGuard
    from agentkit.loop.message_builder import MessageBuilder
    from agentkit.providers.base import Provider
    from agentkit.tools.registry import ToolRegistry


log = get_logger(__name__)


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


def _build_stream_request(
    ctx: TurnContext,
    deps: dict[str, Any],
    registry: "ToolRegistry",
    builder: "MessageBuilder",
    model_override: str | None,
) -> ProviderRequest:
    """Build the provider request for this iteration: apply the per-turn tool
    selector, then build from history, then (one-shot) constrain to the finalize
    tool if a missing-finalize re-prompt asked for it.

    The ``force_finalize_tool_choice`` flag makes the model emit the envelope
    immediately instead of spending another free-form turn (thinking /
    re-narrating) that holds the consumer in a streaming state. It is popped here
    so only the re-prompt turn is forced, and falls back to an unconstrained turn
    when no finalize tool is registered.
    """
    tool_selector = deps.get("tool_selector")
    available_specs = registry.list_specs()
    if tool_selector is not None:
        available_specs = tool_selector(ctx, available_specs)
    request = builder.build(
        system_blocks=deps.get("system_blocks", []),
        history=ctx.history,
        tool_specs=available_specs,
        model_override=model_override,
    )
    if ctx.metadata.pop("force_finalize_tool_choice", False):
        finalize_name = _finalize_tool_name(registry)
        if finalize_name is not None:
            request.tool_choice = NamedToolChoice(name=finalize_name)
    return request


def _record_assistant_message(
    ctx: TurnContext,
    message_id: Any,
    text_blocks: list[TextBlock],
    tool_calls_seen: list[dict[str, Any]],
) -> None:
    """Append the assistant turn (streamed text + tool-use blocks) to history.

    No-op when nothing was streamed (a clean recoverable failure leaves both
    empty), so a retried attempt does not leave an empty assistant message.
    """
    content: list[Any] = list(text_blocks)
    for tc in tool_calls_seen:
        content.append(ToolUseBlock(id=tc["id"], name=tc["name"], arguments=tc["arguments"]))
    if not content:
        return
    ctx.add_message(
        Message(
            id=message_id,
            session_id=ctx.session_id,
            role=MessageRole.ASSISTANT,
            content=content,
            created_at=ctx.clock.now(),
        )
    )


async def _maybe_retry_recoverable_stream(
    ctx: TurnContext,
    deps: dict[str, Any],
    error_event: "Errored | None",
    *,
    emitted_content: bool,
) -> bool:
    """Decide whether a failed stream should be retried, applying the backoff.

    Returns True when the caller should re-enter ``CONTEXT_BUILD`` for another
    attempt. A retry fires only for a *recoverable* error that struck before any
    output reached the consumer (so re-streaming cannot duplicate output), and
    only while the per-attempt retry budget (``max_stream_retries``) is unspent.
    This keeps a long bulk turn alive across a brief provider blip (rate limit,
    timeout, dropped connection at the request boundary) instead of aborting the
    whole worklist. On a retry it sleeps an exponential backoff before returning.
    """
    if error_event is None or not error_event.recoverable or emitted_content:
        return False
    max_retries: int = deps.get("max_stream_retries", 0)
    retry_count: int = ctx.metadata.get("stream_retry_count", 0)
    if retry_count >= max_retries:
        return False
    ctx.metadata["stream_retry_count"] = retry_count + 1
    base: float = deps.get("stream_retry_base_delay_seconds", 0.5)
    log.info(
        "stream_retry",
        attempt=retry_count + 1,
        max_retries=max_retries,
        code=error_event.code.value,
        session_id=str(ctx.session_id),
    )
    await asyncio.sleep(base * (2**retry_count))
    return True


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

    request = _build_stream_request(ctx, deps, registry, builder, model_override)
    mux = StreamMux(ctx, registry=registry)

    text_so_far: list[str] = []
    tool_calls_seen: list[dict[str, Any]] = []
    text_blocks: list[TextBlock] = []
    saw_error = False
    error_event: Errored | None = None
    emitted_content = False

    async for event in mux.translate(provider.stream(request)):
        if isinstance(event, Errored):
            # Hold the error rather than forwarding it immediately: a clean,
            # recoverable failure (see retry block below) is retried silently,
            # and the consumer should never see a flicker for a blip we
            # recover from. It is forwarded only if we end up giving up.
            saw_error = True
            error_event = event
            ctx.metadata["last_error_message"] = event.message
            ctx.metadata["last_error_code"] = event.code.value
            continue
        await queue.put(event)
        if isinstance(event, TextDelta):
            emitted_content = True
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
            emitted_content = True
            tool_calls_seen.append(
                {
                    "id": event.call_id,
                    "name": event.tool_name,
                    "arguments": event.arguments,
                }
            )

    # A recoverable error that struck before any output reached the consumer is
    # a clean transient blip — re-enter from CONTEXT_BUILD for another attempt
    # rather than aborting the turn. See _maybe_retry_recoverable_stream.
    if saw_error and await _maybe_retry_recoverable_stream(
        ctx, deps, error_event, emitted_content=emitted_content
    ):
        return Phase.CONTEXT_BUILD

    # Append assistant message to history with whatever was streamed.
    _record_assistant_message(ctx, mux.message_id, text_blocks, tool_calls_seen)
    ctx.metadata["pending_tool_calls"] = tool_calls_seen

    if saw_error:
        # We did not (or could not) retry: surface the held error now so the
        # consumer sees the genuine failure, then end the turn.
        if error_event is not None:
            await queue.put(error_event)
        return Phase.ERRORED
    # Clean stream: reset the per-attempt retry budget so the next iteration of
    # a multi-step turn gets a fresh allowance against transient blips.
    ctx.metadata["stream_retry_count"] = 0
    if tool_calls_seen:
        return Phase.TOOL_PHASE
    return Phase.FINALIZE_CHECK
