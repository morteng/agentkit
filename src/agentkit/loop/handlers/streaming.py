"""Streaming handler — drive the provider stream and route to TOOL_PHASE/FINALIZE_CHECK.

Responsibilities:
  - Build the ProviderRequest via MessageBuilder.
  - Run StreamMux to translate provider events into user-facing events.
  - Push events into ctx.event_queue (consumer drains).
  - Track pending tool calls in ctx.metadata['pending_tool_calls'] (filled when
    ToolCallStarted is emitted).
  - Bound each chunk await by ``streaming_chunk_timeout_seconds`` so a provider
    that stops sending mid-stream fails the turn instead of hanging it forever.
  - Optionally invoke SuccessClaimGuard on text-so-far; on flag, close the
    stream and transition CONTEXT_BUILD with an injected correction, bounded by
    ``max_claim_corrections``.
  - Decide next phase: TOOL_PHASE if tool calls were emitted, else FINALIZE_CHECK.

Every exit path closes the mux and the underlying provider stream (see
:func:`_aclose_quietly`): an abandoned async generator otherwise keeps the HTTP
response open until the GC gets to it.
"""

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentkit._content import TextBlock, ToolUseBlock
from agentkit._ids import EventId, new_id
from agentkit._logging import get_logger
from agentkit._messages import Message, MessageRole
from agentkit.events import ErrorCode, Errored, ProviderActivity, TextDelta, ToolCallStarted
from agentkit.loop.context import TurnContext
from agentkit.loop.handlers.finalize_check import (
    _inject_correction,  # pyright: ignore[reportPrivateUsage]
)
from agentkit.loop.phase import Phase
from agentkit.loop.stream_mux import StreamMux
from agentkit.providers.base import NamedToolChoice, ProviderRequest
from agentkit.tools.builtin.finalize import FINALIZE_BARE_NAMES

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from agentkit.events.base import BaseEvent
    from agentkit.guards.success_claim import SuccessClaimGuard
    from agentkit.loop.message_builder import MessageBuilder
    from agentkit.providers.base import Provider
    from agentkit.tools.registry import ToolRegistry
    from agentkit.tools.spec import ToolSpec


log = get_logger(__name__)


# Bare tool-name suffixes recognized as the finalize tool — the same
# convention the finalize validator uses (see finalize_validator.py).
# Re-exported from the tool module so this handler and `finalize_check` cannot
# disagree about what counts as a finalize tool.
_FINALIZE_BARE_NAMES = FINALIZE_BARE_NAMES

# Wrapper around the guard's suggested correction. The model has already
# streamed the claim to the user, so the correction has to say what happens
# next, not just what was wrong.
_CLAIM_CORRECTION_TEMPLATE = (
    "Your previous message was interrupted because it claimed work was done "
    "before the work happened.\n\n"
    "{correction}\n\n"
    "The user has already seen the interrupted message. Do the work now by "
    "calling the tool, then say what actually happened."
)


def _finalize_tool_name(specs: "Sequence[ToolSpec]") -> str | None:
    """Resolve the finalize tool's fully-qualified name from the specs OFFERED.

    Returns None when no finalize tool is among them, so the caller falls back
    to an unconstrained re-prompt rather than forcing a tool the request does
    not carry.

    ``specs``, not the registry, and the distinction is the bug this signature
    exists to prevent. The request is built from whatever the per-turn
    ``tool_selector`` left, which may be a subset of the registry or empty. A
    name resolved from the registry can therefore name a tool that is not in
    ``payload["tools"]`` — which providers reject or ignore — and if the
    selector returned nothing at all the builder omits ``tool_choice``
    entirely, because it is only emitted alongside tools. Either way the
    forcing silently does not happen, and the flag has already been popped, so
    there is no second attempt: the re-prompt budget is spent on an
    unconstrained turn that looks from the outside exactly like a forced one
    the model ignored.
    """
    for spec in specs:
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
        # `available_specs`, which is what the request actually carries — see
        # `_finalize_tool_name`. Resolving from the registry here would let a
        # tool_selector produce a request whose tool_choice names a tool its
        # own `tools` list does not contain.
        finalize_name = _finalize_tool_name(available_specs)
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


async def _aclose_quietly(*iterators: Any) -> None:
    """Close every async iterator that supports it, swallowing close-time errors.

    Called from the ``finally`` of the stream loop, including the paths that
    abandon the stream early (success-claim trip, chunk timeout). A close that
    raises — a provider generator that was mid-``await`` when it was cancelled
    is the common case — must not mask the reason we are unwinding.
    """
    for it in iterators:
        aclose = getattr(it, "aclose", None)
        if aclose is None:
            continue
        with contextlib.suppress(Exception):
            await aclose()


def _chunk_timeout(deps: dict[str, Any]) -> float | None:
    """Per-chunk wall-clock budget, or None when the consumer disabled it.

    Mirrors ``AgentConfig.loop.streaming_chunk_timeout_seconds``. A provider
    that accepts the request and then goes quiet (dropped socket that never
    RSTs, an overloaded gateway) otherwise parks the turn forever with the
    consumer stuck in a streaming state; a non-positive value opts out.
    """
    timeout = deps.get("streaming_chunk_timeout_seconds", 60.0)
    if timeout is None or timeout <= 0:
        return None
    return float(timeout)


def _stall_error(ctx: TurnContext, timeout: float) -> Errored:
    """The Errored event for a provider that stopped mid-stream.

    Marked recoverable: a stall before any output reached the consumer is
    exactly the clean transient failure ``_maybe_retry_recoverable_stream``
    exists to absorb.
    """
    return Errored(
        event_id=new_id(EventId),
        session_id=ctx.session_id,
        turn_id=ctx.turn_id,
        ts=ctx.clock.now(),
        sequence=ctx.next_sequence(),
        code=ErrorCode.PROVIDER_FAULT,
        message=f"provider stream stalled: no chunk within {timeout:g}s",
        recoverable=True,
    )


def _spend_claim_correction(
    ctx: TurnContext,
    deps: dict[str, Any],
    correction: str | None,
    *,
    message_id: Any,
    text_blocks: list[TextBlock],
) -> bool:
    """Charge one success-claim correction; True when the turn should retry.

    Three things the pre-0.22 implementation got wrong, fixed here:

    * The correction is appended to ``ctx.history`` as a user-role message (the
      same delivery path ``finalize_check`` uses), because the MessageBuilder
      reads history verbatim — a correction parked in ``ctx.metadata`` never
      reaches the model, so the guard tripped again on the retry, forever.
    * ``max_claim_corrections`` is actually enforced. Once it is spent the
      caller streams on with the guard disabled for the rest of the turn:
      the user is better served by a completed answer than by a turn that
      loops or dies.
    * The text streamed so far is written to history. The user has already
      seen it; a model whose history does not contain its own visible message
      re-narrates it on the retry.

    Partial tool calls from the interrupted stream are deliberately NOT
    recorded: they were never dispatched, and a ``ToolUseBlock`` with no
    matching ``ToolResultBlock`` makes the very next provider request invalid.
    """
    used: int = ctx.metadata.get("claim_corrections", 0)
    budget: int = deps.get("max_claim_corrections", 1)
    if used >= budget:
        ctx.metadata["claim_corrections_exhausted"] = True
        log.warning(
            "success_claim_corrections_exhausted",
            budget=budget,
            session_id=str(ctx.session_id),
            turn_id=str(ctx.turn_id),
        )
        return False
    ctx.metadata["claim_corrections"] = used + 1
    ctx.metadata["claim_correction"] = correction
    log.info(
        "success_claim_correction",
        attempt=used + 1,
        budget=budget,
        session_id=str(ctx.session_id),
        turn_id=str(ctx.turn_id),
    )
    _record_assistant_message(ctx, message_id, text_blocks, [])
    _inject_correction(
        ctx,
        _CLAIM_CORRECTION_TEMPLATE.format(
            correction=correction or "Call the tool that performs the work first."
        ),
    )
    return True


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


@dataclass
class _StreamOutcome:
    """What one drained provider stream produced.

    ``claim_retry`` means the success-claim guard tripped, a correction was
    charged against ``max_claim_corrections`` and already appended to history:
    the caller re-enters CONTEXT_BUILD and must not record the assistant
    message a second time.
    """

    text_blocks: list[TextBlock] = field(default_factory=list)  # type: ignore[reportUnknownVariableType]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # type: ignore[reportUnknownVariableType]
    error_event: Errored | None = None
    emitted_content: bool = False
    claim_retry: bool = False


async def _consume_stream(
    ctx: TurnContext,
    deps: dict[str, Any],
    *,
    events: "AsyncIterator[BaseEvent]",
    queue: "asyncio.Queue[Any]",
    message_id: Any,
    success_claim: "SuccessClaimGuard | None",
) -> _StreamOutcome:
    """Drain one translated provider stream, pushing events to the consumer.

    Returns rather than raises: a stalled or errored stream is data the caller
    weighs against the retry budget. Closing ``events`` is the caller's job.
    """
    out = _StreamOutcome()
    text_so_far: list[str] = []
    claim_guard_spent = False
    chunk_timeout = _chunk_timeout(deps)

    while True:
        try:
            # asyncio.timeout, not wait_for: wait_for would run each __anext__
            # in a *new* task, and provider generators hold anyio cancel scopes
            # (httpx) that must be entered and exited from one task.
            async with asyncio.timeout(chunk_timeout):
                event = await anext(events)
        except StopAsyncIteration:
            break
        except TimeoutError:
            # The provider accepted the request and then went quiet. Hold it as
            # a recoverable error so the retry path can absorb it when nothing
            # has reached the consumer yet.
            stalled_after = chunk_timeout if chunk_timeout is not None else 0.0
            log.warning(
                "stream_chunk_timeout",
                timeout_seconds=stalled_after,
                emitted_content=out.emitted_content,
                session_id=str(ctx.session_id),
                turn_id=str(ctx.turn_id),
            )
            out.error_event = _stall_error(ctx, stalled_after)
            ctx.metadata["last_error_message"] = out.error_event.message
            ctx.metadata["last_error_code"] = out.error_event.code.value
            break

        if isinstance(event, ProviderActivity):
            # Proof of life, not output. Reaching this line already restarted
            # the timeout above, which is the entire purpose of the event; it
            # must not reach the queue, because no consumer knows this type and
            # it carries nothing anyone could render. Dropped before `queue.put`
            # rather than filtered by the consumer, so "never user-visible" is a
            # property of this loop and not of every consumer's dispatcher.
            continue

        if isinstance(event, Errored):
            # Hold the error rather than forwarding it immediately: a clean,
            # recoverable failure (see the retry block in handle_streaming) is
            # retried silently, and the consumer should never see a flicker for
            # a blip we recover from. It is forwarded only if we give up.
            out.error_event = event
            ctx.metadata["last_error_message"] = event.message
            ctx.metadata["last_error_code"] = event.code.value
            continue

        await queue.put(event)
        if isinstance(event, TextDelta):
            out.emitted_content = True
            text_so_far.append(event.delta)
            # Build up text blocks for history insertion.
            if not out.text_blocks:
                out.text_blocks.append(TextBlock(text=event.delta))
            else:
                out.text_blocks[-1] = TextBlock(text=out.text_blocks[-1].text + event.delta)
            if success_claim is not None and not claim_guard_spent:
                verdict = await success_claim.check("".join(text_so_far), ctx)
                if verdict.flag:
                    if _spend_claim_correction(
                        ctx,
                        deps,
                        verdict.suggested_correction,
                        message_id=message_id,
                        text_blocks=out.text_blocks,
                    ):
                        out.claim_retry = True
                        return out
                    # Budget spent: stop checking and let the stream finish.
                    claim_guard_spent = True
        elif isinstance(event, ToolCallStarted):
            out.emitted_content = True
            out.tool_calls.append(
                {
                    "id": event.call_id,
                    "name": event.tool_name,
                    "arguments": event.arguments,
                }
            )
    return out


async def handle_streaming(ctx: TurnContext, deps: dict[str, Any]) -> Phase:
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

    provider_stream = provider.stream(request)
    events: AsyncIterator[BaseEvent] = mux.translate(provider_stream)
    try:
        outcome = await _consume_stream(
            ctx,
            deps,
            events=events,
            queue=queue,
            message_id=mux.message_id,
            success_claim=success_claim,
        )
    finally:
        # Every exit path, including the success-claim trip and the chunk
        # timeout: never leave the provider generator (and its HTTP response)
        # open for the GC to reap.
        await _aclose_quietly(events, provider_stream)

    if outcome.claim_retry:
        # The correction and the already-seen text are in history; go get a
        # fresh stream that has actually read them.
        return Phase.CONTEXT_BUILD

    error_event = outcome.error_event
    # A recoverable error that struck before any output reached the consumer is
    # a clean transient blip — re-enter from CONTEXT_BUILD for another attempt
    # rather than aborting the turn. See _maybe_retry_recoverable_stream.
    if error_event is not None and await _maybe_retry_recoverable_stream(
        ctx, deps, error_event, emitted_content=outcome.emitted_content
    ):
        return Phase.CONTEXT_BUILD

    # Append assistant message to history with whatever was streamed.
    _record_assistant_message(ctx, mux.message_id, outcome.text_blocks, outcome.tool_calls)
    ctx.metadata["pending_tool_calls"] = outcome.tool_calls

    if error_event is not None:
        # We did not (or could not) retry: surface the held error now so the
        # consumer sees the genuine failure, then end the turn.
        await queue.put(error_event)
        return Phase.ERRORED
    # Clean stream: reset the per-attempt retry budget so the next iteration of
    # a multi-step turn gets a fresh allowance against transient blips.
    ctx.metadata["stream_retry_count"] = 0
    if outcome.tool_calls:
        return Phase.TOOL_PHASE
    return Phase.FINALIZE_CHECK
