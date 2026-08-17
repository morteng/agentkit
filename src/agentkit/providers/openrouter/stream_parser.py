"""Translate OpenAI-protocol streaming responses into ProviderEvents."""

import json
from collections.abc import AsyncIterator, Generator
from typing import Any, cast

from agentkit._logging import get_logger
from agentkit._messages import Usage
from agentkit._stream_trace import is_tracing, trace_delta
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
from agentkit.providers.openrouter.model_quirks import parse_finish_reason
from agentkit.providers.openrouter.tool_name_codec import ToolNameCodec
from agentkit.providers.openrouter.tool_translator import parse_tool_args_with_repair
from agentkit.providers.tool_call_errors import (
    INCOMPLETE_TOOL_CALL_CODE,
    INVALID_TOOL_ARGUMENTS_CODE,
)
from agentkit.providers.tool_call_errors import (
    UNNAMED_TOOL as _UNNAMED_TOOL,
)
from agentkit.providers.tool_call_errors import (
    incomplete_call_message as _incomplete_call_message,
)
from agentkit.providers.tool_call_errors import (
    invalid_arguments_message as _invalid_arguments_message,
)

# structlog, not stdlib logging. These three warnings are the only record
# that a tool call was dropped or refused, and under a stdlib logger they
# went nowhere: the application configures structlog, so every one of them
# was mute in production. Found while trying to use
# `pending_tool_calls_dropped` as evidence during an incident and getting
# zero lines — with a control showing zero `openrouter.*` lines of any kind.
logger = get_logger(__name__)

# Re-exported here because these names shipped from this module: the codes and
# the model-facing wording now live in agentkit.providers.tool_call_errors so
# the Anthropic parser resolves the same two failures the same way.
__all__ = ["INCOMPLETE_TOOL_CALL_CODE", "INVALID_TOOL_ARGUMENTS_CODE", "parse_openrouter_stream"]


def _decode_name(name: str | None, name_codec: ToolNameCodec | None) -> str | None:
    """Decode a wire tool name for logging — the name a human greps for."""
    if name is None:
        return None
    return name_codec.decode(name) if name_codec is not None else name


def _fallback_call_id(slot: dict[str, Any], idx: int) -> str:
    """The one call id every event for a slot must agree on.

    The provider is supposed to send an ``id`` with the first tool-call delta;
    when it does not, the slot index is the only stable handle we have. Start,
    delta and complete used to synthesise this differently (``call_{idx}`` vs
    ``""``), which made an id-less call look like two different calls to
    StreamMux's start/complete accounting. One helper, one answer.
    """
    call_id = slot["id"]
    return str(call_id) if call_id else f"call_{idx}"


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


async def parse_openrouter_stream(  # noqa: PLR0912 — chunk-type dispatch + tracing gate
    chunks: AsyncIterator[Any],
    *,
    model: str,
    session_id: str | None = None,
    name_codec: ToolNameCodec | None = None,
) -> AsyncIterator[ProviderEvent]:
    """Map OpenAI ChatCompletionChunk events to ProviderEvents.

    OpenAI streams ``ChatCompletionChunk`` objects with ``choices[0].delta``
    containing one of: text content, tool_calls (partial), or finish_reason.

    Tool calls fail closed. A call is emitted as ``ToolCallComplete`` only when
    the stream said it finished (``finish_reason == "tool_calls"``) *and* its
    arguments parsed into an object. Every other outcome — truncated stream,
    missing function name, irreparable argument JSON — yields an
    :class:`ErrorEvent` (:data:`INCOMPLETE_TOOL_CALL_CODE` /
    :data:`INVALID_TOOL_ARGUMENTS_CODE`) whose ``message`` is written for the
    model to read, and no dispatchable event at all. Arguments are never
    guessed: ``{}`` is a plausible-looking argument set for a destructive tool.

    Args:
        chunks: Async iterator of ChatCompletionChunk objects from the OpenAI SDK.
        model: The model identifier *requested* (e.g. ``"~deepseek/deepseek-v4-flash-latest"``).
            Used only for tracing/log correlation (``STREAM_TRACE_SESSIONS``,
            the dropped-tool-call warning) — never stamped onto the emitted
            :class:`UsageEvent`. OpenRouter supports moving aliases that
            resolve server-side to a concrete model, and each
            ``ChatCompletionChunk`` echoes the *resolved* id back on its own
            ``model`` field (OpenRouter OpenAPI spec: ``ChatStreamChunk.model``,
            required on every chunk). ``UsageEvent.model`` is stamped from that
            wire value, not this parameter — echoing the request here was the
            bug (every usage record said the alias, forever, regardless of
            which concrete model actually answered).
        session_id: Optional session id, forwarded to the per-session stream
            tracer (``agentkit._stream_trace``). When the session is allowlisted
            via ``STREAM_TRACE_SESSIONS``, each text delta is logged at the
            ``translator_in`` checkpoint — the upstream point where the openai
            SDK has just delivered ``delta.content`` and we are about to emit
            ``TextDelta``. Used to attribute chat truncation bugs to a layer
            (a downstream consumer's chat-truncation investigation). No-op when not allowlisted.
    """
    started = False
    finish_reason_raw: str | None = None
    final_usage: Usage | None = None
    # The model actually reported by the wire, not the one requested. Every
    # ``ChatCompletionChunk``/``ChatStreamChunk`` carries a required ``model``
    # field that is OpenRouter's resolved id (see the docstring above) — capture
    # it the same last-wins way as usage, since nothing guarantees which chunk
    # carries it and a later, real value should win over an earlier one. Starts
    # ``None`` and *stays* ``None`` if the wire never sends it — never falls
    # back to the request's ``model`` parameter, which is precisely the bug
    # being fixed.
    response_model: str | None = None
    # The upstream provider that served the request (DeepSeek, Fireworks, ...),
    # read from OpenRouter's opt-in ``openrouter_metadata`` (see adapter.py's
    # ``X-OpenRouter-Metadata: enabled`` header). Undeclared on
    # ``ChatCompletionChunk`` — the openai SDK's ``BaseModel`` has
    # ``extra="allow"``, so it survives as a plain ``dict`` off
    # ``chunk.openrouter_metadata``, not a typed sub-object. Stays ``None``
    # (never a guess, and never the old hardcoded ``"openrouter"`` literal —
    # that name only ever meant "this is the OpenRouter *adapter*", which is
    # already carried by ``Provider.name``, not which inference backend
    # answered) when the metadata is absent, malformed, or no endpoint is
    # marked ``selected``.
    provider_name: str | None = None
    pending_tools: dict[int, dict[str, Any]] = {}  # index -> {"id", "name", "args_buf"}
    # Cache the membership check once per stream — no point repeating the
    # set lookup per chunk, and tracing state is per-session-stable.
    tracing_active = is_tracing(session_id)

    async for chunk in chunks:
        if not started:
            yield MessageStart()
            started = True

        # Last-truthy-wins: an empty/missing ``model`` on one chunk must not
        # clobber a real value already captured from an earlier one.
        if chunk_model := getattr(chunk, "model", None):
            response_model = chunk_model

        if selected := _selected_provider_from_metadata(chunk):
            provider_name = selected

        # Capture usage from any chunk that carries it. The OpenAI canonical
        # ``include_usage`` shape is "empty choices + populated usage on a final
        # chunk", but real OpenRouter responses (verified live against
        # deepseek-chat-v3.1, deepseek-v3.1-terminus, gemini-2.5-flash,
        # gemini-2.5-flash-lite-preview, with and without reasoning) deliver
        # ``usage`` on the SAME chunk as the last delta + ``finish_reason``.
        # Capturing only inside ``if choice is None`` silently dropped every
        # usage chunk on OpenRouter — see the downstream ledger-gap
        # incident (zero ``chat_session`` rows in usage_ledger after deploy).
        # Last-wins semantics are correct here: any chunk that re-publishes a
        # usage object is meant to supersede the prior value.
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            final_usage = _usage_from_openai(usage)

        choice = chunk.choices[0] if chunk.choices else None
        if choice is None:
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
            # Stream-trace checkpoint: ``delta.content`` as the openai SDK
            # just delivered it. Diff against a downstream consumer's ``adapter_in`` and
            # ``adapter_out_cleaned`` to localize chat truncation bugs.
            if tracing_active:
                trace_delta(session_id, "translator_in", text, extra={"model": model})
            yield TextDelta(delta=text, block_index=0)

        # Tool calls — streamed as partial JSON across multiple chunks.
        if tool_calls := getattr(delta, "tool_calls", None):
            for ev in _process_tool_call_deltas(tool_calls, pending_tools, name_codec=name_codec):
                yield ev

        if choice.finish_reason:
            finish_reason_raw = choice.finish_reason

    # End of stream — flush completed tool calls only if finish_reason confirms completion.
    # Tool calls only "complete" when the stream's finish_reason is "tool_calls".
    # If a stream terminates abnormally (None/"length"/"stop" with pending tools),
    # we drop the partial tool calls rather than risk executing with empty args,
    # and emit one ErrorEvent per dropped call so the model is told the action
    # did not happen instead of inferring that it silently succeeded.
    if finish_reason_raw == "tool_calls":
        for ev in _flush_pending_tools(pending_tools, name_codec=name_codec):
            yield ev
    elif pending_tools:
        # The drop itself is deliberate (see above); the SILENCE was not. Without
        # this line the discard is indistinguishable from the model never having
        # called a tool — StreamMux forwards only ``tool_call_complete``, so no
        # downstream layer sees the start either. Never log ``args_buf`` content:
        # argument buffers carry user text and would sidestep the PII firewall.
        logger.warning(
            "openrouter.pending_tool_calls_dropped",
            finish_reason=finish_reason_raw,
            dropped_count=len(pending_tools),
            tools=[
                {
                    "name": _decode_name(slot["name"], name_codec),
                    "has_id": slot["id"] is not None,
                    "args_buf_len": len(slot["args_buf"]),
                }
                for slot in pending_tools.values()
            ],
            model=model,
            session_id=session_id,
        )
        # A log is for us; the ErrorEvent is for the model. A dropped call that
        # produces no event at all reads, from the model's side, as an action it
        # requested simply evaporating — so it narrates the effect as done and
        # moves on. One event per dropped slot, each naming the call so a
        # consumer can pair it with the ToolCallStart it already forwarded.
        for idx, slot in pending_tools.items():
            yield ErrorEvent(
                code=INCOMPLETE_TOOL_CALL_CODE,
                message=_incomplete_call_message(
                    tool_name=_decode_name(slot["name"], name_codec) or _UNNAMED_TOOL,
                    call_id=_fallback_call_id(slot, idx),
                    finish_reason=finish_reason_raw,
                ),
                recoverable=True,
            )

    if final_usage is not None:
        yield UsageEvent(usage=final_usage, model=response_model, provider_name=provider_name)
    yield MessageComplete(finish_reason=parse_finish_reason(finish_reason_raw))


def _process_tool_call_deltas(
    tool_calls: Any,
    pending_tools: dict[int, dict[str, Any]],
    *,
    name_codec: ToolNameCodec | None = None,
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
            tool_name = name_codec.decode(fn.name) if name_codec is not None else fn.name
            yield ToolCallStart(call_id=_fallback_call_id(slot, idx), tool_name=tool_name)
        if fn.arguments:
            slot["args_buf"] += fn.arguments
            yield ToolCallDelta(
                call_id=_fallback_call_id(slot, idx),
                arguments_delta=fn.arguments,
            )


def _flush_pending_tools(
    pending_tools: dict[int, dict[str, Any]],
    *,
    name_codec: ToolNameCodec | None = None,
) -> Generator[ProviderEvent, None, None]:
    """Yield exactly one terminal event per accumulated tool call.

    A slot resolves to either a dispatchable ``ToolCallComplete`` or an
    ``ErrorEvent`` saying why it could not be dispatched. Nothing in between,
    and nothing guessed.
    """
    for idx, slot in pending_tools.items():
        call_id = _fallback_call_id(slot, idx)
        if slot["name"] is None:
            # Arguments may have arrived without the function-name delta ever
            # landing — a protocol violation somewhere in OpenRouter's backend
            # fan-out. Nothing is dispatchable, so skipping is the only option;
            # the log is how we find out whether it ever happens. No
            # ``finish_reason`` field: this flush only runs on "tool_calls".
            # No ToolCallStart was emitted for this slot either (that fires only
            # when a name arrives), so mux-level accounting stays consistent.
            logger.warning(
                "openrouter.nameless_tool_slot_skipped",
                has_id=slot["id"] is not None,
                args_buf_len=len(slot["args_buf"]),
                args_buf_nonempty=bool(slot["args_buf"]),
            )
            yield ErrorEvent(
                code=INCOMPLETE_TOOL_CALL_CODE,
                message=_incomplete_call_message(
                    tool_name=_UNNAMED_TOOL, call_id=call_id, finish_reason="tool_calls"
                ),
                recoverable=True,
            )
            continue
        tool_name: str = name_codec.decode(slot["name"]) if name_codec is not None else slot["name"]
        try:
            parsed: Any = parse_tool_call_arguments(slot["args_buf"] or "")
        except json.JSONDecodeError:
            parsed = None
        if not isinstance(parsed, dict):
            # Unparseable even after json_repair (or parsed to something that is
            # not an argument object at all). It is tempting to emit the call
            # with ``arguments={}`` and let the tool sort it out — do not. For a
            # tool like "delete everything matching filter X" an empty filter is
            # the single most destructive reading of a corrupted intent, and the
            # model never learns its JSON was broken. Refuse the dispatch and
            # hand the model a retryable explanation instead.
            logger.warning(
                "openrouter.tool_args_unparseable_rejected",
                tool_name=tool_name,
                args_buf_len=len(slot["args_buf"]),
            )
            yield ErrorEvent(
                code=INVALID_TOOL_ARGUMENTS_CODE,
                message=_invalid_arguments_message(tool_name=tool_name, call_id=call_id),
                recoverable=True,
            )
            continue
        yield ToolCallComplete(
            call_id=call_id,
            tool_name=tool_name,
            arguments=dict(parsed),  # type: ignore[reportUnknownArgumentType]
        )


def _usage_from_openai(u: Any) -> Usage:
    cached = 0
    details = getattr(u, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    # ``completion_tokens_details.reasoning_tokens`` is the OpenAI-compatible
    # path OpenRouter's usage block actually reports reasoning tokens under
    # (OpenRouter OpenAPI spec: ``ChatUsage.completion_tokens_details
    # .reasoning_tokens``; mirrors openai SDK's own
    # ``CompletionUsage.completion_tokens_details.reasoning_tokens``, which is
    # the type this object is when it comes from the real client). This used
    # to go nowhere: neither a run of ``thinking_delta`` events during the
    # stream nor this usage figure ever reached ``Usage.thinking_tokens``, so
    # it silently read 0 on every turn regardless of how much the model
    # thought.
    #
    # ``thinking_tokens`` stays a plain ``int`` (not ``Optional``), unlike
    # ``model``/``provider_name`` above: 0 is the honest value both when a
    # non-reasoning model genuinely used no reasoning tokens *and* when the
    # provider omits the details block because reasoning doesn't apply — in
    # neither case is there a wrong-looking-right number to manufacture by
    # defaulting to 0, so there's nothing here that ``None`` would express
    # more truthfully.
    thinking = 0
    completion_details = getattr(u, "completion_tokens_details", None)
    if completion_details is not None:
        thinking = getattr(completion_details, "reasoning_tokens", 0) or 0
    return Usage(
        input_tokens=getattr(u, "prompt_tokens", 0),
        output_tokens=getattr(u, "completion_tokens", 0),
        cached_input_tokens=cached,
        thinking_tokens=thinking,
    )


def _selected_provider_from_metadata(chunk: Any) -> str | None:
    """Extract the serving upstream provider from OpenRouter's opt-in routing metadata.

    Reads ``chunk.openrouter_metadata.endpoints.available[]`` for the entry
    marked ``selected`` and returns its ``provider``. The metadata is an
    OpenRouter extension the openai SDK doesn't model — its ``BaseModel`` has
    ``extra="allow"`` (verified: ``openai/_models.py``'s ``construct()``
    stores unmodelled keys as plain parsed JSON, not typed sub-objects), so
    this walks the structure duck-typed against either a ``dict`` (the
    realistic shape) or an object with matching attributes, and returns
    ``None`` on any shape it doesn't recognise rather than guessing.
    """

    def _get(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            d = cast("dict[str, Any]", obj)
            return d.get(key)
        return getattr(obj, key, None)

    meta = getattr(chunk, "openrouter_metadata", None)
    if meta is None:
        return None
    endpoints = _get(meta, "endpoints")
    if endpoints is None:
        return None
    available: list[Any] = _get(endpoints, "available") or []
    for endpoint in available:
        if _get(endpoint, "selected"):
            provider = _get(endpoint, "provider")
            return str(provider) if provider else None
    return None
