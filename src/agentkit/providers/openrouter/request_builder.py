"""Build OpenRouter (OpenAI-Chat-Completions) request payloads."""

import json
from typing import Any

from agentkit._content import ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock
from agentkit._messages import Message, MessageRole
from agentkit.providers.base import NamedToolChoice, ProviderRequest, RoutingPreferences
from agentkit.providers.caching import compute_breakpoints
from agentkit.providers.openrouter.model_quirks import requires_cache_blocks
from agentkit.providers.openrouter.tool_name_codec import ToolNameCodec
from agentkit.providers.openrouter.tool_translator import to_openai_tool


def build_openrouter_request(
    req: ProviderRequest, *, name_codec: ToolNameCodec | None = None
) -> dict[str, Any]:
    name_codec = name_codec or ToolNameCodec.from_request(req)
    use_blocks = requires_cache_blocks(req.model)
    bp = compute_breakpoints(system=req.system, tools=req.tools, messages=req.messages)

    messages_payload: list[dict[str, Any]] = []

    # System: collapse multiple SystemBlocks into one system message.
    if req.system:
        if use_blocks:
            cache_mark = {"cache_control": {"type": "ephemeral"}}
            content: str | list[dict[str, Any]] = [
                {"type": "text", "text": b.text}
                | (cache_mark if (bp.cache_system and b.cache) else {})
                for b in req.system
            ]
        else:
            content = "\n\n".join(b.text for b in req.system)
        messages_payload.append({"role": "system", "content": content})

    # History.
    for i, msg in enumerate(req.messages):
        cache_this = bp.history_cache_index > 0 and i == bp.history_cache_index - 1 and use_blocks
        messages_payload.extend(
            _serialise_message(
                msg,
                attach_cache=cache_this,
                use_blocks=use_blocks,
                name_codec=name_codec,
            )
        )

    payload: dict[str, Any] = {
        "model": req.model,
        "messages": messages_payload,
        "max_tokens": req.max_tokens,
        "stream": True,
    }
    if req.tools:
        payload["tools"] = [to_openai_tool(t, name_codec=name_codec) for t in req.tools]
        tc = _to_openai_tool_choice(req.tool_choice, name_codec=name_codec)
        if tc is not None:
            payload["tool_choice"] = tc
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.stop_when and req.stop_when.stop_sequences:
        payload["stop"] = req.stop_when.stop_sequences
    if req.metadata:
        payload["metadata"] = req.metadata
    if req.routing is not None:
        payload["provider"] = _build_provider_prefs(req.routing)
    return payload


def _build_provider_prefs(routing: RoutingPreferences) -> dict[str, Any]:
    """OpenRouter ``provider`` preferences from routing (doc 01 §7).

    ``allow_fallbacks: false`` is the fail-closed gate: OpenRouter returns an
    error rather than routing to a non-compliant provider, which the firewall
    decorator catches and turns into ``ZdrRouteUnavailable``.
    """
    prefs: dict[str, Any] = {
        "zdr": routing.zdr,
        "data_collection": routing.data_collection,
        "allow_fallbacks": routing.allow_fallbacks,
    }
    if routing.eu_only and routing.only:
        prefs["only"] = routing.only
    return prefs


def _to_openai_tool_choice(
    choice: str | NamedToolChoice, *, name_codec: ToolNameCodec | None = None
) -> str | dict[str, Any] | None:
    """Translate a ProviderRequest.tool_choice into OpenAI/OpenRouter's wire shape.

    OpenAI Chat Completions accepts:
      - ``"auto"`` (default behaviour when tools are present)
      - ``"none"`` (forbid tool calls)
      - ``"required"`` (must call some tool)
      - ``{"type": "function", "function": {"name": "<name>"}}``
    Returns ``None`` for ``"auto"`` so we don't bloat the payload with the
    provider's default.
    """
    if isinstance(choice, NamedToolChoice):
        name = name_codec.encode(choice.name) if name_codec is not None else choice.name
        return {"type": "function", "function": {"name": name}}
    if choice == "required":
        return "required"
    if choice == "none":
        return "none"
    return None


def _serialise_message(
    msg: Message,
    *,
    attach_cache: bool,
    use_blocks: bool,
    name_codec: ToolNameCodec | None = None,
) -> list[dict[str, Any]]:
    """Most messages map to one OpenAI message; tool results map to one per call."""
    if msg.role == MessageRole.USER:
        content = _serialise_user_content(msg, attach_cache=attach_cache, use_blocks=use_blocks)
        return [{"role": "user", "content": content}]

    if msg.role == MessageRole.ASSISTANT:
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for b in msg.content:
            if isinstance(b, TextBlock):
                text_parts.append(b.text)
            elif isinstance(b, ToolUseBlock):
                tool_calls.append(
                    {
                        "id": b.id,
                        "type": "function",
                        "function": {
                            "name": name_codec.encode(b.name) if name_codec is not None else b.name,
                            "arguments": json.dumps(b.arguments),
                        },
                    }
                )
        if not text_parts and not tool_calls:
            # OpenAI rejects assistant messages without content or tool_calls.
            return []
        out: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts) or None}
        if tool_calls:
            out["tool_calls"] = tool_calls
        return [out]

    if msg.role == MessageRole.TOOL:
        # Each ToolResultBlock becomes a separate "tool" message.
        out_msgs: list[dict[str, Any]] = []
        for b in msg.content:
            if isinstance(b, ToolResultBlock):
                # Concatenate text blocks inside the tool result.
                parts = [t.text for t in b.content if isinstance(t, TextBlock)]
                out_msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": b.tool_use_id,
                        "content": "\n".join(parts),
                    }
                )
        return out_msgs

    if msg.role == MessageRole.SYSTEM:
        text = "\n".join(b.text for b in msg.content if isinstance(b, TextBlock))
        return [{"role": "system", "content": text}]

    return []


def _serialise_user_content(
    msg: Message, *, attach_cache: bool, use_blocks: bool
) -> str | list[dict[str, Any]]:
    has_image = any(isinstance(b, ImageBlock) for b in msg.content)
    if not has_image and not use_blocks:
        return "\n".join(b.text for b in msg.content if isinstance(b, TextBlock))
    parts: list[dict[str, Any]] = []
    for b in msg.content:
        if isinstance(b, TextBlock):
            entry: dict[str, Any] = {"type": "text", "text": b.text}
            if attach_cache:
                entry["cache_control"] = {"type": "ephemeral"}
            parts.append(entry)
        elif isinstance(b, ImageBlock):
            url = b.url if b.source == "url" else f"data:{b.media_type};base64,{b.data}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts
