"""Build OpenRouter (OpenAI-Chat-Completions) request payloads."""

import json
from typing import Any

from agentkit._content import ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock
from agentkit._messages import Message, MessageRole
from agentkit.providers.base import ProviderRequest
from agentkit.providers.caching import compute_breakpoints
from agentkit.providers.openrouter.model_quirks import requires_cache_blocks
from agentkit.providers.openrouter.tool_translator import to_openai_tool


def build_openrouter_request(req: ProviderRequest) -> dict[str, Any]:
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
            _serialise_message(msg, attach_cache=cache_this, use_blocks=use_blocks)
        )

    payload: dict[str, Any] = {
        "model": req.model,
        "messages": messages_payload,
        "max_tokens": req.max_tokens,
        "stream": True,
    }
    if req.tools:
        payload["tools"] = [to_openai_tool(t) for t in req.tools]
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.stop_when and req.stop_when.stop_sequences:
        payload["stop"] = req.stop_when.stop_sequences
    if req.metadata:
        payload["metadata"] = req.metadata
    return payload


def _serialise_message(
    msg: Message, *, attach_cache: bool, use_blocks: bool
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
                        "function": {"name": b.name, "arguments": json.dumps(b.arguments)},
                    }
                )
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
