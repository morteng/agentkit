"""Build Anthropic SDK request payloads from agentkit ProviderRequest."""

from typing import Any

from agentkit._content import (
    ImageBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from agentkit._messages import Message, MessageRole
from agentkit.providers.anthropic.tool_translator import to_anthropic_tool
from agentkit.providers.base import NamedToolChoice, ProviderRequest
from agentkit.providers.caching import compute_breakpoints


def build_anthropic_request(req: ProviderRequest) -> dict[str, Any]:
    bp = compute_breakpoints(system=req.system, tools=req.tools, messages=req.messages)

    # System blocks → list of text blocks; cache_control on each cacheable block.
    system_payload: list[dict[str, Any]] = []
    for block in req.system:
        entry: dict[str, Any] = {"type": "text", "text": block.text}
        if bp.cache_system and block.cache:
            entry["cache_control"] = {"type": "ephemeral"}
        system_payload.append(entry)

    # Messages → role + content blocks.
    messages_payload: list[dict[str, Any]] = []
    for i, msg in enumerate(req.messages):
        cache_this = bp.history_cache_index > 0 and i == bp.history_cache_index - 1
        messages_payload.append(
            {
                "role": _role(msg.role),
                "content": _serialise_content(msg, attach_cache=cache_this),
            }
        )

    payload: dict[str, Any] = {
        "model": req.model,
        "messages": messages_payload,
        "max_tokens": req.max_tokens,
    }
    if system_payload:
        payload["system"] = system_payload
    if req.tools:
        payload["tools"] = [to_anthropic_tool(t) for t in req.tools]
        tc = _to_anthropic_tool_choice(req.tool_choice)
        if tc is not None:
            payload["tool_choice"] = tc
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.thinking and req.thinking.enabled:
        payload["thinking"] = {
            "type": "enabled",
            "budget_tokens": req.thinking.budget_tokens or 4096,
        }
    if req.stop_when and req.stop_when.stop_sequences:
        payload["stop_sequences"] = req.stop_when.stop_sequences
    if req.metadata:
        payload["metadata"] = req.metadata
    return payload


def _to_anthropic_tool_choice(
    choice: str | NamedToolChoice,
) -> dict[str, Any] | None:
    """Translate a ProviderRequest.tool_choice into Anthropic's wire shape.

    Anthropic accepts:
      - ``{"type": "auto"}``  (default behaviour)
      - ``{"type": "none"}``  (forbid tool calls)
      - ``{"type": "any"}``   (must call some tool)
      - ``{"type": "tool", "name": "<name>"}``
    Returns ``None`` for ``"auto"`` so we don't bloat the payload with the
    provider's default.
    """
    if isinstance(choice, NamedToolChoice):
        return {"type": "tool", "name": choice.name}
    if choice == "required":
        return {"type": "any"}
    if choice == "none":
        return {"type": "none"}
    # "auto" — omit; matches Anthropic default and keeps payloads minimal.
    return None


def _role(role: MessageRole) -> str:
    # Anthropic accepts "user" and "assistant"; tool/system messages map differently.
    return "user" if role in (MessageRole.USER, MessageRole.TOOL) else "assistant"


def _serialise_block(b: Any) -> dict[str, Any] | None:
    """Serialise a single content block. Returns None for unrecognised types."""
    if isinstance(b, TextBlock):
        return {"type": "text", "text": b.text}
    if isinstance(b, ThinkingBlock):
        entry: dict[str, Any] = {"type": "thinking", "text": b.text}
        if b.signature:
            entry["signature"] = b.signature
        return entry
    if isinstance(b, ImageBlock):
        source: dict[str, Any]
        if b.source == "base64":
            source = {"type": "base64", "media_type": b.media_type, "data": b.data}
        else:
            source = {"type": "url", "url": b.url or ""}
        return {"type": "image", "source": source}
    if isinstance(b, ToolUseBlock):
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.arguments}
    if isinstance(b, ToolResultBlock):
        return _serialise_tool_result(b)
    return None


def _serialise_tool_result(b: ToolResultBlock) -> dict[str, Any]:
    """Serialise a ToolResultBlock; inner content limited to text and images."""
    inner_blocks: list[dict[str, Any]] = []
    for inner in b.content:
        if isinstance(inner, TextBlock):
            inner_blocks.append({"type": "text", "text": inner.text})
        elif isinstance(inner, ImageBlock):
            inner_blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": inner.media_type,
                        "data": inner.data,
                    },
                }
            )
        # Other content types inside tool results are silently skipped.
    return {
        "type": "tool_result",
        "tool_use_id": b.tool_use_id,
        "content": inner_blocks,
        "is_error": b.is_error,
    }


def _serialise_content(msg: Message, *, attach_cache: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for b in msg.content:
        serialised = _serialise_block(b)
        if serialised is not None:
            out.append(serialised)
    if attach_cache and out:
        out[-1] = {**out[-1], "cache_control": {"type": "ephemeral"}}
    return out
