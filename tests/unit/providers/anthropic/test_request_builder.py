from datetime import UTC, datetime

from agentkit._content import TextBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.providers.anthropic.request_builder import build_anthropic_request
from agentkit.providers.base import (
    NamedToolChoice,
    ProviderRequest,
    SystemBlock,
    ToolDefinition,
)


def _msg(role: MessageRole, text: str) -> Message:
    return Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=role,
        content=[TextBlock(text=text)],
        created_at=datetime.now(UTC),
    )


def test_system_blocks_become_list_with_cache_control():
    req = ProviderRequest(
        model="claude-sonnet-4-6",
        system=[SystemBlock(text="You are helpful.")],
        messages=[_msg(MessageRole.USER, "hi")],
        max_tokens=4096,
    )
    payload = build_anthropic_request(req)
    assert isinstance(payload["system"], list)
    assert payload["system"][0]["type"] == "text"
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_tools_translated_with_input_schema_key():
    req = ProviderRequest(
        model="claude-sonnet-4-6",
        messages=[_msg(MessageRole.USER, "hi")],
        tools=[ToolDefinition(name="x", description="d", parameters={"type": "object"})],
        max_tokens=4096,
    )
    payload = build_anthropic_request(req)
    assert payload["tools"][0]["name"] == "x"
    assert payload["tools"][0]["input_schema"] == {"type": "object"}


def test_history_cache_breakpoint_attached_after_first_n_messages():
    msgs = [_msg(MessageRole.USER, str(i)) for i in range(6)]
    req = ProviderRequest(model="m", messages=msgs, max_tokens=4096)
    payload = build_anthropic_request(req)
    # The first 4 messages are cacheable; expect cache_control at index 3.
    cached_indices = [
        i
        for i, m in enumerate(payload["messages"])
        if any(b.get("cache_control") for b in m["content"])
    ]
    assert 3 in cached_indices


from agentkit.providers.base import ToolChoiceMode  # noqa: E402


def _req_with_tool(
    tool_choice: ToolChoiceMode | NamedToolChoice = "auto",
) -> ProviderRequest:
    return ProviderRequest(
        model="claude-sonnet-4-6",
        messages=[_msg(MessageRole.USER, "hi")],
        tools=[ToolDefinition(name="finalize", description="d", parameters={"type": "object"})],
        tool_choice=tool_choice,
        max_tokens=100,
    )


def test_tool_choice_auto_omits_field():
    """`auto` is Anthropic's default — don't pollute the payload."""
    payload = build_anthropic_request(_req_with_tool("auto"))
    assert "tool_choice" not in payload


def test_tool_choice_none_emits_type_none():
    payload = build_anthropic_request(_req_with_tool("none"))
    assert payload["tool_choice"] == {"type": "none"}


def test_tool_choice_required_emits_type_any():
    """OpenAI's `required` maps to Anthropic's `any` — must call some tool."""
    payload = build_anthropic_request(_req_with_tool("required"))
    assert payload["tool_choice"] == {"type": "any"}


def test_tool_choice_named_emits_type_tool_with_name():
    payload = build_anthropic_request(_req_with_tool(NamedToolChoice(name="finalize")))
    assert payload["tool_choice"] == {"type": "tool", "name": "finalize"}


def test_tool_choice_omitted_when_no_tools():
    """Without tools, tool_choice has nothing to constrain — skip it."""
    req = ProviderRequest(
        model="claude-sonnet-4-6",
        messages=[_msg(MessageRole.USER, "hi")],
        tool_choice="required",
        max_tokens=100,
    )
    payload = build_anthropic_request(req)
    assert "tool_choice" not in payload
