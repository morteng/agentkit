from datetime import UTC, datetime

from agentkit._content import TextBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.providers.base import (
    NamedToolChoice,
    ProviderRequest,
    SystemBlock,
    ToolDefinition,
)
from agentkit.providers.openrouter.request_builder import build_openrouter_request


def _msg(role: MessageRole, text: str) -> Message:
    return Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=role,
        content=[TextBlock(text=text)],
        created_at=datetime.now(UTC),
    )


def test_anthropic_model_emits_content_blocks_with_cache_control():
    req = ProviderRequest(
        model="anthropic/claude-sonnet-4-6",
        system=[SystemBlock(text="hi")],
        messages=[_msg(MessageRole.USER, "msg")],
        max_tokens=4096,
    )
    payload = build_openrouter_request(req)
    sys_msg = payload["messages"][0]
    assert sys_msg["role"] == "system"
    assert isinstance(sys_msg["content"], list)
    assert sys_msg["content"][0]["type"] == "text"
    assert sys_msg["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_openai_model_emits_plain_string_content():
    req = ProviderRequest(
        model="openai/gpt-5",
        system=[SystemBlock(text="hi")],
        messages=[_msg(MessageRole.USER, "msg")],
        max_tokens=4096,
    )
    payload = build_openrouter_request(req)
    sys_msg = payload["messages"][0]
    assert sys_msg["role"] == "system"
    assert sys_msg["content"] == "hi"


def test_tools_use_function_envelope():
    req = ProviderRequest(
        model="openai/gpt-5",
        messages=[_msg(MessageRole.USER, "msg")],
        tools=[ToolDefinition(name="x", description="d", parameters={"type": "object"})],
        max_tokens=4096,
    )
    payload = build_openrouter_request(req)
    assert payload["tools"][0]["type"] == "function"
    assert payload["tools"][0]["function"]["name"] == "x"


def test_assistant_message_with_only_thinking_block_is_skipped():
    """Empty assistant messages (no text, no tool_calls) must not be sent —
    OpenAI rejects them. This protects against the case where history replay
    encounters an assistant message containing only ThinkingBlock content."""
    from agentkit._content import ThinkingBlock

    msg = Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=MessageRole.ASSISTANT,
        content=[ThinkingBlock(text="reasoning…")],
        created_at=datetime.now(UTC),
    )
    req = ProviderRequest(model="openai/gpt-5", messages=[msg], max_tokens=100)
    payload = build_openrouter_request(req)
    # The empty assistant message should be omitted entirely.
    assert payload["messages"] == []


def test_assistant_message_with_tool_calls_only_uses_none_content():
    """When assistant has tool_calls but no text, content must be None
    (OpenAI accepts this). The message should still be emitted."""
    from agentkit._content import ToolUseBlock

    msg = Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=MessageRole.ASSISTANT,
        content=[ToolUseBlock(id="call_1", name="add", arguments={"a": 1, "b": 2})],
        created_at=datetime.now(UTC),
    )
    req = ProviderRequest(model="openai/gpt-5", messages=[msg], max_tokens=100)
    payload = build_openrouter_request(req)
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "assistant"
    assert payload["messages"][0]["content"] is None
    assert payload["messages"][0]["tool_calls"][0]["function"]["name"] == "add"


from agentkit.providers.base import ToolChoiceMode  # noqa: E402


def _req_with_tool(
    tool_choice: ToolChoiceMode | NamedToolChoice = "auto",
) -> ProviderRequest:
    return ProviderRequest(
        model="openai/gpt-5",
        messages=[_msg(MessageRole.USER, "hi")],
        tools=[ToolDefinition(name="finalize", description="d", parameters={"type": "object"})],
        tool_choice=tool_choice,
        max_tokens=100,
    )


def test_tool_choice_auto_omits_field():
    """`auto` is OpenAI's default when tools are present — keep payload minimal."""
    payload = build_openrouter_request(_req_with_tool("auto"))
    assert "tool_choice" not in payload


def test_tool_choice_none_emits_string_none():
    payload = build_openrouter_request(_req_with_tool("none"))
    assert payload["tool_choice"] == "none"


def test_tool_choice_required_emits_string_required():
    payload = build_openrouter_request(_req_with_tool("required"))
    assert payload["tool_choice"] == "required"


def test_tool_choice_named_emits_function_envelope():
    payload = build_openrouter_request(_req_with_tool(NamedToolChoice(name="finalize")))
    assert payload["tool_choice"] == {
        "type": "function",
        "function": {"name": "finalize"},
    }


def test_tool_choice_omitted_when_no_tools():
    """Without tools, tool_choice has nothing to constrain — skip it."""
    req = ProviderRequest(
        model="openai/gpt-5",
        messages=[_msg(MessageRole.USER, "hi")],
        tool_choice="required",
        max_tokens=100,
    )
    payload = build_openrouter_request(req)
    assert "tool_choice" not in payload
