from datetime import UTC, datetime

from agentkit._content import TextBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.providers.base import ProviderRequest, SystemBlock, ToolDefinition
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
