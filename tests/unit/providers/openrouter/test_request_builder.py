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
