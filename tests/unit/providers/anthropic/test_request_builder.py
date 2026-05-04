from datetime import UTC, datetime

from agentkit._content import TextBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.providers.anthropic.request_builder import build_anthropic_request
from agentkit.providers.base import ProviderRequest, SystemBlock, ToolDefinition


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
