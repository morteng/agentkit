from datetime import UTC, datetime
from typing import cast

from agentkit._content import TextBlock, ToolUseBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageMetadata, MessageRole, Usage


def test_message_round_trips_json():
    m = Message(
        id=cast(MessageId, new_id(MessageId)),
        session_id=cast(SessionId, new_id(SessionId)),
        role=MessageRole.ASSISTANT,
        content=[TextBlock(text="hi")],
        metadata=MessageMetadata(provider="anthropic", model="claude-sonnet-4-6"),
        created_at=datetime.now(UTC),
    )
    j = m.model_dump_json()
    parsed = Message.model_validate_json(j)
    assert parsed == m


def test_message_holds_mixed_content():
    m = Message(
        id=cast(MessageId, new_id(MessageId)),
        session_id=cast(SessionId, new_id(SessionId)),
        role=MessageRole.ASSISTANT,
        content=[TextBlock(text="here"), ToolUseBlock(id="c1", name="t", arguments={})],
        metadata=MessageMetadata(),
        created_at=datetime.now(UTC),
    )
    assert len(m.content) == 2


def test_usage_defaults_to_zero():
    u = Usage()
    assert u.input_tokens == 0
    assert u.output_tokens == 0
    assert u.cached_input_tokens == 0
