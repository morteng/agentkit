from datetime import UTC, datetime

import pytest

from agentkit._content import ToolUseBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.providers.base import NamedToolChoice, ProviderRequest, ToolDefinition
from agentkit.providers.openrouter.tool_name_codec import (
    UNKNOWN_INVALID_TOOL_NAME,
    ToolNameCodec,
    is_wire_safe_name,
)


def test_qualified_names_encode_and_round_trip():
    codec = ToolNameCodec.from_names(["acme.search", "kit.current_time"])

    assert codec.encode("acme.search") == "acme__search"
    assert codec.encode("kit.current_time") == "kit__current_time"
    assert codec.decode("acme__search") == "acme.search"
    assert codec.decode("kit__current_time") == "kit.current_time"
    assert all(is_wire_safe_name(name) for name in codec.wire_to_canonical)


def test_existing_safe_name_wins_collision_deterministically():
    codec = ToolNameCodec.from_names(["a.b", "a__b"])

    assert codec.encode("a__b") == "a__b"
    assert codec.encode("a.b") == "a__b_2"
    assert codec.decode("a__b_2") == "a.b"


def test_long_or_unicode_name_uses_safe_digest():
    canonical = "søketøy." + ("x" * 100)
    codec = ToolNameCodec.from_names([canonical])
    wire = codec.encode(canonical)

    assert is_wire_safe_name(wire)
    assert codec.decode(wire) == canonical


def test_malformed_provider_name_fails_closed_to_safe_sentinel():
    codec = ToolNameCodec.from_names(["acme.search"])

    assert codec.decode("acme.search") == UNKNOWN_INVALID_TOOL_NAME
    assert is_wire_safe_name(codec.decode("acme.search"))


def test_request_codec_includes_tool_choice_and_replayed_history():
    history = Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=MessageRole.ASSISTANT,
        content=[ToolUseBlock(id="call_1", name="kit.current_time", arguments={})],
        created_at=datetime.now(UTC),
    )
    request = ProviderRequest(
        model="openai/gpt-5.6-luna",
        messages=[history],
        tools=[ToolDefinition(name="acme.search", description="søk", parameters={})],
        tool_choice=NamedToolChoice(name="acme.search"),
    )
    codec = ToolNameCodec.from_request(request)

    assert codec.encode("kit.current_time") == "kit__current_time"
    assert codec.encode("acme.search") == "acme__search"


def test_empty_name_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        ToolNameCodec.from_names([""])
