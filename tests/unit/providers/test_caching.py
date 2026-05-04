from datetime import UTC, datetime

from agentkit._content import TextBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.providers.base import SystemBlock, ToolDefinition
from agentkit.providers.caching import compute_breakpoints


def _msg(role: MessageRole, text: str) -> Message:
    return Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=role,
        content=[TextBlock(text=text)],
        created_at=datetime.now(UTC),
    )


def test_breakpoints_cache_system_and_tools():
    bp = compute_breakpoints(
        system=[SystemBlock(text="x")],
        tools=[ToolDefinition(name="t", description="", parameters={})],
        messages=[_msg(MessageRole.USER, "hi"), _msg(MessageRole.ASSISTANT, "ok")],
    )
    assert bp.cache_system is True
    assert bp.cache_tools is True


def test_breakpoints_cache_history_except_last_two_turns():
    msgs = [_msg(MessageRole.USER, str(i)) for i in range(6)]
    bp = compute_breakpoints(system=[], tools=[], messages=msgs)
    # Six messages -> last two are not cached, first four are cacheable.
    assert bp.history_cache_index == 4


def test_breakpoints_with_short_history_no_cache():
    msgs = [_msg(MessageRole.USER, "hi")]
    bp = compute_breakpoints(system=[], tools=[], messages=msgs)
    assert bp.history_cache_index == 0
