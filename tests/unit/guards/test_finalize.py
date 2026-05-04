import pytest

from agentkit._content import TextBlock, ToolUseBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.guards.finalize import RuleBasedFinalizeValidator
from agentkit.loop.context import TurnContext
from agentkit.tools.spec import ToolCall


def _user(text: str) -> Message:
    from datetime import UTC, datetime

    return Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=MessageRole.USER,
        content=[TextBlock(text=text)],
        created_at=datetime.now(UTC),
    )


def _assistant_with_tool(name: str) -> Message:
    from datetime import UTC, datetime

    return Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=MessageRole.ASSISTANT,
        content=[ToolUseBlock(id="c1", name=name, arguments={})],
        created_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_accept_when_action_request_with_matching_tool():
    v = RuleBasedFinalizeValidator()
    ctx = TurnContext.empty()
    ctx.add_message(_user("turn off the heat pump"))
    ctx.add_message(_assistant_with_tool("ampaera.devices.control"))
    finalize = ToolCall(id="c1", name="kit.finalize", arguments={"reason": "device off"})
    verdict = await v.validate(finalize, ctx)
    assert verdict.accept is True


@pytest.mark.asyncio
async def test_reject_action_request_without_any_tool_call():
    v = RuleBasedFinalizeValidator()
    ctx = TurnContext.empty()
    ctx.add_message(_user("turn off the heat pump"))
    finalize = ToolCall(id="c1", name="kit.finalize", arguments={"reason": "done"})
    verdict = await v.validate(finalize, ctx)
    assert verdict.accept is False
    assert verdict.feedback


@pytest.mark.asyncio
async def test_accept_for_pure_question():
    v = RuleBasedFinalizeValidator()
    ctx = TurnContext.empty()
    ctx.add_message(_user("what was last week's average price?"))
    finalize = ToolCall(id="c1", name="kit.finalize", arguments={"reason": "answered"})
    verdict = await v.validate(finalize, ctx)
    assert verdict.accept is True
