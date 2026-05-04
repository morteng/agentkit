import pytest

from agentkit._content import TextBlock, ToolUseBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.guards.success_claim import RegexSuccessClaimGuard
from agentkit.loop.context import TurnContext


def _assistant_text(text: str) -> Message:
    from datetime import UTC, datetime

    return Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=MessageRole.ASSISTANT,
        content=[TextBlock(text=text)],
        created_at=datetime.now(UTC),
    )


def _tool_call_msg() -> Message:
    from datetime import UTC, datetime

    return Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=MessageRole.ASSISTANT,
        content=[ToolUseBlock(id="c1", name="ampaera.rules.create", arguments={})],
        created_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_flag_when_claim_without_write_tool():
    g = RegexSuccessClaimGuard()
    ctx = TurnContext.empty()
    text = "I've created the rule for you."
    verdict = await g.check(text, ctx)
    assert verdict.flag is True


@pytest.mark.asyncio
async def test_no_flag_when_claim_backed_by_write_tool():
    g = RegexSuccessClaimGuard()
    ctx = TurnContext.empty()
    ctx.add_message(_tool_call_msg())
    verdict = await g.check("I've created the rule for you.", ctx)
    assert verdict.flag is False


@pytest.mark.asyncio
async def test_no_flag_for_neutral_text():
    g = RegexSuccessClaimGuard()
    ctx = TurnContext.empty()
    verdict = await g.check("Last week the average price was 1.2 NOK/kWh.", ctx)
    assert verdict.flag is False
