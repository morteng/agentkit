import pytest

from agentkit._content import TextBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.guards.intent import (
    ContentBlocklistCheck,
    DefaultIntentGate,
    InMemoryRateLimitCheck,
    MaxMessageLengthCheck,
)
from agentkit.loop.context import TurnContext


def _user(text: str) -> Message:
    from datetime import UTC, datetime

    return Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=MessageRole.USER,
        content=[TextBlock(text=text)],
        created_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_no_checks_allows_everything():
    gate = DefaultIntentGate(checks=[])
    ctx = TurnContext.empty()
    ctx.add_message(_user("hi"))
    decision = await gate.evaluate(ctx)
    assert decision.allow is True


@pytest.mark.asyncio
async def test_max_length_rejects_oversize_message():
    gate = DefaultIntentGate(checks=[MaxMessageLengthCheck(max_chars=10)])
    ctx = TurnContext.empty()
    ctx.add_message(_user("x" * 100))
    decision = await gate.evaluate(ctx)
    assert decision.allow is False
    assert "max" in (decision.reason or "").lower()


@pytest.mark.asyncio
async def test_blocklist_rejects_match():
    gate = DefaultIntentGate(checks=[ContentBlocklistCheck(patterns=[r"forbidden"])])
    ctx = TurnContext.empty()
    ctx.add_message(_user("this contains a forbidden phrase"))
    decision = await gate.evaluate(ctx)
    assert decision.allow is False


@pytest.mark.asyncio
async def test_rate_limit_after_threshold():
    check = InMemoryRateLimitCheck(turns_per_minute=2)
    gate = DefaultIntentGate(checks=[check])
    ctx = TurnContext.empty()
    ctx.add_message(_user("hi"))
    ctx.metadata["owner"] = "u1"
    assert (await gate.evaluate(ctx)).allow
    assert (await gate.evaluate(ctx)).allow
    assert not (await gate.evaluate(ctx)).allow
