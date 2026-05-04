import pytest

from agentkit._content import TextBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.guards.intent import DefaultIntentGate, MaxMessageLengthCheck
from agentkit.loop.context import TurnContext
from agentkit.loop.handlers.context_build import handle_context_build
from agentkit.loop.handlers.intent_gate import handle_intent_gate
from agentkit.loop.phase import Phase


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
async def test_intent_gate_passes_through_to_context_build():
    ctx = TurnContext.empty()
    ctx.add_message(_user("hi"))
    deps = {"intent_gate": DefaultIntentGate(checks=[])}
    next_ = await handle_intent_gate(ctx, deps)
    assert next_ is Phase.CONTEXT_BUILD


@pytest.mark.asyncio
async def test_intent_gate_rejection_routes_to_errored():
    ctx = TurnContext.empty()
    ctx.add_message(_user("x" * 1000))
    deps = {"intent_gate": DefaultIntentGate(checks=[MaxMessageLengthCheck(max_chars=10)])}
    next_ = await handle_intent_gate(ctx, deps)
    assert next_ is Phase.ERRORED


@pytest.mark.asyncio
async def test_context_build_returns_streaming():
    ctx = TurnContext.empty()
    deps = {}
    next_ = await handle_context_build(ctx, deps)
    assert next_ is Phase.STREAMING
