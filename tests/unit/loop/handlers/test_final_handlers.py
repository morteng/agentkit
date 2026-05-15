import pytest

from agentkit._content import TextBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.guards.finalize import StructuralFinalizeValidator
from agentkit.loop.context import TurnContext
from agentkit.loop.handlers.finalize_check import handle_finalize_check
from agentkit.loop.handlers.memory_extract import handle_memory_extract
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
async def test_finalize_check_accepts_valid_finalize():
    ctx = TurnContext.empty()
    ctx.add_message(_user("what time is it?"))
    ctx.finalize_called = True
    ctx.finalize_args = {"status": "done", "intent_kind": "answer", "actions_performed": [], "answer_evidence": "general_knowledge"}
    deps = {"finalize_validator": StructuralFinalizeValidator()}
    next_ = await handle_finalize_check(ctx, deps)
    assert next_ is Phase.MEMORY_EXTRACT


@pytest.mark.asyncio
async def test_finalize_check_rejects_invalid_finalize_with_retry():
    ctx = TurnContext.empty()
    ctx.add_message(_user("turn off the heat pump"))
    ctx.finalize_called = True
    # intent_kind=action + empty actions_performed is a structural violation (empty_on_done)
    ctx.finalize_args = {"status": "done", "intent_kind": "action", "actions_performed": []}
    ctx.metadata["finalize_retries"] = 0
    deps = {"finalize_validator": StructuralFinalizeValidator(), "max_finalize_retries": 2}
    next_ = await handle_finalize_check(ctx, deps)
    assert next_ is Phase.CONTEXT_BUILD
    assert "finalize_correction" in ctx.metadata


@pytest.mark.asyncio
async def test_finalize_check_passes_through_when_no_finalize_called():
    """No finalize call yet — accept and let the conversation end naturally."""
    ctx = TurnContext.empty()
    ctx.add_message(_user("hi"))
    deps = {"finalize_validator": StructuralFinalizeValidator()}
    next_ = await handle_finalize_check(ctx, deps)
    assert next_ is Phase.MEMORY_EXTRACT


@pytest.mark.asyncio
async def test_memory_extract_returns_turn_ended():
    ctx = TurnContext.empty()
    next_ = await handle_memory_extract(ctx, {})
    assert next_ is Phase.TURN_ENDED
