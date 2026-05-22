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
    ctx.finalize_args = {
        "status": "done",
        "intent_kind": "answer",
        "actions_performed": [],
        "answer_evidence": "general_knowledge",
    }
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
    # The correction must reach the model: a user-role message carrying the
    # validator feedback is appended to history before re-streaming.
    last = ctx.history[-1]
    assert last.role is MessageRole.USER
    block = last.content[0]
    assert isinstance(block, TextBlock)
    assert "rejected" in block.text
    assert ctx.finalize_called is False


@pytest.mark.asyncio
async def test_finalize_check_reprompts_when_finalize_missing():
    """No finalize call -> re-prompt the model once to finalize properly."""
    ctx = TurnContext.empty()
    ctx.add_message(_user("does the article have a hero image?"))
    deps = {"finalize_validator": StructuralFinalizeValidator()}
    next_ = await handle_finalize_check(ctx, deps)
    assert next_ is Phase.CONTEXT_BUILD
    assert ctx.metadata["missing_finalize_reprompts"] == 1
    last = ctx.history[-1]
    assert last.role is MessageRole.USER
    block = last.content[0]
    assert isinstance(block, TextBlock)
    assert "finalize_response" in block.text
    assert "clarify" in block.text


@pytest.mark.asyncio
async def test_finalize_check_ends_turn_after_reprompt_budget_spent():
    """Re-prompt budget exhausted -> let the turn end so the consumer synthesizes."""
    ctx = TurnContext.empty()
    ctx.add_message(_user("hi"))
    ctx.metadata["missing_finalize_reprompts"] = 1
    deps = {
        "finalize_validator": StructuralFinalizeValidator(),
        "max_missing_finalize_reprompts": 1,
    }
    next_ = await handle_finalize_check(ctx, deps)
    assert next_ is Phase.MEMORY_EXTRACT
    assert ctx.metadata["finalize_missing"] is True


@pytest.mark.asyncio
async def test_finalize_check_passes_through_when_no_validator():
    """No finalize validator configured -> consumer opted out; accept and end."""
    ctx = TurnContext.empty()
    ctx.add_message(_user("hi"))
    next_ = await handle_finalize_check(ctx, {})
    assert next_ is Phase.MEMORY_EXTRACT


@pytest.mark.asyncio
async def test_memory_extract_returns_turn_ended():
    ctx = TurnContext.empty()
    next_ = await handle_memory_extract(ctx, {})
    assert next_ is Phase.TURN_ENDED
