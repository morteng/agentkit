"""Helpers for constructing isolated child contexts.

A child subagent gets its own session_id, turn_id, history, and event queue —
its operations don't leak into the parent's view.
"""

from agentkit._content import TextBlock
from agentkit._ids import MessageId, SessionId, TurnId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.loop.context import TurnContext


def fresh_child_context(parent: TurnContext, *, prompt: str) -> TurnContext:
    from datetime import UTC, datetime

    child_session = new_id(SessionId)
    user_msg = Message(
        id=new_id(MessageId),
        session_id=child_session,
        role=MessageRole.USER,
        content=[TextBlock(text=prompt)],
        created_at=datetime.now(UTC),
    )
    child = TurnContext(
        session_id=child_session,
        turn_id=new_id(TurnId),
        call_id="",
        history=[user_msg],
        clock=parent.clock,
        memory_store=parent.memory_store,
        memory_scope=parent.memory_scope,
    )
    return child
