import asyncio
from datetime import UTC, datetime

import pytest

from agentkit._content import TextBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.events import MessageCompleted
from agentkit.events import TextDelta as PubTextDelta
from agentkit.events import ToolCallStarted as PubToolCallStarted
from agentkit.loop.context import TurnContext
from agentkit.loop.handlers.streaming import handle_streaming
from agentkit.loop.message_builder import MessageBuilder
from agentkit.loop.phase import Phase
from agentkit.providers.fakes import FakeProvider
from agentkit.tools.registry import ToolRegistry


def _user(text: str) -> Message:
    return Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=MessageRole.USER,
        content=[TextBlock(text=text)],
        created_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_streaming_with_text_response_returns_finalize_check():
    """Plain text response transitions FINALIZE_CHECK because the stream ended without a tool call.

    NB: in v0.1 we treat finish_reason == 'end_turn' WITH no kit.finalize call as
    "let finalize_check decide" — the validator may reject and force another
    iteration via CONTEXT_BUILD.
    """
    provider = FakeProvider().script(FakeProvider.text("hello"))
    queue: asyncio.Queue = asyncio.Queue()
    ctx = TurnContext.empty()
    ctx.add_message(_user("hi"))
    ctx.event_queue = queue

    builder = MessageBuilder(model="m", max_tokens=128)
    deps = {
        "provider": provider,
        "message_builder": builder,
        "registry": ToolRegistry(),
        "system_blocks": [],
        "success_claim": None,
    }
    next_ = await handle_streaming(ctx, deps)
    assert next_ is Phase.FINALIZE_CHECK

    # Drain emitted events.
    emitted = []
    while not queue.empty():
        emitted.append(queue.get_nowait())
    assert any(isinstance(e, PubTextDelta) for e in emitted)
    assert any(isinstance(e, MessageCompleted) for e in emitted)


@pytest.mark.asyncio
async def test_streaming_with_tool_call_returns_tool_phase():
    provider = FakeProvider().script(FakeProvider.tool_call("kit.x", {"a": 1}))
    queue: asyncio.Queue = asyncio.Queue()
    ctx = TurnContext.empty()
    ctx.add_message(_user("do x"))
    ctx.event_queue = queue

    deps = {
        "provider": provider,
        "message_builder": MessageBuilder(model="m", max_tokens=128),
        "registry": ToolRegistry(),
        "system_blocks": [],
        "success_claim": None,
    }
    next_ = await handle_streaming(ctx, deps)
    assert next_ is Phase.TOOL_PHASE

    emitted = []
    while not queue.empty():
        emitted.append(queue.get_nowait())
    starts = [e for e in emitted if isinstance(e, PubToolCallStarted)]
    assert starts and starts[0].tool_name == "kit.x"
