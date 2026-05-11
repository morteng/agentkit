import asyncio

import pytest

from agentkit._content import TextBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.events import (
    MessageCompleted,
    PhaseChanged,
    TextDelta,
    TurnEnded,
    TurnEndReason,
)
from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.guards.finalize import StructuralFinalizeValidator
from agentkit.loop.context import TurnContext
from agentkit.loop.handlers.approval_wait import handle_approval_wait
from agentkit.loop.handlers.context_build import handle_context_build
from agentkit.loop.handlers.finalize_check import handle_finalize_check
from agentkit.loop.handlers.intent_gate import handle_intent_gate
from agentkit.loop.handlers.memory_extract import handle_memory_extract
from agentkit.loop.handlers.streaming import handle_streaming
from agentkit.loop.handlers.tool_executing import handle_tool_executing
from agentkit.loop.handlers.tool_phase import handle_tool_phase
from agentkit.loop.handlers.tool_results import handle_tool_results
from agentkit.loop.message_builder import MessageBuilder
from agentkit.loop.orchestrator import Loop
from agentkit.loop.phase import Phase
from agentkit.loop.tool_dispatcher import DispatchPolicy, ToolDispatcher
from agentkit.providers.fakes import FakeProvider
from agentkit.tools.registry import ToolRegistry

pytestmark = pytest.mark.e2e


def _user(text: str) -> Message:
    from datetime import UTC, datetime

    return Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=MessageRole.USER,
        content=[TextBlock(text=text)],
        created_at=datetime.now(UTC),
    )


def _all_handlers():
    return {
        Phase.INTENT_GATE: handle_intent_gate,
        Phase.CONTEXT_BUILD: handle_context_build,
        Phase.STREAMING: handle_streaming,
        Phase.TOOL_PHASE: handle_tool_phase,
        Phase.APPROVAL_WAIT: handle_approval_wait,
        Phase.TOOL_EXECUTING: handle_tool_executing,
        Phase.TOOL_RESULTS: handle_tool_results,
        Phase.FINALIZE_CHECK: handle_finalize_check,
        Phase.MEMORY_EXTRACT: handle_memory_extract,
    }


@pytest.mark.asyncio
async def test_text_only_turn_completes():
    provider = FakeProvider().script(FakeProvider.text("hello world"))
    queue: asyncio.Queue = asyncio.Queue()
    registry = ToolRegistry()
    ctx = TurnContext.empty()
    ctx.event_queue = queue
    ctx.add_message(_user("hi"))

    deps = {
        "provider": provider,
        "message_builder": MessageBuilder(model="m", max_tokens=128),
        "registry": registry,
        "system_blocks": [],
        "intent_gate": None,
        "approval_gate": RiskBasedApprovalGate(),
        "dispatcher": ToolDispatcher(registry=registry, policy=DispatchPolicy()),
        "finalize_validator": StructuralFinalizeValidator(),
        "max_finalize_retries": 2,
        "max_iterations": 10,
    }
    loop = Loop(ctx=ctx, handlers=_all_handlers(), deps=deps)

    events = [ev async for ev in loop.run()]
    types = [type(e).__name__ for e in events]

    assert types[0] == "TurnStarted"
    assert isinstance(events[-1], TurnEnded)
    assert events[-1].reason is TurnEndReason.COMPLETED
    # Phase log should include INTENT_GATE -> CONTEXT_BUILD -> STREAMING ->
    # FINALIZE_CHECK -> MEMORY_EXTRACT.
    assert any(isinstance(e, PhaseChanged) and e.from_ is Phase.STREAMING for e in events)

    # Streaming events from the queue.
    streamed = []
    while not queue.empty():
        streamed.append(queue.get_nowait())
    assert any(isinstance(e, TextDelta) for e in streamed)
    assert any(isinstance(e, MessageCompleted) for e in streamed)
