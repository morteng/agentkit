import asyncio

import pytest

from agentkit.events import ToolCallResult as PubToolCallResult
from agentkit.events import TurnEnded, TurnEndReason
from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.guards.finalize import StructuralFinalizeValidator
from agentkit.loop.context import TurnContext
from agentkit.loop.message_builder import MessageBuilder
from agentkit.loop.orchestrator import Loop
from agentkit.loop.tool_dispatcher import DispatchPolicy, ToolDispatcher
from agentkit.providers.fakes import FakeProvider
from agentkit.tools.builtin import DEFAULT_BUILTINS
from agentkit.tools.registry import ToolRegistry

# Reuse handler map from text-only e2e test.
from tests.e2e.test_loop_text_only import _all_handlers, _user

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_finalize_tool_completes_turn():
    """Provider asks to call kit.finalize -> registry executes -> finalize_check accepts."""
    provider = FakeProvider().script(
        FakeProvider.tool_call(
            "kit.finalize",
            {
                "status": "done",
                "intent_kind": "answer",
                "summary": "It is just past noon.",
                "answer_evidence": "general_knowledge",
            },
        ),
    )

    queue: asyncio.Queue = asyncio.Queue()
    registry = ToolRegistry()
    for spec, handler in DEFAULT_BUILTINS:
        registry.register_builtin(spec, handler)

    ctx = TurnContext.empty()
    ctx.event_queue = queue
    ctx.add_message(_user("what time is it?"))

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
    assert isinstance(events[-1], TurnEnded)
    assert events[-1].reason is TurnEndReason.COMPLETED

    # ToolCallResult event should appear for kit.finalize.
    streamed = []
    while not queue.empty():
        streamed.append(queue.get_nowait())
    results = [e for e in streamed if isinstance(e, PubToolCallResult)]
    assert results and results[0].status == "ok"


@pytest.mark.asyncio
async def test_missing_finalize_reprompt_then_finalizes():
    """Model streams text without finalizing -> re-prompt -> model finalizes."""
    provider = FakeProvider().script(
        # First response: just an answer, no finalize call.
        FakeProvider.text("Yes, the article already has a hero image."),
        # After the missing-finalize re-prompt, the model finalizes.
        FakeProvider.tool_call(
            "kit.finalize",
            {
                "status": "done",
                "intent_kind": "answer",
                "summary": "The article has a hero image.",
                "answer_evidence": "general_knowledge",
            },
        ),
    )

    queue: asyncio.Queue = asyncio.Queue()
    registry = ToolRegistry()
    for spec, handler in DEFAULT_BUILTINS:
        registry.register_builtin(spec, handler)

    ctx = TurnContext.empty()
    ctx.event_queue = queue
    ctx.add_message(_user("does the article have a hero image?"))

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
        "max_missing_finalize_reprompts": 1,
        "max_iterations": 10,
    }
    loop = Loop(ctx=ctx, handlers=_all_handlers(), deps=deps)

    events = [ev async for ev in loop.run()]
    assert isinstance(events[-1], TurnEnded)
    assert events[-1].reason is TurnEndReason.COMPLETED
    # The re-prompt fired exactly once and the model then finalized cleanly.
    assert ctx.metadata["missing_finalize_reprompts"] == 1
    assert "finalize_missing" not in ctx.metadata
    assert ctx.finalize_called is True
