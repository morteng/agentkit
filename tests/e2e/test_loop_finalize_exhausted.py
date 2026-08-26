"""A turn whose finalize envelope is rejected until the retry budget is spent.

The value under test is the one that crosses the boundary: ``TurnEnded.reason``.
A consumer branching on that enum is the only thing standing between "the model
finished" and "the model ran out of finalize retries", and for as long as this
path emitted ``COMPLETED`` the two were the same value on the wire.
"""

import asyncio

import pytest

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
from tests.e2e.test_loop_text_only import _all_handlers, _user

pytestmark = pytest.mark.e2e

# Schema-legal (``status`` and ``intent_kind`` are present, so the registry
# dispatches it and ``finalize_handler`` runs) but structurally invalid: an
# ``intent_kind="answer"`` envelope must carry ``answer_evidence``. The model
# therefore genuinely finalizes and the validator genuinely rejects, which is
# the state this path exists to describe.
_REJECTED_ENVELOPE = {
    "status": "done",
    "intent_kind": "answer",
    "summary": "It is just past noon.",
}


def _loop_with_rejected_finalizes(max_finalize_retries: int) -> tuple[Loop, TurnContext]:
    # One attempt plus one response per retry: the last one is the call that
    # finds the budget spent.
    provider = FakeProvider().script(
        *(
            FakeProvider.tool_call("kit.finalize", dict(_REJECTED_ENVELOPE))
            for _ in range(max_finalize_retries + 1)
        )
    )
    registry = ToolRegistry()
    for spec, handler in DEFAULT_BUILTINS:
        registry.register_builtin(spec, handler)

    ctx = TurnContext.empty()
    ctx.event_queue = asyncio.Queue()
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
        "max_finalize_retries": max_finalize_retries,
        "max_missing_finalize_reprompts": 1,
        "max_iterations": 10,
    }
    return Loop(ctx=ctx, handlers=_all_handlers(), deps=deps), ctx


@pytest.mark.asyncio
async def test_finalize_rejection_budget_spent_is_not_completed():
    """The terminal event names the degraded outcome, not ``COMPLETED``.

    ``COMPLETED`` asserts the model finished. Here it finalized three times and
    every envelope was rejected, so nothing the user is looking at was ever
    accepted as an answer. A consumer that renders the terminal event cannot
    tell the reader that unless the reason says so.
    """
    loop, ctx = _loop_with_rejected_finalizes(max_finalize_retries=2)

    events = [ev async for ev in loop.run()]
    ended = events[-1]

    assert isinstance(ended, TurnEnded)
    # Assert the value, not merely that it differs from COMPLETED: a reason of
    # ERROR or NO_RESPONSE would also be "not COMPLETED" and would both be
    # wrong — the turn did not crash, and the model did answer and did call
    # finalize_response.
    assert ended.reason is TurnEndReason.FINALIZE_REJECTED
    assert ended.reason.value == "finalize_rejected"
    # Belt and braces on the wire shape: a consumer reads the serialized event,
    # so the string is what actually reaches it.
    assert ended.model_dump(mode="json")["reason"] == "finalize_rejected"

    # The precondition — this is the exhausted branch and not some other exit.
    assert ctx.metadata["finalize_exhausted"] is True
    assert ctx.metadata["finalize_retries"] == 2
    assert ctx.finalize_called is True


@pytest.mark.asyncio
async def test_accepted_finalize_still_completes():
    """Control: the same wiring with a valid envelope still ends ``COMPLETED``.

    Without this, a fix that hard-coded the new reason for every turn would
    pass the test above.
    """
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
    registry = ToolRegistry()
    for spec, handler in DEFAULT_BUILTINS:
        registry.register_builtin(spec, handler)

    ctx = TurnContext.empty()
    ctx.event_queue = asyncio.Queue()
    ctx.add_message(_user("what time is it?"))

    loop = Loop(
        ctx=ctx,
        handlers=_all_handlers(),
        deps={
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
        },
    )

    events = [ev async for ev in loop.run()]
    assert isinstance(events[-1], TurnEnded)
    assert events[-1].reason is TurnEndReason.COMPLETED
    assert "finalize_exhausted" not in ctx.metadata


@pytest.mark.asyncio
async def test_earlier_suspend_reason_keeps_priority_over_finalize_rejection():
    """An already-recorded reason outranks the finalize rejection.

    ``AWAITING_APPROVAL`` and ``MAX_ITERATIONS`` explain more about why the turn
    is ending than a rejected envelope does, and the sibling early-exit paths
    all defer to whatever was recorded first. This pins that the new reason
    joins that rule rather than overwriting it.
    """
    loop, ctx = _loop_with_rejected_finalizes(max_finalize_retries=2)
    ctx.metadata["suspend_reason"] = TurnEndReason.MAX_ITERATIONS.value

    events = [ev async for ev in loop.run()]

    assert isinstance(events[-1], TurnEnded)
    assert events[-1].reason is TurnEndReason.MAX_ITERATIONS
    assert ctx.metadata["finalize_exhausted"] is True
