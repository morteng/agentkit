"""SuccessClaimGuard corrections must terminate, and must reach the model.

The pre-0.22 handler returned CONTEXT_BUILD on every guard trip: the correction
was parked in ``ctx.metadata`` where the MessageBuilder never looks, the
streamed text the user had already seen was dropped from history, and
``max_claim_corrections`` was never consulted — so a model that kept claiming
success looped until something else killed the turn.
"""

import asyncio
from datetime import UTC, datetime

import pytest

from agentkit._content import TextBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import INJECTED_CORRECTION_ANNOTATION, Message, MessageRole
from agentkit.events import TurnEnded, TurnEndReason
from agentkit.guards.success_claim import ClaimVerdict
from agentkit.loop.context import TurnContext
from agentkit.loop.handlers.streaming import handle_streaming
from agentkit.loop.message_builder import MessageBuilder
from agentkit.loop.orchestrator import Loop
from agentkit.loop.phase import Phase
from agentkit.providers.fakes import FakeProvider
from agentkit.tools.registry import ToolRegistry

_CORRECTION = "You claimed success without calling the tool."


class _AlwaysFlags:
    """A guard that flags every chunk — the pathological case."""

    def __init__(self) -> None:
        self.calls = 0

    async def check(self, text_so_far: str, ctx: TurnContext) -> ClaimVerdict:
        self.calls += 1
        return ClaimVerdict(flag=True, suggested_correction=_CORRECTION)


def _user(text: str) -> Message:
    return Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=MessageRole.USER,
        content=[TextBlock(text=text)],
        created_at=datetime.now(UTC),
    )


def _deps(provider: FakeProvider, guard, **overrides):
    deps = {
        "provider": provider,
        "message_builder": MessageBuilder(model="m", max_tokens=128),
        "registry": ToolRegistry(),
        "system_blocks": [],
        "success_claim": guard,
        "max_claim_corrections": 1,
        "max_stream_retries": 0,
        "stream_retry_base_delay_seconds": 0.0,
    }
    deps.update(overrides)
    return deps


def _corrections(ctx: TurnContext) -> list[Message]:
    return [
        m
        for m in ctx.history
        if m.role is MessageRole.USER and m.metadata.annotations.get(INJECTED_CORRECTION_ANNOTATION)
    ]


@pytest.mark.asyncio
async def test_claim_trip_puts_text_and_correction_in_history():
    provider = FakeProvider().script(FakeProvider.text("I have created the file"))
    ctx = TurnContext.empty()
    ctx.add_message(_user("make the file"))
    ctx.event_queue = asyncio.Queue()

    next_ = await handle_streaming(ctx, _deps(provider, _AlwaysFlags()))

    assert next_ is Phase.CONTEXT_BUILD
    # The user already saw the partial text; the model must see it too.
    assistant = [m for m in ctx.history if m.role is MessageRole.ASSISTANT]
    assert len(assistant) == 1
    assert "".join(b.text for b in assistant[0].content if isinstance(b, TextBlock))
    # The correction is a real message, not a metadata note the model never reads.
    corrections = _corrections(ctx)
    assert len(corrections) == 1
    # Narrow before reading .text: content is a block union, and a correction
    # delivered as anything other than text is a correction the model does not
    # read as a correction.
    first_block = corrections[0].content[0]
    assert isinstance(first_block, TextBlock)
    assert _CORRECTION in first_block.text
    assert ctx.metadata["claim_corrections"] == 1
    # Kept for consumers that already read it.
    assert ctx.metadata["claim_correction"] == _CORRECTION


@pytest.mark.asyncio
async def test_claim_budget_is_enforced_and_the_stream_finishes():
    """Second trip in the same turn: budget spent, guard stands down."""
    provider = FakeProvider().script(FakeProvider.text("I have created the file"))
    ctx = TurnContext.empty()
    ctx.add_message(_user("make the file"))
    ctx.event_queue = asyncio.Queue()
    ctx.metadata["claim_corrections"] = 1  # a previous iteration already spent it

    next_ = await handle_streaming(ctx, _deps(provider, _AlwaysFlags()))

    assert next_ is Phase.FINALIZE_CHECK
    assert ctx.metadata["claim_corrections_exhausted"] is True
    assert _corrections(ctx) == []


@pytest.mark.asyncio
async def test_zero_budget_never_interrupts_the_stream():
    provider = FakeProvider().script(FakeProvider.text("I have created the file"))
    ctx = TurnContext.empty()
    ctx.add_message(_user("make the file"))
    ctx.event_queue = asyncio.Queue()

    next_ = await handle_streaming(ctx, _deps(provider, _AlwaysFlags(), max_claim_corrections=0))

    assert next_ is Phase.FINALIZE_CHECK
    assert ctx.metadata.get("claim_corrections", 0) == 0


@pytest.mark.asyncio
async def test_guard_is_checked_only_once_per_delta_after_the_budget_is_spent():
    """Once spent, the guard stops being consulted for the rest of the stream."""
    provider = FakeProvider().script(FakeProvider.text("I have created the file"))
    guard = _AlwaysFlags()
    ctx = TurnContext.empty()
    ctx.add_message(_user("make the file"))
    ctx.event_queue = asyncio.Queue()
    ctx.metadata["claim_corrections"] = 1

    await handle_streaming(ctx, _deps(provider, guard))

    assert guard.calls == 1


@pytest.mark.asyncio
async def test_always_tripping_guard_terminates_the_turn():
    """End to end: a provider that always claims success cannot loop forever."""
    provider = FakeProvider().script(
        *[FakeProvider.text("I have created the file") for _ in range(6)]
    )
    ctx = TurnContext.empty()
    ctx.add_message(_user("make the file"))
    ctx.event_queue = asyncio.Queue()

    async def to_streaming(ctx, deps):
        return Phase.STREAMING

    async def to_memory_extract(ctx, deps):
        return Phase.MEMORY_EXTRACT

    async def to_turn_ended(ctx, deps):
        return Phase.TURN_ENDED

    async def to_context_build(ctx, deps):
        return Phase.CONTEXT_BUILD

    handlers = {
        Phase.INTENT_GATE: to_context_build,
        Phase.CONTEXT_BUILD: to_streaming,
        Phase.STREAMING: handle_streaming,
        Phase.FINALIZE_CHECK: to_memory_extract,
        Phase.MEMORY_EXTRACT: to_turn_ended,
    }
    loop = Loop(
        ctx=ctx,
        handlers=handlers,
        deps=_deps(provider, _AlwaysFlags()),
        end_reason=TurnEndReason.COMPLETED,
    )

    events = await asyncio.wait_for(_collect(loop), timeout=10)

    assert isinstance(events[-1], TurnEnded)
    assert events[-1].reason is TurnEndReason.COMPLETED
    streams = [frm for (frm, _to, _ms) in ctx.phase_log if frm == Phase.STREAMING.value]
    # One correction budget: attempt, correction, one retry that is allowed to finish.
    assert len(streams) == 2
    assert len(_corrections(ctx)) == 1


async def _collect(loop: Loop) -> list:
    return [ev async for ev in loop.run()]
