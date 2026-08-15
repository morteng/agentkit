"""Session durability: a turn that goes wrong must still leave a usable record.

Covers the persistence path shared by ``run()`` and the resume entry points —
bounded, tool-pair-safe history loading; end-of-turn persistence tied to the
``async with`` block rather than to generator garbage collection; and the
synthesized ``cancelled`` tool results that stop one broken turn from
poisoning every later turn in the session.
"""

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import Any

import pytest
import structlog

from agentkit import AgentConfig, AgentSession
from agentkit._content import ContentBlock, TextBlock, ToolResultBlock, ToolUseBlock
from agentkit._ids import MessageId, OwnerId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.events import ToolCallStarted, TurnStarted
from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.loop.context import TurnContext
from agentkit.providers.fakes import FakeProvider
from agentkit.session import (
    DEFAULT_HISTORY_LIMIT,
    SYNTHETIC_TOOL_RESULT_ANNOTATION,
    _repair_dangling_tool_uses,
    _tool_safe_start,
    _unresolved_tool_use_ids,
)
from agentkit.store.fakes import FakeCheckpointStore, FakeMemoryStore, FakeSessionStore
from agentkit.tools.registry import ToolRegistry
from agentkit.tools.spec import (
    ApprovalPolicy,
    ContentBlockOut,
    RiskLevel,
    SideEffects,
    ToolResult,
    ToolSpec,
)

_FINALIZE_ARGS = {
    "status": "done",
    "intent_kind": "answer",
    "summary": "Done.",
    "answer_evidence": "general_knowledge",
}


# --- transcript helpers -----------------------------------------------------


def _msg(session_id: SessionId, role: MessageRole, content: list[ContentBlock]) -> Message:
    return Message(
        id=new_id(MessageId),
        session_id=session_id,
        role=role,
        content=content,
        created_at=datetime.now(UTC),
    )


def _user(session_id: SessionId, text: str) -> Message:
    return _msg(session_id, MessageRole.USER, [TextBlock(text=text)])


def _assistant_text(session_id: SessionId, text: str) -> Message:
    return _msg(session_id, MessageRole.ASSISTANT, [TextBlock(text=text)])


def _assistant_tool_use(session_id: SessionId, *call_ids: str) -> Message:
    return _msg(
        session_id,
        MessageRole.ASSISTANT,
        [ToolUseBlock(id=cid, name="t.demo", arguments={}) for cid in call_ids],
    )


def _tool_result(session_id: SessionId, call_id: str) -> Message:
    return _msg(
        session_id,
        MessageRole.TOOL,
        [ToolResultBlock(tool_use_id=call_id, content=[TextBlock(text="ok")])],
    )


def _dangling_tool_use_ids(messages: list[Message]) -> list[str]:
    """Independent well-formedness check (deliberately not the implementation)."""
    used = [b.id for m in messages for b in m.content if isinstance(b, ToolUseBlock)]
    answered = {
        b.tool_use_id for m in messages for b in m.content if isinstance(b, ToolResultBlock)
    }
    return [call_id for call_id in used if call_id not in answered]


# --- pure helpers -----------------------------------------------------------


def test_unresolved_tool_use_ids_reports_only_the_unanswered_calls():
    """Two parallel calls in one message, one answered: only the other is unresolved."""
    sid = new_id(SessionId)
    history = [
        _user(sid, "go"),
        _assistant_tool_use(sid, "call-a", "call-b"),
        _tool_result(sid, "call-a"),
    ]
    assert _unresolved_tool_use_ids(history) == ["call-b"]


def test_unresolved_tool_use_ids_empty_for_a_complete_transcript():
    sid = new_id(SessionId)
    history = [
        _user(sid, "go"),
        _assistant_tool_use(sid, "call-a"),
        _tool_result(sid, "call-a"),
        _assistant_text(sid, "done"),
    ]
    assert _unresolved_tool_use_ids(history) == []


def test_tool_safe_start_moves_past_a_result_whose_use_would_be_dropped():
    sid = new_id(SessionId)
    window = [
        _assistant_tool_use(sid, "call-a"),  # 0 — would be cut away
        _tool_result(sid, "call-a"),  # 1 — naive cut lands here, orphaning it
        _assistant_text(sid, "after"),  # 2
    ]
    assert _tool_safe_start(window, 1) == 2


def test_tool_safe_start_moves_past_a_result_whose_use_is_outside_the_window():
    """The tool_use fell off the front of the fetched window entirely."""
    sid = new_id(SessionId)
    window = [
        _user(sid, "noise"),  # 0
        _tool_result(sid, "call-gone"),  # 1 — no tool_use anywhere in the window
        _assistant_text(sid, "after"),  # 2
    ]
    assert _tool_safe_start(window, 1) == 2


def test_tool_safe_start_keeps_a_cut_that_splits_nothing():
    sid = new_id(SessionId)
    window = [
        _assistant_text(sid, "old"),
        _user(sid, "next"),
        _assistant_tool_use(sid, "call-a"),
        _tool_result(sid, "call-a"),
    ]
    assert _tool_safe_start(window, 1) == 1
    assert _tool_safe_start(window, 0) == 0


def test_repair_dangling_tool_uses_inserts_the_result_directly_after_the_call():
    """Position matters: providers want the result in the next message, not at the end."""
    sid = new_id(SessionId)
    history = [
        _user(sid, "go"),
        _assistant_tool_use(sid, "call-a"),
        _user(sid, "still there?"),
    ]
    repaired, count = _repair_dangling_tool_uses(history, sid)

    assert count == 1
    assert len(repaired) == 4
    inserted = repaired[2]
    assert inserted.role is MessageRole.TOOL
    block = inserted.content[0]
    assert isinstance(block, ToolResultBlock)
    assert block.tool_use_id == "call-a"
    assert block.is_error is True
    assert inserted.metadata.annotations[SYNTHETIC_TOOL_RESULT_ANNOTATION] == "cancelled"
    assert _dangling_tool_use_ids(repaired) == []


def test_repair_dangling_tool_uses_is_a_noop_for_a_clean_transcript():
    sid = new_id(SessionId)
    history = [_user(sid, "go"), _assistant_tool_use(sid, "call-a"), _tool_result(sid, "call-a")]
    repaired, count = _repair_dangling_tool_uses(history, sid)
    assert count == 0
    assert repaired is history


# --- session fixtures -------------------------------------------------------


def _make_session(
    provider: FakeProvider,
    *,
    history_limit: int = DEFAULT_HISTORY_LIMIT,
    registry: ToolRegistry | None = None,
) -> AgentSession:
    config = AgentConfig()
    config.guards.approval = RiskBasedApprovalGate()
    config.stores.session = FakeSessionStore()
    config.stores.memory = FakeMemoryStore()
    config.stores.checkpoint = FakeCheckpointStore()

    if registry is None:
        registry = ToolRegistry()
        registry.register_default_builtins()

    return AgentSession(
        owner=OwnerId("u:1"),
        config=config,
        provider=provider,
        registry=registry,
        model="m",
        history_limit=history_limit,
    )


async def _seed(session: AgentSession, messages: list[Message]) -> None:
    """Put a pre-existing transcript in the store before the session initializes."""
    store = session.config.stores.session
    await store.create(session.id, session.owner)
    for msg in messages:
        await store.append_message(session.id, msg)


async def _run_capturing_context(session: AgentSession, text: str) -> TurnContext:
    """Run one turn and return its (post-turn) TurnContext."""
    ctx, _loaded = await _run_capturing_loaded_history(session, text)
    return ctx


async def _run_capturing_loaded_history(
    session: AgentSession, text: str
) -> tuple[TurnContext, list[Message]]:
    """Run one turn, returning its TurnContext and the history it *started* with.

    The context keeps mutating for the rest of the turn, so the loaded window
    has to be snapshotted at the first iteration to be asserted on.
    """
    seen: list[TurnContext] = []
    loaded: list[list[Message]] = []

    async def hook(ctx: TurnContext) -> None:
        seen.append(ctx)
        loaded.append(list(ctx.history))

    session.config.on_iteration_start = hook
    async with session.run(text) as stream:
        async for _ev in stream:
            pass
    assert seen, "on_iteration_start never fired — no TurnContext captured"
    return seen[0], loaded[0]


def _blocking_tool(started: asyncio.Event, finished: list[str]) -> tuple[ToolSpec, Any]:
    """A tool that parks forever, so a turn can be cancelled mid-execution."""
    spec = ToolSpec(
        name="demo.block",
        description="blocks until cancelled",
        parameters={"type": "object"},
        returns=None,
        risk=RiskLevel.LOW_WRITE,
        idempotent=False,
        side_effects=SideEffects.LOCAL,
        requires_approval=ApprovalPolicy.NEVER,
        cache_ttl_seconds=None,
        timeout_seconds=30.0,
    )

    async def handler(args, ctx):
        started.set()
        await asyncio.sleep(30)
        finished.append(ctx.call_id)
        return ToolResult(
            call_id=ctx.call_id,
            status="ok",
            content=[ContentBlockOut(type="text", text="never")],
            error=None,
            duration_ms=0,
            cached=False,
        )

    return spec, handler


async def _abandon_mid_tool(session: AgentSession, text: str, started: asyncio.Event) -> None:
    """Run a turn until its tool is parked mid-execution, then walk away.

    Waiting on ``started`` inside the loop matters: ``ToolCallStarted`` is
    emitted while the model's stream is still being parsed, so breaking on the
    event alone would race the dispatcher and sometimes cancel before the
    assistant's tool_use message was ever recorded.
    """
    async with session.run(text) as stream:
        async for ev in stream:
            if isinstance(ev, ToolCallStarted):
                await asyncio.wait_for(started.wait(), timeout=5)
                break


# --- history loading (F: run() truncated silently, and could orphan a pair) --


@pytest.mark.asyncio
async def test_run_truncates_at_an_explicit_limit_and_never_orphans_a_result():
    """The naive cut would keep a tool_result whose tool_use was dropped."""
    provider = FakeProvider().script(FakeProvider.text("hi"))
    session = _make_session(provider, history_limit=2)
    sid = session.id
    await _seed(
        session,
        [
            _user(sid, "old"),
            _assistant_tool_use(sid, "call-a"),
            _tool_result(sid, "call-a"),
            _assistant_text(sid, "tail"),
        ],
    )

    with structlog.testing.capture_logs() as logs:
        ctx, loaded = await _run_capturing_loaded_history(session, "next")

    # The pair went out together rather than the result being kept alone.
    assert [m.role for m in loaded] == [MessageRole.ASSISTANT, MessageRole.USER]
    kept = loaded[0].content[0]
    assert isinstance(kept, TextBlock)
    assert kept.text == "tail"
    assert _dangling_tool_use_ids(loaded) == []

    # Truncation is announced, not silent.
    assert ctx.metadata["history_load"]["truncated"] is True
    assert ctx.metadata["history_load"]["limit"] == 2
    assert any(entry["event"] == "history_truncated" for entry in logs)


@pytest.mark.asyncio
async def test_run_loads_everything_and_warns_about_nothing_when_under_the_limit():
    provider = FakeProvider().script(FakeProvider.text("hi"))
    session = _make_session(provider, history_limit=50)
    sid = session.id
    await _seed(session, [_user(sid, "old"), _assistant_text(sid, "older reply")])

    with structlog.testing.capture_logs() as logs:
        ctx, loaded = await _run_capturing_loaded_history(session, "next")

    assert len(loaded) == 3  # two seeded + this turn's user message
    assert "history_load" not in ctx.metadata
    assert not [entry for entry in logs if entry["event"] == "history_truncated"]


@pytest.mark.asyncio
async def test_run_repairs_a_legacy_transcript_with_a_dangling_tool_use():
    """A transcript broken before this fix existed must not break the next turn."""
    provider = FakeProvider().script(FakeProvider.text("hi"))
    session = _make_session(provider)
    sid = session.id
    await _seed(session, [_user(sid, "old"), _assistant_tool_use(sid, "call-orphan")])

    ctx, loaded = await _run_capturing_loaded_history(session, "next")

    assert _dangling_tool_use_ids(loaded) == []
    repaired = loaded[2]
    assert repaired.role is MessageRole.TOOL
    assert repaired.metadata.annotations[SYNTHETIC_TOOL_RESULT_ANNOTATION] == "cancelled"
    assert ctx.metadata["history_load"]["repaired_tool_uses"] == 1

    # The in-memory repair is not written back to the store.
    stored = await session.config.stores.session.list_messages(sid, limit=100)
    assert [m.role for m in stored[:2]] == [MessageRole.USER, MessageRole.ASSISTANT]
    assert all(
        m.metadata.annotations.get(SYNTHETIC_TOOL_RESULT_ANNOTATION) is None for m in stored[:2]
    )


# --- cleanup + persistence on the context manager ---------------------------


@pytest.mark.asyncio
async def test_abandoning_the_stream_stops_the_turn_at_context_manager_exit():
    """Leaving ``async with`` must end the turn, not detach it."""
    started = asyncio.Event()
    finished: list[str] = []
    registry = ToolRegistry()
    registry.register_default_builtins()
    registry.register_builtin(*_blocking_tool(started, finished))

    provider = FakeProvider().script(
        FakeProvider.tool_call("demo.block", {}),
        FakeProvider.tool_call("kit.finalize", _FINALIZE_ARGS),
    )
    session = _make_session(provider, registry=registry)

    await _abandon_mid_tool(session, "do the slow thing", started)

    assert started.is_set(), "the tool never got as far as running"
    # Give a detached loop every chance to keep going; it must not.
    await asyncio.sleep(0.05)
    assert finished == [], "the turn kept running after the context manager exited"


@pytest.mark.asyncio
async def test_turn_is_recorded_when_the_consuming_task_is_cancelled():
    """The web-request-goes-away case: the task holding the ``async with`` is
    cancelled outright, and the turn must still reach the store."""
    started = asyncio.Event()
    consuming = asyncio.Event()
    finished: list[str] = []
    registry = ToolRegistry()
    registry.register_default_builtins()
    registry.register_builtin(*_blocking_tool(started, finished))

    provider = FakeProvider().script(
        FakeProvider.tool_call("demo.block", {}),
        FakeProvider.tool_call("kit.finalize", _FINALIZE_ARGS),
    )
    session = _make_session(provider, registry=registry)

    async def consume() -> None:
        async with session.run("do the slow thing") as stream:
            async for ev in stream:
                if isinstance(ev, ToolCallStarted):
                    await asyncio.wait_for(started.wait(), timeout=5)
                    consuming.set()
                    await asyncio.sleep(30)  # still inside the block

    task = asyncio.create_task(consume())
    await asyncio.wait_for(consuming.wait(), timeout=5)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.05)  # let a shielded write land

    stored = await session.config.stores.session.list_messages(session.id, limit=100)
    assert any(isinstance(b, ToolUseBlock) for m in stored for b in m.content)
    assert _dangling_tool_use_ids(stored) == []
    assert finished == []


@pytest.mark.asyncio
async def test_cancelled_turn_persists_a_cancelled_result_for_the_open_call():
    """The tool_use is on record, so its result has to be too — providers reject
    a history whose tool_use is unanswered, permanently breaking the session."""
    started = asyncio.Event()
    finished: list[str] = []
    registry = ToolRegistry()
    registry.register_default_builtins()
    registry.register_builtin(*_blocking_tool(started, finished))

    provider = FakeProvider().script(
        FakeProvider.tool_call("demo.block", {}),
        FakeProvider.tool_call("kit.finalize", _FINALIZE_ARGS),
    )
    session = _make_session(provider, registry=registry)

    await _abandon_mid_tool(session, "do the slow thing", started)

    stored = await session.config.stores.session.list_messages(session.id, limit=100)
    assert any(isinstance(b, ToolUseBlock) for m in stored for b in m.content), (
        "the assistant's tool_use never reached the store"
    )
    assert _dangling_tool_use_ids(stored) == []
    closers = [m for m in stored if m.metadata.annotations.get(SYNTHETIC_TOOL_RESULT_ANNOTATION)]
    assert len(closers) == 1
    block = closers[0].content[0]
    assert isinstance(block, ToolResultBlock)
    assert block.is_error is True
    assert closers[0].metadata.annotations[SYNTHETIC_TOOL_RESULT_ANNOTATION] == "cancelled"


@pytest.mark.asyncio
async def test_turn_after_a_cancelled_turn_runs_on_well_formed_history():
    """The regression that made this HIGH severity: turn N+1 sends turn N's mess."""
    started = asyncio.Event()
    finished: list[str] = []
    registry = ToolRegistry()
    registry.register_default_builtins()
    registry.register_builtin(*_blocking_tool(started, finished))

    provider = FakeProvider().script(
        FakeProvider.tool_call("demo.block", {}),
        FakeProvider.text("second turn reply"),
    )
    session = _make_session(provider, registry=registry)

    await _abandon_mid_tool(session, "do the slow thing", started)

    ctx = await _run_capturing_context(session, "are you still there?")

    assert _dangling_tool_use_ids(ctx.history) == []
    # Nothing needed repairing on load — the cancelled turn persisted cleanly.
    assert "history_load" not in ctx.metadata

    stored = await session.config.stores.session.list_messages(session.id, limit=100)
    assert _dangling_tool_use_ids(stored) == []


@pytest.mark.asyncio
async def test_completed_turn_persists_each_message_exactly_once():
    """Draining the stream AND exiting the context manager must not double-write.

    The store must end up holding precisely the turn's in-memory history — no
    gaps, no repeats, same order.
    """
    provider = FakeProvider().script(FakeProvider.tool_call("kit.finalize", _FINALIZE_ARGS))
    session = _make_session(provider)

    ctx = await _run_capturing_context(session, "hi")

    stored = await session.config.stores.session.list_messages(session.id, limit=100)
    assert [m.id for m in stored] == [m.id for m in ctx.history]
    assert _dangling_tool_use_ids(stored) == []


@pytest.mark.asyncio
async def test_turn_is_persisted_even_when_the_consumer_raises():
    """An exception inside ``async with`` still ends the turn through the same path."""
    provider = FakeProvider().script(FakeProvider.tool_call("kit.finalize", _FINALIZE_ARGS))
    session = _make_session(provider)

    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        async with session.run("hi") as stream:
            async for ev in stream:
                if isinstance(ev, TurnStarted):
                    raise Boom

    stored = await session.config.stores.session.list_messages(session.id, limit=100)
    assert stored, "nothing was persisted for the aborted turn"
    assert stored[0].role is MessageRole.USER
    assert len({m.id for m in stored}) == len(stored)
    assert _dangling_tool_use_ids(stored) == []
