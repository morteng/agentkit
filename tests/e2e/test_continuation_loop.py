"""End-to-end goal-continuation test driving a real :class:`AgentSession`.

A scripted ``FakeProvider`` calls ``kit.finalize`` on each turn so the loop
reaches a terminal envelope. A fake evaluator rejects twice, then accepts —
asserting that three turns run, the right events fire in the right order,
and the synthesised continuation messages appear in history.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agentkit import AgentConfig, AgentSession
from agentkit._content import TextBlock
from agentkit._ids import MessageId, OwnerId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.continuation import (
    ContinuationDecision,
    ContinuationRequest,
    TriggerMode,
)
from agentkit.events import (
    GoalAbandoned,
    GoalAchieved,
    GoalEvaluated,
    GoalSet,
    TurnEnded,
    TurnStarted,
)
from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.guards.finalize import StructuralFinalizeValidator
from agentkit.providers.fakes import FakeProvider, ScriptedResponse
from agentkit.store.fakes import FakeCheckpointStore, FakeMemoryStore, FakeSessionStore
from agentkit.tools.builtin import DEFAULT_BUILTINS
from agentkit.tools.registry import ToolRegistry

pytestmark = pytest.mark.e2e

_FINALIZE_ARGS = {
    "status": "done",
    "intent_kind": "answer",
    "summary": "Step complete.",
    "answer_evidence": "general_knowledge",
}


def _finalize_call() -> ScriptedResponse:
    return FakeProvider.tool_call("kit.finalize", _FINALIZE_ARGS)


class _ScriptedEvaluator:
    """Returns each scripted decision in order; raises if drained."""

    trigger: TriggerMode = "every_turn"

    def __init__(self, *decisions: ContinuationDecision) -> None:
        self._queue = list(decisions)
        self.calls: list[ContinuationRequest] = []

    async def __call__(self, request: ContinuationRequest) -> ContinuationDecision:
        self.calls.append(request)
        if not self._queue:
            raise RuntimeError("evaluator script exhausted")
        return self._queue.pop(0)


def _make_session(provider: FakeProvider, evaluator: object) -> AgentSession:
    cfg = AgentConfig()
    cfg.guards.approval = RiskBasedApprovalGate()
    cfg.guards.finalize = StructuralFinalizeValidator()
    cfg.stores.session = FakeSessionStore()
    cfg.stores.memory = FakeMemoryStore()
    cfg.stores.checkpoint = FakeCheckpointStore()
    cfg.continuation_evaluator = evaluator

    registry = ToolRegistry()
    for spec, handler in DEFAULT_BUILTINS:
        registry.register_builtin(spec, handler)

    return AgentSession(
        owner=OwnerId("u:1"),
        config=cfg,
        provider=provider,
        registry=registry,
        model="m",
    )


async def test_evaluator_rejects_twice_then_accepts():
    """Three terminal envelopes; three evaluator calls; the third returns met=True."""
    provider = FakeProvider().script(
        _finalize_call(),
        _finalize_call(),
        _finalize_call(),
    )
    evaluator = _ScriptedEvaluator(
        ContinuationDecision(met=False, reason="still 2 of 3"),
        ContinuationDecision(met=False, reason="still 1 of 3"),
        ContinuationDecision(met=True, reason="all 3 triaged"),
    )
    session = _make_session(provider, evaluator)
    session.set_goal("triagér alle 3 åpne tilbakemeldinger")

    events: list = []
    async with session.run("legg i gang") as stream:
        async for ev in stream:
            events.append(ev)

    # GoalSet fires exactly once, before the first TurnStarted.
    goal_set = [e for e in events if isinstance(e, GoalSet)]
    assert len(goal_set) == 1
    assert goal_set[0].condition == "triagér alle 3 åpne tilbakemeldinger"

    # Three turns ran.
    turn_starts = [e for e in events if isinstance(e, TurnStarted)]
    turn_ends = [e for e in events if isinstance(e, TurnEnded)]
    assert len(turn_starts) == 3
    assert len(turn_ends) == 3

    # GoalEvaluated fires after each TurnEnded.
    evaluations = [e for e in events if isinstance(e, GoalEvaluated)]
    assert len(evaluations) == 3
    assert [e.met for e in evaluations] == [False, False, True]
    assert [e.turn_count for e in evaluations] == [1, 2, 3]

    # Stream ends with GoalAchieved, not just TurnEnded.
    achieved = [e for e in events if isinstance(e, GoalAchieved)]
    assert len(achieved) == 1
    assert achieved[0].reason == "all 3 triaged"
    assert achieved[0].turn_count == 3
    assert events[-1] is achieved[0]

    # Session goal cleared after achievement.
    assert session.goal is None
    assert session._goal_set_pending is False


async def test_continuation_synthesises_system_message_between_turns():
    """Rejected evaluator decisions append a system-role guidance message."""
    provider = FakeProvider().script(_finalize_call(), _finalize_call())
    evaluator = _ScriptedEvaluator(
        ContinuationDecision(met=False, reason="need to publish second article"),
        ContinuationDecision(met=True, reason="both published"),
    )
    session = _make_session(provider, evaluator)
    session.set_goal("publiser begge artiklene")

    async with session.run("start") as stream:
        async for _ in stream:
            pass

    history = await session.config.stores.session.list_messages(session.id)
    system_msgs = [m for m in history if m.role is MessageRole.SYSTEM]
    assert len(system_msgs) == 1
    text_block = system_msgs[0].content[0]
    assert "[goal continuation:" in text_block.text  # type: ignore[attr-defined]
    assert "need to publish second article" in text_block.text  # type: ignore[attr-defined]
    assert system_msgs[0].metadata.annotations.get("goal_continuation") is True


async def test_evaluator_sees_only_history_since_set_at():
    """The evaluator's transcript window starts at ``goal.set_at`` so prior
    chat turns don't leak in."""
    provider = FakeProvider().script(_finalize_call())
    evaluator = _ScriptedEvaluator(
        ContinuationDecision(met=True, reason="done"),
    )
    session = _make_session(provider, evaluator)

    # Pre-populate history with a prior unrelated user turn.
    await session.initialize()
    pre = Message(
        id=new_id(MessageId),
        session_id=session.id,
        role=MessageRole.USER,
        content=[TextBlock(text="hva er i dag?")],
        created_at=datetime.now(UTC) - timedelta(hours=1),
    )
    await session.config.stores.session.append_message(session.id, pre)

    session.set_goal("ny oppgave")
    async with session.run("legg i gang") as stream:
        async for _ in stream:
            pass

    # Evaluator got at least one call; the transcript window must NOT contain
    # the pre-existing message.
    assert evaluator.calls, "evaluator never ran"
    seen_texts = [
        b.text
        for req in evaluator.calls
        for m in req.transcript
        for b in m.content
        if isinstance(b, TextBlock)
    ]
    assert all("hva er i dag" not in t for t in seen_texts)


async def test_clear_goal_emits_abandoned_at_turn_boundary():
    """When the consumer clears the goal mid-stream (between turns), the next
    boundary emits ``GoalAbandoned(cause='cleared')`` and the stream ends
    without running the evaluator again."""
    provider = FakeProvider().script(_finalize_call())
    # The evaluator should never be called — clear happens before the
    # evaluator dispatch path on the first turn boundary.
    evaluator = _ScriptedEvaluator()
    session = _make_session(provider, evaluator)
    session.set_goal("aldri ferdig")

    events: list = []
    async with session.run("start") as stream:
        async for ev in stream:
            events.append(ev)
            # Clear the goal as soon as we see TurnEnded for turn 1 — the
            # next boundary check inside run() must honor it.
            if isinstance(ev, TurnEnded):
                session.clear_goal()

    abandoned = [e for e in events if isinstance(e, GoalAbandoned)]
    assert len(abandoned) == 1
    assert abandoned[0].cause == "cleared"
    assert evaluator.calls == []
    assert session.goal is None


async def test_evaluator_returning_budget_exceeded_maps_to_abandoned():
    """Reason strings reserved in ``GoalAbandoned.cause`` short-circuit to
    abandoned instead of achieved — matches the consumer-side budget policy
    described in the spec."""
    provider = FakeProvider().script(_finalize_call())
    evaluator = _ScriptedEvaluator(
        ContinuationDecision(met=True, reason="budget_exceeded"),
    )
    session = _make_session(provider, evaluator)
    session.set_goal("dyrt mål")

    events: list = []
    async with session.run("start") as stream:
        async for ev in stream:
            events.append(ev)

    abandoned = [e for e in events if isinstance(e, GoalAbandoned)]
    achieved = [e for e in events if isinstance(e, GoalAchieved)]
    assert len(abandoned) == 1
    assert abandoned[0].cause == "budget_exceeded"
    assert achieved == []
    assert session.goal is None


async def test_no_goal_means_single_turn_behaviour_unchanged():
    """Backward-compat: without a goal, ``run()`` ends on first TurnEnded
    and emits no Goal* events."""
    provider = FakeProvider().script(_finalize_call())
    session = _make_session(provider, _ScriptedEvaluator())  # evaluator never fires

    events: list = []
    async with session.run("hei") as stream:
        async for ev in stream:
            events.append(ev)

    assert isinstance(events[-1], TurnEnded)
    goal_events = [
        e for e in events if isinstance(e, GoalSet | GoalEvaluated | GoalAchieved | GoalAbandoned)
    ]
    assert goal_events == []


async def test_self_declared_trigger_rejected_in_v0_10_0():
    """Forward-compat literal is exposed but dispatch is reserved for v0.11.0."""

    class _SelfDeclared:
        trigger: TriggerMode = "self_declared"

        async def __call__(self, request: ContinuationRequest) -> ContinuationDecision:
            return ContinuationDecision(met=True, reason="done")

    provider = FakeProvider().script(_finalize_call())
    session = _make_session(provider, _SelfDeclared())
    session.set_goal("x")

    with pytest.raises(RuntimeError, match="self_declared"):
        async with session.run("start") as stream:
            async for _ in stream:
                pass
