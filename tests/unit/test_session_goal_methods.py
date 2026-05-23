"""``AgentSession.set_goal`` / ``clear_goal`` semantics — no I/O, pure state mutation."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from agentkit import AgentConfig, AgentSession
from agentkit._ids import OwnerId
from agentkit.continuation import GoalState
from agentkit.providers.fakes import FakeProvider
from agentkit.tools.registry import ToolRegistry


def _make_session() -> AgentSession:
    cfg = AgentConfig()
    return AgentSession(
        owner=OwnerId("u:1"),
        config=cfg,
        provider=FakeProvider(),
        registry=ToolRegistry(),
        model="fake/test",
    )


def test_set_goal_activates_state_and_stamps_pending_flag():
    sess = _make_session()
    assert sess.goal is None
    assert sess._goal_set_pending is False

    sess.set_goal("alle åpne tilbakemeldinger er triagert")

    assert sess.goal is not None
    assert sess.goal.condition == "alle åpne tilbakemeldinger er triagert"
    assert sess.goal.turn_count == 0
    assert sess._goal_set_pending is True


def test_set_goal_with_state_id_round_trips():
    sess = _make_session()
    sid = uuid4()
    sess.set_goal("triagér feedback", state_id=sid)
    assert sess.goal is not None
    assert sess.goal.state_id == sid


def test_set_goal_with_resume_from_rehydrates_counters():
    """Resume path skips counter reset — consumer feeds prior progress
    back in. Mirrors Pikkolo's Card-bound goal rehydration."""
    sess = _make_session()
    snapshot = GoalState(
        condition="x",
        set_at=datetime(2026, 5, 1, tzinfo=UTC),
        turn_count=7,
        iteration_count=2,
        token_spend=12_345,
        last_reason="still 3 reports open",
        state_id=uuid4(),
    )

    sess.set_goal("x", resume_from=snapshot)

    assert sess.goal is snapshot
    assert sess.goal is not None  # narrow for the assertions below
    assert sess.goal.turn_count == 7
    assert sess.goal.iteration_count == 2
    assert sess.goal.last_reason == "still 3 reports open"


def test_set_goal_rejects_empty_condition():
    sess = _make_session()
    with pytest.raises(ValueError, match="non-empty"):
        sess.set_goal("")


def test_set_goal_rejects_whitespace_only_condition():
    sess = _make_session()
    with pytest.raises(ValueError, match="non-empty"):
        sess.set_goal("   \n\t  ")


def test_set_goal_replaces_prior_goal_silently():
    """One goal per session; subsequent calls overwrite."""
    sess = _make_session()
    sess.set_goal("first")
    sess.set_goal("second")
    assert sess.goal is not None
    assert sess.goal.condition == "second"
    assert sess._goal_set_pending is True


def test_set_goal_resets_clear_pending_flag():
    """A new goal supersedes a pending clear."""
    sess = _make_session()
    sess.set_goal("first")
    sess.clear_goal()
    assert sess._goal_clear_pending is True

    sess.set_goal("second")
    assert sess._goal_clear_pending is False
    assert sess.goal is not None
    assert sess.goal.condition == "second"


def test_clear_goal_marks_pending_when_goal_active():
    sess = _make_session()
    sess.set_goal("x")
    sess.clear_goal()
    # The goal stays on the session — abandonment fires at the next turn
    # boundary inside run().
    assert sess.goal is not None
    assert sess._goal_clear_pending is True


def test_clear_goal_is_noop_without_active_goal():
    sess = _make_session()
    sess.clear_goal()
    assert sess.goal is None
    assert sess._goal_clear_pending is False
