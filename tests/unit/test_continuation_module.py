"""Shape tests for the continuation extension point.

Covers ``ContinuationRequest`` / ``ContinuationDecision`` / ``GoalState``
construction, the ``ContinuationEvaluator`` runtime-checkable Protocol, and
the public ``TriggerMode`` literal values.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from typing import get_args
from uuid import UUID, uuid4

import pytest

from agentkit.continuation import (
    ContinuationDecision,
    ContinuationEvaluator,
    ContinuationRequest,
    GoalState,
    TriggerMode,
)


def test_trigger_mode_exposes_both_literal_values():
    """v0.10.0 ships every_turn dispatch; self_declared is reserved for v0.11.0
    but already appears in the type so consumer code is forward-compatible."""
    assert set(get_args(TriggerMode)) == {"every_turn", "self_declared"}


def test_continuation_request_is_frozen():
    req = ContinuationRequest(
        condition="alle åpne tilbakemeldinger er triagert",
        transcript=(),
        turn_count=2,
        iteration_count=0,
        token_spend=1234,
        set_at=datetime(2026, 5, 23, tzinfo=UTC),
        state_id=None,
    )
    with pytest.raises(FrozenInstanceError):
        req.turn_count = 99  # type: ignore[misc]


def test_continuation_decision_is_frozen():
    dec = ContinuationDecision(met=False, reason="still 3 reports open")
    with pytest.raises(FrozenInstanceError):
        dec.met = True  # type: ignore[misc]


def test_continuation_request_carries_state_id_unchanged():
    """``state_id`` is opaque to agentkit — round-trips through the request."""
    sid = uuid4()
    req = ContinuationRequest(
        condition="x",
        transcript=(),
        turn_count=0,
        iteration_count=0,
        token_spend=0,
        set_at=datetime.now(UTC),
        state_id=sid,
    )
    assert req.state_id == sid


def test_goal_state_defaults_to_zero_counters():
    g = GoalState(condition="x", set_at=datetime.now(UTC))
    assert g.turn_count == 0
    assert g.iteration_count == 0
    assert g.token_spend == 0
    assert g.last_reason is None
    assert g.state_id is None


def test_goal_state_is_mutable_for_in_loop_counters():
    """``GoalState`` is intentionally not frozen — the runtime mutates
    counters between turns."""
    g = GoalState(condition="x", set_at=datetime.now(UTC))
    g.turn_count += 1
    g.last_reason = "still working"
    assert g.turn_count == 1
    assert g.last_reason == "still working"


def test_evaluator_protocol_runtime_check_recognises_implementer():
    """Anything exposing ``trigger`` and async ``__call__`` satisfies the
    runtime-checkable Protocol."""

    class _Impl:
        trigger: TriggerMode = "every_turn"

        async def __call__(self, request: ContinuationRequest) -> ContinuationDecision:
            return ContinuationDecision(met=True, reason="done")

    impl = _Impl()
    assert isinstance(impl, ContinuationEvaluator)


def test_evaluator_protocol_rejects_missing_call():
    class _NoCall:
        trigger: TriggerMode = "every_turn"

    assert not isinstance(_NoCall(), ContinuationEvaluator)


def test_uuid_state_id_type_is_uuid():
    """Documented type is ``UUID | None``; consumers pass real UUIDs."""
    sid = uuid4()
    g = GoalState(condition="x", set_at=datetime.now(UTC), state_id=sid)
    assert isinstance(g.state_id, UUID)
