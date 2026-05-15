"""Pydantic schema tests for agentkit.envelope.Envelope."""

import pytest
from pydantic import ValidationError

from agentkit.envelope import Action, Envelope, PendingConfirmation


def test_envelope_minimal_action():
    e = Envelope(status="done", intent_kind="action")
    assert e.status == "done"
    assert e.intent_kind == "action"
    assert e.actions_performed == []
    assert e.expected_count is None


def test_envelope_intent_kind_required():
    with pytest.raises(ValidationError):
        Envelope(status="done")  # type: ignore[call-arg]


def test_envelope_intent_kind_rejects_unknown_value():
    with pytest.raises(ValidationError):
        Envelope(status="done", intent_kind="lookup")  # type: ignore[arg-type]


def test_envelope_intent_kind_accepts_action_answer_clarify():
    for kind in ("action", "answer", "clarify"):
        Envelope(status="done", intent_kind=kind)


def test_envelope_expected_count_zero_invalid():
    with pytest.raises(ValidationError):
        Envelope(status="done", intent_kind="action", expected_count=0)


def test_envelope_expected_count_positive_ok():
    e = Envelope(status="done", intent_kind="action", expected_count=3)
    assert e.expected_count == 3


def test_envelope_expected_count_null_default():
    e = Envelope(status="done", intent_kind="action")
    assert e.expected_count is None


def test_envelope_with_actions():
    e = Envelope(
        status="done",
        intent_kind="action",
        actions_performed=[
            Action(tool="patch_content", target="My article", description="Updated body"),
        ],
        expected_count=1,
    )
    assert len(e.actions_performed) == 1
    assert e.actions_performed[0].tool == "patch_content"


def test_envelope_blocked_with_pending_confirmation():
    e = Envelope(
        status="blocked",
        intent_kind="clarify",
        pending_confirmation=PendingConfirmation(
            question="Should I publish?",
            kind="confirm",
        ),
    )
    assert e.pending_confirmation is not None
    assert e.pending_confirmation.question == "Should I publish?"


def test_action_requires_tool_and_description():
    with pytest.raises(ValidationError):
        Action(target="X")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Action(tool="patch_content")  # type: ignore[call-arg]


def test_envelope_status_only_three_values():
    for status in ("done", "partial", "blocked"):
        Envelope(status=status, intent_kind="action")
    with pytest.raises(ValidationError):
        Envelope(status="finished", intent_kind="action")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# answer_evidence field (added in agentkit v0.7.0)
# ---------------------------------------------------------------------------


def test_envelope_answer_evidence_defaults_to_none():
    e = Envelope(status="done", intent_kind="action")
    assert e.answer_evidence is None


def test_envelope_answer_evidence_accepts_three_literals():
    for value in ("tool_results", "context", "general_knowledge"):
        e = Envelope(status="done", intent_kind="answer", answer_evidence=value)
        assert e.answer_evidence == value


def test_envelope_answer_evidence_rejects_unknown_value():
    with pytest.raises(ValidationError):
        Envelope(
            status="done",
            intent_kind="answer",
            answer_evidence="from_memory",  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        )
