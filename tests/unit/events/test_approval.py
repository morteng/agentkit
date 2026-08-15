"""Approval event shapes: reversibility, taint provenance, and resolution."""

from datetime import UTC, datetime

from agentkit._ids import EventId, SessionId, TurnId, new_id
from agentkit.events import EVENT_ADAPTER, ApprovalNeeded, ApprovalResolved
from agentkit.events.approval import SYSTEM_RESOLVER, UNKNOWN_CLASSIFICATION
from agentkit.guards.taint import TaintSource
from agentkit.tools.spec import SideEffects


def _common(seq: int = 0) -> dict:
    return {
        "event_id": new_id(EventId),
        "session_id": new_id(SessionId),
        "turn_id": new_id(TurnId),
        "ts": datetime.now(UTC),
        "sequence": seq,
    }


def _needed(**kwargs) -> ApprovalNeeded:
    fields = {
        "call_id": "c1",
        "tool_name": "vaultwarden_user_delete",
        "arguments": {"email": "REDACTED@example.test"},
        "risk": "destructive",
        "timeout_at": datetime.now(UTC),
        **kwargs,
    }
    return ApprovalNeeded(**_common(), **fields)


def test_side_effects_defaults_to_unknown():
    """An event built without a classification must not read as harmless."""
    assert _needed().side_effects == UNKNOWN_CLASSIFICATION


def test_side_effects_serialises_as_the_enum_value():
    """The wire carries "external_irreversible", never "SideEffects.…"."""
    ev = _needed(side_effects=SideEffects.EXTERNAL_IRREVERSIBLE.value)
    assert ev.model_dump(mode="json")["side_effects"] == "external_irreversible"


def test_side_effects_distinguishes_reversibility_at_equal_risk():
    reversible = _needed(risk="high_write", side_effects=SideEffects.EXTERNAL_REVERSIBLE.value)
    irreversible = _needed(risk="high_write", side_effects=SideEffects.EXTERNAL_IRREVERSIBLE.value)
    assert reversible.risk == irreversible.risk
    assert reversible.side_effects != irreversible.side_effects


def test_taint_defaults_empty_and_round_trips_through_the_union():
    assert _needed().taint == []

    ev = _needed(
        taint=[TaintSource(call_id="c0", tool_name="web.fetch", kind="untrusted")],
    )
    dumped = ev.model_dump(mode="json")
    assert dumped["taint"] == [
        {"call_id": "c0", "tool_name": "web.fetch", "kind": "untrusted"},
    ]

    parsed = EVENT_ADAPTER.validate_python(dumped)
    assert isinstance(parsed, ApprovalNeeded)
    assert parsed.taint[0].tool_name == "web.fetch"


def test_approval_resolved_round_trips_through_the_union():
    ev = ApprovalResolved(
        **_common(3),
        call_id="c1",
        decision="approve",
        resolved_by="u:morten",
        edited_args={"category": "movies"},
    )
    parsed = EVENT_ADAPTER.validate_python(ev.model_dump(mode="json"))
    assert isinstance(parsed, ApprovalResolved)
    assert parsed.type == "approval_resolved"
    assert parsed.decision == "approve"
    assert parsed.resolved_by == "u:morten"
    assert parsed.edited_args == {"category": "movies"}
    assert parsed.expired is False
    assert parsed.reason is None


def test_approval_resolved_expiry_is_a_denial():
    ev = ApprovalResolved(
        **_common(),
        call_id="c1",
        decision="deny",
        resolved_by=SYSTEM_RESOLVER,
        reason="approval window expired",
        expired=True,
    )
    assert ev.decision == "deny"
    assert ev.expired is True
    assert ev.model_dump(mode="json")["resolved_by"] == "system"
