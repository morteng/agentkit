"""Wire snapshot tests for every concrete event class agentkit emits.

Two parts:
1. Parametrized snapshot test — constructs a canonical instance of each
   event class, dumps via model_dump(mode="json"), and asserts the dict
   matches the JSON snapshot under tests/wire/snapshots/<name>.json.
2. Meta-test — discovers every concrete BaseEvent subclass and fails if
   any is missing from EVENT_FIXTURES.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

import agentkit.events  # noqa: F401 — ensures all submodules are imported so subclasses register
from agentkit._messages import Usage
from agentkit.events.approval import (
    ApprovalDenied,
    ApprovalGranted,
    ApprovalNeeded,
    ApprovalResolved,
)
from agentkit.events.base import BaseEvent
from agentkit.events.lifecycle import (
    ErrorCode,
    Errored,
    TurnEnded,
    TurnEndReason,
    TurnMetrics,
    TurnStarted,
)
from agentkit.events.phase import PhaseChanged
from agentkit.events.streaming import (
    MessageCompleted,
    MessageStarted,
    TextDelta,
    ThinkingDelta,
    UsageRecorded,
)
from agentkit.events.subagent import SubagentEnded, SubagentEvent, SubagentStarted
from agentkit.events.tool import ToolCallProgress, ToolCallResult, ToolCallStarted
from agentkit.guards.taint import TaintSource
from agentkit.loop.phase import Phase
from tests.wire._snapshot_helper import assert_event_snapshot

# ---------------------------------------------------------------------------
# Canonical base-field values — pinned so snapshots capture the full wire shape
# ---------------------------------------------------------------------------

BASE_KWARGS: dict[str, Any] = {
    "event_id": "ev_canonical",
    "session_id": "sess_canonical",
    "turn_id": "turn_canonical",
    "ts": datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC),
    "sequence": 1,
}

# Shared canonical IDs for event-specific fields
MSG_ID = "msg_canonical"
CALL_ID = "call_canonical"
SUBAGENT_ID = "subagent_canonical"

# ---------------------------------------------------------------------------
# EVENT_FIXTURES: (EventClass, snapshot_name, event_kwargs)
# One entry per concrete event class in the Event union; the meta-test below
# fails if a class is added without one.
# ---------------------------------------------------------------------------

EVENT_FIXTURES: list[tuple[type[BaseEvent], str, dict[str, Any]]] = [
    # --- Lifecycle ---
    (
        TurnStarted,
        "turn_started",
        {"user_message_id": MSG_ID},
    ),
    (
        TurnEnded,
        "turn_ended",
        {
            "reason": TurnEndReason.COMPLETED,
            "metrics": TurnMetrics(),
            "summary": None,
        },
    ),
    (
        Errored,
        "errored",
        {
            "code": ErrorCode.PROVIDER_FAULT,
            "message": "upstream provider returned 500",
            "recoverable": True,
        },
    ),
    # --- Phase ---
    (
        PhaseChanged,
        "phase_changed",
        {
            "from_": Phase.IDLE,
            "to": Phase.STREAMING,
            "duration_ms": 42,
        },
    ),
    # --- Streaming ---
    (
        MessageStarted,
        "message_started",
        {"message_id": MSG_ID},
    ),
    (
        TextDelta,
        "text_delta",
        {
            "message_id": MSG_ID,
            "delta": "Hello, world!",
            "block_index": 0,
        },
    ),
    (
        ThinkingDelta,
        "thinking_delta",
        {
            "message_id": MSG_ID,
            "delta": "Let me think...",
        },
    ),
    (
        MessageCompleted,
        "message_completed",
        {
            "message_id": MSG_ID,
            "finish_reason": "end_turn",
        },
    ),
    (
        UsageRecorded,
        "usage_recorded",
        {
            "message_id": MSG_ID,
            "model": "openai/gpt-5",
            "usage": Usage(input_tokens=100, output_tokens=50, cached_input_tokens=20),
            "provider_name": "openrouter",
        },
    ),
    # --- Tool ---
    (
        ToolCallStarted,
        "tool_call_started",
        {
            "call_id": CALL_ID,
            "tool_name": "acme.search",
            "arguments": {"query": "Oslo Fjord"},
            "risk": "read",
        },
    ),
    (
        ToolCallProgress,
        "tool_call_progress",
        {
            "call_id": CALL_ID,
            "message": "Fetching results…",
            "progress": 0.5,
            "total": 1.0,
        },
    ),
    (
        ToolCallResult,
        "tool_call_result",
        {
            "call_id": CALL_ID,
            # Outcome events name the tool, same as the events that open the
            # call. Consumers were each rebuilding a call_id -> name map to
            # recover it; pinning it here is what stops it going missing again.
            "tool_name": "acme.search_content",
            "status": "ok",
            "content_summary": "Found 3 results",
            "duration_ms": 128,
            "cached": False,
            "error": None,
            "content": [],
        },
    ),
    # --- Approval ---
    (
        ApprovalNeeded,
        "approval_needed",
        {
            "call_id": CALL_ID,
            "tool_name": "acme.delete_content",
            "arguments": {"content_id": "cnt_abc123"},
            "rationale": "Destructive operation requires confirmation",
            "risk": "destructive",
            # Both non-default: a snapshot of the empty case would not show a
            # consumer what the reversibility axis or a taint entry looks like
            # on the wire, which is the whole point of pinning this event.
            "side_effects": "external_irreversible",
            "taint": [
                TaintSource(
                    call_id="call_web_fetch",
                    tool_name="web.fetch",
                    kind="untrusted",
                )
            ],
            "timeout_at": datetime(2026, 1, 1, 0, 5, 0, tzinfo=UTC),
        },
    ),
    (
        ApprovalGranted,
        "approval_granted",
        {
            "call_id": CALL_ID,
            "tool_name": "acme.delete_content",
            "edited_args": None,
        },
    ),
    (
        ApprovalDenied,
        "approval_denied",
        {
            "call_id": CALL_ID,
            "tool_name": "acme.delete_content",
            "reason": "User declined",
        },
    ),
    (
        ApprovalResolved,
        "approval_resolved",
        {
            "call_id": CALL_ID,
            # Named even on the expiry path. This is the row that used to audit
            # as a bare id, which is precisely when a reader needs the name.
            "tool_name": "acme.delete_content",
            # The expiry case, deliberately: it is the outcome the other two
            # approval events cannot express, and the reason this event exists.
            "decision": "deny",
            "resolved_by": "system",
            "edited_args": None,
            "reason": "approval window closed",
            "expired": True,
        },
    ),
    # --- Subagent ---
    (
        SubagentStarted,
        "subagent_started",
        {
            "subagent_id": SUBAGENT_ID,
            "parent_call_id": CALL_ID,
            "purpose": "Translate content to English",
        },
    ),
    (
        SubagentEvent,
        "subagent_event",
        {
            "subagent_id": SUBAGENT_ID,
            "inner": {"type": "text_delta", "delta": "…"},
        },
    ),
    (
        SubagentEnded,
        "subagent_ended",
        {
            "subagent_id": SUBAGENT_ID,
            "reason": TurnEndReason.COMPLETED,
            "summary": "Translation complete",
        },
    ),
]

# ---------------------------------------------------------------------------
# Parametrized snapshot test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls, snapshot_name, extra_kwargs",
    [(cls, name, kwargs) for cls, name, kwargs in EVENT_FIXTURES],
    ids=[name for _, name, _ in EVENT_FIXTURES],
)
def test_event_snapshot(
    cls: type[BaseEvent], snapshot_name: str, extra_kwargs: dict[str, Any]
) -> None:
    """Construct a canonical event instance and assert it matches the JSON snapshot."""
    kwargs = {**BASE_KWARGS, **extra_kwargs}
    event = cls(**kwargs)
    payload = event.model_dump(mode="json")
    assert_event_snapshot(payload, snapshot_name)


# ---------------------------------------------------------------------------
# Meta-test: every concrete BaseEvent subclass must have a snapshot fixture
# ---------------------------------------------------------------------------


def _all_concrete_event_classes() -> set[type[BaseEvent]]:
    """Return every concrete (non-abstract) direct subclass of BaseEvent.

    The import of agentkit.events at module top ensures all submodules are
    loaded and subclasses are registered before this is called.
    """
    return {
        cls for cls in BaseEvent.__subclasses__() if cls.__module__.startswith("agentkit.events.")
    }


def test_no_event_class_lacks_snapshot() -> None:
    """Fail if a concrete BaseEvent subclass is not covered by EVENT_FIXTURES."""
    declared_classes = {cls for cls, _, _ in EVENT_FIXTURES}
    discovered = _all_concrete_event_classes()
    missing = discovered - declared_classes
    assert not missing, (
        f"Event classes have no snapshot fixture: "
        f"{sorted(c.__name__ for c in missing)}\n"
        f"Add them to EVENT_FIXTURES in test_event_snapshots.py."
    )
