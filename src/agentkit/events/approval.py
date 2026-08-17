"""Approval-gate events."""

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from agentkit.events.base import BaseEvent
from agentkit.guards.taint import TaintSource

UNKNOWN_CLASSIFICATION = "unknown"
"""Value for ``risk``/``side_effects`` when the call's spec is not in the
registry. Consumers treat it at destructive friction — an unclassified tool is
not a safe tool."""

SYSTEM_RESOLVER = "system"
"""``ApprovalResolved.resolved_by`` for a verdict agentkit reached on its own
rather than one a human gave."""

UNNAMED_TOOL = ""
"""``tool_name`` when the call was not in any of the turn's call buckets.

Not expected on any path agentkit itself emits — the name is looked up from
the same metadata the verdict was applied to. It exists so the lookup has a
total return type rather than an optional one, and so a consumer that does
render the name has a single falsy case to handle."""


class ApprovalNeeded(BaseEvent):
    type: Literal["approval_needed"] = Field(default="approval_needed")  # type: ignore[reportIncompatibleVariableOverride]
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    rationale: str | None = None
    risk: str  # RiskLevel value as str
    side_effects: str = UNKNOWN_CLASSIFICATION
    """SideEffects value as str, or ``"unknown"``.

    Reversibility is a separate axis from ``risk``: a HIGH_WRITE that renames a
    playlist and an EXTERNAL_IRREVERSIBLE that deletes an account carry the same
    risk level, and a consumer that cannot tell them apart either over-warns on
    everything or under-warns on the one that matters. ToolSpec already knows;
    without it here every consumer rebuilds a hand-maintained
    tool-to-reversibility map that drifts the moment a tool is added.
    """
    taint: list[TaintSource] = Field(default_factory=list)  # type: ignore[reportUnknownVariableType]
    """Untrusted-content results this turn ingested before the call was
    proposed, in ingestion order. Empty for a clean turn. Non-empty means the
    model read third-party text before proposing this action — the case a human
    approver has to look at hardest. See :mod:`agentkit.guards.taint`."""
    timeout_at: datetime


class ApprovalGranted(BaseEvent):
    type: Literal["approval_granted"] = Field(default="approval_granted")  # type: ignore[reportIncompatibleVariableOverride]
    call_id: str
    tool_name: str
    """What was approved, not just which call id.

    :class:`ApprovalNeeded` carries the name; every event reporting the
    *outcome* used to drop it, so a consumer holding a verdict frame could not
    say what it authorised without keeping its own ``call_id -> name`` map back
    to the request. Three independent consumers built that map before this
    field existed, and agentkit builds it twice internally
    (``AgentSession._tool_name_for_call``, and ``name_by_id`` in the
    tool-results handler) — so the name was never missing, only unpublished.

    :data:`UNNAMED_TOOL` when the lookup found nothing.
    """
    edited_args: dict[str, Any] | None = None


class ApprovalDenied(BaseEvent):
    type: Literal["approval_denied"] = Field(default="approval_denied")  # type: ignore[reportIncompatibleVariableOverride]
    call_id: str
    tool_name: str
    """What was refused. See :attr:`ApprovalGranted.tool_name`."""
    reason: str | None = None


class ApprovalResolved(BaseEvent):
    """A pending approval reached a terminal state — however it got there.

    Emitted alongside :class:`ApprovalGranted`/:class:`ApprovalDenied`, which
    stay as they are for existing consumers, and — unlike them — **also on the
    timeout path**. An approval that expires emits nothing today, which a
    client cannot distinguish from a hung turn, so its card stays on screen
    forever. One event type covers every outcome, so a card has exactly one
    thing to listen for.
    """

    type: Literal["approval_resolved"] = Field(default="approval_resolved")  # type: ignore[reportIncompatibleVariableOverride]
    call_id: str
    tool_name: str
    """What resolved. See :attr:`ApprovalGranted.tool_name`.

    Carried on the expiry path too, where it is the only place the name is
    available at all: expiry emits no verdict event and the checkpoint that
    held the pending call is gone by the time a consumer sees this.
    """
    decision: Literal["approve", "deny"]
    """Normalised. The session treats anything that is not exactly ``"approve"``
    as a denial, so this reports which of the two actually happened rather than
    echoing what the client sent."""
    resolved_by: str
    """Who ruled: the session owner for a human verdict, ``"system"`` for one
    agentkit reached on its own (expiry, or a call the resume did not rule on)."""
    edited_args: dict[str, Any] | None = None
    reason: str | None = None
    expired: bool = False
    """True when the approval window closed before anyone answered. ``decision``
    is then ``"deny"`` — an unanswered approval is not consent."""
