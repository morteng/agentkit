"""ApprovalGate — decide which tool calls require user approval."""

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from agentkit.tools.spec import (
    ApprovalPolicy,
    RiskLevel,
    ToolCall,
    ToolSpec,
)


class ApprovalDecision(StrEnum):
    AUTO_APPROVE = "auto_approve"
    NEEDS_USER = "needs_user"
    AUTO_DENY = "auto_deny"


@runtime_checkable
class ApprovalGate(Protocol):
    async def decide(
        self,
        call: ToolCall,
        spec: ToolSpec,
        ctx: Any,  # TurnContext — typed loosely to avoid circular import
    ) -> ApprovalDecision: ...


DEFAULT_APPROVAL_POLICY: dict[RiskLevel, ApprovalDecision] = {
    RiskLevel.READ: ApprovalDecision.AUTO_APPROVE,
    RiskLevel.LOW_WRITE: ApprovalDecision.AUTO_APPROVE,
    RiskLevel.HIGH_WRITE: ApprovalDecision.NEEDS_USER,
    RiskLevel.DESTRUCTIVE: ApprovalDecision.NEEDS_USER,
}


class RiskBasedApprovalGate(ApprovalGate):
    """Decide based on ToolSpec.requires_approval and RiskLevel.

    Spec-level policy ALWAYS / NEVER short-circuits the risk table.
    BY_RISK looks up the risk table, then per-tool overrides apply.
    """

    def __init__(
        self,
        *,
        risk_policy: dict[RiskLevel, ApprovalDecision] | None = None,
        policy_overrides: dict[str, ApprovalDecision] | None = None,
    ) -> None:
        self._risk_policy = {**DEFAULT_APPROVAL_POLICY, **(risk_policy or {})}
        self._overrides = policy_overrides or {}

    async def decide(self, call: ToolCall, spec: ToolSpec, ctx: Any) -> ApprovalDecision:
        if call.name in self._overrides:
            return self._overrides[call.name]
        if spec.requires_approval is ApprovalPolicy.ALWAYS:
            return ApprovalDecision.NEEDS_USER
        if spec.requires_approval is ApprovalPolicy.NEVER:
            return ApprovalDecision.AUTO_APPROVE
        return self._risk_policy.get(spec.risk, ApprovalDecision.NEEDS_USER)
