"""ApprovalGate — decide which tool calls require user approval.

READ THIS BEFORE SHIPPING A CONSUMER
------------------------------------
:data:`DEFAULT_APPROVAL_POLICY` auto-approves ``READ`` **and** ``LOW_WRITE``.
The honesty of your risk classification is therefore the whole boundary: every
tool registered ``LOW_WRITE`` runs unattended, so a config change, a message to
a third party, or a device actuation filed under ``LOW_WRITE`` runs unattended
too. ``LOW_WRITE`` means "reversible, small blast radius, the user would not
want to be asked" — nothing weaker.

Two presets ship:

* :data:`DEFAULT_APPROVAL_POLICY` — READ and LOW_WRITE auto-approve. The
  interactive default; prompting on every low write makes an assistant
  unusable.
* :data:`STRICT_APPROVAL_POLICY` — only READ auto-approves. Select it in one
  line with :meth:`RiskBasedApprovalGate.strict`, which additionally stops
  trusting a third-party tool's own ``ApprovalPolicy.NEVER`` declaration.

Tools arriving from an external MCP server are classified ``HIGH_WRITE`` /
``ApprovalPolicy.ALWAYS`` until a consumer classifies them (see
``agentkit.mcp_client.stdio.UNCLASSIFIED_RISK``), so they never ride in on the
LOW_WRITE row.
"""

from collections.abc import Sequence
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


#: The interactive default: READ and LOW_WRITE run unattended, anything with a
#: real blast radius asks. See the module docstring — this table is only as
#: sound as the risk levels the registry declares.
DEFAULT_APPROVAL_POLICY: dict[RiskLevel, ApprovalDecision] = {
    RiskLevel.READ: ApprovalDecision.AUTO_APPROVE,
    RiskLevel.LOW_WRITE: ApprovalDecision.AUTO_APPROVE,
    RiskLevel.HIGH_WRITE: ApprovalDecision.NEEDS_USER,
    RiskLevel.DESTRUCTIVE: ApprovalDecision.NEEDS_USER,
}

#: Security-sensitive preset: only reads run unattended. Every write asks,
#: including the ones somebody classified as small. Use it when the tool
#: catalogue is large, third-party, or changes without review — and accept the
#: prompt volume that comes with it.
STRICT_APPROVAL_POLICY: dict[RiskLevel, ApprovalDecision] = {
    RiskLevel.READ: ApprovalDecision.AUTO_APPROVE,
    RiskLevel.LOW_WRITE: ApprovalDecision.NEEDS_USER,
    RiskLevel.HIGH_WRITE: ApprovalDecision.NEEDS_USER,
    RiskLevel.DESTRUCTIVE: ApprovalDecision.NEEDS_USER,
}

#: Tool-name prefixes whose ``ApprovalPolicy.NEVER`` a strict gate still
#: honours: the runtime's own loop-control and bookkeeping tools (finalize,
#: note, memory, subagent spawn). They ship with this library rather than
#: being a third party describing itself.
KIT_NAMESPACE_PREFIXES: tuple[str, ...] = ("kit.",)


class RiskBasedApprovalGate(ApprovalGate):
    """Decide based on ToolSpec.requires_approval and RiskLevel.

    Resolution order:

    1. ``policy_overrides[call.name]`` — the consumer's per-tool last word.
    2. Spec-level ``ApprovalPolicy.ALWAYS`` -> NEEDS_USER.
    3. Spec-level ``ApprovalPolicy.NEVER`` -> AUTO_APPROVE, subject to
       ``spec_never_prefixes``.
    4. The risk table (``risk_policy``, defaulting to
       :data:`DEFAULT_APPROVAL_POLICY`), with NEEDS_USER for an unknown risk.

    ``spec_never_prefixes`` restricts step 3 to tools whose name starts with one
    of the given prefixes. ``None`` (the default) honours ``NEVER`` for every
    tool, which is the historical behaviour; :meth:`strict` narrows it to
    :data:`KIT_NAMESPACE_PREFIXES`, because a server declaring "no approval
    needed" *about itself* is not evidence.
    """

    def __init__(
        self,
        *,
        risk_policy: dict[RiskLevel, ApprovalDecision] | None = None,
        policy_overrides: dict[str, ApprovalDecision] | None = None,
        spec_never_prefixes: Sequence[str] | None = None,
    ) -> None:
        self._risk_policy = {**DEFAULT_APPROVAL_POLICY, **(risk_policy or {})}
        self._overrides = policy_overrides or {}
        self._never_prefixes = (
            tuple(spec_never_prefixes) if spec_never_prefixes is not None else None
        )

    @classmethod
    def strict(
        cls,
        *,
        policy_overrides: dict[str, ApprovalDecision] | None = None,
        spec_never_prefixes: Sequence[str] | None = KIT_NAMESPACE_PREFIXES,
    ) -> "RiskBasedApprovalGate":
        """The one-line security-sensitive preset.

        ``config.guards.approval = RiskBasedApprovalGate.strict()`` — only READ
        auto-approves, and only ``kit.*`` tools keep their spec-level
        ``ApprovalPolicy.NEVER``. Pass ``spec_never_prefixes=None`` to keep
        honouring ``NEVER`` everywhere while still tightening the risk table.
        """
        return cls(
            risk_policy=dict(STRICT_APPROVAL_POLICY),
            policy_overrides=policy_overrides,
            spec_never_prefixes=spec_never_prefixes,
        )

    def _honours_spec_never(self, name: str) -> bool:
        if self._never_prefixes is None:
            return True
        return name.startswith(self._never_prefixes)

    async def decide(self, call: ToolCall, spec: ToolSpec, ctx: Any) -> ApprovalDecision:
        if call.name in self._overrides:
            return self._overrides[call.name]
        if spec.requires_approval is ApprovalPolicy.ALWAYS:
            return ApprovalDecision.NEEDS_USER
        if spec.requires_approval is ApprovalPolicy.NEVER and self._honours_spec_never(call.name):
            return ApprovalDecision.AUTO_APPROVE
        return self._risk_policy.get(spec.risk, ApprovalDecision.NEEDS_USER)
