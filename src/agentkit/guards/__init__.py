"""Guards — pluggable behaviour gates."""

from agentkit.guards.approval import (
    DEFAULT_APPROVAL_POLICY,
    KIT_NAMESPACE_PREFIXES,
    STRICT_APPROVAL_POLICY,
    ApprovalDecision,
    ApprovalGate,
    RiskBasedApprovalGate,
)
from agentkit.guards.finalize import (
    FinalizeValidator,
    FinalizeVerdict,
    StructuralFinalizeValidator,
)
from agentkit.guards.intent import (
    DEFAULT_TURNS_PER_MINUTE,
    ContentBlocklistCheck,
    DefaultIntentGate,
    InMemoryRateLimitCheck,
    IntentCheck,
    IntentDecision,
    IntentGate,
    MaxMessageLengthCheck,
    verified_principal,
)
from agentkit.guards.success_claim import (
    ClaimVerdict,
    RegexSuccessClaimGuard,
    SuccessClaimGuard,
)
from agentkit.guards.taint import (
    TAINT_DENIAL_CODE,
    TAINT_DENIAL_MESSAGE,
    NullTaintPolicy,
    RiskBasedTaintPolicy,
    TaintPolicy,
    TaintSource,
    is_tainted,
    mark_taint,
    taint_sources,
)

__all__ = [
    "DEFAULT_APPROVAL_POLICY",
    "DEFAULT_TURNS_PER_MINUTE",
    "KIT_NAMESPACE_PREFIXES",
    "STRICT_APPROVAL_POLICY",
    "TAINT_DENIAL_CODE",
    "TAINT_DENIAL_MESSAGE",
    "ApprovalDecision",
    "ApprovalGate",
    "ClaimVerdict",
    "ContentBlocklistCheck",
    "DefaultIntentGate",
    "FinalizeValidator",
    "FinalizeVerdict",
    "InMemoryRateLimitCheck",
    "IntentCheck",
    "IntentDecision",
    "IntentGate",
    "MaxMessageLengthCheck",
    "NullTaintPolicy",
    "RegexSuccessClaimGuard",
    "RiskBasedApprovalGate",
    "RiskBasedTaintPolicy",
    "StructuralFinalizeValidator",
    "SuccessClaimGuard",
    "TaintPolicy",
    "TaintSource",
    "is_tainted",
    "mark_taint",
    "taint_sources",
    "verified_principal",
]
