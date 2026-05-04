import pytest

from agentkit.guards.approval import (
    DEFAULT_APPROVAL_POLICY,
    ApprovalDecision,
    RiskBasedApprovalGate,
)
from agentkit.tools.spec import (
    ApprovalPolicy,
    RiskLevel,
    SideEffects,
    ToolCall,
    ToolSpec,
)


def _spec(name: str, risk: RiskLevel, policy: ApprovalPolicy = ApprovalPolicy.BY_RISK) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="",
        parameters={"type": "object"},
        returns=None,
        risk=risk,
        idempotent=True,
        side_effects=SideEffects.NONE,
        requires_approval=policy,
        cache_ttl_seconds=None,
        timeout_seconds=10.0,
    )


@pytest.mark.asyncio
async def test_default_policy_auto_approves_reads():
    gate = RiskBasedApprovalGate()
    decision = await gate.decide(
        ToolCall(id="c1", name="x", arguments={}),
        _spec("x", RiskLevel.READ),
        ctx=None,
    )
    assert decision is ApprovalDecision.AUTO_APPROVE


@pytest.mark.asyncio
async def test_default_policy_requires_user_for_high_writes():
    gate = RiskBasedApprovalGate()
    decision = await gate.decide(
        ToolCall(id="c1", name="x", arguments={}),
        _spec("x", RiskLevel.HIGH_WRITE),
        ctx=None,
    )
    assert decision is ApprovalDecision.NEEDS_USER


@pytest.mark.asyncio
async def test_per_tool_override_takes_precedence():
    gate = RiskBasedApprovalGate(
        policy_overrides={"ampaera.devices.control": ApprovalDecision.NEEDS_USER},
    )
    decision = await gate.decide(
        ToolCall(id="c1", name="ampaera.devices.control", arguments={}),
        _spec("ampaera.devices.control", RiskLevel.READ),
        ctx=None,
    )
    assert decision is ApprovalDecision.NEEDS_USER


@pytest.mark.asyncio
async def test_spec_approval_policy_always_overrides_risk_policy():
    gate = RiskBasedApprovalGate()
    decision = await gate.decide(
        ToolCall(id="c1", name="x", arguments={}),
        _spec("x", RiskLevel.READ, policy=ApprovalPolicy.ALWAYS),
        ctx=None,
    )
    assert decision is ApprovalDecision.NEEDS_USER


@pytest.mark.asyncio
async def test_spec_approval_policy_never_overrides_risk_policy():
    gate = RiskBasedApprovalGate()
    decision = await gate.decide(
        ToolCall(id="c1", name="x", arguments={}),
        _spec("x", RiskLevel.HIGH_WRITE, policy=ApprovalPolicy.NEVER),
        ctx=None,
    )
    assert decision is ApprovalDecision.AUTO_APPROVE


def test_default_policy_table_complete():
    assert RiskLevel.READ in DEFAULT_APPROVAL_POLICY
    assert RiskLevel.LOW_WRITE in DEFAULT_APPROVAL_POLICY
    assert RiskLevel.HIGH_WRITE in DEFAULT_APPROVAL_POLICY
    assert RiskLevel.DESTRUCTIVE in DEFAULT_APPROVAL_POLICY
