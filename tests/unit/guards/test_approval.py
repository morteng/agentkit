import pytest

from agentkit.guards.approval import (
    DEFAULT_APPROVAL_POLICY,
    STRICT_APPROVAL_POLICY,
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
        policy_overrides={"REDACTED.devices.control": ApprovalDecision.NEEDS_USER},
    )
    decision = await gate.decide(
        ToolCall(id="c1", name="REDACTED.devices.control", arguments={}),
        _spec("REDACTED.devices.control", RiskLevel.READ),
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


@pytest.mark.asyncio
async def test_strict_preset_requires_user_for_low_writes():
    """The one-line security-sensitive preset: only READ runs unattended."""
    gate = RiskBasedApprovalGate.strict()
    assert (
        await gate.decide(
            ToolCall(id="c1", name="srv.write", arguments={}),
            _spec("srv.write", RiskLevel.LOW_WRITE),
            ctx=None,
        )
        is ApprovalDecision.NEEDS_USER
    )
    assert (
        await gate.decide(
            ToolCall(id="c2", name="srv.read", arguments={}),
            _spec("srv.read", RiskLevel.READ),
            ctx=None,
        )
        is ApprovalDecision.AUTO_APPROVE
    )


@pytest.mark.asyncio
async def test_strict_preset_ignores_a_third_party_never_declaration():
    """A server saying 'no approval needed' about itself is not evidence."""
    gate = RiskBasedApprovalGate.strict()
    decision = await gate.decide(
        ToolCall(id="c1", name="thirdparty.send_email", arguments={}),
        _spec("thirdparty.send_email", RiskLevel.LOW_WRITE, policy=ApprovalPolicy.NEVER),
        ctx=None,
    )
    assert decision is ApprovalDecision.NEEDS_USER


@pytest.mark.asyncio
async def test_strict_preset_still_honours_kit_never():
    """The runtime's own loop-control tools keep running unattended."""
    gate = RiskBasedApprovalGate.strict()
    decision = await gate.decide(
        ToolCall(id="c1", name="kit.memory.save", arguments={}),
        _spec("kit.memory.save", RiskLevel.LOW_WRITE, policy=ApprovalPolicy.NEVER),
        ctx=None,
    )
    assert decision is ApprovalDecision.AUTO_APPROVE


@pytest.mark.asyncio
async def test_strict_preset_keeps_always_and_overrides():
    gate = RiskBasedApprovalGate.strict(
        policy_overrides={"srv.safe": ApprovalDecision.AUTO_APPROVE}
    )
    assert (
        await gate.decide(
            ToolCall(id="c1", name="srv.safe", arguments={}),
            _spec("srv.safe", RiskLevel.DESTRUCTIVE),
            ctx=None,
        )
        is ApprovalDecision.AUTO_APPROVE
    )
    assert (
        await gate.decide(
            ToolCall(id="c2", name="kit.thing", arguments={}),
            _spec("kit.thing", RiskLevel.READ, policy=ApprovalPolicy.ALWAYS),
            ctx=None,
        )
        is ApprovalDecision.NEEDS_USER
    )


def test_default_policy_is_unchanged_by_the_strict_preset():
    """Existing consumers keep the interactive default; strict is opt-in."""
    assert DEFAULT_APPROVAL_POLICY[RiskLevel.LOW_WRITE] is ApprovalDecision.AUTO_APPROVE
    assert STRICT_APPROVAL_POLICY[RiskLevel.LOW_WRITE] is ApprovalDecision.NEEDS_USER
    assert STRICT_APPROVAL_POLICY[RiskLevel.READ] is ApprovalDecision.AUTO_APPROVE
    assert set(STRICT_APPROVAL_POLICY) == set(DEFAULT_APPROVAL_POLICY)
