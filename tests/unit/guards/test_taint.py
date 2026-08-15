"""Taint guard: once untrusted content lands in a turn, writes are off."""

from typing import Any

from agentkit._content import Provenance
from agentkit.guards.taint import (
    TAINT_DENIAL_MESSAGE,
    NullTaintPolicy,
    RiskBasedTaintPolicy,
    TaintSource,
    denied_result,
    is_tainted,
    mark_taint,
    taint_sources,
)
from agentkit.loop.context import TurnContext
from agentkit.tools.spec import (
    ApprovalPolicy,
    ContentBlockOut,
    RiskLevel,
    SideEffects,
    ToolResult,
    ToolSpec,
)

WRITE_RISKS = [RiskLevel.LOW_WRITE, RiskLevel.HIGH_WRITE, RiskLevel.DESTRUCTIVE]


def _spec(name: str, risk: RiskLevel) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="d",
        parameters={"type": "object"},
        returns=None,
        risk=risk,
        idempotent=True,
        side_effects=SideEffects.NONE,
        requires_approval=ApprovalPolicy.NEVER,
        cache_ttl_seconds=None,
        timeout_seconds=10.0,
    )


def _result(provenance: Provenance = Provenance.SYSTEM, call_id: str = "c1") -> ToolResult:
    return ToolResult(
        call_id=call_id,
        status="ok",
        content=[ContentBlockOut(type="text", text="body")],
        provenance=provenance,
    )


def test_untainted_turn_allows_every_risk_level():
    policy = RiskBasedTaintPolicy()
    ctx = TurnContext.empty()
    for risk in [RiskLevel.READ, *WRITE_RISKS]:
        assert policy.denial_reason(_spec("t", risk), ctx) is None


def test_tainted_turn_denies_everything_above_read():
    policy = RiskBasedTaintPolicy()
    ctx = TurnContext.empty()
    ctx.tainted = True

    assert policy.denial_reason(_spec("kit.read", RiskLevel.READ), ctx) is None
    for risk in WRITE_RISKS:
        reason = policy.denial_reason(_spec("kit.write", risk), ctx)
        assert reason is not None
        # The message has to be narratable: the model must be able to tell the
        # user what happened and what to do about it.
        assert reason.startswith(TAINT_DENIAL_MESSAGE)
        assert "new turn" in reason
        assert "kit.write" in reason


def test_ceiling_is_configurable():
    """A consumer can allow low writes but still block the dangerous tiers."""
    policy = RiskBasedTaintPolicy(max_risk_when_tainted=RiskLevel.LOW_WRITE)
    ctx = TurnContext.empty()
    ctx.tainted = True

    assert policy.denial_reason(_spec("t", RiskLevel.LOW_WRITE), ctx) is None
    assert policy.denial_reason(_spec("t", RiskLevel.HIGH_WRITE), ctx) is not None
    assert policy.denial_reason(_spec("t", RiskLevel.DESTRUCTIVE), ctx) is not None


def test_unknown_risk_level_fails_closed():
    """A risk value the rank table does not know counts as the worst case."""

    class _WeirdSpec:
        name = "vendor.mystery"
        risk = "vendor_specific_risk"

    ctx = TurnContext.empty()
    ctx.tainted = True
    reason = RiskBasedTaintPolicy().denial_reason(_WeirdSpec(), ctx)  # type: ignore[arg-type]
    assert reason is not None


def test_null_policy_never_denies():
    ctx = TurnContext.empty()
    ctx.tainted = True
    for risk in [RiskLevel.READ, *WRITE_RISKS]:
        assert NullTaintPolicy().denial_reason(_spec("t", risk), ctx) is None


def test_is_tainted_tolerates_contexts_without_the_attribute():
    class _Bare:
        call_id = "c1"

    assert is_tainted(_Bare()) is False


def test_mark_taint_latches_on_first_untrusted_result():
    ctx = TurnContext.empty()
    assert ctx.tainted is False

    assert mark_taint(ctx, _result(Provenance.SYSTEM)) is False
    assert mark_taint(ctx, _result(Provenance.PRINCIPAL)) is False
    assert ctx.tainted is False

    # First untrusted result flips it and reports the transition...
    assert mark_taint(ctx, _result(Provenance.UNTRUSTED)) is True
    assert ctx.tainted is True
    # ...every later one is a no-op, so callers log the event once.
    assert mark_taint(ctx, _result(Provenance.UNTRUSTED)) is False
    assert ctx.tainted is True

    # And taint never clears mid-turn, whatever arrives afterwards.
    mark_taint(ctx, _result(Provenance.SYSTEM))
    assert ctx.tainted is True


def test_mark_taint_survives_a_context_that_rejects_attributes():
    class _Slotted:
        __slots__ = ()

    obj: Any = _Slotted()
    assert mark_taint(obj, _result(Provenance.UNTRUSTED)) is False


def test_denied_result_is_a_normal_readable_result():
    res = denied_result("call-7", "denied: nope")
    assert res.status == "denied"
    assert res.call_id == "call-7"
    assert res.content[0].text == "denied: nope"
    assert res.error is not None
    assert res.error.code == "tainted_turn"
    # Retrying inside the same turn cannot succeed — do not invite it.
    assert res.error.retryable is False


def test_mark_taint_records_every_untrusted_source_in_order():
    """The approval card needs what tainted the turn, not just that it is tainted."""
    ctx = TurnContext.empty()
    assert taint_sources(ctx) == []

    mark_taint(ctx, _result(Provenance.SYSTEM, call_id="c0"), tool_name="kit.clock")
    assert taint_sources(ctx) == []

    mark_taint(ctx, _result(Provenance.UNTRUSTED, call_id="c1"), tool_name="web.fetch")
    # Later untrusted results are still recorded even though mark_taint has
    # already latched and returns False for them.
    mark_taint(ctx, _result(Provenance.UNTRUSTED, call_id="c2"), tool_name="mail.read")

    assert taint_sources(ctx) == [
        TaintSource(call_id="c1", tool_name="web.fetch", kind="untrusted"),
        TaintSource(call_id="c2", tool_name="mail.read", kind="untrusted"),
    ]


def test_repeated_identical_result_is_recorded_once():
    ctx = TurnContext.empty()
    for _ in range(3):
        mark_taint(ctx, _result(Provenance.UNTRUSTED), tool_name="web.fetch")
    assert len(taint_sources(ctx)) == 1


def test_taint_sources_tolerates_contexts_without_the_attribute():
    class _Bare:
        call_id = "c1"

    bare = _Bare()
    # Neither reading nor writing may raise on a context object that predates
    # the field — same tolerance is_tainted has.
    assert taint_sources(bare) == []
    mark_taint(bare, _result(Provenance.UNTRUSTED), tool_name="web.fetch")
    assert taint_sources(bare) == []


def test_taint_sources_returns_a_copy():
    """A consumer mutating the returned list must not rewrite turn state."""
    ctx = TurnContext.empty()
    mark_taint(ctx, _result(Provenance.UNTRUSTED), tool_name="web.fetch")
    taint_sources(ctx).clear()
    assert len(ctx.taint_sources) == 1
