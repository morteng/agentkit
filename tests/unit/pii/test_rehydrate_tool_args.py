"""rehydrate_tool_args: DENY passthrough vs ALLOW allowlisted + refusal."""

import pytest

from agentkit.pii.firewall import Firewall, RehydrationRefused
from agentkit.pii.policy import PiiPolicy
from agentkit.pii.types import RehydratePolicy
from agentkit.tools.spec import ApprovalPolicy, RiskLevel, SideEffects, ToolSpec

from .conftest import FakeDetector, FakeTokenMap


def _fw(detector: FakeDetector) -> Firewall:
    return Firewall(detector=detector, policy=PiiPolicy())


def _tool(name: str, rehydrate: RehydratePolicy) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="d",
        parameters={"type": "object"},
        returns=None,
        risk=RiskLevel.LOW_WRITE,
        idempotent=True,
        side_effects=SideEffects.LOCAL,
        requires_approval=ApprovalPolicy.NEVER,
        cache_ttl_seconds=None,
        timeout_seconds=10.0,
        rehydrate=rehydrate,
    )


def test_default_rehydrate_is_deny():
    t = _tool("x", RehydratePolicy.DENY)
    assert t.rehydrate is RehydratePolicy.DENY


def test_deny_returns_args_unchanged(detector: FakeDetector, tmap: FakeTokenMap):
    fw = _fw(detector)
    tmap.token_for("kari@example.no", "EMAIL")  # [EMAIL_1]
    args = {"body": "reach me at [EMAIL_1]"}
    out = fw.rehydrate_tool_args(_tool("send", RehydratePolicy.DENY), args, tmap)
    # Tokens stay tokens — exfil channel closed.
    assert out["body"] == "reach me at [EMAIL_1]"


def test_allow_rehydrates_known_tokens(detector: FakeDetector, tmap: FakeTokenMap):
    fw = _fw(detector)
    tmap.token_for("kari@example.no", "EMAIL")  # [EMAIL_1]
    args = {"field_html": "<p>[EMAIL_1]</p>"}
    out = fw.rehydrate_tool_args(_tool("fill_form", RehydratePolicy.ALLOW), args, tmap)
    assert out["field_html"] == "<p>kari@example.no</p>"


def test_allow_ignores_unknown_tokens(detector: FakeDetector, tmap: FakeTokenMap):
    fw = _fw(detector)
    args = {"field_html": "[EMAIL_99] not in map"}
    out = fw.rehydrate_tool_args(_tool("fill_form", RehydratePolicy.ALLOW), args, tmap)
    assert out["field_html"] == "[EMAIL_99] not in map"  # unchanged


@pytest.mark.parametrize("dest_field", ["url", "callback_url", "webhook", "recipient", "dest"])
def test_allow_refuses_destination_fields(
    detector: FakeDetector, tmap: FakeTokenMap, dest_field: str
):
    fw = _fw(detector)
    tmap.token_for("kari@example.no", "EMAIL")
    args = {dest_field: "https://evil.no/?e=[EMAIL_1]"}
    with pytest.raises(RehydrationRefused):
        fw.rehydrate_tool_args(_tool("bad", RehydratePolicy.ALLOW), args, tmap)
