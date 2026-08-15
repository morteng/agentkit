"""An external MCP server must not get unattended write access by default.

The stdio client used to stamp every subprocess tool ``LOW_WRITE`` +
``BY_RISK``, which ``DEFAULT_APPROVAL_POLICY`` auto-approves — so connecting a
third-party server silently granted it the right to write without ever
prompting the user. These tests pin the fail-closed replacement.
"""

import sys
from typing import Any

import pytest

from agentkit.guards.approval import ApprovalDecision, RiskBasedApprovalGate
from agentkit.mcp_client.stdio import (
    UNCLASSIFIED_APPROVAL,
    UNCLASSIFIED_RISK,
    StdioMCPClient,
    _mcp_tool_to_spec,
)
from agentkit.tools.spec import (
    ApprovalPolicy,
    RiskLevel,
    SideEffects,
    ToolCall,
    ToolSpec,
)


class _FakeTool:
    def __init__(self, name: str, description: str = "d", schema: Any = None) -> None:
        self.name = name
        self.description = description
        self.inputSchema = schema or {"type": "object"}  # MCP wire name


class _FakeListResult:
    def __init__(self, tools: list[_FakeTool]) -> None:
        self.tools = tools


class _FakeSession:
    """Stands in for the MCP ClientSession; only list_tools is exercised."""

    def __init__(self, tools: list[_FakeTool]) -> None:
        self._tools = tools

    async def list_tools(self) -> _FakeListResult:
        return _FakeListResult(self._tools)


def _client(tools: list[_FakeTool], **kwargs: Any) -> StdioMCPClient:
    client = StdioMCPClient(name="srv", command=[sys.executable, "-c", "pass"], **kwargs)
    # No subprocess: the transport is irrelevant to classification.
    client._session = _FakeSession(tools)  # type: ignore[assignment]
    return client


def _classified(spec: ToolSpec) -> ToolSpec:
    return spec.model_copy(
        update={
            "risk": RiskLevel.READ,
            "idempotent": True,
            "side_effects": SideEffects.NONE,
            "requires_approval": ApprovalPolicy.BY_RISK,
        }
    )


def test_unclassified_mcp_tool_is_high_write_and_always_approved():
    spec = _mcp_tool_to_spec(_FakeTool("delete_everything"))
    assert spec.risk is UNCLASSIFIED_RISK
    assert spec.requires_approval is UNCLASSIFIED_APPROVAL
    assert spec.risk is not RiskLevel.LOW_WRITE, "LOW_WRITE is auto-approved by default"
    assert spec.side_effects is SideEffects.EXTERNAL_IRREVERSIBLE
    assert spec.idempotent is False


async def test_default_gate_sends_an_unclassified_mcp_tool_to_the_user():
    """The property that actually matters: the default approval gate must not
    wave this through. Asserting the risk enum alone would not catch a future
    change to the risk table."""
    spec = _mcp_tool_to_spec(_FakeTool("wipe_disk"))
    gate = RiskBasedApprovalGate()
    decision = await gate.decide(ToolCall(id="c1", name="wipe_disk", arguments={}), spec, None)
    assert decision is ApprovalDecision.NEEDS_USER


async def test_a_permissive_risk_table_cannot_re_open_the_hole():
    """``ALWAYS`` beats the risk table. A deployment that auto-approves its own
    HIGH_WRITE builtins must not thereby auto-approve third-party MCP tools."""
    spec = _mcp_tool_to_spec(_FakeTool("wipe_disk"))
    gate = RiskBasedApprovalGate(risk_policy={RiskLevel.HIGH_WRITE: ApprovalDecision.AUTO_APPROVE})
    decision = await gate.decide(ToolCall(id="c1", name="wipe_disk", arguments={}), spec, None)
    assert decision is ApprovalDecision.NEEDS_USER


async def test_list_tools_defaults_every_tool_to_unclassified():
    client = _client([_FakeTool("a"), _FakeTool("b")])
    specs = await client.list_tools()
    assert {s.name for s in specs} == {"a", "b"}
    assert all(s.risk is UNCLASSIFIED_RISK for s in specs)
    assert all(s.requires_approval is UNCLASSIFIED_APPROVAL for s in specs)


async def test_classifier_can_downgrade_a_tool_it_vouches_for():
    """Opting a tool down to auto-approvable must be an explicit act."""
    client = _client([_FakeTool("read_file")], classifier=_classified)
    (spec,) = await client.list_tools()
    assert spec.risk is RiskLevel.READ
    gate = RiskBasedApprovalGate()
    decision = await gate.decide(ToolCall(id="c1", name="read_file", arguments={}), spec, None)
    assert decision is ApprovalDecision.AUTO_APPROVE


async def test_classifier_returning_none_leaves_the_tool_fail_closed():
    client = _client([_FakeTool("mystery")], classifier=lambda _spec: None)
    (spec,) = await client.list_tools()
    assert spec.risk is UNCLASSIFIED_RISK
    assert spec.requires_approval is UNCLASSIFIED_APPROVAL


async def test_classifier_may_not_rename_a_tool():
    """A renamed spec would be registered under a name the server will not
    answer to — a silent mis-wiring, so it is an error, not a warning."""
    client = _client(
        [_FakeTool("real")],
        classifier=lambda spec: spec.model_copy(update={"name": "other"}),
    )
    with pytest.raises(ValueError, match="must not rename"):
        await client.list_tools()


async def test_require_classification_hides_unclassified_tools():
    client = _client(
        [_FakeTool("known"), _FakeTool("mystery")],
        classifier=lambda spec: _classified(spec) if spec.name == "known" else None,
        require_classification=True,
    )
    specs = await client.list_tools()
    assert [s.name for s in specs] == ["known"]


async def test_require_classification_refuses_the_call_even_if_the_spec_leaks():
    """Hiding a tool from list_tools is not enough on its own: a caller holding
    a stale spec must still be refused, and refused with a result rather than
    an exception so the model can read it."""
    client = _client(
        [_FakeTool("mystery")],
        classifier=lambda _spec: None,
        require_classification=True,
    )
    await client.list_tools()
    result = await client.call_tool("mystery", {})
    assert result.status == "denied"
    assert result.error is not None
    assert result.error.code == "mcp_tool_unclassified"


async def test_call_tool_is_not_gated_when_classification_is_not_required():
    """The default mode must not need list_tools to have run first — that path
    is guarded by approval, not by an allowlist."""
    client = _client([_FakeTool("echo")])
    result = await client.call_tool("echo", {})
    # _FakeSession has no call_tool, so the transport error is what comes back:
    # proof the call reached the transport instead of being refused upstream.
    assert result.error is not None
    assert result.error.code == "mcp_call_failed"
