"""The WebSocket transport must let a human answer EVERY pending approval.

These tests drive the transport, not ``AgentSession``. That distinction is the
whole point: ``resume_with_approval_batch`` was fully implemented and covered
by its own e2e tests for months while no transport ever called it, so a turn
that suspended on two tool calls could only ever receive one verdict — the
singular ``respond_to_approval`` handler resumed on one call_id and
``_deny_unresolved`` closed the rest out as ``"not resolved in this resume"``.
A test of the session method could not have caught that. A test of the frame
can.
"""

from collections.abc import Awaitable, Callable

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from agentkit import AgentConfig, AgentSession
from agentkit._ids import OwnerId
from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.providers.fakes import FakeProvider
from agentkit.store.fakes import FakeCheckpointStore, FakeMemoryStore, FakeSessionStore
from agentkit.tools.registry import ToolRegistry
from agentkit.tools.spec import (
    ApprovalPolicy,
    ContentBlockOut,
    RiskLevel,
    SideEffects,
    ToolResult,
    ToolSpec,
)
from agentkit.transports.websocket import mount_websocket_route

pytestmark = pytest.mark.integration

#: See ``test_websocket.py`` — TestClient sends no Origin header.
NO_ORIGIN = [""]

#: The auto-deny reason ``AgentSession._deny_unresolved`` stamps on every
#: pending call a resume did not rule on. Duplicated as a literal on purpose:
#: it is what the *client* sees on the wire, and asserting against the
#: constant would pass even if the constant changed out from under the
#: contract that documents this string.
UNRESOLVED = "not resolved in this resume"


class _StubAuth:
    async def authenticate(self, ws: WebSocket) -> bool:
        return True


def _two_approval_factory() -> tuple[Callable[[WebSocket], Awaitable[AgentSession]], list[dict]]:
    """Session factory whose first turn emits TWO HIGH_WRITE calls, so the turn
    suspends with two pending approvals.

    Returns the factory and the list every execution of the guarded tool
    appends to, so a test can assert on what actually ran rather than only on
    frames.
    """
    executions: list[dict] = []

    spec = ToolSpec(
        name="globex.devices.delete",
        description="delete device (irreversible)",
        parameters={"type": "object"},
        returns=None,
        risk=RiskLevel.HIGH_WRITE,
        idempotent=False,
        side_effects=SideEffects.EXTERNAL_IRREVERSIBLE,
        requires_approval=ApprovalPolicy.BY_RISK,
        cache_ttl_seconds=None,
        timeout_seconds=10.0,
    )

    async def handler(args, ctx):
        executions.append(dict(args))
        return ToolResult(
            call_id=ctx.call_id,
            status="ok",
            content=[ContentBlockOut(type="text", text="deleted")],
            error=None,
            duration_ms=1,
            cached=False,
        )

    async def session_factory(ws: WebSocket) -> AgentSession:
        config = AgentConfig()
        config.guards.approval = RiskBasedApprovalGate()
        config.stores.session = FakeSessionStore()
        config.stores.memory = FakeMemoryStore()
        config.stores.checkpoint = FakeCheckpointStore()

        registry = ToolRegistry()
        registry.register_default_builtins()
        registry.register_builtin(spec, handler)

        provider = FakeProvider().script(
            FakeProvider.tool_calls(
                [
                    ("globex.devices.delete", {"id": "a"}),
                    ("globex.devices.delete", {"id": "b"}),
                ]
            ),
            FakeProvider.tool_call(
                "kit.finalize",
                {
                    "status": "done",
                    "intent_kind": "action",
                    "summary": "Done.",
                    "actions_performed": [],
                },
            ),
        )
        return AgentSession(
            owner=OwnerId("u:test"),
            config=config,
            provider=provider,
            registry=registry,
            model="m",
        )

    return session_factory, executions


def _two_approval_app(**mount_kwargs) -> tuple[FastAPI, list[dict]]:
    """The factory above, mounted at ``/ws/agent``."""
    session_factory, executions = _two_approval_factory()
    app = FastAPI()
    mount_websocket_route(
        app,
        path="/ws/agent",
        session_factory=session_factory,
        origin_allowlist=NO_ORIGIN,
        auth=_StubAuth(),
        **mount_kwargs,
    )
    return app, executions


def _drain_turn(ws) -> list[dict]:
    """Read frames until the turn ends. Returns every frame including the last."""
    frames: list[dict] = []
    while True:
        frame = ws.receive_json()
        frames.append(frame)
        if frame["type"] == "turn_ended":
            return frames


def _suspend_on_two(ws) -> list[dict]:
    """Drive one turn to its two-approval checkpoint; return the two frames."""
    ws.send_json({"type": "send_message", "text": "delete a and b"})
    frames = _drain_turn(ws)
    needed = [f for f in frames if f["type"] == "approval_needed"]
    assert len(needed) == 2, f"expected two checkpoints, got {[f['type'] for f in frames]}"
    return needed


def test_respond_to_approvals_answers_every_pending_checkpoint():
    """The decisive test: two checkpoints, two verdicts, nothing auto-denied."""
    app, executions = _two_approval_app()
    client = TestClient(app)
    with client.websocket_connect("/ws/agent") as ws:
        needed = _suspend_on_two(ws)
        assert executions == []  # nothing ran before the human answered

        ws.send_json(
            {
                "type": "respond_to_approvals",
                "turn_id": needed[0]["turn_id"],
                "decisions": [
                    {"call_id": needed[0]["call_id"], "decision": "approve"},
                    {"call_id": needed[1]["call_id"], "decision": "approve"},
                ],
            }
        )
        frames = _drain_turn(ws)

    granted = {f["call_id"] for f in frames if f["type"] == "approval_granted"}
    assert granted == {needed[0]["call_id"], needed[1]["call_id"]}

    # Neither checkpoint was closed out by agentkit on the human's behalf.
    assert [f for f in frames if f.get("reason") == UNRESOLVED] == []
    assert all(f.get("resolved_by") != "system" for f in frames if f["type"] == "approval_resolved")

    # And the approvals meant something: both tools actually ran.
    assert sorted(e["id"] for e in executions) == ["a", "b"]


def test_respond_to_approvals_carries_per_call_deny_and_edited_args():
    """One frame, two different verdicts — the mixed case a single card needs."""
    app, executions = _two_approval_app()
    client = TestClient(app)
    with client.websocket_connect("/ws/agent") as ws:
        needed = _suspend_on_two(ws)
        ws.send_json(
            {
                "type": "respond_to_approvals",
                "turn_id": needed[0]["turn_id"],
                "decisions": [
                    {
                        "call_id": needed[0]["call_id"],
                        "decision": "approve",
                        "edited_args": {"id": "a-edited"},
                    },
                    {
                        "call_id": needed[1]["call_id"],
                        "decision": "deny",
                        "reason": "not that one",
                    },
                ],
            }
        )
        frames = _drain_turn(ws)

    granted = [f for f in frames if f["type"] == "approval_granted"]
    denied = [f for f in frames if f["type"] == "approval_denied"]
    assert [f["call_id"] for f in granted] == [needed[0]["call_id"]]
    assert [f["call_id"] for f in denied] == [needed[1]["call_id"]]
    assert denied[0]["reason"] == "not that one"
    assert denied[0]["reason"] != UNRESOLVED
    # The edit reached execution, and the denied call never ran.
    assert executions == [{"id": "a-edited"}]


def test_singular_respond_to_approval_still_auto_denies_the_others():
    """Control. The old frame keeps its old behaviour — that is the bug's shape.

    This is what proves the test above could have failed: same turn, same two
    checkpoints, and the singular frame still strands the second one. If the
    plural handler were wired to the singular session method, this test would
    keep passing while the one above broke.
    """
    app, executions = _two_approval_app()
    client = TestClient(app)
    with client.websocket_connect("/ws/agent") as ws:
        needed = _suspend_on_two(ws)
        ws.send_json(
            {
                "type": "respond_to_approval",
                "turn_id": needed[0]["turn_id"],
                "call_id": needed[0]["call_id"],
                "decision": "approve",
            }
        )
        frames = _drain_turn(ws)

    granted = [f for f in frames if f["type"] == "approval_granted"]
    denied = [f for f in frames if f["type"] == "approval_denied"]
    assert [f["call_id"] for f in granted] == [needed[0]["call_id"]]
    assert [f["call_id"] for f in denied] == [needed[1]["call_id"]]
    assert denied[0]["reason"] == UNRESOLVED
    # The human answered one card; only that call ran.
    assert executions == [{"id": "a"}]


def test_respond_to_approvals_is_gated_by_the_approval_authority():
    """The plural frame goes through the same authority as the singular one.

    A second command shape that bypassed ``approval_authority`` would be an
    approval-gate hole, not an ergonomics gap.
    """
    seen: list[dict] = []

    class _RefuseAll:
        async def authorize_approval(self, ws: WebSocket, command: dict) -> bool:
            seen.append(command)
            return False

    app, executions = _two_approval_app(approval_authority=_RefuseAll())
    client = TestClient(app)
    with client.websocket_connect("/ws/agent") as ws:
        needed = _suspend_on_two(ws)
        ws.send_json(
            {
                "type": "respond_to_approvals",
                "turn_id": needed[0]["turn_id"],
                "decisions": [
                    {"call_id": needed[0]["call_id"], "decision": "approve"},
                    {"call_id": needed[1]["call_id"], "decision": "approve"},
                ],
            }
        )
        frame = ws.receive_json()

    assert frame["type"] == "errored"
    assert frame["code"] == "approval_not_authorized"
    assert seen and seen[0]["type"] == "respond_to_approvals"
    # The refused frame ran nothing and left the turn resumable.
    assert executions == []
