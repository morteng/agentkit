"""The WebSocket bridge's security posture: auth is required, origins are strict.

The audit's HIGH finding was that ``mount_websocket_route`` defaulted to an
allow-all authenticator, so a consumer that simply forgot ``auth=`` shipped an
unauthenticated socket onto its network with the session's full tool surface
behind it. These tests pin the new contract:

* omitting ``auth`` is a ``TypeError`` at mount time, not a silent open socket;
* choosing an open socket requires naming :class:`InsecureAllowAllAuth`, which
  warns;
* ``"*"`` origins raise unless ``dev_mode=True`` is passed on purpose;
* ``respond_to_approval`` can be bound to a different principal than the one
  driving the model.
"""

import warnings

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from agentkit import AgentConfig, AgentSession
from agentkit._ids import OwnerId
from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.providers.fakes import FakeProvider
from agentkit.store.fakes import FakeCheckpointStore, FakeMemoryStore, FakeSessionStore
from agentkit.tools.registry import ToolRegistry
from agentkit.transports.websocket import (
    InsecureAllowAllAuth,
    InsecureTransportWarning,
    SameSocketApprovalAuthority,
    mount_websocket_route,
)

pytestmark = pytest.mark.integration

NO_ORIGIN = [""]


class _TokenAuth:
    """Accepts only a peer presenting the right ``x-agent-token`` header."""

    def __init__(self, token: str) -> None:
        self._token = token
        self.calls = 0

    async def authenticate(self, ws: WebSocket) -> bool:
        self.calls += 1
        return ws.headers.get("x-agent-token") == self._token


class _RecordingApprovalAuthority:
    """Approval authority that only trusts frames carrying an approver token."""

    def __init__(self, *, token: str | None) -> None:
        self._token = token
        self.seen: list[dict] = []

    async def authorize_approval(self, ws: WebSocket, command: dict) -> bool:
        self.seen.append(command)
        return self._token is not None and command.get("approval_token") == self._token


async def _session_factory(_ws: WebSocket) -> AgentSession:
    config = AgentConfig()
    config.guards.approval = RiskBasedApprovalGate()
    config.stores.session = FakeSessionStore()
    config.stores.memory = FakeMemoryStore()
    config.stores.checkpoint = FakeCheckpointStore()
    registry = ToolRegistry()
    registry.register_default_builtins()
    return AgentSession(
        owner=OwnerId("u:test"),
        config=config,
        provider=FakeProvider().script(FakeProvider.text("hi")),
        registry=registry,
        model="m",
    )


# ---- auth is required -------------------------------------------------------


def test_mounting_without_auth_is_a_type_error():
    """The whole finding in one assertion: no auth argument, no route.

    Previously this mounted a working, unauthenticated endpoint.
    """
    app = FastAPI()
    with pytest.raises(TypeError):
        mount_websocket_route(  # type: ignore[call-arg]
            app,
            path="/ws/agent",
            session_factory=_session_factory,
            origin_allowlist=NO_ORIGIN,
        )


def test_mounting_with_auth_none_is_a_type_error_naming_the_opt_in():
    """Passing ``auth=None`` explicitly must not resurrect the old default."""
    app = FastAPI()
    with pytest.raises(TypeError, match="InsecureAllowAllAuth"):
        mount_websocket_route(
            app,
            path="/ws/agent",
            session_factory=_session_factory,
            origin_allowlist=NO_ORIGIN,
            auth=None,  # type: ignore[arg-type]
        )


def test_mounting_with_a_non_wsauth_object_is_a_type_error():
    app = FastAPI()
    with pytest.raises(TypeError, match="authenticate"):
        mount_websocket_route(
            app,
            path="/ws/agent",
            session_factory=_session_factory,
            origin_allowlist=NO_ORIGIN,
            auth=object(),  # type: ignore[arg-type]
        )


def test_failed_authentication_closes_the_socket_before_any_session_is_built():
    app = FastAPI()
    auth = _TokenAuth("s3cret")
    built: list[str] = []

    async def _tracking_factory(ws: WebSocket) -> AgentSession:
        built.append("yes")
        return await _session_factory(ws)

    mount_websocket_route(
        app,
        path="/ws/agent",
        session_factory=_tracking_factory,
        origin_allowlist=NO_ORIGIN,
        auth=auth,
    )
    client = TestClient(app)

    with pytest.raises(Exception), client.websocket_connect("/ws/agent") as ws:  # noqa: B017
        ws.receive_json()
    assert auth.calls == 1
    assert built == [], "session factory must not run for an unauthenticated peer"


def test_successful_authentication_runs_a_turn():
    app = FastAPI()
    mount_websocket_route(
        app,
        path="/ws/agent",
        session_factory=_session_factory,
        origin_allowlist=NO_ORIGIN,
        auth=_TokenAuth("s3cret"),
    )
    client = TestClient(app)
    with client.websocket_connect("/ws/agent", headers={"x-agent-token": "s3cret"}) as ws:
        ws.send_json({"type": "send_message", "text": "hello"})
        types = []
        while True:
            ev = ws.receive_json()
            types.append(ev["type"])
            if ev["type"] == "turn_ended":
                break
    assert "text_delta" in types


# ---- the opt-in escape hatch is loud ----------------------------------------


def test_insecure_allow_all_auth_warns_on_construction():
    with pytest.warns(InsecureTransportWarning, match="without authentication"):
        InsecureAllowAllAuth()


def test_insecure_allow_all_auth_still_works_when_chosen_deliberately():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InsecureTransportWarning)
        auth = InsecureAllowAllAuth()
    app = FastAPI()
    mount_websocket_route(
        app,
        path="/ws/agent",
        session_factory=_session_factory,
        origin_allowlist=NO_ORIGIN,
        auth=auth,
    )
    client = TestClient(app)
    with client.websocket_connect("/ws/agent") as ws:
        ws.send_json({"type": "cancel"})
        assert ws.receive_json()["type"] == "cancelled"


def test_insecure_transport_warning_can_be_promoted_to_an_error():
    """A deployment can fail its build on demo-grade configuration."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", InsecureTransportWarning)
        with pytest.raises(InsecureTransportWarning):
            InsecureAllowAllAuth()


# ---- origins ----------------------------------------------------------------


def test_wildcard_origin_is_rejected_outside_dev_mode():
    app = FastAPI()
    with pytest.raises(ValueError, match="dev_mode"):
        mount_websocket_route(
            app,
            path="/ws/agent",
            session_factory=_session_factory,
            origin_allowlist=["*"],
            auth=_TokenAuth("t"),
        )


def test_wildcard_origin_warns_when_dev_mode_is_explicit():
    app = FastAPI()
    with pytest.warns(InsecureTransportWarning, match="origin check is disabled"):
        mount_websocket_route(
            app,
            path="/ws/agent",
            session_factory=_session_factory,
            origin_allowlist=["*"],
            auth=_TokenAuth("t"),
            dev_mode=True,
        )


def test_disallowed_origin_is_closed():
    app = FastAPI()
    mount_websocket_route(
        app,
        path="/ws/agent",
        session_factory=_session_factory,
        origin_allowlist=["https://your.site"],
        auth=_TokenAuth("t"),
    )
    client = TestClient(app)
    with (
        pytest.raises(Exception),  # noqa: B017 - starlette refuses the handshake
        client.websocket_connect("/ws/agent", headers={"origin": "https://evil.test"}) as ws,
    ):
        ws.receive_json()


# ---- approval authority -----------------------------------------------------


def test_approval_authority_can_reject_a_self_granted_approval():
    """The driving socket is not automatically the approving principal.

    With an authority installed, a ``respond_to_approval`` frame that carries
    no approver token is refused and the suspended turn is left alone.
    """
    app = FastAPI()
    authority = _RecordingApprovalAuthority(token="approver-token")
    mount_websocket_route(
        app,
        path="/ws/agent",
        session_factory=_session_factory,
        origin_allowlist=NO_ORIGIN,
        auth=_TokenAuth("t"),
        approval_authority=authority,
    )
    client = TestClient(app)
    with client.websocket_connect("/ws/agent", headers={"x-agent-token": "t"}) as ws:
        ws.send_json(
            {
                "type": "respond_to_approval",
                "turn_id": "t1",
                "call_id": "c1",
                "decision": "approve",
            }
        )
        ev = ws.receive_json()
    assert ev["type"] == "errored"
    assert ev["code"] == "approval_not_authorized"
    assert authority.seen[0]["call_id"] == "c1"


def test_approval_authority_sees_the_raw_frame_and_can_admit_it():
    """A frame carrying the out-of-band approver token gets through the gate.

    It then fails on the missing checkpoint, which is the point: the authority
    is a gate in front of the session, not a replacement for it.
    """
    app = FastAPI()
    authority = _RecordingApprovalAuthority(token="approver-token")
    mount_websocket_route(
        app,
        path="/ws/agent",
        session_factory=_session_factory,
        origin_allowlist=NO_ORIGIN,
        auth=_TokenAuth("t"),
        approval_authority=authority,
    )
    client = TestClient(app)
    with (
        pytest.raises(Exception),  # noqa: B017 - CheckpointMissing tears the socket down
        client.websocket_connect("/ws/agent", headers={"x-agent-token": "t"}) as ws,
    ):
        ws.send_json(
            {
                "type": "respond_to_approval",
                "turn_id": "t1",
                "call_id": "c1",
                "decision": "approve",
                "approval_token": "approver-token",
            }
        )
        ws.receive_json()
    assert authority.seen[0]["approval_token"] == "approver-token"


async def test_same_socket_approval_authority_admits_everything():
    """The documented-hazard default is a real object, not an implicit None."""
    authority = SameSocketApprovalAuthority()
    assert await authority.authorize_approval(None, {"type": "respond_to_approval"}) is True  # type: ignore[arg-type]
