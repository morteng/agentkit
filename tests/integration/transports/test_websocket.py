import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from agentkit import AgentConfig, AgentSession
from agentkit._ids import OwnerId
from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.providers.fakes import FakeProvider
from agentkit.store.fakes import FakeCheckpointStore, FakeMemoryStore, FakeSessionStore
from agentkit.tools.builtin import DEFAULT_BUILTINS
from agentkit.tools.registry import ToolRegistry
from agentkit.transports.websocket import mount_websocket_route

pytestmark = pytest.mark.integration


def _make_app() -> FastAPI:
    app = FastAPI()

    async def session_factory(ws: WebSocket) -> AgentSession:
        config = AgentConfig()
        config.guards.approval = RiskBasedApprovalGate()
        config.stores.session = FakeSessionStore()
        config.stores.memory = FakeMemoryStore()
        config.stores.checkpoint = FakeCheckpointStore()

        registry = ToolRegistry()
        for spec, handler in DEFAULT_BUILTINS:
            registry.register_builtin(spec, handler)

        return AgentSession(
            owner=OwnerId("u:test"),
            config=config,
            provider=FakeProvider().script(FakeProvider.text("hi from server")),
            registry=registry,
            model="m",
        )

    mount_websocket_route(
        app, path="/ws/agent", session_factory=session_factory, origin_allowlist=["*"]
    )
    return app


def test_send_message_streams_back_text_delta_events():
    app = _make_app()
    client = TestClient(app)
    with client.websocket_connect("/ws/agent") as ws:
        ws.send_json({"type": "send_message", "text": "hello"})
        events = []
        while True:
            ev = ws.receive_json()
            events.append(ev)
            if ev["type"] == "turn_ended":
                break
    types = [e["type"] for e in events]
    assert "turn_started" in types
    assert "text_delta" in types
    assert "turn_ended" in types
