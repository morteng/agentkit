"""MCP client lifecycle must be owned by one task on the WebSocket bridge.

The route used to build the session and then let the *first turn's* stream
task be the one that triggered ``AgentSession.initialize()`` — lazily, from
inside ``session.run()`` — while ``session.shutdown()`` ran from the route
task's ``finally``. A stdio MCP client opens anyio task-scoped resources on
initialize, and anyio raises ``RuntimeError: Attempted to exit cancel scope in
a different task`` when they are closed from anywhere else.

The bridge now initializes eagerly in the route task, before any
``create_task``. These tests assert exactly that property: the task identity
observed at ``initialize()`` is the one observed at ``shutdown()``, and it is
not the per-turn stream task.
"""

import asyncio
from typing import Any

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from agentkit import AgentConfig, AgentSession
from agentkit._ids import OwnerId
from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.providers.fakes import FakeProvider
from agentkit.store.fakes import FakeMemoryStore, FakeSessionStore
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


class _TaskAffineMCPClient:
    """Stands in for a stdio MCP client: records which task ran each lifecycle call.

    A real ``StdioMCPClient`` would *raise* on a cross-task exit; recording the
    task names lets the test show the violation directly instead of depending
    on anyio's error message.
    """

    name = "affine"

    def __init__(self) -> None:
        self.init_task: asyncio.Task[Any] | None = None
        self.shutdown_task: asyncio.Task[Any] | None = None
        self.init_count = 0

    async def initialize(self) -> None:
        self.init_count += 1
        self.init_task = asyncio.current_task()

    async def list_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="ping",
                description="ping",
                parameters={"type": "object", "properties": {}},
                returns=None,
                risk=RiskLevel.READ,
                idempotent=True,
                side_effects=SideEffects.NONE,
                requires_approval=ApprovalPolicy.NEVER,
                cache_ttl_seconds=None,
                timeout_seconds=5.0,
            )
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any], **_: Any) -> ToolResult:
        return ToolResult(
            call_id="",
            status="ok",
            content=[ContentBlockOut(type="text", text="pong")],
            error=None,
            duration_ms=0,
            cached=False,
        )

    async def shutdown(self) -> None:
        self.shutdown_task = asyncio.current_task()

    async def health_check(self) -> bool:
        return True


class _StubAuth:
    async def authenticate(self, ws: WebSocket) -> bool:
        return True


def _make_app(client: _TaskAffineMCPClient) -> FastAPI:
    app = FastAPI()

    async def session_factory(_ws: WebSocket) -> AgentSession:
        config = AgentConfig()
        config.guards.approval = RiskBasedApprovalGate()
        config.stores.session = FakeSessionStore()
        config.stores.memory = FakeMemoryStore()
        registry = ToolRegistry()
        registry.register_default_builtins()
        registry.register_mcp_server("affine", client)  # type: ignore[arg-type]
        return AgentSession(
            owner=OwnerId("u:test"),
            config=config,
            provider=FakeProvider().script(FakeProvider.text("first"), FakeProvider.text("second")),
            registry=registry,
            model="m",
        )

    mount_websocket_route(
        app,
        path="/ws/agent",
        session_factory=session_factory,
        origin_allowlist=[""],
        auth=_StubAuth(),
    )
    return app


def _run_turn(ws, text: str) -> None:
    ws.send_json({"type": "send_message", "text": text})
    while ws.receive_json()["type"] != "turn_ended":
        pass


def test_mcp_client_is_initialized_and_shut_down_by_the_same_task():
    client = _TaskAffineMCPClient()
    app = _make_app(client)
    with TestClient(app) as http, http.websocket_connect("/ws/agent") as ws:
        _run_turn(ws, "hello")

    assert client.init_task is not None, "MCP client was never initialized"
    assert client.shutdown_task is not None, "MCP client was never shut down"
    assert client.init_task is client.shutdown_task, (
        "MCP client lifecycle crossed tasks: initialized in "
        f"{client.init_task!r}, shut down in {client.shutdown_task!r}. "
        "anyio raises 'Attempted to exit cancel scope in a different task' here."
    )


def test_mcp_client_is_initialized_before_the_first_turn_task_runs():
    """Eager init: the client is already up when the first stream task starts.

    Two turns run on one socket; ``initialize`` must have happened exactly
    once, in the route task, not once per turn and not inside either stream
    task.
    """
    client = _TaskAffineMCPClient()
    app = _make_app(client)
    with TestClient(app) as http, http.websocket_connect("/ws/agent") as ws:
        _run_turn(ws, "one")
        _run_turn(ws, "two")

    assert client.init_count == 1
    assert client.init_task is client.shutdown_task
