"""F9, F10, F11: WebSocket bridge cancel/close-code/error-code regressions."""

import asyncio
import time

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from agentkit import AgentConfig, AgentSession
from agentkit._ids import OwnerId
from agentkit._messages import Usage
from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.providers.base import (
    MessageComplete,
    MessageStart,
    ProviderRequest,
    TextDelta,
    UsageEvent,
)
from agentkit.providers.fakes import FakeProvider
from agentkit.store.fakes import FakeCheckpointStore, FakeMemoryStore, FakeSessionStore
from agentkit.tools.registry import ToolRegistry
from agentkit.transports.websocket import mount_websocket_route

pytestmark = pytest.mark.integration


class _SlowProvider:
    """A provider that drips out text deltas with a configurable delay so the test
    can race ``cancel`` against the stream deterministically."""

    name = "slow"
    capabilities = FakeProvider.capabilities

    def __init__(self, *, delta_count: int = 50, delay_per_delta: float = 0.05) -> None:
        self._delta_count = delta_count
        self._delay = delay_per_delta

    async def stream(self, request: ProviderRequest):
        yield MessageStart()
        for i in range(self._delta_count):
            await asyncio.sleep(self._delay)
            yield TextDelta(delta=f"chunk-{i} ", block_index=0)
        yield UsageEvent(usage=Usage(input_tokens=1, output_tokens=1))
        yield MessageComplete(finish_reason="end_turn")

    def estimate_tokens(self, _messages):
        return 0

    def estimate_cost(self, _usage):
        from decimal import Decimal

        return Decimal("0")


def _make_app(provider) -> FastAPI:
    app = FastAPI()

    async def session_factory(_ws: WebSocket) -> AgentSession:
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
            provider=provider,
            registry=registry,
            model="m",
        )

    mount_websocket_route(
        app, path="/ws/agent", session_factory=session_factory, origin_allowlist=["*"]
    )
    return app


def test_cancel_during_turn_stops_stream_and_acks():
    """F9: a cancel sent mid-turn aborts the stream and the server acks `cancelled`.

    Total stream length without cancel would be 50 deltas at 40ms each = 2s. We send
    cancel after seeing 3 deltas; if cancel works the remaining ~45 deltas are
    skipped. We bound how many events we'll consume so a broken cancel still
    fails the test (rather than hanging indefinitely).
    """
    provider = _SlowProvider(delta_count=50, delay_per_delta=0.04)
    app = _make_app(provider)
    client = TestClient(app)

    with client.websocket_connect("/ws/agent") as ws:
        ws.send_json({"type": "send_message", "text": "stream lots"})
        deltas_seen_before_cancel = 0
        # Read until we have a few deltas; safety cap protects from infinite loop.
        for _ in range(10):
            ev = ws.receive_json()
            if ev.get("type") == "text_delta":
                deltas_seen_before_cancel += 1
                if deltas_seen_before_cancel >= 3:
                    break
        assert deltas_seen_before_cancel >= 3

        t0 = time.monotonic()
        ws.send_json({"type": "cancel", "reason": "test"})

        events_after_cancel: list[dict] = []
        cancelled_seen = False
        # Hard cap: if cancel did nothing we'd see ~47 more deltas + turn_ended.
        # If cancel works we'd see at most a handful before "cancelled".
        for _ in range(15):
            ev = ws.receive_json()
            events_after_cancel.append(ev)
            if ev.get("type") == "cancelled":
                cancelled_seen = True
                break

        elapsed = time.monotonic() - t0
        assert cancelled_seen, (
            f"never saw 'cancelled' ack within 15 events; got types: "
            f"{[e.get('type') for e in events_after_cancel]}"
        )
        # Cancel must arrive faster than the natural turn end (~1.7s remaining).
        assert elapsed < 1.5, f"cancel took {elapsed:.2f}s — too slow"


def test_unknown_command_uses_invalid_command_code_and_keeps_socket_alive():
    """F11: unknown command type reports `invalid_command`, not `internal`."""
    provider = FakeProvider().script(FakeProvider.text("hello"))
    app = _make_app(provider)
    client = TestClient(app)

    with client.websocket_connect("/ws/agent") as ws:
        ws.send_json({"type": "frobnicate"})
        ev = ws.receive_json()
        assert ev["type"] == "errored"
        assert ev["code"] == "invalid_command"
        assert "frobnicate" in ev["message"]
        # Socket still alive: real command works.
        ws.send_json({"type": "send_message", "text": "hi"})
        seen_end = False
        for _ in range(40):
            evt = ws.receive_json()
            if evt.get("type") == "turn_ended":
                seen_end = True
                break
        assert seen_end


def test_cancel_with_no_active_turn_is_acknowledged():
    """F9: cancel between turns should ack rather than be silently swallowed."""
    provider = FakeProvider()
    app = _make_app(provider)
    client = TestClient(app)

    with client.websocket_connect("/ws/agent") as ws:
        ws.send_json({"type": "cancel"})
        ev = ws.receive_json()
        assert ev["type"] == "cancelled"
        assert ev["reason"] == "no_active_turn"
