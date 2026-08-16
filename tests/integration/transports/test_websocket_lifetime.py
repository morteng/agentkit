"""`max_connection_seconds`: bound a connection's life without truncating a turn.

A WebSocket carries the authorization decision made at its handshake for as
long as it stays open — there are no headers after the upgrade, so nothing can
re-check it in place. The only available control is to end the connection and
make the client come back through the handshake.

That control is only worth having if it cannot eat a running turn. These tests
assert both halves, because the dangerous half is the one a "it closed, good"
test leaves unproven:

* an idle connection is closed at the bound (the control works), and
* a turn that starts before the bound and ends after it streams to completion
  first, and the close lands after ``turn_ended`` (the control is safe).
"""

import asyncio

import pytest
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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
from agentkit.transports.websocket import (
    WS_CLOSE_LIFETIME_EXCEEDED,
    mount_websocket_route,
)

pytestmark = pytest.mark.integration


class _SlowProvider:
    """Streams ``delta_count`` deltas ``delay`` apart, so a turn's wall-clock
    length is known and can be made to straddle the lifetime bound."""

    name = "slow"
    capabilities = FakeProvider.capabilities

    def __init__(self, *, delta_count: int = 10, delay: float = 0.05) -> None:
        self._delta_count = delta_count
        self._delay = delay

    async def stream(self, request: ProviderRequest):
        yield MessageStart()
        for i in range(self._delta_count):
            await asyncio.sleep(self._delay)
            yield TextDelta(delta=f"chunk-{i} ", block_index=0)
        yield UsageEvent(
            usage=Usage(input_tokens=1, output_tokens=1),
            model="fake/test",
            provider_name="fake",
        )
        yield MessageComplete(finish_reason="end_turn")

    def estimate_tokens(self, _messages):
        return 0

    def estimate_cost(self, _usage):
        from decimal import Decimal

        return Decimal("0")


class _StubAuth:
    """`auth=` is required — there is no allow-all default to fall back on."""

    async def authenticate(self, ws: WebSocket) -> bool:
        return True


def _make_app(provider, *, max_connection_seconds: float | None) -> FastAPI:
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
        app,
        path="/ws/agent",
        session_factory=session_factory,
        # TestClient sends no Origin header; "" is the deliberate allowlist
        # entry for non-browser clients ("*" is rejected outside dev mode).
        origin_allowlist=[""],
        auth=_StubAuth(),
        max_connection_seconds=max_connection_seconds,
    )
    return app


def test_idle_connection_is_closed_at_the_bound_with_the_lifetime_code():
    """The control works: nothing sent, and the server hangs up on schedule."""
    app = _make_app(FakeProvider().script(FakeProvider.text("hi")), max_connection_seconds=0.05)
    client = TestClient(app)

    with client.websocket_connect("/ws/agent") as ws, pytest.raises(WebSocketDisconnect) as exc:
        ws.receive_json()

    assert exc.value.code == WS_CLOSE_LIFETIME_EXCEEDED, (
        "an expired idle connection must close with the lifetime code, not a "
        f"generic one; got {exc.value.code}"
    )


def test_a_turn_running_past_the_bound_completes_before_the_close():
    """The control is safe: the close waits for ``turn_ended``.

    The turn is deliberately several times longer than the bound. If the close
    were raced against the stream instead of checked between turns, the client
    would see the deltas stop partway and the socket drop — a staleness control
    turned into visible data loss. So this asserts the *whole* turn arrives,
    and only then the close.
    """
    # ~0.6s of streaming against a 0.15s bound: the deadline passes while the
    # turn is mid-flight, several deltas before it ends.
    provider = _SlowProvider(delta_count=12, delay=0.05)
    app = _make_app(provider, max_connection_seconds=0.15)
    client = TestClient(app)

    deltas = 0
    with client.websocket_connect("/ws/agent") as ws:
        ws.send_json({"type": "send_message", "text": "stream past the bound"})
        saw_turn_ended = False
        for _ in range(60):
            ev = ws.receive_json()
            if ev.get("type") == "text_delta":
                deltas += 1
            if ev.get("type") == "turn_ended":
                saw_turn_ended = True
                break
        assert saw_turn_ended, (
            "the turn was cut off by the lifetime bound — the close must be "
            f"deferred until the stream ends (saw {deltas} deltas)"
        )
        # ``>=`` rather than ``==``: the session may re-prompt the provider once
        # inside a single turn (finalize validation), which replays the script.
        # The invariant under test is that nothing was *truncated*.
        assert deltas >= 12, f"expected at least 12 deltas before the close, got {deltas}"

        # …and only now, with the socket idle again, does the bound apply.
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
    assert exc.value.code == WS_CLOSE_LIFETIME_EXCEEDED


def test_a_command_buffered_during_the_expired_turn_still_runs():
    """A command that arrived mid-turn is queued work, not a new idle period.

    ``_stream_with_cancel_watch`` buffers any non-cancel command that lands
    during a turn. Dropping that buffered command on expiry would lose a
    message the client believes it sent and got no error for — the same data
    loss as a mid-stream close, one frame earlier.
    """
    provider = _SlowProvider(delta_count=6, delay=0.05)
    app = _make_app(provider, max_connection_seconds=0.15)
    client = TestClient(app)

    with client.websocket_connect("/ws/agent") as ws:
        ws.send_json({"type": "send_message", "text": "first"})
        # Land the second message inside the first turn's stream.
        ws.receive_json()
        ws.send_json({"type": "send_message", "text": "second (buffered)"})

        turns_ended = 0
        for _ in range(120):
            try:
                ev = ws.receive_json()
            except WebSocketDisconnect as disconnect:
                assert disconnect.code == WS_CLOSE_LIFETIME_EXCEEDED
                break
            if ev.get("type") == "turn_ended":
                turns_ended += 1
        assert turns_ended == 2, (
            "the buffered second message must still run before the connection "
            f"is retired; only {turns_ended} turn(s) completed"
        )


def test_default_is_unbounded_so_existing_callers_are_unaffected():
    """`None` is today's behaviour: an idle socket stays open."""
    app = _make_app(FakeProvider().script(FakeProvider.text("hi")), max_connection_seconds=None)
    client = TestClient(app)

    with client.websocket_connect("/ws/agent") as ws:
        # Idle well past what the bounded cases above close at, then prove the
        # socket still serves a turn.
        ws.send_json({"type": "cancel"})
        assert ws.receive_json()["reason"] == "no_active_turn"
        ws.send_json({"type": "send_message", "text": "still here"})
        saw_turn_ended = False
        for _ in range(40):
            if ws.receive_json().get("type") == "turn_ended":
                saw_turn_ended = True
                break
        assert saw_turn_ended


@pytest.mark.parametrize("bad", [0, -1.0])
def test_non_positive_bound_is_rejected_at_mount_time(bad):
    """Fail at import/startup, not on the first doomed connection."""
    with pytest.raises(ValueError, match="max_connection_seconds must be positive"):
        _make_app(FakeProvider(), max_connection_seconds=bad)
