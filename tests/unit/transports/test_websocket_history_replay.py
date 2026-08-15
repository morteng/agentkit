"""Opt-in history replay on WS connect.

The replay is context; the live socket is the product. So the assertions here
are as much about what a failing replay must NOT do — close the socket, raise
into the route loop — as about the frame it sends when it works.
"""

from datetime import UTC, datetime
from typing import Any

import pytest

from agentkit import AgentConfig, AgentSession
from agentkit._content import TextBlock
from agentkit._ids import MessageId, OwnerId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.providers.fakes import FakeProvider
from agentkit.store.fakes import FakeMemoryStore, FakeSessionStore
from agentkit.tools.registry import ToolRegistry
from agentkit.transports.websocket import _replay_history  # pyright: ignore[reportPrivateUsage]

_SESSION = SessionId("REDACTEDmorten")


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)


class _ExplodingStore(FakeSessionStore):
    async def list_messages(self, session_id: SessionId, *, limit: int = 200) -> list[Message]:
        raise RuntimeError("store unavailable")


def _session(store: Any) -> AgentSession:
    config = AgentConfig()
    config.stores.session = store
    config.stores.memory = FakeMemoryStore()
    return AgentSession(
        owner=OwnerId("REDACTEDmorten"),
        config=config,
        provider=FakeProvider().script(FakeProvider.text("hi")),
        registry=ToolRegistry(),
        model="m",
        session_id=_SESSION,
    )


def _user(text: str) -> Message:
    return Message(
        id=new_id(MessageId),
        session_id=_SESSION,
        role=MessageRole.USER,
        content=[TextBlock(text=text)],
        created_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_the_replay_frame_carries_the_stored_transcript():
    store = FakeSessionStore()
    await store.create(_SESSION, OwnerId("REDACTEDmorten"))
    await store.append_message(_SESSION, _user("find something to watch"))
    ws = _FakeWS()

    await _replay_history(ws, _session(store), limit=50, path="/ws/agent")  # type: ignore[arg-type]

    assert len(ws.sent) == 1
    frame = ws.sent[0]
    assert frame["type"] == "history"
    assert frame["session_id"] == "REDACTEDmorten"
    assert frame["count"] == 1
    assert frame["items"][0]["kind"] == "user"
    assert frame["items"][0]["text"] == "find something to watch"


@pytest.mark.asyncio
async def test_an_empty_session_still_gets_a_frame():
    """The empty page is what a client renders its first-run state from; a
    missing frame is indistinguishable from a replay that never ran."""
    ws = _FakeWS()
    await _replay_history(ws, _session(FakeSessionStore()), limit=50, path="/ws/agent")  # type: ignore[arg-type]

    assert ws.sent[0]["count"] == 0
    assert ws.sent[0]["items"] == []


@pytest.mark.asyncio
async def test_the_limit_reaches_the_store():
    store = FakeSessionStore()
    await store.create(_SESSION, OwnerId("REDACTEDmorten"))
    for i in range(5):
        await store.append_message(_SESSION, _user(f"m{i}"))
    ws = _FakeWS()

    await _replay_history(ws, _session(store), limit=2, path="/ws/agent")  # type: ignore[arg-type]

    assert ws.sent[0]["count"] == 2
    assert ws.sent[0]["truncated"] is True


@pytest.mark.asyncio
async def test_no_session_store_skips_the_replay_without_failing_the_connection():
    ws = _FakeWS()
    await _replay_history(ws, _session(None), limit=50, path="/ws/agent")  # type: ignore[arg-type]
    assert ws.sent == []


@pytest.mark.asyncio
async def test_a_failing_store_read_does_not_take_the_socket_down():
    ws = _FakeWS()
    await _replay_history(ws, _session(_ExplodingStore()), limit=50, path="/ws/agent")  # type: ignore[arg-type]
    assert ws.sent == []


def test_the_flag_is_opt_in():
    """A replay is a second delivery of content the client may already have;
    only the consumer knows whether its client dedupes."""
    import inspect

    from agentkit.transports.websocket import mount_websocket_route

    params = inspect.signature(mount_websocket_route).parameters
    assert params["replay_history"].default is False
    assert params["replay_limit"].default == 50
