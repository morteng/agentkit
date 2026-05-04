"""FastAPI WebSocket bridge — optional convenience for consumers.

Translates inbound JSON ``ClientCommand``s to AgentSession calls and outbound
events to JSON frames. Origin check + auth pluggable.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from agentkit.events import Event, TurnEnded
from agentkit.session import AgentSession


async def _stream_to_ws(ws: WebSocket, stream: AsyncIterator[Event]) -> None:
    async for event in stream:
        await ws.send_json(event.model_dump(mode="json"))
        if isinstance(event, TurnEnded):
            break


async def _handle_send_message(ws: WebSocket, session: AgentSession, cmd: dict[str, Any]) -> None:
    text = str(cmd.get("text", ""))
    async with session.run(text) as stream:
        await _stream_to_ws(ws, stream)


async def _handle_respond_to_approval(
    ws: WebSocket, session: AgentSession, cmd: dict[str, Any]
) -> None:
    async with session.resume_with_approval(
        cmd["turn_id"],
        cmd["call_id"],
        decision=cmd.get("decision", "deny"),
        edited_args=cmd.get("edited_args"),
        reason=cmd.get("reason"),
    ) as stream:
        await _stream_to_ws(ws, stream)


class WSAuth(Protocol):
    async def authenticate(self, ws: WebSocket) -> bool: ...


class _AllowAllAuth:
    async def authenticate(self, ws: WebSocket) -> bool:
        return True


def mount_websocket_route(
    app: FastAPI,
    *,
    path: str = "/ws/agent",
    session_factory: Callable[[WebSocket], Awaitable[AgentSession]],
    origin_allowlist: list[str],
    auth: WSAuth | None = None,
    heartbeat_interval: float = 30.0,
) -> None:
    auth_impl: WSAuth = auth or _AllowAllAuth()

    @app.websocket(path)
    async def _ws_route(ws: WebSocket) -> None:  # pyright: ignore[reportUnusedFunction]
        # Origin check before auth.
        origin = ws.headers.get("origin", "")
        if "*" not in origin_allowlist and origin not in origin_allowlist:
            await ws.close(code=4003)
            return
        if not await auth_impl.authenticate(ws):
            await ws.close(code=4001)
            return
        await ws.accept()

        try:
            session = await session_factory(ws)
        except Exception:
            await ws.close(code=1011)
            return

        try:
            while True:
                cmd = await ws.receive_json()
                ctype = cmd.get("type")
                if ctype == "send_message":
                    await _handle_send_message(ws, session, cmd)
                elif ctype == "respond_to_approval":
                    await _handle_respond_to_approval(ws, session, cmd)
                elif ctype == "cancel":
                    # In v0.1, cancel just closes the socket. The AgentSession
                    # cancellation surface is async-context-driven; closing the
                    # iterator triggers cleanup via the asynccontextmanager.
                    break
                else:
                    await ws.send_json(
                        {
                            "type": "errored",
                            "code": "internal",
                            "message": f"unknown command: {ctype}",
                        }
                    )
        except WebSocketDisconnect:
            pass
        finally:
            await session.shutdown()
