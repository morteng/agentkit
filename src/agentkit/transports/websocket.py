"""FastAPI WebSocket bridge — optional convenience for consumers.

Translates inbound JSON ``ClientCommand``s to AgentSession calls and outbound
events to JSON frames. Origin check + auth pluggable.

Concurrency model
-----------------
While a turn is streaming we run the agent stream as a background task and
poll for a single inbound message in parallel. A ``cancel`` arriving during
the turn aborts the streaming task; any other message during a turn is
buffered as the next command. Outside a turn the route awaits ``receive_json``
in a tight loop.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from typing import Any, Protocol

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from agentkit.events import Event, TurnEnded
from agentkit.session import AgentSession


async def _drain_stream_into_ws(
    ws: WebSocket, stream: AsyncIterator[Event], cancel_event: asyncio.Event
) -> None:
    """Forward events from ``stream`` to ``ws`` until TurnEnded or cancel.

    Stops early when ``cancel_event`` is set; the surrounding task will be
    cancelled by the caller, which propagates into the AgentSession's
    asynccontextmanager and triggers cleanup.
    """
    async for event in stream:
        await ws.send_json(event.model_dump(mode="json"))
        if isinstance(event, TurnEnded):
            return
        if cancel_event.is_set():
            return


async def _stream_with_cancel_watch(
    ws: WebSocket,
    stream_factory: Callable[[asyncio.Event], Coroutine[Any, Any, None]],
) -> dict[str, Any] | None:
    """Run ``stream_factory`` while watching for an inbound ``cancel`` command.

    Returns the next non-cancel command if one arrived during the turn (so the
    outer loop can dispatch it next), or ``None`` if the turn ended on its
    own.
    """
    cancel_event = asyncio.Event()
    deferred_cmd: dict[str, Any] | None = None
    stream_task = asyncio.create_task(stream_factory(cancel_event))
    receive_task = asyncio.create_task(ws.receive_json())

    while not stream_task.done():
        done, _ = await asyncio.wait(
            {stream_task, receive_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if receive_task in done:
            try:
                cmd = receive_task.result()
            except (WebSocketDisconnect, RuntimeError):
                # Client disconnected; cancel the stream so AgentSession cleans up.
                stream_task.cancel()
                with _suppress_cancel():
                    await stream_task
                raise
            if cmd.get("type") == "cancel":
                cancel_event.set()
                stream_task.cancel()
                with _suppress_cancel():
                    await stream_task
                await ws.send_json(
                    {"type": "cancelled", "reason": cmd.get("reason", "user_cancel")}
                )
                return None
            # Any other command during a turn: buffer and let the turn finish.
            deferred_cmd = cmd
            receive_task = asyncio.create_task(_never_completes())
            continue
        # stream_task is done — clean up the receive task.
        if not receive_task.done():
            receive_task.cancel()
            with _suppress_cancel():
                await receive_task
        break
    # Surface any exception that occurred inside the stream task.
    exc = stream_task.exception()
    if exc is not None:
        raise exc
    return deferred_cmd


async def _never_completes() -> dict[str, Any]:
    await asyncio.Event().wait()
    return {}  # unreachable


class _suppress_cancel:
    """Tiny context manager — swallow ``CancelledError`` from a target await.

    Equivalent to ``contextlib.suppress(asyncio.CancelledError)`` but written
    inline so the cancellation contract is visible at the call site.
    """

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: object,
    ) -> bool:
        return exc_type is asyncio.CancelledError


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
            await ws.close(code=1011, reason="session factory failed")
            return

        pending_cmd: dict[str, Any] | None = None
        try:
            while True:
                cmd: dict[str, Any] = (
                    pending_cmd if pending_cmd is not None else await ws.receive_json()
                )
                pending_cmd = None
                ctype = cmd.get("type")
                if ctype == "send_message":
                    text = str(cmd.get("text", ""))

                    async def _run(cancel_evt: asyncio.Event, _t: str = text) -> None:
                        async with session.run(_t) as stream:
                            await _drain_stream_into_ws(ws, stream, cancel_evt)

                    pending_cmd = await _stream_with_cancel_watch(ws, _run)

                elif ctype == "respond_to_approval":

                    async def _resume(
                        cancel_evt: asyncio.Event,
                        _c: dict[str, Any] = cmd,
                    ) -> None:
                        async with session.resume_with_approval(
                            _c["turn_id"],
                            _c["call_id"],
                            decision=str(_c.get("decision", "deny")),
                            edited_args=_c.get("edited_args"),
                            reason=_c.get("reason"),
                        ) as stream:
                            await _drain_stream_into_ws(ws, stream, cancel_evt)

                    pending_cmd = await _stream_with_cancel_watch(ws, _resume)

                elif ctype == "cancel":
                    # No active turn — nothing to cancel. Acknowledge and continue.
                    await ws.send_json({"type": "cancelled", "reason": "no_active_turn"})

                else:
                    await ws.send_json(
                        {
                            "type": "errored",
                            "code": "invalid_command",
                            "message": f"unknown command: {ctype}",
                        }
                    )
        except WebSocketDisconnect:
            pass
        finally:
            await session.shutdown()
            # Already closed by the client / framework? That's fine.
            with contextlib.suppress(RuntimeError):
                await ws.close(code=1000)
