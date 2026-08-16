"""FastAPI WebSocket bridge — optional convenience for consumers.

Translates inbound JSON ``ClientCommand``s to AgentSession calls and outbound
events to JSON frames.

Concurrency model
-----------------
While a turn is streaming we run the agent stream as a background task and
poll for a single inbound message in parallel. A ``cancel`` arriving during
the turn aborts the streaming task; any other message during a turn is
buffered as the next command. Outside a turn the route awaits ``receive_json``
in a tight loop.

The AgentSession is **initialized eagerly, in the route task**, before any
``create_task``. That is a correctness requirement, not a style choice:
``AgentSession.initialize`` starts the registry's MCP clients, and a stdio MCP
client opens anyio task-scoped resources (a subprocess + its cancel scope).
Starting one inside the per-turn stream task and closing it from the route
task's ``finally`` crosses task boundaries and raises
``RuntimeError: Attempted to exit cancel scope in a different task``. Owning
both ends here keeps start and stop in one task.

Security
--------
**Authentication is mandatory.** ``auth`` is a required keyword argument with
no default. There is no implicit allow-all: a bridge whose consumer forgot to
pass an authenticator used to hand every peer that could reach the port the
session's entire tool surface, privileged tools included. If you genuinely
want an open socket — a local demo, a test — pass
:class:`InsecureAllowAllAuth` explicitly; it warns on construction so the
choice shows up in logs and in review.

**Origins.** ``"*"`` in ``origin_allowlist`` disables the origin check and is
rejected unless ``dev_mode=True`` is also passed (which itself warns). Note
that a client sending no ``Origin`` header at all (most non-browser clients)
presents ``""``; add ``""`` to the allowlist deliberately if you want to
admit those, rather than reaching for ``"*"``.

**Approval hazard — read this before shipping.** ``respond_to_approval``
arrives on the *same* channel that drives the model. Whoever can send
``send_message`` can therefore also grant the approvals that message
provoked, so the human-in-the-loop gate protects against a confused model,
not against a compromised or coerced client. If the driving channel is
scriptable (a browser tab reachable by XSS, an automation key, a relay that
forwards model-adjacent content), treat every approval on it as
self-granted.

Bind approvals to a different principal by passing ``approval_authority``:
every ``respond_to_approval`` frame is checked by
:meth:`WSApprovalAuthority.authorize_approval` before it reaches the session,
so a consumer can require a second factor carried in the frame (a signed
one-time approval token minted by an out-of-band approver UI), a different
header/cookie identity than the one ``auth`` accepted, or an operator role
the driving principal does not hold. The default,
:class:`SameSocketApprovalAuthority`, preserves the historical behaviour and
is named so that accepting it is visible.
"""

import asyncio
import contextlib
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from typing import Any, Protocol, runtime_checkable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from agentkit._logging import get_logger
from agentkit.events import Event, TurnEnded
from agentkit.history import DEFAULT_HISTORY_PAGE_LIMIT, history_frame, load_history_page
from agentkit.session import AgentSession

log = get_logger(__name__)

#: Wildcard entry that disables the origin check. Only honoured under
#: ``dev_mode=True`` — see :func:`mount_websocket_route`.
ANY_ORIGIN = "*"

#: Close code sent when ``max_connection_seconds`` elapses. In the private-use
#: range (4000-4999) so it cannot collide with an RFC 6455 code, and distinct
#: from 4001 (auth rejected) and 4003 (origin rejected) so a client can tell
#: "reconnect, you are expected back" from "do not bother".
WS_CLOSE_LIFETIME_EXCEEDED = 4008

#: Reason string paired with :data:`WS_CLOSE_LIFETIME_EXCEEDED`. Plain enough
#: for a UI to show, and it says the connection ended, not that anything failed.
WS_CLOSE_LIFETIME_EXCEEDED_REASON = "reconnect required"


class InsecureTransportWarning(UserWarning):
    """Warns that the WebSocket bridge is mounted without a real security control.

    Its own category (rather than a bare ``UserWarning``) so deployments can
    turn it into an error — ``warnings.simplefilter("error",
    InsecureTransportWarning)`` — and fail the build if a demo-grade
    configuration ever reaches production.
    """


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


@runtime_checkable
class WSAuth(Protocol):
    """Decides whether a peer may open the agent socket at all.

    Called once per connection, before ``accept()``. Return ``False`` to
    close with code 4001. Read whatever the handshake carried —
    ``ws.headers``, ``ws.cookies``, ``ws.query_params`` — and resolve it to a
    principal your ``session_factory`` will agree with; the two must not
    disagree about who is connected.
    """

    async def authenticate(self, ws: WebSocket) -> bool: ...


@runtime_checkable
class WSApprovalAuthority(Protocol):
    """Decides whether *this* socket may answer an approval prompt.

    Consulted for every ``respond_to_approval`` frame, after :class:`WSAuth`
    has already admitted the connection. Returning ``False`` rejects the frame
    with an ``approval_not_authorized`` error and leaves the suspended turn
    intact, so a legitimate approver can still resolve it.

    ``command`` is the raw inbound frame, so an implementation can require an
    out-of-band credential inside it (e.g. a one-time approval token minted by
    a separate approver UI and bound to ``command["call_id"]``).
    """

    async def authorize_approval(self, ws: WebSocket, command: dict[str, Any]) -> bool: ...


class InsecureAllowAllAuth:
    """Authenticator that admits **every** peer. Not for production.

    Any process that can reach the port gets a full agent session — which
    means the session's whole tool surface, privileged tools included. This
    class exists so that choosing an open socket is an explicit, greppable,
    reviewable line of code instead of an omission; it is deliberately not the
    default and never will be.

    Constructing it emits an :class:`InsecureTransportWarning`.
    """

    def __init__(self) -> None:
        warnings.warn(
            "InsecureAllowAllAuth admits every peer without authentication. "
            "Anyone who can reach this WebSocket gets the session's full tool "
            "surface. Use it for local development and tests only.",
            InsecureTransportWarning,
            stacklevel=2,
        )
        log.warning(
            "insecure_ws_auth_selected",
            auth="InsecureAllowAllAuth",
            hint="every peer is admitted without authentication",
        )

    async def authenticate(self, ws: WebSocket) -> bool:
        return True


class SameSocketApprovalAuthority:
    """Default approval authority: the driving socket may grant its own approvals.

    This is the historical behaviour and it is a real hazard — see the module
    docstring. Whoever can send ``send_message`` on this socket can also
    approve the HIGH_WRITE/DESTRUCTIVE calls that message produced, so the
    approval gate defends against a confused model and not against a
    compromised client. Pass a different :class:`WSApprovalAuthority` to bind
    approvals to a second principal.
    """

    async def authorize_approval(self, ws: WebSocket, command: dict[str, Any]) -> bool:
        return True


def _validate_auth(auth: WSAuth) -> WSAuth:
    """Reject a missing/unusable authenticator at mount time, not at connect time."""
    if auth is None:  # type: ignore[reportUnnecessaryComparison]
        raise TypeError(
            "mount_websocket_route() requires an explicit auth= authenticator. "
            "There is no allow-all default: an unauthenticated agent socket "
            "exposes the session's entire tool surface to any peer that can "
            "reach the port. Pass your own WSAuth implementation, or "
            "InsecureAllowAllAuth() if you really want an open socket."
        )
    if not callable(getattr(auth, "authenticate", None)):
        raise TypeError(
            f"auth= must implement WSAuth.authenticate(ws) -> bool; got {type(auth).__name__}"
        )
    return auth


def _validate_origin_allowlist(origin_allowlist: list[str], *, dev_mode: bool) -> None:
    """Reject the wildcard origin outside dev mode; warn about it inside it."""
    if ANY_ORIGIN not in origin_allowlist:
        return
    if not dev_mode:
        raise ValueError(
            'origin_allowlist contains "*", which disables the origin check and '
            "lets any web page a victim visits drive this agent from their "
            "browser (their cookies and headers ride along). List the exact "
            "origins you serve. If this really is a local demo, pass "
            "dev_mode=True to acknowledge it. To admit non-browser clients "
            'that send no Origin header, allowlist "" instead.'
        )
    warnings.warn(
        'origin_allowlist contains "*" — the WebSocket origin check is disabled. '
        "Every browser origin can drive this agent. Development only.",
        InsecureTransportWarning,
        stacklevel=3,
    )
    log.warning(
        "insecure_ws_origin_policy",
        origin_allowlist=list(origin_allowlist),
        dev_mode=dev_mode,
        hint="wildcard origin accepted because dev_mode=True",
    )


async def _send_error(ws: WebSocket, code: str, message: str) -> None:
    await ws.send_json({"type": "errored", "code": code, "message": message})


async def _handshake(ws: WebSocket, *, origin_allowlist: list[str], auth_impl: WSAuth) -> bool:
    """Origin check, then auth, then ``accept()``. False means already closed."""
    origin = ws.headers.get("origin", "")
    if ANY_ORIGIN not in origin_allowlist and origin not in origin_allowlist:
        await ws.close(code=4003)
        return False
    if not await auth_impl.authenticate(ws):
        await ws.close(code=4001)
        return False
    await ws.accept()
    return True


async def _open_session(
    ws: WebSocket,
    *,
    session_factory: Callable[[WebSocket], Awaitable[AgentSession]],
    path: str,
) -> AgentSession | None:
    """Build the session and initialize it **in the caller's task**.

    Eager initialization is the fix for the cancel-scope violation described
    in the module docstring: ``AgentSession.initialize`` starts MCP clients,
    whose anyio scopes must be exited by the task that entered them. Doing it
    here means the route task owns both ends, and no per-turn stream task ever
    starts a client the route task will later shut down.

    Returns ``None`` (socket already closed) when either step fails.
    """
    try:
        session = await session_factory(ws)
    except Exception:
        log.exception("ws_session_factory_failed", path=path)
        await ws.close(code=1011, reason="session factory failed")
        return None
    try:
        await session.initialize()
    except Exception:
        log.exception("ws_session_initialize_failed", path=path)
        with contextlib.suppress(Exception):
            await session.shutdown()
        await ws.close(code=1011, reason="session initialization failed")
        return None
    return session


async def _replay_history(ws: WebSocket, session: AgentSession, *, limit: int, path: str) -> None:
    """Send one ``history`` frame with the session's stored transcript.

    Opt-in, and best-effort by design. A reconnecting client that cannot get
    its history back should still get a working socket: the replay is context,
    the live stream is the product. So a missing store or a failing read is
    logged and skipped rather than closing the connection.

    The frame is the REST body of the same page plus a ``type`` — see
    :mod:`agentkit.history`. It is not a :class:`~agentkit.events.base.BaseEvent`
    and carries no ``event_id``/``turn_id``/``sequence``: it belongs to no turn.
    """
    store = session.config.stores.session
    if store is None:
        log.warning(
            "ws_history_replay_skipped",
            path=path,
            session_id=str(session.id),
            hint="replay_history=True but config.stores.session is None",
        )
        return
    try:
        page = await load_history_page(store, session.id, limit=limit)
    except Exception:
        log.exception("ws_history_replay_failed", path=path, session_id=str(session.id))
        return
    await ws.send_json(history_frame(page))


def _lifetime_deadline(max_connection_seconds: float | None) -> float | None:
    """Absolute monotonic time this connection must be closed at, if bounded."""
    if max_connection_seconds is None:
        return None
    return asyncio.get_running_loop().time() + max_connection_seconds


async def _receive_or_expire(ws: WebSocket, deadline: float | None) -> dict[str, Any] | None:
    """Await the next inbound command, or ``None`` if the lifetime bound elapsed.

    Called only from the *between-turns* position in the receive loop, which is
    what makes the bound safe: an expiry decided here can never truncate a
    stream the user is watching. A turn that starts before the deadline and
    runs past it is not interrupted — the loop comes back here when it ends,
    finds the budget already spent, and closes then. That deferral is the whole
    point. A close landing mid-stream would turn a staleness control into
    visible data loss, which is strictly worse than the staleness it buys.
    """
    if deadline is None:
        return await ws.receive_json()
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        # The previous turn outlived the bound. Do not even start a receive.
        return None
    try:
        return await asyncio.wait_for(ws.receive_json(), remaining)
    except TimeoutError:
        return None


async def _close_expired(ws: WebSocket, *, path: str, session: AgentSession) -> None:
    """Send the lifetime-exceeded close frame. Errors are logged, never raised.

    The surrounding ``finally`` still runs :func:`_close_session`, whose own
    ``ws.close(1000)`` is a suppressed no-op once this frame has gone out. We
    send from here rather than from the ``finally`` so the client learns *why*
    immediately, instead of after however long session shutdown takes.
    """
    log.info(
        "ws_connection_lifetime_exceeded",
        path=path,
        session_id=str(session.id),
        code=WS_CLOSE_LIFETIME_EXCEEDED,
    )
    with contextlib.suppress(RuntimeError, WebSocketDisconnect):
        await ws.close(code=WS_CLOSE_LIFETIME_EXCEEDED, reason=WS_CLOSE_LIFETIME_EXCEEDED_REASON)


async def _close_session(session: AgentSession, ws: WebSocket, *, path: str) -> None:
    """Shut the session down in the route task, then close the socket.

    Shutdown is suppressed-and-logged so a failing MCP teardown cannot stop
    the close frame from going out.
    """
    try:
        await session.shutdown()
    except Exception:
        log.exception("ws_session_shutdown_failed", path=path)
    # Already closed by the client / framework? That's fine.
    with contextlib.suppress(RuntimeError):
        await ws.close(code=1000)


def mount_websocket_route(
    app: FastAPI,
    *,
    path: str = "/ws/agent",
    session_factory: Callable[[WebSocket], Awaitable[AgentSession]],
    origin_allowlist: list[str],
    auth: WSAuth,
    approval_authority: WSApprovalAuthority | None = None,
    max_connection_seconds: float | None = None,
    replay_history: bool = False,
    replay_limit: int = DEFAULT_HISTORY_PAGE_LIMIT,
    dev_mode: bool = False,
) -> None:
    """Mount the agent WebSocket endpoint on ``app``.

    ``auth`` is **required** — see the module docstring. ``origin_allowlist``
    is matched exactly against the handshake's ``Origin`` header; ``"*"``
    raises unless ``dev_mode=True``.

    ``approval_authority`` gates ``respond_to_approval`` frames independently
    of ``auth``, so approvals can be bound to a different principal than the
    one driving the model. Defaults to
    :class:`SameSocketApprovalAuthority` — same principal, documented hazard.

    ``max_connection_seconds`` bounds how long one connection may live.
    ``None`` (the default) is unbounded — the historical behaviour, unchanged
    for every existing caller. When set, the socket is closed with
    :data:`WS_CLOSE_LIFETIME_EXCEEDED` once the wall clock passes the bound,
    **but only between turns**: a turn already streaming runs to completion and
    the close happens when it ends. Pass this when the handshake carried an
    authorization decision that the connection then holds forever — a
    forward-auth header, a cookie, a group membership. Nothing can re-check
    those on an open socket (there are no headers after the handshake), so the
    only control available is to make the client come back and be re-checked.
    The client sees an ordinary close and its existing reconnect path re-runs
    the handshake; it needs no change to be correct, and can special-case the
    code to skip a backoff delay it does not need.

    ``replay_history=True`` sends one ``history`` frame right after the socket
    opens, carrying the newest ``replay_limit`` stored messages projected to
    the slim shapes in :mod:`agentkit.history`. Off by default: it is a second
    delivery of content the client may already have, and only the consumer
    knows whether its client dedupes. A consumer that would rather fetch the
    same page over HTTP calls :func:`~agentkit.history.load_history_page` from
    its own route — same shapes, so the client parses one thing either way.

    Raises ``TypeError`` for a missing or malformed ``auth`` and
    ``ValueError`` for a wildcard origin outside dev mode; both at mount time,
    so a misconfiguration fails at import/startup rather than on the first
    connection.
    """
    if max_connection_seconds is not None and max_connection_seconds <= 0:
        raise ValueError(
            "max_connection_seconds must be positive, or None for an unbounded "
            f"connection; got {max_connection_seconds!r}. A zero or negative "
            "bound would close every connection before it could carry a turn."
        )
    auth_impl: WSAuth = _validate_auth(auth)
    _validate_origin_allowlist(origin_allowlist, dev_mode=dev_mode)
    approval_impl: WSApprovalAuthority = approval_authority or SameSocketApprovalAuthority()

    async def _dispatch(
        ws: WebSocket, session: AgentSession, cmd: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Run one inbound command. Returns a command deferred during a turn."""
        ctype = cmd.get("type")
        if ctype == "send_message":
            text = str(cmd.get("text", ""))

            async def _run(cancel_evt: asyncio.Event) -> None:
                async with session.run(text) as stream:
                    await _drain_stream_into_ws(ws, stream, cancel_evt)

            return await _stream_with_cancel_watch(ws, _run)

        if ctype == "respond_to_approval":
            if not await approval_impl.authorize_approval(ws, cmd):
                log.warning(
                    "ws_approval_not_authorized",
                    path=path,
                    call_id=cmd.get("call_id"),
                    turn_id=cmd.get("turn_id"),
                )
                await _send_error(
                    ws,
                    "approval_not_authorized",
                    "this connection is not permitted to answer approval prompts",
                )
                return None

            async def _resume(cancel_evt: asyncio.Event) -> None:
                async with session.resume_with_approval(
                    cmd["turn_id"],
                    cmd["call_id"],
                    decision=str(cmd.get("decision", "deny")),
                    edited_args=cmd.get("edited_args"),
                    reason=cmd.get("reason"),
                ) as stream:
                    await _drain_stream_into_ws(ws, stream, cancel_evt)

            return await _stream_with_cancel_watch(ws, _resume)

        if ctype == "cancel":
            # No active turn — nothing to cancel. Acknowledge and continue.
            await ws.send_json({"type": "cancelled", "reason": "no_active_turn"})
            return None

        await _send_error(ws, "invalid_command", f"unknown command: {ctype}")
        return None

    @app.websocket(path)
    async def _ws_route(ws: WebSocket) -> None:  # pyright: ignore[reportUnusedFunction]
        if not await _handshake(ws, origin_allowlist=origin_allowlist, auth_impl=auth_impl):
            return
        # Clocked from the handshake, because what the bound is really ageing
        # out is the authorization decision _handshake just made.
        deadline = _lifetime_deadline(max_connection_seconds)
        session = await _open_session(ws, session_factory=session_factory, path=path)
        if session is None:
            return
        pending_cmd: dict[str, Any] | None = None
        try:
            # Inside the try: a client that vanishes mid-replay must still take
            # the ``finally`` path that shuts the session down.
            if replay_history:
                await _replay_history(ws, session, limit=replay_limit, path=path)
            while True:
                cmd: dict[str, Any] | None
                if pending_cmd is not None:
                    # A command buffered during the turn we just finished. It is
                    # already queued work, so it runs even past the deadline —
                    # the expiry is checked when the socket is genuinely idle.
                    cmd = pending_cmd
                else:
                    cmd = await _receive_or_expire(ws, deadline)
                    if cmd is None:
                        await _close_expired(ws, path=path, session=session)
                        break
                pending_cmd = await _dispatch(ws, session, cmd)
        except WebSocketDisconnect:
            pass
        finally:
            await _close_session(session, ws, path=path)
