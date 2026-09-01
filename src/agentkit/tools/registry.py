"""ToolRegistry — single source of truth for what tools exist this session.

Three registration sources:
1. Built-in tools: registered with a Python handler. Use the ``kit.`` namespace.
2. In-process MCP clients: ``InProcessMCPClient`` instances; their tools come
   under ``<server_name>.<tool_name>``.
3. Stdio MCP clients: subprocess-backed MCP servers; same naming rule.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from agentkit._logging import get_logger
from agentkit.errors import ToolError as ToolErr
from agentkit.guards.taint import (
    RiskBasedTaintPolicy,
    TaintPolicy,
    denied_result,
    mark_taint,
)
from agentkit.loop.context import TurnContext
from agentkit.tools.spec import (
    DEFAULT_TOOL_TIMEOUT_SECONDS,
    ContentBlockOut,
    ToolCall,
    ToolResult,
    ToolSpec,
    invalid_arguments_message,
    unknown_tool_message,
    validate_tool_arguments,
)
from agentkit.tools.spec import ToolError as ToolErrorModel

if TYPE_CHECKING:
    from agentkit.mcp_client.base import MCPClient

log = get_logger(__name__)


# A built-in handler signature: (arguments, ctx) -> ToolResult.
BuiltinHandler = Callable[[dict[str, Any], TurnContext], Awaitable[ToolResult]]


@runtime_checkable
class ToolAuthorizer(Protocol):
    """Execution-time gate: may this context invoke this tool at all?

    Visibility filtering (the ``tool_selector`` / ToolPlane hook) shapes only
    what is *advertised*. A model that names a tool it was never shown — from
    an earlier turn, a system prompt, a hallucination, or an injected
    instruction — otherwise reaches the unfiltered registry. An authorizer
    closes that gap by re-running the same decision at invoke time.

    Return ``None`` to permit the call, or a model-readable reason to deny it.
    ``ctx`` is the ``TurnContext``, typed loosely to avoid a circular import.
    """

    def authorize(self, spec: ToolSpec, ctx: Any) -> str | None: ...


class ToolRegistry:
    """Registry of invocable tools, and the enforcement point in front of them.

    Every gate that decides whether a call may run lives in :meth:`invoke`, not
    in the listing API: advertisement is advisory, execution is authoritative.

    ``authorizer`` — optional execution-time visibility/capability gate; see
    :class:`ToolAuthorizer` and
    :class:`agentkit.toolplane.plane.ToolPlaneAuthorizer`. ``None`` (the
    default) means no visibility policy is configured for this registry, which
    matches the advertisement side: nothing is filtered there either.

    ``taint_policy`` — anti-prompt-injection guard, ON by default. Pass
    :class:`agentkit.guards.taint.NullTaintPolicy` to opt out explicitly, or
    your own :class:`~agentkit.guards.taint.TaintPolicy` to change the ceiling.

    ``validate_arguments`` — structural check of model-produced arguments
    against ``ToolSpec.parameters`` before dispatch. On by default.

    ``default_timeout_seconds`` — used when a spec declares no usable
    ``timeout_seconds``.
    """

    def __init__(
        self,
        *,
        authorizer: ToolAuthorizer | None = None,
        taint_policy: TaintPolicy | None = None,
        validate_arguments: bool = True,
        default_timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
    ) -> None:
        self._builtins: dict[str, tuple[ToolSpec, BuiltinHandler]] = {}
        self._mcp_servers: dict[str, MCPClient] = {}
        self._mcp_specs: dict[str, ToolSpec] = {}  # qualified_name -> spec
        self._failed_servers: dict[str, str] = {}  # name -> error message; tools unavailable
        self._authorizer = authorizer
        self._taint_policy: TaintPolicy = (
            RiskBasedTaintPolicy() if taint_policy is None else taint_policy
        )
        self._validate_arguments = validate_arguments
        self._default_timeout_seconds = default_timeout_seconds

    def set_authorizer(self, authorizer: ToolAuthorizer | None) -> None:
        """Install (or clear) the execution-time authorization gate.

        Available as a setter because the natural authorizer — a ToolPlane —
        is often constructed after the registry it guards.
        """
        self._authorizer = authorizer

    @property
    def authorizer(self) -> ToolAuthorizer | None:
        """The installed execution-time gate, if any.

        Readable so a caller that wants to *add* a gate can compose it with
        whatever is already there (see
        :func:`agentkit.subagents.isolation.chain_authorizers`) instead of
        silently replacing another layer's enforcement.
        """
        return self._authorizer

    # ---- Registration ------------------------------------------------------

    def register_builtin(self, spec: ToolSpec, handler: BuiltinHandler) -> None:
        if spec.name in self._builtins:
            raise ToolErr(f"duplicate builtin registration: {spec.name}")
        self._builtins[spec.name] = (spec, handler)

    def register_default_builtins(self) -> None:
        """Register every entry of :data:`DEFAULT_BUILTINS`.

        Convenience for the common case — equivalent to looping over
        ``DEFAULT_BUILTINS`` and calling :meth:`register_builtin` for each entry.
        """
        # Local import to avoid a module-load cycle (builtin handlers may import
        # from ``agentkit.tools`` indirectly).
        from agentkit.tools.builtin import DEFAULT_BUILTINS  # noqa: PLC0415

        for spec, handler in DEFAULT_BUILTINS:
            self.register_builtin(spec, handler)

    def register_mcp_server(self, name: str, client: MCPClient) -> None:
        """Register an MCP server. Call ``initialize_mcp_servers`` after to load tools."""
        if name in self._mcp_servers:
            raise ToolErr(f"duplicate MCP server registration: {name}")
        self._mcp_servers[name] = client

    async def initialize_mcp_servers(self) -> None:
        """Connect to every registered MCP server and import its tools.

        A server whose ``initialize()`` or ``list_tools()`` raises is marked as
        failed (see :attr:`failed_servers`) but does **not** abort the loop —
        other servers continue to load. The session can still run; the failed
        server's tools simply aren't available.
        """
        for server_name, client in self._mcp_servers.items():
            try:
                await client.initialize()
                specs = await client.list_tools()
            except Exception as exc:
                self._failed_servers[server_name] = f"{type(exc).__name__}: {exc}"
                # Best-effort cleanup — shutdown may itself raise on a partially
                # initialized client; we record the original error regardless.
                with contextlib.suppress(Exception):
                    await client.shutdown()
                continue
            for spec in specs:
                qualified = f"{server_name}.{spec.name}"
                if qualified in self._mcp_specs or qualified in self._builtins:
                    raise ToolErr(f"namespace collision: {qualified}")
                self._mcp_specs[qualified] = spec.model_copy(update={"name": qualified})

    @property
    def failed_servers(self) -> dict[str, str]:
        """Map of MCP server name to the error message from initialization.

        Empty when all registered servers initialized cleanly.
        """
        return dict(self._failed_servers)

    async def shutdown(self) -> None:
        for client in self._mcp_servers.values():
            await client.shutdown()

    # ---- Listing -----------------------------------------------------------

    def list_specs(self, *, available_to_provider: bool = True) -> list[ToolSpec]:
        # available_to_provider is forward API for risk-based filtering once the
        # ApprovalGate / TurnContext lands (Phase 7+). Today it is a no-op.
        _ = available_to_provider
        out = [spec for spec, _ in self._builtins.values()]
        out.extend(self._mcp_specs.values())
        return out

    def spec_for(self, name: str) -> ToolSpec | None:
        """The spec registered under ``name``, or ``None`` if there is none.

        O(1) lookup — prefer it over scanning :meth:`list_specs`.
        """
        entry = self._builtins.get(name)
        if entry is not None:
            return entry[0]
        return self._mcp_specs.get(name)

    # ---- Invocation --------------------------------------------------------

    async def invoke(self, call: ToolCall, ctx: TurnContext) -> ToolResult:
        """Run one tool call, applying every execution-time gate.

        Order: resolve -> authorize -> taint guard -> argument validation ->
        timeout-bounded execution. Each gate returns a normal ``ToolResult``
        the model can read and react to; nothing here raises for a condition
        the model caused, because a raised exception ends the turn with no
        result and the model never learns what went wrong.
        """
        spec = self.spec_for(call.name)
        if spec is None:
            # Unknown tool name (commonly a model hallucination). Return a
            # status="error" ToolResult instead of raising — raising bubbles to
            # the orchestrator and ends the turn with no result, so the model
            # never sees the mistake. As an error result it flows to the model,
            # which can self-correct with the right tool name.
            msg = unknown_tool_message(call.name, [*self._builtins, *self._mcp_specs])
            return ToolResult(
                call_id=call.id,
                status="error",
                content=[ContentBlockOut(type="text", text=msg)],
                error=ToolErrorModel(code="unknown_tool", message=msg, retryable=True),
                duration_ms=0,
                cached=False,
            )

        # 1. Authorization: the tool exists, but is it reachable for this
        # context? Naming a hidden tool must not run it.
        reason = self._authorization_denial(spec, ctx)
        if reason is not None:
            log.warning("tool_call_unauthorized", tool=call.name, reason=reason)
            return denied_result(call.id, reason, code="not_authorized")

        # 2. Taint: has this turn already ingested untrusted content?
        reason = self._taint_denial(spec, ctx)
        if reason is not None:
            log.warning("tool_call_denied_tainted_turn", tool=call.name, risk=str(spec.risk))
            return denied_result(call.id, reason)

        # 3. Arguments: never dispatch a call whose arguments the schema
        # already rejects — the tool would fail deeper and less legibly.
        if self._validate_arguments:
            problems = validate_tool_arguments(spec.parameters, call.arguments)
            if problems:
                msg = invalid_arguments_message(call.name, problems)
                # The only one of the three gates above that used to return in
                # silence, which made a schema-rejected call the one refusal
                # with no trace anywhere: not in the logs, and not in the
                # persisted messages either, because the store records tool_use
                # arguments as null. `problems` names argument keys and types,
                # never their values, so it is safe to log.
                log.warning("tool_call_invalid_arguments", tool=call.name, problems=problems)
                # Recorded on the turn as well as logged: a caller downstream
                # may need to tell "the model never called this tool" from "the
                # model called it and we refused" — for the finalize tool those
                # are opposite situations that both leave `finalize_called`
                # False, since this gate runs before the handler that sets it.
                rejected = ctx.metadata.setdefault("invalid_argument_calls", {})
                if isinstance(rejected, dict):
                    cast("dict[str, str]", rejected)[call.name] = msg
                return ToolResult(
                    call_id=call.id,
                    status="error",
                    content=[ContentBlockOut(type="text", text=msg)],
                    error=ToolErrorModel(code="invalid_arguments", message=msg, retryable=True),
                    duration_ms=0,
                    cached=False,
                )

        runner = self._runner_for(call, ctx)
        return await self._execute(call, spec, runner, ctx)

    def _runner_for(self, call: ToolCall, ctx: TurnContext) -> Callable[[], Awaitable[ToolResult]]:
        """Return a nullary callable that performs the actual invocation.

        Deferred (rather than an already-created coroutine) so a gate can
        decline without leaving an un-awaited coroutine behind.
        """
        builtin = self._builtins.get(call.name)
        if builtin is not None:
            _spec, handler = builtin

            def _run_builtin() -> Awaitable[ToolResult]:
                return handler(call.arguments, ctx)

            return _run_builtin

        server, _, bare = call.name.partition(".")
        client = self._mcp_servers.get(server)
        if client is None:
            # Registry-internal inconsistency (a spec whose server vanished),
            # not something the model did — surface it as a real exception.
            raise ToolErr(f"server not found for tool: {call.name}")

        # Bridge MCP server-side progress notifications to user-facing
        # ToolCallProgress events. The dispatcher sets ctx.call_id to the
        # current call before invoking; report_tool_progress no-ops if
        # ctx has no event_queue (e.g., subagent-internal contexts).
        async def _on_progress(message: str, progress: float | None, total: float | None) -> None:
            await ctx.report_tool_progress(message, call_id=call.id, progress=progress, total=total)

        def _run_mcp() -> Awaitable[ToolResult]:
            return client.call_tool(bare, call.arguments, on_progress=_on_progress)

        return _run_mcp

    async def _execute(
        self,
        call: ToolCall,
        spec: ToolSpec,
        runner: Callable[[], Awaitable[ToolResult]],
        ctx: TurnContext,
    ) -> ToolResult:
        """Await ``runner`` under a timeout, converting failure into a result."""
        timeout = self._timeout_for(spec)
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(runner(), timeout)
        except TimeoutError:
            elapsed = int((time.monotonic() - started) * 1000)
            msg = (
                f"tool {call.name} timed out after {timeout:g}s and was cancelled. "
                f"Retry, or tell the user the tool is not responding."
            )
            log.warning("tool_call_timeout", tool=call.name, timeout_seconds=timeout)
            return ToolResult(
                call_id=call.id,
                status="timeout",
                content=[ContentBlockOut(type="text", text=msg)],
                error=ToolErrorModel(code="timeout", message=msg, retryable=True),
                duration_ms=elapsed,
                cached=False,
            )
        except Exception as exc:
            # A raising handler must never kill the turn: siblings dispatched
            # alongside it would be orphaned and the model would see nothing.
            elapsed = int((time.monotonic() - started) * 1000)
            msg = f"tool {call.name} raised {type(exc).__name__}: {exc}"
            log.exception("tool_call_exception", tool=call.name, error_type=type(exc).__name__)
            return ToolResult(
                call_id=call.id,
                status="error",
                content=[ContentBlockOut(type="text", text=msg)],
                error=ToolErrorModel(code="tool_exception", message=msg, retryable=True),
                duration_ms=elapsed,
                cached=False,
            )

        # Built-in and MCP handlers each construct their own ToolResult; the
        # MCP ``call_tool`` signature does not even carry the call_id. Stamp
        # ``call.id`` onto the returned result here so downstream serialisation
        # always emits a well-formed ``tool_call_id`` — Gemini rejects empty
        # strings outright ("Tool message must have either name or
        # tool_call_id"); other providers tolerate the gap silently.
        result = result.model_copy(update={"call_id": call.id})
        if mark_taint(ctx, result, tool_name=call.name):
            log.info("turn_tainted", tool=call.name)
        return result

    def _timeout_for(self, spec: ToolSpec) -> float:
        declared = spec.timeout_seconds
        if declared is None or declared <= 0:  # type: ignore[reportUnnecessaryComparison]
            return self._default_timeout_seconds
        return declared

    def _authorization_denial(self, spec: ToolSpec, ctx: TurnContext) -> str | None:
        if self._authorizer is None:
            return None
        try:
            return self._authorizer.authorize(spec, ctx)
        except Exception as exc:
            # Fail closed: an authorizer that cannot answer denies.
            log.exception("tool_authorizer_failed", tool=spec.name, error_type=type(exc).__name__)
            return (
                f"denied: the authorization check for {spec.name} failed "
                f"({type(exc).__name__}), so the call was not executed."
            )

    def _taint_denial(self, spec: ToolSpec, ctx: TurnContext) -> str | None:
        try:
            return self._taint_policy.denial_reason(spec, ctx)
        except Exception as exc:
            # Fail closed, same rationale as the authorizer.
            log.exception("taint_policy_failed", tool=spec.name, error_type=type(exc).__name__)
            return (
                f"denied: the taint check for {spec.name} failed "
                f"({type(exc).__name__}), so the call was not executed."
            )
