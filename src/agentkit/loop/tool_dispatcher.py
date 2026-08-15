"""ToolDispatcher — runs tool calls with the right concurrency policy."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from agentkit._logging import get_logger
from agentkit._redaction import argument_preview, clean_text
from agentkit.audit import (
    ACTION_TOOL_CALL,
    SOURCE_AGENT,
    AuditRecord,
    AuditSink,
    NullAuditSink,
    record_audit,
)
from agentkit.events.approval import UNKNOWN_CLASSIFICATION
from agentkit.guards.taint import is_tainted, mark_taint
from agentkit.tools.registry import ToolRegistry
from agentkit.tools.spec import (
    ContentBlockOut,
    RiskLevel,
    ToolCall,
    ToolError,
    ToolResult,
    ToolSpec,
)

log = get_logger(__name__)

#: Cap on the error message copied into an audit record's detail.
_MAX_AUDIT_ERROR_CHARS = 200


@dataclass(frozen=True)
class DispatchPolicy:
    max_parallel: int = 8
    """Cap on concurrent tool executions when running in parallel mode."""


def _exception_result(call: ToolCall, exc: BaseException) -> ToolResult:
    """Map an escaped exception to the result the model should see.

    A cancelled child keeps its own status rather than being reported as a
    tool failure — the tool did not fail, it was stopped.
    """
    if isinstance(exc, asyncio.CancelledError):
        msg = f"tool {call.name} was cancelled before it produced a result."
        return ToolResult(
            call_id=call.id,
            status="cancelled",
            content=[ContentBlockOut(type="text", text=msg)],
            error=ToolError(code="cancelled", message=msg, retryable=True),
            duration_ms=0,
            cached=False,
        )
    msg = f"tool {call.name} raised {type(exc).__name__}: {exc}"
    return ToolResult(
        call_id=call.id,
        status="error",
        content=[ContentBlockOut(type="text", text=msg)],
        error=ToolError(code="tool_exception", message=msg, retryable=True),
        duration_ms=0,
        cached=False,
    )


def _audit_record(
    call: ToolCall, result: ToolResult, spec: ToolSpec | None, ctx: Any
) -> AuditRecord:
    """Build the audit record for one dispatched call.

    Every call, not only the risky ones: "which tools are audited" is a
    question with one answer here, and a per-tool answer is how a vault
    deletion ends up unrecorded while a torrent add is not. ``risk`` and
    ``side_effects`` come off the spec so a reader can tell a rename from a
    destruction without a lookup table.
    """
    metadata: dict[str, Any] = getattr(ctx, "metadata", None) or {}
    preview, args_truncated = argument_preview(call.arguments)
    detail: dict[str, Any] = {
        "status": result.status,
        "risk": spec.risk.value if spec is not None else UNKNOWN_CLASSIFICATION,
        "side_effects": spec.side_effects.value if spec is not None else UNKNOWN_CLASSIFICATION,
        "arguments": preview,
        "arguments_truncated": args_truncated,
        "duration_ms": result.duration_ms,
        "cached": result.cached,
        "tainted": is_tainted(ctx),
    }
    if result.error is not None:
        detail["error"] = {
            "code": result.error.code,
            "message": clean_text(result.error.message, limit=_MAX_AUDIT_ERROR_CHARS),
        }
    return AuditRecord(
        ts=datetime.now(UTC),
        actor=str(metadata.get("owner", "")),
        action=ACTION_TOOL_CALL,
        target=call.name,
        detail=detail,
        source=SOURCE_AGENT,
        session_id=str(getattr(ctx, "session_id", "")) or None,
        turn_id=str(getattr(ctx, "turn_id", "")) or None,
        call_id=call.id,
    )


class ToolDispatcher:
    def __init__(
        self,
        *,
        registry: ToolRegistry,
        policy: DispatchPolicy,
        audit: AuditSink | None = None,
    ) -> None:
        self._registry = registry
        self._policy = policy
        # Never None internally: the audit call is unconditional, so there is
        # no branch a future edit can forget to take.
        self._audit: AuditSink = audit or NullAuditSink()

    async def run(self, calls: Sequence[ToolCall], ctx: Any) -> list[ToolResult]:
        if not calls:
            return []
        if self._safe_for_parallel(calls):
            return await self._run_parallel(calls, ctx)
        return await self._run_sequential(calls, ctx)

    def _safe_for_parallel(self, calls: Sequence[ToolCall]) -> bool:
        """All calls must be READ + idempotent for parallel dispatch.

        An unknown tool name has no spec — treat it as not-parallel-safe so
        dispatch falls to the sequential path and ``registry.invoke`` produces
        the error ToolResult. Never raise here: a raised exception bubbles to
        the orchestrator and kills the turn with no result for the model.
        """
        for call in calls:
            spec = self._find_spec(call.name)
            if spec is None or spec.risk != RiskLevel.READ or not spec.idempotent:
                return False
        return True

    def _find_spec(self, name: str) -> ToolSpec | None:
        return self._registry.spec_for(name)

    async def _invoke_guarded(self, call: ToolCall, ctx: Any) -> ToolResult:
        """Invoke one call, converting any escaping exception into a result.

        ``registry.invoke`` already turns a raising handler into an error
        result; this is the outer belt for everything else it can raise
        (registry-internal invariants, a broken authorizer, a client that
        blows up before the handler is reached). One bad call must never
        orphan its parallel siblings or end the turn with no results at all.
        """
        try:
            result = await self._registry.invoke(call, ctx)
        except Exception as exc:
            log.exception("tool_dispatch_exception", tool=call.name, error_type=type(exc).__name__)
            result = _exception_result(call, exc)
        else:
            # The registry marks taint too; doing it here keeps the invariant
            # true for any registry (or test double) that does not.
            if mark_taint(ctx, result, tool_name=call.name):
                log.info("turn_tainted", tool=call.name)
        # Audit last, and on both paths: a call that blew up is exactly the one
        # a reader needs the record for. record_audit never raises.
        spec = self._find_spec(call.name)
        await record_audit(self._audit, _audit_record(call, result, spec, ctx))
        return result

    async def _run_parallel(self, calls: Sequence[ToolCall], ctx: Any) -> list[ToolResult]:
        sem = asyncio.Semaphore(self._policy.max_parallel)

        async def _bounded(call: ToolCall) -> ToolResult:
            async with sem:
                return await self._invoke_guarded(call, ctx)

        # return_exceptions=True is defence in depth: if anything still escapes
        # ``_invoke_guarded``, gather returns it in that call's slot instead of
        # cancelling the siblings, and we map it to an error result in place so
        # results stay positionally aligned with ``calls``.
        gathered = await asyncio.gather(*(_bounded(c) for c in calls), return_exceptions=True)
        results: list[ToolResult] = []
        for call, outcome in zip(calls, gathered, strict=True):
            if isinstance(outcome, BaseException):
                log.exception(
                    "tool_dispatch_escaped_exception",
                    tool=call.name,
                    error_type=type(outcome).__name__,
                    exc_info=outcome,
                )
                results.append(_exception_result(call, outcome))
            else:
                results.append(outcome)
        return results

    async def _run_sequential(self, calls: Sequence[ToolCall], ctx: Any) -> list[ToolResult]:
        results: list[ToolResult] = []
        for call in calls:
            results.append(await self._invoke_guarded(call, ctx))
        return results
