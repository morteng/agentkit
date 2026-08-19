"""ToolDispatcher — runs tool calls with the right concurrency policy."""

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from pydantic import ValidationError

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

#: ``ctx.metadata`` key holding this turn's ledger of executed non-idempotent
#: calls: ``{identity -> ToolResult as a JSON-safe dict}``.
#:
#: **Turn-scoped by construction, and that is load-bearing.**
#: ``TurnContext.metadata`` is a ``field(default_factory=dict)`` on a context
#: ``AgentSession.send`` builds fresh — new ``TurnId`` and all — for every turn,
#: so a ledger filled in one turn cannot be visible in the next, and the
#: principal asking for the same write again tomorrow still gets it. The single
#: path that carries metadata across a context boundary is an approval
#: suspend/resume, which rebuilds the context under the *same* ``turn_id`` from
#: the checkpoint: the same turn, so the ledger *should* survive it, and does.
#:
#: Entries are stored as JSON-safe dicts rather than ``ToolResult`` objects for
#: that resume path specifically: ``loop.context.to_checkpoint_payload``
#: serialises metadata with ``json.dumps(..., default=str)``, which would
#: flatten a model instance to its repr and leave the resumed turn holding a
#: string it cannot read back.
_REPLAY_LEDGER_KEY = "nonidempotent_call_ledger"

#: Prefixed to the content a suppressed repeat returns.
#:
#: Plain language, and it says the effect *happened* rather than that the call
#: was refused. A model told only "denied" concludes the action did not land and
#: goes looking for a third route to the same effect — which is the failure this
#: guard exists to prevent, arriving by a different door.
_REPLAY_NOTE = (
    "This exact call already ran earlier in this turn and its effect has already "
    "happened, so it was not run a second time. What follows is the result from "
    "that first run. Treat the action as done — do not attempt another way to "
    "achieve it."
)


@dataclass(frozen=True)
class DispatchPolicy:
    max_parallel: int = 8
    """Cap on concurrent tool executions when running in parallel mode."""

    guard_nonidempotent_replay: bool = True
    """Suppress a repeat of a call this turn already ran successfully, for tools
    declaring ``idempotent=False``. See
    :attr:`agentkit.config.ToolDispatchConfig.guard_nonidempotent_replay` for
    why the loop produces such repeats in the first place."""


def _call_identity(call: ToolCall) -> str | None:
    """A stable key for "this exact call", or ``None`` when one cannot be made.

    ``sort_keys`` so the same arguments spelled in a different key order are
    still recognised as the same effect — a model re-emitting a call after a
    correction routinely reorders its JSON, and a key-order-sensitive identity
    would miss every one of those, which are precisely the repeats worth
    catching. ``default=str`` so a value ``json`` cannot encode natively (a
    ``datetime`` an MCP bridge left in the arguments) degrades to something
    comparable instead of raising.

    Returning ``None`` rather than raising is the safe direction: no identity
    means the guard abstains and the call executes, which is exactly the
    behaviour that existed before the guard. Raising here would turn an
    un-encodable argument into a dead turn.
    """
    try:
        canonical = json.dumps(call.arguments, sort_keys=True, default=str)
    except Exception:  # abstain on ANY encoding failure; see docstring
        log.warning("replay_identity_unavailable", tool=call.name)
        return None
    # NUL as the separator: ``json.dumps`` never emits a raw NUL byte (one
    # inside a string comes out escaped, as six printable characters), so no
    # tool name and argument blob can collide by straddling the boundary.
    return f"{call.name}\x00{canonical}"


def _replay_result(call: ToolCall, recorded: Any) -> ToolResult:
    """Build the result a suppressed repeat should see, from the ledger entry."""
    try:
        earlier = ToolResult.model_validate(recorded)
    except ValidationError:
        # Losing the earlier content is a shame; running the tool again is a
        # second real-world effect. What the ledger proves is that the call
        # ALREADY RAN, and that is not in doubt here — only its payload is — so
        # still suppress, and say plainly that the earlier output is gone rather
        # than inventing one. Status "ok" is truthful: only successful calls are
        # ever recorded, so the call this stands in for did succeed.
        log.warning("replay_ledger_entry_unreadable", tool=call.name, call_id=call.id)
        return ToolResult(
            call_id=call.id,
            status="ok",
            content=[
                ContentBlockOut(
                    type="text",
                    text=f"{_REPLAY_NOTE} Its output is no longer available to repeat here.",
                )
            ],
        )
    # The recorded ``call_id`` belongs to the first call and would leave the
    # model holding a result it cannot match to the request it just made — and,
    # on the wire, an id the provider never issued for this turn's message.
    return earlier.model_copy(
        update={
            "call_id": call.id,
            "content": [ContentBlockOut(type="text", text=_REPLAY_NOTE), *earlier.content],
        }
    )


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
    call: ToolCall, result: ToolResult, spec: ToolSpec | None, ctx: Any, *, replayed: bool = False
) -> AuditRecord:
    """Build the audit record for one dispatched call.

    Every call, not only the risky ones: "which tools are audited" is a
    question with one answer here, and a per-tool answer is how a vault
    deletion ends up unrecorded while a torrent add is not. ``risk`` and
    ``side_effects`` come off the spec so a reader can tell a rename from a
    destruction without a lookup table.

    ``replayed`` marks the row as a repeat the replay guard suppressed. Always
    present, never conditional: a reader auditing a non-idempotent tool needs to
    tell "ran twice" from "was asked for twice and ran once", and a key that
    appears only on the interesting rows is one an aggregator reading older rows
    will read as absent-means-no when it actually means unknown.
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
        "replayed": replayed,
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

    def _replay_ledger(self, ctx: Any) -> dict[str, Any] | None:
        """This turn's ledger of executed non-idempotent calls, created on demand.

        ``None`` means "cannot guard", and cannot-guard means execute. Two ways
        to get there: the guard is switched off, or ``ctx`` carries no
        ``metadata`` mapping to keep a ledger in — a subagent stub or a test
        double, which the rest of this module already tolerates (see
        ``_audit_record``'s ``getattr``).
        """
        if not self._policy.guard_nonidempotent_replay:
            return None
        metadata = getattr(ctx, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        typed = cast("dict[str, Any]", metadata)
        ledger = typed.get(_REPLAY_LEDGER_KEY)
        if not isinstance(ledger, dict):
            # Missing, or restored from a checkpoint as something else. Either
            # way a fresh dict is the only thing safe to write into.
            ledger = {}
            typed[_REPLAY_LEDGER_KEY] = ledger
        return cast("dict[str, Any]", ledger)

    def _replayed_result(
        self, call: ToolCall, spec: ToolSpec | None, ctx: Any
    ) -> ToolResult | None:
        """The earlier result for ``call`` when this turn already ran it, else ``None``.

        ``None`` — go execute — for every case the guard does not own:

        * **``spec is None``.** An unknown name has no declared semantics to
          guard by, and suppressing it would swallow ``registry.invoke``'s
          unknown-tool error, which is the only thing that tells the model it
          used a name that does not exist.
        * **``spec.idempotent``.** Repeating an idempotent call is by definition
          harmless, and answering a deliberate re-read with a cached value would
          hide exactly the change the model re-read to find.
        * no ledger, or an identity that could not be canonicalised.
        * a first sighting of this identity — the common case.
        """
        if spec is None or spec.idempotent:
            return None
        ledger = self._replay_ledger(ctx)
        if ledger is None:
            return None
        identity = _call_identity(call)
        if identity is None:
            return None
        recorded = ledger.get(identity)
        if recorded is None:
            return None
        log.info(
            "tool_replay_suppressed",
            tool=call.name,
            call_id=call.id,
            risk=str(spec.risk),
            side_effects=str(spec.side_effects),
        )
        return _replay_result(call, recorded)

    def _record_execution(
        self, call: ToolCall, spec: ToolSpec | None, result: ToolResult, ctx: Any
    ) -> None:
        """Remember a successful non-idempotent call so a repeat can be answered.

        Success only, tested as ``status == "ok"`` — the same test
        ``loop.handlers.tool_results`` uses to decide whether a result is an
        error. A denied, timed-out, cancelled or failed call has consumed
        nothing the model must be stopped from asking for again, and recording
        it would turn a transient failure into a permanent one for the rest of
        the turn: the retry the model is entitled to would come back as the
        original failure, forever.
        """
        if spec is None or spec.idempotent or result.status != "ok":
            return
        ledger = self._replay_ledger(ctx)
        if ledger is None:
            return
        identity = _call_identity(call)
        if identity is None:
            return
        ledger[identity] = result.model_dump(mode="json")

    async def _invoke_guarded(self, call: ToolCall, ctx: Any) -> ToolResult:
        """Invoke one call, converting any escaping exception into a result.

        ``registry.invoke`` already turns a raising handler into an error
        result; this is the outer belt for everything else it can raise
        (registry-internal invariants, a broken authorizer, a client that
        blows up before the handler is reached). One bad call must never
        orphan its parallel siblings or end the turn with no results at all.

        It is also the per-turn replay guard's only seat, because it is the one
        funnel every invocation passes through: ``_run_parallel`` and
        ``_run_sequential`` both come here, so there is no path that reaches a
        handler around it.
        """
        # One lookup, shared by the replay guard, the ledger write and the audit
        # record. Safe to read once and reuse across the await: the registry is
        # populated at construction and nothing in the codebase registers a tool
        # during a dispatch, so the spec cannot change under us mid-call.
        spec = self._find_spec(call.name)

        replayed = self._replayed_result(call, spec, ctx)
        if replayed is not None:
            # Audited like any other dispatch, and flagged: "the model asked a
            # second time and we did not run it" is precisely the event a
            # reader auditing a non-idempotent tool came for, and the absence of
            # a row would read as the request never having been made.
            await record_audit(self._audit, _audit_record(call, replayed, spec, ctx, replayed=True))
            return replayed

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
        # Outside the try/except and before the audit, so an escaped exception
        # takes the same route — it arrives here as status="error", which
        # ``_record_execution`` declines to record.
        self._record_execution(call, spec, result, ctx)
        # Audit last, and on both paths: a call that blew up is exactly the one
        # a reader needs the record for. record_audit never raises.
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
