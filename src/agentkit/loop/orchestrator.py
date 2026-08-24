"""Loop — drive a single turn through the phase machine.

Handlers live in ``loop/handlers/*``. The orchestrator picks the right handler
for the current phase, awaits it, validates the transition, emits PhaseChanged,
and continues. Terminal phases (TURN_ENDED, ERRORED) end the run.

Every path into ERRORED is *observable*. A handler that raises, a phase with no
handler, and an illegal transition all produce the same three artefacts before
the run ends: a structlog record with the traceback, ``ctx.metadata["turn_error"]``
for the persisted turn, and an :class:`Errored` event on ``ctx.event_queue`` so
the consumer sees a failure instead of a turn that simply stops. An
AI-administered system whose turns can fail with zero diagnostics is not
auditable, so silence here is treated as a bug.
"""

import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from decimal import Decimal
from typing import Any, cast

from agentkit._content import ToolUseBlock
from agentkit._ids import EventId, MessageId, new_id
from agentkit._logging import get_logger
from agentkit._messages import Usage
from agentkit.errors import InvalidPhaseTransition, error_code_for
from agentkit.events import (
    ErrorCode,
    Errored,
    PhaseChanged,
    TurnEnded,
    TurnEndReason,
    TurnMetrics,
    TurnStarted,
)
from agentkit.events.base import BaseEvent
from agentkit.loop.context import TurnContext
from agentkit.loop.phase import Phase, is_terminal, validate_transition

log = get_logger(__name__)

PhaseHandler = Callable[[TurnContext, dict[str, Any]], Awaitable[Phase]]


class Loop:
    """One Loop per turn. Handlers and dependencies are injected at construction.

    ``deps`` is a free-form bag passed to every handler — used by handlers that
    need access to provider, registry, dispatcher, guards, stores. The
    AgentSession wires this up; tests pass minimal stubs.

    ``publish_phase_changed`` mirrors ``AgentConfig.events.publish_phase_changed``:
    set it False to keep :class:`PhaseChanged` off the event stream (the phase
    log on the context is still recorded, so nothing observability-side is lost
    for consumers that only want user-facing events on the wire).
    """

    def __init__(
        self,
        *,
        ctx: TurnContext,
        handlers: Mapping[Phase, PhaseHandler],
        deps: dict[str, Any] | None = None,
        starting_phase: Phase = Phase.INTENT_GATE,
        end_reason: TurnEndReason = TurnEndReason.COMPLETED,
        publish_phase_changed: bool = True,
    ) -> None:
        self._ctx = ctx
        self._handlers = handlers
        self._deps = deps or {}
        self._starting = starting_phase
        self._end_reason = end_reason
        self._publish_phase_changed = publish_phase_changed

    async def run(self) -> AsyncIterator[BaseEvent]:
        # Synthesise a user_message_id placeholder for TurnStarted events; if the
        # handlers populated ctx.history with the user message, use its id.
        user_msg_id = self._ctx.history[-1].id if self._ctx.history else new_id(MessageId)
        yield self._mk(TurnStarted, user_message_id=user_msg_id)

        turn_started = time.perf_counter()
        history_start = len(self._ctx.history)

        phase = self._starting
        while not is_terminal(phase):
            handler = self._handlers.get(phase)
            if handler is None:
                await self._record_turn_error(
                    phase,
                    kind="NoHandler",
                    message=f"no handler registered for phase {phase.value}",
                )
                if (evt := self._mk_phase_changed(phase, Phase.ERRORED, 0)) is not None:
                    yield evt
                phase = Phase.ERRORED
                break

            started = time.perf_counter()
            try:
                deps_for_handler = {**self._deps, "current_phase": phase}
                next_phase = await handler(self._ctx, deps_for_handler)
            except Exception as exc:
                duration = int((time.perf_counter() - started) * 1000)
                # log.exception carries the traceback; the metadata + event carry
                # the diagnosis to the persisted turn and to the consumer.
                log.exception(
                    "loop_handler_exception",
                    phase=phase.value,
                    error_type=type(exc).__name__,
                    session_id=str(self._ctx.session_id),
                    turn_id=str(self._ctx.turn_id),
                )
                await self._record_turn_error(
                    phase,
                    kind=type(exc).__name__,
                    message=str(exc),
                    log_it=False,
                    exc=exc,
                )
                if (evt := self._mk_phase_changed(phase, Phase.ERRORED, duration)) is not None:
                    yield evt
                phase = Phase.ERRORED
                break

            duration = int((time.perf_counter() - started) * 1000)
            try:
                validate_transition(phase, next_phase)
            except InvalidPhaseTransition as exc:
                await self._record_turn_error(
                    phase,
                    kind=type(exc).__name__,
                    message=str(exc),
                    exc=exc,
                )
                if (evt := self._mk_phase_changed(phase, Phase.ERRORED, duration)) is not None:
                    yield evt
                phase = Phase.ERRORED
                break

            if (evt := self._mk_phase_changed(phase, next_phase, duration)) is not None:
                yield evt
            self._ctx.phase_log.append((phase.value, next_phase.value, duration))
            phase = next_phase

        suspend_reason_str = self._ctx.metadata.get("suspend_reason")
        suspend_reason = TurnEndReason(suspend_reason_str) if suspend_reason_str else None
        if phase is Phase.TURN_ENDED:
            reason = suspend_reason if suspend_reason is not None else self._end_reason
        else:
            reason = TurnEndReason.ERROR
        yield self._mk(
            TurnEnded,
            reason=reason,
            metrics=self._turn_metrics(turn_started, history_start),
            summary=self._ctx.finalize_reason,
        )

    async def _record_turn_error(
        self,
        phase: Phase,
        *,
        kind: str,
        message: str,
        log_it: bool = True,
        exc: BaseException | None = None,
    ) -> None:
        """Make a turn failure observable: log, stash, emit.

        Called from every path that forces ERRORED. ``log_it=False`` is passed by
        the handler-exception path, which already logged with the traceback via
        ``log.exception``.

        ``exc`` is the exception that ended the turn, when there is one, and it
        is read for one thing: the :class:`ErrorCode` it asks for. This used to
        be hard-coded ``INTERNAL`` for every path through here, which made a
        deliberate refusal — the PII firewall declining to route a call, say —
        indistinguishable on the wire from an unhandled crash, and left a
        consumer no option but to pattern-match exception names out of the free
        text in ``message``. Paths with no exception (a phase with no handler)
        pass nothing and still get ``INTERNAL``, which is what they mean.
        """
        if log_it:
            log.error(
                "loop_turn_error",
                phase=phase.value,
                error_type=kind,
                error=message,
                session_id=str(self._ctx.session_id),
                turn_id=str(self._ctx.turn_id),
            )
        code = error_code_for(exc) if exc is not None else ErrorCode.INTERNAL
        # Plain strings only: this rides along in the checkpoint payload. The
        # code is stored too, so a turn read back from history discriminates the
        # failure the same way the live event did.
        self._ctx.metadata["turn_error"] = {
            "phase": phase.value,
            "type": kind,
            "message": message,
            "code": code.value,
        }
        queue = self._ctx.event_queue
        if queue is None:
            return
        evt = self._mk(
            Errored,
            code=code,
            message=f"{phase.value}: {kind}: {message}",
            recoverable=False,
        )
        await queue.put(evt)

    def _turn_metrics(self, turn_started: float, history_start: int) -> TurnMetrics:
        """Summarise the turn: tokens, cost, tool calls, iterations, duration.

        Token counts come from the ``usage`` provider events the StreamMux parked
        on ``ctx.metadata["usages"]``. ``cost_usd`` is what makes
        ``Provider.estimate_cost`` load-bearing rather than decorative — it is the
        only place the runtime prices a turn. A provider that cannot price the
        usage (unknown model, no pricing table) returns 0 and the field stays 0;
        a provider that raises must not take the turn down with it.
        """
        usages = [u for u in self._ctx.metadata.get("usages", []) if isinstance(u, Usage)]
        cost = sum((u.cost_usd for u in usages), Decimal("0"))
        provider = self._deps.get("provider")
        estimate = getattr(provider, "estimate_cost", None)
        if not cost and estimate is not None:
            try:
                cost = sum((Decimal(estimate(u)) for u in usages), Decimal("0"))
            except Exception:
                # Pricing is decoration; it must never take a turn down.
                log.warning(
                    "turn_cost_estimate_failed",
                    provider=getattr(provider, "name", type(provider).__name__),
                    session_id=str(self._ctx.session_id),
                )
                cost = Decimal("0")
        tool_calls = sum(
            1
            for msg in self._ctx.history[history_start:]
            for block in msg.content
            if isinstance(block, ToolUseBlock)
        )
        iterations = sum(1 for (_f, to, _d) in self._ctx.phase_log if to == Phase.STREAMING.value)
        return TurnMetrics(
            input_tokens=sum(u.input_tokens for u in usages),
            output_tokens=sum(u.output_tokens for u in usages),
            cached_input_tokens=sum(u.cached_input_tokens for u in usages),
            thinking_tokens=sum(u.thinking_tokens for u in usages),
            cost_usd=cost,
            duration_ms=int((time.perf_counter() - turn_started) * 1000),
            tool_calls=tool_calls,
            iterations=iterations,
        )

    def _mk(self, cls: type[BaseEvent], **payload: Any) -> BaseEvent:
        return cls(
            event_id=new_id(EventId),
            session_id=self._ctx.session_id,
            turn_id=self._ctx.turn_id,
            ts=self._ctx.clock.now(),
            sequence=self._ctx.next_sequence(),
            **payload,
        )

    def _mk_phase_changed(self, from_: Phase, to: Phase, duration_ms: int) -> PhaseChanged | None:
        """The PhaseChanged event for this transition, or None when suppressed."""
        if not self._publish_phase_changed:
            return None
        evt = self._mk(PhaseChanged, from_=from_, to=to, duration_ms=duration_ms)
        return cast("PhaseChanged", evt)
