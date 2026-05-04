"""Loop — drive a single turn through the phase machine.

Handlers live in ``loop/handlers/*``. The orchestrator picks the right handler
for the current phase, awaits it, validates the transition, emits PhaseChanged,
and continues. Terminal phases (TURN_ENDED, ERRORED) end the run.
"""

import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, cast

from agentkit._ids import EventId, MessageId, new_id
from agentkit.errors import InvalidPhaseTransition
from agentkit.events import PhaseChanged, TurnEnded, TurnEndReason, TurnMetrics, TurnStarted
from agentkit.events.base import BaseEvent
from agentkit.loop.context import TurnContext
from agentkit.loop.phase import Phase, is_terminal, validate_transition

PhaseHandler = Callable[[TurnContext, dict[str, Any]], Awaitable[Phase]]


class Loop:
    """One Loop per turn. Handlers and dependencies are injected at construction.

    ``deps`` is a free-form bag passed to every handler — used by handlers that
    need access to provider, registry, dispatcher, guards, stores. The
    AgentSession wires this up; tests pass minimal stubs.
    """

    def __init__(
        self,
        *,
        ctx: TurnContext,
        handlers: Mapping[Phase, PhaseHandler],
        deps: dict[str, Any] | None = None,
        starting_phase: Phase = Phase.INTENT_GATE,
        end_reason: TurnEndReason = TurnEndReason.COMPLETED,
    ) -> None:
        self._ctx = ctx
        self._handlers = handlers
        self._deps = deps or {}
        self._starting = starting_phase
        self._end_reason = end_reason
        self._sequence = 0

    async def run(self) -> AsyncIterator[BaseEvent]:
        # Synthesise a user_message_id placeholder for TurnStarted events; if the
        # handlers populated ctx.history with the user message, use its id.
        user_msg_id = self._ctx.history[-1].id if self._ctx.history else new_id(MessageId)
        yield self._mk(TurnStarted, user_message_id=user_msg_id)

        phase = self._starting
        while not is_terminal(phase):
            handler = self._handlers.get(phase)
            if handler is None:
                yield self._mk_phase_changed(phase, Phase.ERRORED, 0)
                phase = Phase.ERRORED
                break

            started = time.perf_counter()
            try:
                deps_for_handler = {**self._deps, "current_phase": phase}
                next_phase = await handler(self._ctx, deps_for_handler)
            except Exception:
                duration = int((time.perf_counter() - started) * 1000)
                yield self._mk_phase_changed(phase, Phase.ERRORED, duration)
                phase = Phase.ERRORED
                break

            duration = int((time.perf_counter() - started) * 1000)
            try:
                validate_transition(phase, next_phase)
            except InvalidPhaseTransition:
                yield self._mk_phase_changed(phase, Phase.ERRORED, duration)
                phase = Phase.ERRORED
                break

            yield self._mk_phase_changed(phase, next_phase, duration)
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
            metrics=TurnMetrics(),
        )

    def _mk(self, cls: type[BaseEvent], **payload: Any) -> BaseEvent:
        seq = self._sequence
        self._sequence += 1
        return cls(
            event_id=new_id(EventId),
            session_id=self._ctx.session_id,
            turn_id=self._ctx.turn_id,
            ts=self._ctx.clock.now(),
            sequence=seq,
            **payload,
        )

    def _mk_phase_changed(self, from_: Phase, to: Phase, duration_ms: int) -> PhaseChanged:
        evt = self._mk(PhaseChanged, from_=from_, to=to, duration_ms=duration_ms)
        return cast("PhaseChanged", evt)
