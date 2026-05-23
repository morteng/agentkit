"""Continuation evaluator extension point.

A ``ContinuationEvaluator`` is a consumer-supplied hook that decides — after
the loop reaches a terminal envelope — whether the session's current goal is
met. If not, the runtime synthesises a ``[goal continuation: <reason>]`` system
message and starts another turn, repeating until the evaluator says ``met=True``
or the consumer clears the goal.

The hook is purposely state-light:

* The evaluator is stateless (Protocol). It sees only the request payload
  and returns a structured decision.
* ``GoalState`` lives on :class:`agentkit.session.AgentSession`. agentkit
  treats ``state_id`` as opaque — consumers persisting goal state externally
  (e.g. a Pikkolo Task row) carry their own projection.
* Cost-cap / quiet-hours policy lives in the consumer's evaluator. agentkit
  doesn't know about money.

v0.10.0 ships ``every_turn`` dispatch only. ``self_declared`` is reserved in
the literal type for forward-compatibility (v0.11.0 will add the dispatch
path and the ``self_declare_done`` envelope intent_kind that drives it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from uuid import UUID

    from agentkit._messages import Message


# When the evaluator fires.
# - every_turn: run the evaluator after every terminal envelope (Claude Code style).
#   Right for chat-shaped /goal where the editor is watching and progress narration matters.
# - self_declared: run the evaluator only when the agent emits a self-declared "done"
#   envelope. Right for background work like editorial Cards where the evaluator is
#   expensive and the agent knows when it has produced the artifact under test.
#   *Reserved in v0.10.0; dispatch lands in v0.11.0.*
TriggerMode = Literal["every_turn", "self_declared"]


@dataclass(frozen=True, slots=True)
class ContinuationRequest:
    """Payload handed to the evaluator on each evaluation."""

    condition: str
    transcript: Sequence[Message]
    turn_count: int  # total turns under this goal — broad budget control
    iteration_count: int  # rejected self-declared-done cycles; v0.10.0 always 0
    token_spend: int
    set_at: datetime
    state_id: UUID | None  # opaque consumer-side key; None for ephemeral chat goals


@dataclass(frozen=True, slots=True)
class ContinuationDecision:
    """Structured return value from the evaluator."""

    met: bool
    reason: str


@runtime_checkable
class ContinuationEvaluator(Protocol):
    """Consumer-supplied async callable that judges goal completion.

    Implementations expose a ``trigger`` class- or instance-attribute that
    declares when the runtime should call them.
    """

    trigger: TriggerMode

    async def __call__(self, request: ContinuationRequest) -> ContinuationDecision: ...


@dataclass(slots=True)
class GoalState:
    """In-memory state for the active goal on an :class:`AgentSession`.

    agentkit owns this object's lifetime within a process. External
    persistence (e.g. a Pikkolo Task row) is the consumer's responsibility;
    ``state_id`` is the opaque key the consumer uses to correlate.

    Session resume restores ``condition``, ``set_at``, and ``state_id``;
    ``turn_count``, ``iteration_count``, ``token_spend``, and ``last_reason``
    reset to zero/None. Consumers that want to carry counters across resumes
    can rehydrate by passing ``resume_from=GoalState(...)`` to
    :meth:`AgentSession.set_goal`.
    """

    condition: str
    set_at: datetime
    turn_count: int = 0
    iteration_count: int = 0
    token_spend: int = 0
    last_reason: str | None = None
    state_id: UUID | None = None


__all__ = [
    "ContinuationDecision",
    "ContinuationEvaluator",
    "ContinuationRequest",
    "GoalState",
    "TriggerMode",
]
