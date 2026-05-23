"""Goal lifecycle events.

Emitted by the runtime when a session's goal state changes. Consumers
translate these into UI envelopes (e.g. Pikkolo renders ``GoalSet`` as a
panel header, ``GoalEvaluated`` as a continuation narration line, etc.).

The ``state_id`` field carries the consumer's opaque correlation key
(e.g. a Pikkolo Task row id). agentkit does not read or write it; it
flows through unchanged so consumer event translators can route on it.
"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field

from agentkit.events.base import BaseEvent


class GoalSet(BaseEvent):
    """A goal was set on the session.

    Emitted by :meth:`AgentSession.set_goal` before the next turn runs.
    """

    type: Literal["goal_set"] = Field(default="goal_set")  # type: ignore[reportIncompatibleVariableOverride]
    condition: str
    set_at: datetime
    state_id: UUID | None = None


class GoalEvaluated(BaseEvent):
    """The continuation evaluator returned a decision for the active goal.

    Always paired with either a :class:`GoalAchieved` (when ``met=True``)
    or the next turn's :class:`TurnStarted` (when ``met=False``).
    """

    type: Literal["goal_evaluated"] = Field(default="goal_evaluated")  # type: ignore[reportIncompatibleVariableOverride]
    condition: str
    state_id: UUID | None = None
    met: bool
    reason: str
    turn_count: int
    iteration_count: int


class GoalAchieved(BaseEvent):
    """The goal's completion condition was met.

    Terminal for the goal lifecycle; the session clears its goal state
    before this event is yielded.
    """

    type: Literal["goal_achieved"] = Field(default="goal_achieved")  # type: ignore[reportIncompatibleVariableOverride]
    condition: str
    state_id: UUID | None = None
    reason: str
    turn_count: int
    iteration_count: int
    total_tokens: int
    duration_s: float


class GoalAbandoned(BaseEvent):
    """The goal stopped without being met.

    ``cause`` is the structured outcome:

    * ``cleared`` — the caller invoked :meth:`AgentSession.clear_goal`.
    * ``budget_exceeded`` — the evaluator returned this reason (consumer policy).
    * ``max_turns`` — the evaluator returned this reason (consumer policy).
    * ``max_iterations`` — reserved for v0.11.0 ``self_declared`` mode.
    """

    type: Literal["goal_abandoned"] = Field(default="goal_abandoned")  # type: ignore[reportIncompatibleVariableOverride]
    condition: str
    state_id: UUID | None = None
    cause: Literal["cleared", "budget_exceeded", "max_turns", "max_iterations"]
    turn_count: int
    iteration_count: int
