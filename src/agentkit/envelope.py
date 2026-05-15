"""Canonical Envelope schema for terminal `finalize_response`-style tools.

Consumers (e.g. Pikkolo) register a thin tool wrapper whose input dict
parses to ``Envelope``; the structural validator (see
``agentkit.finalize_validator``) inspects the parsed envelope plus the
turn's tool-call log to decide whether the agent's "I'm done" claim is
internally consistent.

Design intent: the model self-declares ``intent_kind`` per turn so the
runtime never inspects user message text to infer whether work was
expected. See ``docs/superpowers/specs/2026-05-10-envelope-intent-kind-design.md``
in the Pikkolo repo.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Action(BaseModel):
    """One write performed during the turn, as reported by the agent."""

    model_config = ConfigDict(frozen=True)

    tool: str = Field(..., min_length=1, description="Real tool name from this turn's call log.")
    target: str | None = Field(
        None, description="User-readable identifier (title, name) — not a UUID."
    )
    description: str = Field(..., min_length=1, description="What this specific action did.")


class PendingConfirmation(BaseModel):
    """Question awaiting the editor when status='partial' or 'blocked'."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(..., min_length=1)
    kind: Literal["confirm", "choose", "free_text"] = "confirm"


class Envelope(BaseModel):
    """The agent's structured "I'm done" statement.

    The agent self-classifies its turn via ``intent_kind`` and (when
    applicable) names ``expected_count``. The validator never re-derives
    these from user prose.
    """

    model_config = ConfigDict(frozen=True)

    status: Literal["done", "partial", "blocked"]
    intent_kind: Literal["action", "answer", "clarify"]
    summary: str | None = None
    actions_performed: list[Action] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    pending_confirmation: PendingConfirmation | None = None
    expected_count: int | None = Field(
        default=None,
        ge=1,
        description="When the user named a count (e.g. '3 articles'), the model "
        "fills this. 0 is invalid — use null for 'no count implied'.",
    )
    answer_evidence: Literal["tool_results", "context", "general_knowledge"] | None = Field(
        default=None,
        description=(
            "When intent_kind='answer', the model self-declares what kind of "
            "evidence the answer rests on. 'tool_results' = grounded in a read "
            "tool called this turn; 'context' = answer is in the system prompt "
            "(page state, brand voice, prior turn) — NOT the memory index summary; "
            "'general_knowledge' = training data (math, language, world facts). "
            "Required for intent_kind='answer' (enforced by validate_envelope), "
            "ignored otherwise."
        ),
    )
    proposed_autonomous_scope: dict[str, object] | None = None


class Violation(BaseModel):
    """One rule failure produced by ``validate_envelope``."""

    model_config = ConfigDict(frozen=True)

    rule: str
    detail: str


class ToolCallSummary(BaseModel):
    """Slim view of a tool call this turn, passed to the validator."""

    model_config = ConfigDict(frozen=True)

    name: str
    is_error: bool = False
    is_write: bool = True


class ValidationResult(BaseModel):
    """Aggregate result returned by ``validate_envelope``."""

    model_config = ConfigDict(frozen=True)

    ok: bool
    violations: list[Violation] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
