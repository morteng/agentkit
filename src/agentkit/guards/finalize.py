"""FinalizeValidator — gate the agent's "I'm done" claim via the envelope.

Structural validator: parses the finalize tool call's input dict into an
``Envelope``, walks the turn's tool-call history to build the call log,
runs ``validate_envelope``, and turns the result into a ``FinalizeVerdict``.

No regex. No user-message inspection. The model self-classifies via
``Envelope.intent_kind``; the validator checks structural consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import ValidationError

from agentkit._content import ToolResultBlock, ToolUseBlock
from agentkit.envelope import Envelope, ToolCallSummary, Violation
from agentkit.finalize_validator import (
    _is_default_write,  # pyright: ignore[reportPrivateUsage]
    _summaries_since_last_user_turn,  # pyright: ignore[reportPrivateUsage]
    validate_envelope,
)

if TYPE_CHECKING:
    from agentkit.loop.context import TurnContext
    from agentkit.tools.spec import ToolCall


@dataclass(frozen=True)
class FinalizeVerdict:
    accept: bool
    feedback: str | None = None


@runtime_checkable
class FinalizeValidator(Protocol):
    async def validate(self, finalize_call: ToolCall, ctx: TurnContext) -> FinalizeVerdict: ...


def _ctx_to_summaries(ctx: TurnContext) -> list[ToolCallSummary]:
    """Walk ctx.history to build a ToolCallSummary list for the validator."""
    use_names: dict[str, str] = {}
    result_errors: dict[str, bool] = {}
    for msg in ctx.history:
        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                use_names[block.id] = block.name
            elif isinstance(block, ToolResultBlock):
                result_errors[block.tool_use_id] = block.is_error

    summaries: list[ToolCallSummary] = []
    for use_id, name in use_names.items():
        # Skip the finalize_response call itself — it's not "work".
        bare = name.split(".", 1)[-1]
        if bare in ("finalize_response", "finalize"):
            continue
        summaries.append(
            ToolCallSummary(
                name=bare,
                is_error=result_errors.get(use_id, False),
                is_write=_is_default_write(name),
            )
        )
    return summaries


def _format_violations(violations: list[Violation]) -> str:
    if not violations:
        return ""
    lines = [f"- {v.rule}: {v.detail}" for v in violations]
    return "Envelope failed structural validation:\n" + "\n".join(lines)


class StructuralFinalizeValidator:
    """Default structural validator. Parses the envelope, runs validate_envelope.

    Rejects when the envelope fails Pydantic parsing OR when the validator
    returns any blocking violation. The feedback string lists the rule
    names so the agent can self-correct on retry.
    """

    async def validate(self, finalize_call: ToolCall, ctx: TurnContext) -> FinalizeVerdict:
        try:
            envelope = Envelope.model_validate(finalize_call.arguments)
        except ValidationError as e:
            missing = [  # pyright: ignore[reportUnknownMemberType]
                str(err.get("loc", ["?"])[0]) for err in e.errors()
            ]
            return FinalizeVerdict(
                accept=False,
                feedback=(
                    "Envelope failed schema validation. "
                    f"Required field issues: {', '.join(missing) or 'unknown'}. "
                    "intent_kind must be one of: action, answer, clarify."
                ),
            )

        summaries = _ctx_to_summaries(ctx)
        turn_summaries = _summaries_since_last_user_turn(ctx.history)
        result = validate_envelope(envelope, summaries, turn_summaries=turn_summaries)
        if result.ok:
            return FinalizeVerdict(accept=True)
        return FinalizeVerdict(accept=False, feedback=_format_violations(result.violations))
