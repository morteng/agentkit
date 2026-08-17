"""FinalizeValidator — gate the agent's "I'm done" claim via the envelope.

Structural validator: parses the finalize tool call's input dict into an
``Envelope``, walks the turn's tool-call history to build the call log,
runs ``validate_envelope``, and turns the result into a ``FinalizeVerdict``.

No regex. No user-message inspection. The model self-classifies via
``Envelope.intent_kind``; the validator checks structural consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import ValidationError

from agentkit._content import ToolResultBlock, ToolUseBlock
from agentkit.compaction import PRIOR_TOOL_CALLS_ANNOTATION
from agentkit.envelope import Envelope, ToolCallSummary, Violation
from agentkit.finalize_validator import (
    RiskFor,
    _is_default_write,  # pyright: ignore[reportPrivateUsage]
    _is_write,  # pyright: ignore[reportPrivateUsage]
    _summaries_since_last_user_turn,  # pyright: ignore[reportPrivateUsage]
    validate_envelope,
)

# ``_is_default_write`` is no longer called here — ``_is_write`` wraps it — but
# it stays reachable from this module, which is where callers and tests have
# always imported it from. Dropping the name would break them for a refactor
# they have no stake in. Named in ``__all__`` because that is the re-export
# both linters accept: a bare import reads as unused, and `X as X` reads as a
# useless alias.
__all__ = [
    "FinalizeValidator",
    "FinalizeVerdict",
    "StructuralFinalizeValidator",
    "_is_default_write",
]

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


def _ctx_to_summaries(ctx: TurnContext, risk_for: RiskFor | None = None) -> list[ToolCallSummary]:
    """Walk ctx.history to build a ToolCallSummary list for the validator."""
    use_names: dict[str, str] = {}
    result_errors: dict[str, bool] = {}
    compacted_names: set[str] = set()
    for msg in ctx.history:
        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                use_names[block.id] = block.name
            elif isinstance(block, ToolResultBlock):
                result_errors[block.tool_use_id] = block.is_error
        # A compaction summary message carries the names of successful writes
        # from the turn(s) it replaced (see PRIOR_TOOL_CALLS_ANNOTATION) — the
        # ToolUseBlock/ToolResultBlock pair itself is gone, but the write
        # genuinely happened this session, so Rule 1 must still be able to
        # credit it rather than reading a compacted turn as a fabricated one.
        carried = msg.metadata.annotations.get(PRIOR_TOOL_CALLS_ANNOTATION)
        if carried:
            compacted_names.update(carried)

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
                is_write=_is_write(name, risk_for),
            )
        )
    for name in compacted_names:
        bare = name.split(".", 1)[-1]
        if bare in ("finalize_response", "finalize"):
            continue
        summaries.append(
            ToolCallSummary(name=bare, is_error=False, is_write=_is_write(name, risk_for))
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

    def __init__(self, registry: Any | None = None) -> None:
        """``registry`` supplies each tool's declared risk level.

        Optional, and typed loosely, so every existing construction site and
        test double keeps working — without one the validator falls back to
        the name heuristic exactly as before. Passing it is what makes the
        write-mandate rules able to tell a read from a write for consumers
        whose tool names the heuristic cannot parse; see ``_is_write``.
        """
        self._registry = registry

    def _risk_for(self, name: str) -> str | None:
        """Registered risk for ``name``, trying the qualified form then bare.

        The call log records whichever form the model emitted; the registry is
        keyed by the form the tool was registered under. Trying both is the
        difference between resolving a spec and silently falling back to the
        heuristic this exists to replace.
        """
        registry = self._registry
        if registry is None:
            return None
        spec_for = getattr(registry, "spec_for", None)
        if spec_for is None:
            return None
        for candidate in (name, name.split(".", 1)[-1]):
            spec = spec_for(candidate)
            if spec is not None:
                return str(spec.risk)
        return None

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

        summaries = _ctx_to_summaries(ctx, self._risk_for)
        turn_summaries = _summaries_since_last_user_turn(ctx.history, self._risk_for)
        result = validate_envelope(envelope, summaries, turn_summaries=turn_summaries)
        if result.ok:
            return FinalizeVerdict(accept=True)
        return FinalizeVerdict(accept=False, feedback=_format_violations(result.violations))
