"""Finalize-check handler — gate the agent's "I'm done" claim via the envelope.

Three outcomes, all reached only when a ``finalize_validator`` is configured
(consumers without one don't enforce the finalize contract — see below):

* finalize_response was called and the validator accepts -> MEMORY_EXTRACT.
* finalize_response was called but the validator rejects -> CONTEXT_BUILD,
  bounded by ``max_finalize_retries``.
* finalize_response was NOT called -> CONTEXT_BUILD with a re-prompt asking
  the model to finalize, bounded by ``max_missing_finalize_reprompts``. Once
  that budget is spent the turn ends as ``TurnEndReason.NO_RESPONSE``. A turn
  that simply stops mid-thought — often by asking the user a question — would
  otherwise settle with no envelope at all; the re-prompt gives the model an
  explicit chance to classify it (e.g. ``intent_kind="clarify"``).

  This used to end as COMPLETED, on the reasoning that the consumer would
  synthesize a terminal envelope from the tool log. Treat that as a warning
  about writing a contract into a docstring: no consumer implemented it, and
  because COMPLETED is also what a successful turn emits, nothing anywhere
  could tell the two apart. A downstream UI rendered a stalled turn as a
  finished one for as long as the code said this.

When no validator is configured the handler is a pass-through: the consumer
has opted out of the finalize contract entirely, so a missing or unchecked
finalize is accepted as the conversation naturally ending.

Corrections reach the model as an appended user-role message — see
:func:`_inject_correction`. The MessageBuilder reads ``ctx.history``
verbatim, so a correction only stashed in ``ctx.metadata`` never reaches the
provider; appending a real message is the delivery path. Each injected
message is tagged ``metadata.annotations[INJECTED_CORRECTION_ANNOTATION]`` so
turn-boundary walkers (e.g. ``_summaries_since_last_user_turn``) can tell it
apart from a genuine human prompt and not mistake it for a new turn.
"""

from typing import TYPE_CHECKING, Any

from agentkit._content import TextBlock
from agentkit._ids import MessageId, new_id
from agentkit._logging import get_logger
from agentkit._messages import (
    INJECTED_CORRECTION_ANNOTATION,
    Message,
    MessageMetadata,
    MessageRole,
)
from agentkit.events.lifecycle import TurnEndReason
from agentkit.loop.context import TurnContext
from agentkit.loop.phase import Phase
from agentkit.tools.spec import ToolCall

if TYPE_CHECKING:
    from agentkit.guards.finalize import FinalizeValidator

log = get_logger(__name__)


_MISSING_FINALIZE_REPROMPT = (
    "You ended your turn without calling finalize_response. Every turn MUST "
    "end with exactly one finalize_response call so the system can record "
    "the outcome — including turns where you stopped to ask the user a "
    "question.\n\n"
    "Call finalize_response now. Do not repeat your previous message:\n"
    '  • If you completed work that wrote data, use intent_kind="action" '
    "and list every write in actions_performed.\n"
    "  • If you answered a question and made no writes, use "
    'intent_kind="answer".\n'
    "  • If your message asked the user a question, offered them a choice, "
    "or needs their decision before you can continue, use "
    'intent_kind="clarify" with status="blocked" and put the question in '
    "pending_confirmation."
)


def _inject_correction(ctx: TurnContext, text: str) -> None:
    """Append a user-role correction message so the model sees it on retry.

    Tagged with ``annotations[INJECTED_CORRECTION_ANNOTATION]`` so it is not
    mistaken for a fresh human prompt by code that infers the turn boundary
    from the most recent USER message.
    """
    ctx.add_message(
        Message(
            id=new_id(MessageId),
            session_id=ctx.session_id,
            role=MessageRole.USER,
            content=[TextBlock(text=text)],
            metadata=MessageMetadata(annotations={INJECTED_CORRECTION_ANNOTATION: True}),
            created_at=ctx.clock.now(),
        )
    )


async def handle_finalize_check(ctx: TurnContext, deps: dict[str, Any]) -> Phase:
    validator: FinalizeValidator | None = deps.get("finalize_validator")

    # No validator configured -> consumer has opted out of the finalize
    # contract; accept whatever happened and end the turn.
    if validator is None:
        return Phase.MEMORY_EXTRACT

    if not ctx.finalize_called:
        # The model ended the turn without finalizing. Re-prompt it once so
        # it emits a real envelope (it may need intent_kind="clarify").
        reprompts = ctx.metadata.get("missing_finalize_reprompts", 0)
        max_reprompts = deps.get("max_missing_finalize_reprompts", 1)
        if reprompts >= max_reprompts:
            # Budget spent — let the turn end. A consumer MAY synthesize a
            # terminal envelope from the tool log, but most will not, so the
            # turn end has to be self-describing: mark it NO_RESPONSE rather
            # than letting it settle as COMPLETED, which asserts the model
            # finished. Previously the two were identical on the wire and a UI
            # had no way to distinguish "here is your answer" from "the model
            # stopped mid-thought" — see TurnEndReason.NO_RESPONSE.
            #
            # setdefault, not assignment: an existing suspend_reason
            # (AWAITING_APPROVAL, MAX_ITERATIONS) explains more about why the
            # turn is ending than the absent envelope does, and has priority.
            ctx.metadata["finalize_missing"] = True
            ctx.metadata.setdefault("suspend_reason", TurnEndReason.NO_RESPONSE.value)
            log.warning(
                "finalize_missing_budget_spent",
                session_id=str(ctx.session_id),
                turn_id=str(ctx.turn_id),
                reprompts=reprompts,
                max_reprompts=max_reprompts,
            )
            return Phase.MEMORY_EXTRACT
        ctx.metadata["missing_finalize_reprompts"] = reprompts + 1
        if deps.get("force_finalize_on_missing_reprompt"):
            # Constrain the re-prompt turn to the finalize tool so the model
            # emits the envelope immediately instead of spending another
            # free-form turn. The streaming handler consumes this flag.
            ctx.metadata["force_finalize_tool_choice"] = True
        _inject_correction(ctx, _MISSING_FINALIZE_REPROMPT)
        return Phase.CONTEXT_BUILD

    finalize_call = ToolCall(
        id="finalize",
        name="kit.finalize",
        arguments=ctx.finalize_args
        if ctx.finalize_args is not None
        else {"reason": ctx.finalize_reason or ""},
    )
    verdict = await validator.validate(finalize_call, ctx)
    if verdict.accept:
        return Phase.MEMORY_EXTRACT

    retries = ctx.metadata.get("finalize_retries", 0)
    max_retries = deps.get("max_finalize_retries", 2)
    if retries >= max_retries:
        # Deliberately NOT marked NO_RESPONSE. The model did answer and did
        # finalize; only the envelope failed validation, so the user has
        # content in front of them and "no response" would be a lie. The turn
        # is still ending in a degraded state that nothing else records, hence
        # the warning — this path was previously invisible in the logs.
        ctx.metadata["finalize_exhausted"] = True
        log.warning(
            "finalize_validation_exhausted",
            session_id=str(ctx.session_id),
            turn_id=str(ctx.turn_id),
            retries=retries,
            max_retries=max_retries,
            last_correction=ctx.metadata.get("finalize_correction"),
        )
        return Phase.MEMORY_EXTRACT
    ctx.metadata["finalize_retries"] = retries + 1
    correction = verdict.feedback or "Reconsider before finalizing."
    ctx.metadata["finalize_correction"] = correction
    _inject_correction(
        ctx,
        "Your finalize_response call was rejected:\n\n"
        f"{correction}\n\n"
        "Fix the problem and call finalize_response again.",
    )
    ctx.finalize_called = False
    ctx.finalize_reason = None
    return Phase.CONTEXT_BUILD
