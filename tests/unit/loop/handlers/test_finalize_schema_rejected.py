"""A finalize call the schema rejects is not a turn that never finalized.

``ToolRegistry`` validates arguments before dispatch; ``finalize_handler`` sets
``ctx.finalize_called`` inside dispatch. So a finalize call missing a required
argument leaves that flag False — the same state as a model that never called
finalize at all — and the loop used to treat the two identically. They are
opposite situations:

* never called: tell the model it forgot, once (``max_missing_finalize_reprompts``).
* called and refused: tell the model *which argument* was wrong, and give it
  the finalize retry budget (``max_finalize_retries``).

Charging the second to the first ends the turn as ``NO_RESPONSE`` after a single
generic nudge, which a chat UI renders as an empty reply.

The other half is that the re-prompt could produce the rejection it was
correcting: it named ``intent_kind`` in all three of its branches and ``status``
in one, while the schema required both.
"""

from datetime import UTC, datetime
from typing import Any

import pytest

from agentkit._content import TextBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.events.lifecycle import TurnEndReason
from agentkit.guards.finalize import StructuralFinalizeValidator
from agentkit.loop.context import TurnContext
from agentkit.loop.handlers.finalize_check import handle_finalize_check
from agentkit.loop.phase import Phase
from agentkit.tools.builtin.finalize import FINALIZE_SPEC


def _user(text: str) -> Message:
    return Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=MessageRole.USER,
        content=[TextBlock(text=text)],
        created_at=datetime.now(UTC),
    )


def _injected_text(ctx: TurnContext) -> str:
    """The last correction appended to history, as the model would read it."""
    for message in reversed(ctx.history):
        if message.role is MessageRole.USER:
            return "".join(b.text for b in message.content if isinstance(b, TextBlock))
    raise AssertionError("no correction was injected")


def _ctx_with_rejected_finalize(message: str) -> TurnContext:
    """A turn where the registry refused to dispatch the model's finalize call."""
    ctx = TurnContext.empty()
    ctx.add_message(_user("turn the kitchen lights down"))
    ctx.finalize_called = False  # the handler that sets it never ran
    ctx.metadata["invalid_argument_calls"] = {"kit.finalize": message}
    return ctx


_REJECTION = (
    "invalid arguments for kit.finalize: missing required argument(s): status. "
    "The call was not executed — retry with arguments matching the tool's schema."
)


# ---- the re-prompt must not ask for a call the schema will refuse ----------


@pytest.mark.asyncio
async def test_every_branch_of_the_reprompt_names_every_required_argument():
    """Each bullet is a complete instruction, so each must name every required field.

    Asserting the field appears *somewhere* in the correction is the check that
    misses this: the old text named ``status`` once, in the clarify bullet, and
    a model following the "answer" or "action" bullet to the letter produced a
    call the registry refused. Split on the bullets and require each one to
    stand on its own.

    Against ``FINALIZE_SPEC``'s own ``required`` list, not a copy of it here: a
    hand-copied enumeration cannot notice the schema gaining a third field.
    """
    ctx = TurnContext.empty()
    ctx.add_message(_user("how many days to clear the backlog?"))
    await handle_finalize_check(ctx, {"finalize_validator": StructuralFinalizeValidator()})

    text = _injected_text(ctx)
    required = list((FINALIZE_SPEC.parameters or {}).get("required", []))
    assert required, "the finalize spec declares no required arguments"

    bullets = [b for b in text.split("•")[1:] if b.strip()]
    assert len(bullets) >= 2, "expected the correction to offer per-intent branches"
    for bullet in bullets:
        missing = [name for name in required if name not in bullet]
        assert not missing, f"this branch never mentions {missing}: {bullet.strip()!r}"


@pytest.mark.asyncio
async def test_missing_finalize_reprompt_reads_the_registered_spec():
    """A consumer's own finalize tool, with its own required list, is honoured.

    Pins the mechanism rather than the current wording: the text is derived
    from whatever finalize tool the registry offers, so it cannot go stale the
    way a literal list did.
    """

    class _Spec:
        def __init__(self) -> None:
            self.name = "house.finalize"
            self.parameters: dict[str, Any] = {
                "type": "object",
                "required": ["status", "intent_kind", "confidence"],
            }

    class _Registry:
        def list_specs(self) -> list[Any]:
            return [_Spec()]

    ctx = TurnContext.empty()
    ctx.add_message(_user("how many days to clear the backlog?"))
    await handle_finalize_check(
        ctx,
        {
            "finalize_validator": StructuralFinalizeValidator(),
            "registry": _Registry(),
        },
    )
    assert "confidence" in _injected_text(ctx)


# ---- a rejected call is routed to the retry budget, not the reprompt one ----


@pytest.mark.asyncio
async def test_schema_rejected_finalize_retries_after_the_reprompt_budget_is_gone():
    """The missing-finalize budget is spent; the model still gets a retry.

    Its call was refused, not absent, so it is charged to ``max_finalize_retries``.
    Before the fix this returned MEMORY_EXTRACT and ended the turn.
    """
    ctx = _ctx_with_rejected_finalize(_REJECTION)
    ctx.metadata["missing_finalize_reprompts"] = 1  # budget already spent
    deps = {
        "finalize_validator": StructuralFinalizeValidator(),
        "max_missing_finalize_reprompts": 1,
        "max_finalize_retries": 2,
    }
    assert await handle_finalize_check(ctx, deps) is Phase.CONTEXT_BUILD
    assert ctx.metadata.get("finalize_retries") == 1
    assert "finalize_missing" not in ctx.metadata


@pytest.mark.asyncio
async def test_schema_rejected_finalize_correction_names_the_refused_argument():
    """The model is told which argument was wrong, not just that it should finalize."""
    ctx = _ctx_with_rejected_finalize(_REJECTION)
    deps = {
        "finalize_validator": StructuralFinalizeValidator(),
        "max_finalize_retries": 2,
    }
    await handle_finalize_check(ctx, deps)
    text = _injected_text(ctx)
    assert "status" in text
    assert "was rejected" in text
    assert "You ended your turn without calling finalize_response" not in text


@pytest.mark.asyncio
async def test_schema_rejected_finalize_out_of_retries_is_not_no_response():
    """Out of retries it ends FINALIZE_REJECTED: the model answered, and did call finalize.

    NO_RESPONSE reaches a chat UI as "I stopped without writing a reply", which
    is a different and untrue account of what happened.
    """
    ctx = _ctx_with_rejected_finalize(_REJECTION)
    ctx.metadata["finalize_retries"] = 2
    deps = {
        "finalize_validator": StructuralFinalizeValidator(),
        "max_finalize_retries": 2,
    }
    assert await handle_finalize_check(ctx, deps) is Phase.MEMORY_EXTRACT
    assert ctx.metadata["suspend_reason"] == TurnEndReason.FINALIZE_REJECTED.value


@pytest.mark.asyncio
async def test_the_rejection_marker_does_not_re_route_the_next_iteration():
    """Cleared when read, so one refused call cannot spend the whole retry budget."""
    ctx = _ctx_with_rejected_finalize(_REJECTION)
    deps = {
        "finalize_validator": StructuralFinalizeValidator(),
        "max_finalize_retries": 2,
    }
    await handle_finalize_check(ctx, deps)
    assert not ctx.metadata["invalid_argument_calls"]

    # Second pass: nothing was refused this time, so this is a genuine
    # missing finalize and takes the re-prompt path.
    await handle_finalize_check(ctx, deps)
    assert ctx.metadata.get("missing_finalize_reprompts") == 1


@pytest.mark.asyncio
async def test_a_rejected_call_to_some_other_tool_is_still_a_missing_finalize():
    """The control: only a refused *finalize* call changes the routing."""
    ctx = TurnContext.empty()
    ctx.add_message(_user("what is on tonight?"))
    ctx.metadata["invalid_argument_calls"] = {"media.search": "invalid arguments"}
    deps = {"finalize_validator": StructuralFinalizeValidator()}
    assert await handle_finalize_check(ctx, deps) is Phase.CONTEXT_BUILD
    assert ctx.metadata.get("missing_finalize_reprompts") == 1
    assert "finalize_retries" not in ctx.metadata
