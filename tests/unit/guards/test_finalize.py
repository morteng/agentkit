"""Tests for StructuralFinalizeValidator (replaces RuleBasedFinalizeValidator)."""

from datetime import UTC, datetime

import pytest

from agentkit._content import TextBlock, ToolUseBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.guards.finalize import (
    StructuralFinalizeValidator,
)
from agentkit.loop.context import TurnContext
from agentkit.tools.spec import ToolCall


def _make_finalize_call(args: dict) -> ToolCall:
    return ToolCall(id="test-call-1", name="finalize_response", arguments=args)


def _msg(role: MessageRole, content: list) -> Message:
    return Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=role,
        content=content,
        created_at=datetime.now(UTC),
    )


def _make_ctx(
    history: list[Message] | None = None,
    *,
    tool_use_blocks: list[ToolUseBlock] | None = None,
) -> TurnContext:
    """Build a minimal TurnContext. The validator only reads ctx.history."""
    ctx = TurnContext.empty()
    for msg in history or []:
        ctx.add_message(msg)
    if tool_use_blocks:
        ctx.add_message(_msg(MessageRole.ASSISTANT, list(tool_use_blocks)))
    return ctx


@pytest.mark.asyncio
async def test_validator_accepts_valid_action_envelope():
    validator = StructuralFinalizeValidator()
    args = {
        "status": "done",
        "intent_kind": "action",
        "actions_performed": [
            {"tool": "patch_content", "target": "X", "description": "ok"},
        ],
    }
    ctx = _make_ctx(
        tool_use_blocks=[
            ToolUseBlock(id="u1", name="patch_content", arguments={}),
        ],
    )
    verdict = await validator.validate(_make_finalize_call(args), ctx)
    assert verdict.accept is True


@pytest.mark.asyncio
async def test_validator_rejects_empty_on_done_action():
    validator = StructuralFinalizeValidator()
    args = {
        "status": "done",
        "intent_kind": "action",
        "actions_performed": [],
    }
    ctx = _make_ctx()
    verdict = await validator.validate(_make_finalize_call(args), ctx)
    assert verdict.accept is False
    assert verdict.feedback is not None
    assert "empty_on_done" in verdict.feedback or "actions_performed" in verdict.feedback


@pytest.mark.asyncio
async def test_validator_accepts_empty_on_done_answer():
    validator = StructuralFinalizeValidator()
    args = {
        "status": "done",
        "intent_kind": "answer",
        "actions_performed": [],
        "answer_evidence": "general_knowledge",
    }
    verdict = await validator.validate(_make_finalize_call(args), _make_ctx())
    assert verdict.accept is True


@pytest.mark.asyncio
async def test_validator_rejects_unparseable_envelope():
    validator = StructuralFinalizeValidator()
    args = {"status": "done"}  # missing intent_kind
    verdict = await validator.validate(_make_finalize_call(args), _make_ctx())
    assert verdict.accept is False
    assert verdict.feedback is not None
    assert "intent_kind" in verdict.feedback


@pytest.mark.asyncio
async def test_validator_does_not_inspect_user_messages():
    """Regression: the validator MUST NOT use user message text for any decision."""
    validator = StructuralFinalizeValidator()
    args = {
        "status": "done",
        "intent_kind": "answer",
        "actions_performed": [],
        "answer_evidence": "general_knowledge",
    }
    user_msg = _msg(
        MessageRole.USER,
        [TextBlock(text="please publish all the articles right now")],
    )
    ctx = _make_ctx([user_msg])
    verdict = await validator.validate(_make_finalize_call(args), ctx)
    # intent_kind=answer + done + no writes is structurally fine; the
    # legacy regex would have rejected this because "publish all" looks
    # like an action request. Structural validator must accept.
    assert verdict.accept is True


def test_no_action_verbs_regex_remains():
    """Regression: the _ACTION_VERBS regex must be deleted."""
    import agentkit.guards.finalize as mod

    assert not hasattr(mod, "_ACTION_VERBS")
    assert not hasattr(mod, "_is_action_request")
    assert not hasattr(mod, "_latest_user_message")
    assert not hasattr(mod, "_has_non_kit_tool_call")
    assert not hasattr(mod, "RuleBasedFinalizeValidator")


def test_recall_memories_classifies_as_read():
    """recall_memories is a read tool, not a write. See Task A2 in plan
    2026-05-15-answer-evidence-envelope.md — this was a latent
    misclassification fixed alongside the answer_evidence work."""
    from agentkit.guards.finalize import _is_default_write

    assert _is_default_write("recall_memories") is False
    assert _is_default_write("acme.recall_memories") is False
    # Sanity: actual writes still classify as writes.
    assert _is_default_write("patch_content") is True
    assert _is_default_write("create_content") is True


@pytest.mark.asyncio
async def test_structural_validator_rejects_tool_results_with_no_turn_reads():
    """End-to-end: model declares answer_evidence='tool_results' but the
    current turn has no successful read tool call. The validator pulls
    turn-scoped summaries from ctx.history and rejects."""
    from agentkit._content import TextBlock, ToolResultBlock, ToolUseBlock
    from agentkit._messages import MessageRole
    from agentkit.guards.finalize import StructuralFinalizeValidator
    from agentkit.loop.context import TurnContext
    from agentkit.tools.spec import ToolCall

    history = [
        # Prior turn read (should NOT count for Rule 9)
        _msg(MessageRole.USER, [TextBlock(text="prior question")]),
        _msg(
            MessageRole.ASSISTANT,
            [ToolUseBlock(id="old1", name="search", arguments={})],
        ),
        _msg(
            MessageRole.USER,
            [
                ToolResultBlock(
                    tool_use_id="old1",
                    content=[TextBlock(text="ok")],
                    is_error=False,
                )
            ],
        ),
        # Current turn: no read tools.
        _msg(MessageRole.USER, [TextBlock(text="what do you know about me?")]),
    ]
    ctx = TurnContext.empty()
    for msg in history:
        ctx.add_message(msg)
    call = ToolCall(
        id="f1",
        name="finalize_response",
        arguments={
            "status": "done",
            "intent_kind": "answer",
            "answer_evidence": "tool_results",
        },
    )
    verdict = await StructuralFinalizeValidator().validate(call, ctx)
    assert verdict.accept is False
    assert "answer_evidence_consistent" in (verdict.feedback or "")


@pytest.mark.asyncio
async def test_structural_validator_accepts_general_knowledge_with_no_reads():
    from agentkit._content import TextBlock
    from agentkit._messages import MessageRole
    from agentkit.guards.finalize import StructuralFinalizeValidator
    from agentkit.loop.context import TurnContext
    from agentkit.tools.spec import ToolCall

    history = [
        _msg(MessageRole.USER, [TextBlock(text="hi")]),
    ]
    ctx = TurnContext.empty()
    for msg in history:
        ctx.add_message(msg)
    call = ToolCall(
        id="f1",
        name="finalize_response",
        arguments={
            "status": "done",
            "intent_kind": "answer",
            "answer_evidence": "general_knowledge",
        },
    )
    verdict = await StructuralFinalizeValidator().validate(call, ctx)
    assert verdict.accept is True


@pytest.mark.parametrize(
    "tool_name",
    ["read_blocks", "fetch_page", "query_index", "get_profile", "list_rooms", "search_notes"],
)
@pytest.mark.asyncio
async def test_structural_validator_accepts_a_truthful_tool_results_claim_after_a_read(
    tool_name: str,
):
    """A turn whose only tool call is a plainly-named read must be allowed to
    say so.

    Rule 9 classifies a tool as a write unless its name matches a read prefix,
    and ``read_``/``fetch_``/``query_`` were missing from that list — so a turn
    that called ``read_blocks``, got a result, and truthfully reported
    ``answer_evidence="tool_results"`` was rejected. The consequences are worse
    than a spurious error: the model either burns its finalize retries on an
    envelope that cannot pass, or "corrects" itself to
    ``answer_evidence="context"``, which is false. A validator that punishes an
    accurate claim teaches the model to make inaccurate ones.

    Driven through ``StructuralFinalizeValidator.validate`` rather than
    ``validate_envelope``, and asserting on the verdict: the rule-9 tests
    alongside this one hand ``ToolCallSummary(is_write=...)`` in directly, so
    they never exercise the name-based classification where the bug lived. The
    prefixes already present are parametrized here too, so a future edit to the
    list cannot quietly drop one.
    """
    from agentkit._content import TextBlock, ToolResultBlock, ToolUseBlock
    from agentkit._messages import MessageRole
    from agentkit.guards.finalize import StructuralFinalizeValidator
    from agentkit.loop.context import TurnContext
    from agentkit.tools.spec import ToolCall

    history = [
        _msg(MessageRole.USER, [TextBlock(text="what is in the collab room?")]),
        _msg(MessageRole.ASSISTANT, [ToolUseBlock(id="r1", name=tool_name, arguments={})]),
        _msg(
            MessageRole.USER,
            [ToolResultBlock(tool_use_id="r1", content=[TextBlock(text="…")], is_error=False)],
        ),
    ]
    ctx = TurnContext.empty()
    for msg in history:
        ctx.add_message(msg)
    call = ToolCall(
        id="f1",
        name="finalize_response",
        arguments={
            "status": "done",
            "intent_kind": "answer",
            "answer_evidence": "tool_results",
        },
    )

    verdict = await StructuralFinalizeValidator().validate(call, ctx)

    assert verdict.accept is True, (
        f"{tool_name} is a read; a truthful tool_results claim must be accepted. "
        f"Got: {verdict.feedback}"
    )


@pytest.mark.asyncio
async def test_rule9_still_rejects_tool_results_when_the_turn_only_wrote():
    """The half of rule 9 that survives widening the read-prefix list.

    Worth pinning next to the test above: the point of adding read verbs is to
    stop rejecting honest claims, not to stop checking. A turn whose only call
    was a write still cannot claim its answer came from tool results.
    """
    from agentkit._content import TextBlock, ToolResultBlock, ToolUseBlock
    from agentkit._messages import MessageRole
    from agentkit.guards.finalize import StructuralFinalizeValidator
    from agentkit.loop.context import TurnContext
    from agentkit.tools.spec import ToolCall

    history = [
        _msg(MessageRole.USER, [TextBlock(text="delete the draft")]),
        _msg(MessageRole.ASSISTANT, [ToolUseBlock(id="w1", name="delete_block", arguments={})]),
        _msg(
            MessageRole.USER,
            [ToolResultBlock(tool_use_id="w1", content=[TextBlock(text="ok")], is_error=False)],
        ),
    ]
    ctx = TurnContext.empty()
    for msg in history:
        ctx.add_message(msg)
    call = ToolCall(
        id="f1",
        name="finalize_response",
        arguments={
            "status": "done",
            "intent_kind": "answer",
            "answer_evidence": "tool_results",
        },
    )

    verdict = await StructuralFinalizeValidator().validate(call, ctx)

    assert verdict.accept is False
    assert "answer_evidence_consistent" in (verdict.feedback or "")


@pytest.mark.asyncio
async def test_rule9_still_rejects_tool_results_when_the_only_read_errored():
    """The other surviving half: a read that failed is not evidence."""
    from agentkit._content import TextBlock, ToolResultBlock, ToolUseBlock
    from agentkit._messages import MessageRole
    from agentkit.guards.finalize import StructuralFinalizeValidator
    from agentkit.loop.context import TurnContext
    from agentkit.tools.spec import ToolCall

    history = [
        _msg(MessageRole.USER, [TextBlock(text="what is in the room?")]),
        _msg(MessageRole.ASSISTANT, [ToolUseBlock(id="r1", name="read_blocks", arguments={})]),
        _msg(
            MessageRole.USER,
            [ToolResultBlock(tool_use_id="r1", content=[TextBlock(text="boom")], is_error=True)],
        ),
    ]
    ctx = TurnContext.empty()
    for msg in history:
        ctx.add_message(msg)
    call = ToolCall(
        id="f1",
        name="finalize_response",
        arguments={
            "status": "done",
            "intent_kind": "answer",
            "answer_evidence": "tool_results",
        },
    )

    verdict = await StructuralFinalizeValidator().validate(call, ctx)

    assert verdict.accept is False
    assert "answer_evidence_consistent" in (verdict.feedback or "")
