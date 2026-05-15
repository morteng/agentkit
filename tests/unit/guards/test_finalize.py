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
    assert _is_default_write("pikkolo.recall_memories") is False
    # Sanity: actual writes still classify as writes.
    assert _is_default_write("patch_content") is True
    assert _is_default_write("create_content") is True


import pytest


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
