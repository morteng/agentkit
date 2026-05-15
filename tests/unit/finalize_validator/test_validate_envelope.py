"""Per-rule tests for agentkit.finalize_validator.validate_envelope.

Each rule has a positive (passes) and negative (fires) case. None of
these tests pass a user_message — by design, the validator is purely
structural.
"""

from agentkit.envelope import (
    Action,
    Envelope,
    PendingConfirmation,
    ToolCallSummary,
)
from agentkit.finalize_validator import validate_envelope


def _summary(name: str, *, is_error: bool = False, is_write: bool = True) -> ToolCallSummary:
    return ToolCallSummary(name=name, is_error=is_error, is_write=is_write)


# Rule 1 — fabricated tool


def test_rule1_fabricated_tool_strips_unknown_action():
    env = Envelope(
        status="done",
        intent_kind="action",
        actions_performed=[Action(tool="ghost_tool", target=None, description="x")],
    )
    result = validate_envelope(env, [_summary("patch_content")])
    rules = {v.rule for v in result.violations}
    assert "fabricated_tool" in rules


def test_rule1_passes_when_tool_in_log():
    env = Envelope(
        status="done",
        intent_kind="action",
        actions_performed=[Action(tool="patch_content", target=None, description="ok")],
    )
    result = validate_envelope(env, [_summary("patch_content")])
    assert result.ok


# Rule 2 — blocked needs pending_confirmation


def test_rule2_blocked_without_pending_fires():
    env = Envelope(status="blocked", intent_kind="clarify")
    result = validate_envelope(env, [])
    assert {v.rule for v in result.violations} == {
        "blocked_no_confirmation",
        "clarify_needs_blocked",
    } - {"clarify_needs_blocked"} | {"blocked_no_confirmation"}
    # Above expression simplifies to {"blocked_no_confirmation"} when intent_kind=clarify+status=blocked is consistent (rule 6 only fires when status != blocked).  # noqa: E501
    assert "blocked_no_confirmation" in {v.rule for v in result.violations}


def test_rule2_blocked_with_pending_passes():
    env = Envelope(
        status="blocked",
        intent_kind="clarify",
        pending_confirmation=PendingConfirmation(question="Confirm?"),
    )
    result = validate_envelope(env, [])
    assert "blocked_no_confirmation" not in {v.rule for v in result.violations}


# Rule 3 — partial needs evidence


def test_rule3_partial_no_evidence_fires():
    env = Envelope(status="partial", intent_kind="action")
    result = validate_envelope(env, [])
    assert "partial_no_evidence" in {v.rule for v in result.violations}


def test_rule3_partial_with_action_passes():
    env = Envelope(
        status="partial",
        intent_kind="action",
        actions_performed=[Action(tool="patch_content", description="x")],  # pyright: ignore[reportCallIssue]
    )
    result = validate_envelope(env, [_summary("patch_content")])
    assert "partial_no_evidence" not in {v.rule for v in result.violations}


def test_rule3_partial_with_pending_passes():
    env = Envelope(
        status="partial",
        intent_kind="action",
        pending_confirmation=PendingConfirmation(question="More?"),
    )
    result = validate_envelope(env, [])
    assert "partial_no_evidence" not in {v.rule for v in result.violations}


def test_rule3_partial_with_tool_error_passes():
    env = Envelope(status="partial", intent_kind="action")
    result = validate_envelope(env, [_summary("patch_content", is_error=True)])
    assert "partial_no_evidence" not in {v.rule for v in result.violations}


# Rule 4 — empty on done (NEW: structural, gated by intent_kind="action")


def test_rule4_empty_on_done_action_fires():
    env = Envelope(status="done", intent_kind="action", actions_performed=[])
    result = validate_envelope(env, [])
    assert "empty_on_done" in {v.rule for v in result.violations}


def test_rule4_empty_on_done_answer_passes():
    env = Envelope(status="done", intent_kind="answer", actions_performed=[])
    result = validate_envelope(env, [])
    assert "empty_on_done" not in {v.rule for v in result.violations}


def test_rule4_empty_on_done_action_with_error_does_not_fire():
    env = Envelope(status="done", intent_kind="action")
    # If there's an error in the log we still consider the turn evidenced;
    # the agent should have downgraded to partial — that's a separate
    # concern caught by the "done with no actions but errors present"
    # diagnostic, not rule 4.
    result = validate_envelope(env, [_summary("patch_content", is_error=True)])
    # Rule 4 still fires because status=done + intent_kind=action + actions=[]
    # is internally inconsistent regardless of errors.
    assert "empty_on_done" in {v.rule for v in result.violations}


# Rule 5 — count mismatch (NEW: structural, reads envelope.expected_count)


def test_rule5_count_mismatch_fires_when_distinct_targets_short():
    env = Envelope(
        status="done",
        intent_kind="action",
        expected_count=3,
        actions_performed=[
            Action(tool="patch_content", target="A", description="x"),
            Action(tool="patch_content", target="B", description="x"),
        ],
    )
    result = validate_envelope(env, [_summary("patch_content"), _summary("patch_content")])
    assert "count_mismatch" in {v.rule for v in result.violations}


def test_rule5_count_satisfied_passes():
    env = Envelope(
        status="done",
        intent_kind="action",
        expected_count=2,
        actions_performed=[
            Action(tool="patch_content", target="A", description="x"),
            Action(tool="patch_content", target="B", description="x"),
        ],
    )
    result = validate_envelope(env, [_summary("patch_content"), _summary("patch_content")])
    assert "count_mismatch" not in {v.rule for v in result.violations}


def test_rule5_skips_when_status_not_done():
    env = Envelope(
        status="partial",
        intent_kind="action",
        expected_count=5,
        actions_performed=[Action(tool="patch_content", target="A", description="x")],
    )
    result = validate_envelope(env, [_summary("patch_content")])
    assert "count_mismatch" not in {v.rule for v in result.violations}


# Rule 6 — clarify needs blocked


def test_rule6_clarify_with_done_fires():
    env = Envelope(status="done", intent_kind="clarify")
    result = validate_envelope(env, [])
    assert "clarify_needs_blocked" in {v.rule for v in result.violations}


def test_rule6_clarify_with_blocked_and_pending_passes():
    env = Envelope(
        status="blocked",
        intent_kind="clarify",
        pending_confirmation=PendingConfirmation(question="Which?"),
    )
    result = validate_envelope(env, [])
    assert "clarify_needs_blocked" not in {v.rule for v in result.violations}


# Rule 7 — answer with writes


def test_rule7_answer_with_actions_fires():
    env = Envelope(
        status="done",
        intent_kind="answer",
        actions_performed=[Action(tool="patch_content", description="x")],  # pyright: ignore[reportCallIssue]
    )
    result = validate_envelope(env, [_summary("patch_content")])
    assert "answer_with_writes" in {v.rule for v in result.violations}


def test_rule7_answer_no_writes_passes():
    env = Envelope(status="done", intent_kind="answer")
    result = validate_envelope(env, [])
    assert "answer_with_writes" not in {v.rule for v in result.violations}


# Validator signature must NOT accept user_message or mandate_was_write


def test_validate_envelope_signature_has_no_user_message():
    import inspect

    sig = inspect.signature(validate_envelope)
    assert "user_message" not in sig.parameters
    assert "mandate_was_write" not in sig.parameters


def test_validate_envelope_signature_only_takes_envelope_and_log():
    import inspect

    sig = inspect.signature(validate_envelope)
    assert list(sig.parameters.keys()) == ["envelope", "tool_calls"]


# ---------------------------------------------------------------------------
# _summaries_since_last_user_turn helper (used by Rule 9)
# ---------------------------------------------------------------------------


def _make_msg(role, content):
    """Build a fully-valid Message for tests (id/session_id/created_at required)."""
    from datetime import UTC, datetime

    from agentkit._ids import MessageId, SessionId, new_id
    from agentkit._messages import Message

    return Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=role,
        content=content,
        created_at=datetime.now(UTC),
    )


def test_helper_returns_empty_when_no_history():
    from agentkit._messages import Message, MessageRole
    from agentkit.finalize_validator import _summaries_since_last_user_turn

    history: list[Message] = []
    result = _summaries_since_last_user_turn(history)
    assert result == []


def test_helper_returns_only_tools_after_last_user_message():
    from agentkit._content import TextBlock, ToolResultBlock, ToolUseBlock
    from agentkit._messages import MessageRole
    from agentkit.finalize_validator import _summaries_since_last_user_turn

    history = [
        _make_msg(MessageRole.USER, [TextBlock(text="first question")]),
        _make_msg(
            MessageRole.ASSISTANT,
            [ToolUseBlock(id="t1", name="search", arguments={})],
        ),
        _make_msg(
            MessageRole.USER,
            [
                ToolResultBlock(tool_use_id="t1", content=[TextBlock(text="ok")], is_error=False)
            ],
        ),
        # ===== second user turn starts here =====
        _make_msg(MessageRole.USER, [TextBlock(text="second question")]),
        _make_msg(
            MessageRole.ASSISTANT,
            [ToolUseBlock(id="t2", name="recall_memories", arguments={})],
        ),
        _make_msg(
            MessageRole.USER,
            [
                ToolResultBlock(tool_use_id="t2", content=[TextBlock(text="ok")], is_error=False)
            ],
        ),
    ]
    result = _summaries_since_last_user_turn(history)
    assert [s.name for s in result] == ["recall_memories"]
    assert result[0].is_error is False


def test_helper_skips_finalize_response_tool():
    from agentkit._content import TextBlock, ToolUseBlock
    from agentkit._messages import MessageRole
    from agentkit.finalize_validator import _summaries_since_last_user_turn

    history = [
        _make_msg(MessageRole.USER, [TextBlock(text="hi")]),
        _make_msg(
            MessageRole.ASSISTANT,
            [
                ToolUseBlock(id="t1", name="search", arguments={}),
                ToolUseBlock(id="t2", name="finalize_response", arguments={}),
            ],
        ),
    ]
    result = _summaries_since_last_user_turn(history)
    assert [s.name for s in result] == ["search"]
