"""Pure-function structural validator for the finalize envelope.

Inspects the parsed ``Envelope`` plus the turn's tool-call log. Never
reads user message text. See the design spec:
``docs/superpowers/specs/2026-05-10-envelope-intent-kind-design.md``
in the Pikkolo repo.
"""

from __future__ import annotations

from agentkit._content import ToolResultBlock, ToolUseBlock
from agentkit._messages import Message, MessageRole
from agentkit.envelope import Envelope, ToolCallSummary, ValidationResult, Violation


def _summaries_since_last_user_turn(history: list[Message]) -> list[ToolCallSummary]:
    """Walk history backwards; return ToolCallSummary entries for tool calls
    made AFTER the most recent USER message.

    Used by Rule 9 (answer_evidence_consistent) so the model can't claim
    ``answer_evidence='tool_results'`` based on stale reads from prior turns.
    The existing ``_ctx_to_summaries`` in ``guards/finalize.py`` walks the
    full history and remains unchanged — its write-mandate semantics
    legitimately span turns.

    The ``finalize_response`` tool itself is filtered out: it's the
    terminal call being validated, not "evidence".
    """
    # Find the index of the most recent USER message that represents a genuine
    # human prompt. In Anthropic-style history, a USER message may carry
    # ToolResultBlock entries for tools the prior assistant turn called —
    # those belong to the PRIOR turn. We skip USER messages that consist
    # entirely of ToolResultBlocks (tool-return carriers) and keep scanning
    # until we find a USER message with non-ToolResultBlock content (or any
    # message after which the agent's tool calls belong to the current turn).
    last_user_idx = -1
    for i in range(len(history) - 1, -1, -1):
        if history[i].role is MessageRole.USER:
            # If this USER message contains only ToolResultBlocks, it is a
            # tool-return carrier for the current turn — keep scanning.
            if all(isinstance(b, ToolResultBlock) for b in history[i].content):
                continue
            # Otherwise this is the genuine prompt boundary.
            last_user_idx = i
            break

    if last_user_idx < 0:
        return []

    use_names: dict[str, str] = {}
    result_errors: dict[str, bool] = {}
    for msg in history[last_user_idx + 1 :]:
        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                use_names[block.id] = block.name
            elif isinstance(block, ToolResultBlock):
                result_errors[block.tool_use_id] = block.is_error

    summaries: list[ToolCallSummary] = []
    for use_id, name in use_names.items():
        bare = name.split(".", 1)[-1]
        if bare in ("finalize_response", "finalize"):
            continue
        # Read classification mirrors guards/finalize.py: anything not on
        # the conservative read-prefix list counts as a write. For Rule 9
        # we only care about the read/write flag; reuse the same heuristic.
        from agentkit.guards.finalize import _is_default_write

        summaries.append(
            ToolCallSummary(
                name=bare,
                is_error=result_errors.get(use_id, False),
                is_write=_is_default_write(name),
            )
        )
    return summaries


def validate_envelope(  # noqa: PLR0912
    envelope: Envelope,
    tool_calls: list[ToolCallSummary],
    turn_summaries: list[ToolCallSummary] | None = None,
) -> ValidationResult:
    """Validate an envelope against the turn's tool-call log.

    ``turn_summaries`` (optional) scopes Rule 9 to this turn's reads only.
    When omitted, falls back to ``tool_calls`` for backwards compatibility.

    Rules (see spec section "Validator rules"):

    1. fabricated_tool: every actions_performed[].tool must be a successful
       write call in tool_calls.
    2. blocked_no_confirmation: status='blocked' requires pending_confirmation.
    3. partial_no_evidence: status='partial' requires at least one action,
       one errored call, or pending_confirmation.
    4. empty_on_done: status='done' AND intent_kind='action' AND
       actions_performed=[] is inconsistent.
    5. count_mismatch: status='done' AND intent_kind='action' AND
       expected_count > distinct(actions_performed[].target).
    6. clarify_needs_blocked: intent_kind='clarify' requires status='blocked'
       AND pending_confirmation.
    7. answer_with_writes: intent_kind='answer' AND actions_performed != [].
    8. answer_evidence_required: intent_kind='answer' requires answer_evidence
       to be set.
    9. answer_evidence_consistent: answer_evidence='tool_results' requires at
       least one successful read tool call in this turn's tool log.

    Rule 8 (CTA coverage) is consumer-specific and lives in Pikkolo's
    adapter, not here.
    """
    violations: list[Violation] = []

    successful_writes: dict[str, int] = {}
    any_error = False
    for c in tool_calls:
        if c.is_error:
            any_error = True
            continue
        if c.is_write:
            successful_writes[c.name] = successful_writes.get(c.name, 0) + 1

    # Rule 1
    consumed: dict[str, int] = {}
    for action in envelope.actions_performed:
        available = successful_writes.get(action.tool, 0) - consumed.get(action.tool, 0)
        if available <= 0:
            violations.append(
                Violation(
                    rule="fabricated_tool",
                    detail=f"no successful {action.tool} call this turn",
                )
            )
        else:
            consumed[action.tool] = consumed.get(action.tool, 0) + 1

    # Rule 2
    if envelope.status == "blocked" and envelope.pending_confirmation is None:
        violations.append(
            Violation(
                rule="blocked_no_confirmation",
                detail="status=blocked but pending_confirmation is null",
            )
        )

    # Rule 3
    if (
        envelope.status == "partial"
        and not envelope.actions_performed
        and not any_error
        and envelope.pending_confirmation is None
    ):
        violations.append(
            Violation(
                rule="partial_no_evidence",
                detail=(
                    "status=partial with no actions, no tool errors, and no pending_confirmation"
                ),
            )
        )

    # Rule 4
    if (
        envelope.status == "done"
        and envelope.intent_kind == "action"
        and not envelope.actions_performed
    ):
        violations.append(
            Violation(
                rule="empty_on_done",
                detail="status=done with intent_kind=action but actions_performed is empty",
            )
        )

    # Rule 5
    if (
        envelope.status == "done"
        and envelope.intent_kind == "action"
        and envelope.expected_count is not None
    ):
        distinct_targets = {a.target for a in envelope.actions_performed if a.target}
        if len(distinct_targets) < envelope.expected_count:
            violations.append(
                Violation(
                    rule="count_mismatch",
                    detail=(
                        f"expected_count={envelope.expected_count} but envelope "
                        f"shows {len(distinct_targets)} distinct targets"
                    ),
                )
            )

    # Rule 6
    if envelope.intent_kind == "clarify" and envelope.status != "blocked":
        violations.append(
            Violation(
                rule="clarify_needs_blocked",
                detail="intent_kind=clarify requires status=blocked and pending_confirmation",
            )
        )

    # Rule 7
    if envelope.intent_kind == "answer" and envelope.actions_performed:
        violations.append(
            Violation(
                rule="answer_with_writes",
                detail="intent_kind=answer but actions_performed is non-empty",
            )
        )

    # Rule 8
    if envelope.intent_kind == "answer" and envelope.answer_evidence is None:
        violations.append(
            Violation(
                rule="answer_evidence_required",
                detail=(
                    "intent_kind=answer requires answer_evidence. Pick one: "
                    "'tool_results' (you called a read tool this turn), "
                    "'context' (the answer is in your system prompt — page "
                    "state, brand voice, prior turn), or 'general_knowledge' "
                    "(training data — math, language, world facts)."
                ),
            )
        )

    # Rule 9
    if envelope.answer_evidence == "tool_results":
        scope = turn_summaries if turn_summaries is not None else tool_calls
        has_successful_read = any(
            (not c.is_write) and (not c.is_error) for c in scope
        )
        if not has_successful_read:
            violations.append(
                Violation(
                    rule="answer_evidence_consistent",
                    detail=(
                        "answer_evidence='tool_results' requires at least one "
                        "successful read tool this turn. The turn's tool log "
                        "has no successful reads. Either call a read tool (e.g. "
                        "recall_memories, search, smart_search, get_*) or "
                        "change answer_evidence to 'context' / "
                        "'general_knowledge'."
                    ),
                )
            )

    return ValidationResult(ok=not violations, violations=violations)
