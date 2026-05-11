"""Pure-function structural validator for the finalize envelope.

Inspects the parsed ``Envelope`` plus the turn's tool-call log. Never
reads user message text. See the design spec:
``docs/superpowers/specs/2026-05-10-envelope-intent-kind-design.md``
in the Pikkolo repo.
"""

from __future__ import annotations

from agentkit.envelope import Envelope, ToolCallSummary, ValidationResult, Violation


def validate_envelope(  # noqa: PLR0912
    envelope: Envelope,
    tool_calls: list[ToolCallSummary],
) -> ValidationResult:
    """Validate an envelope against the turn's tool-call log.

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

    return ValidationResult(ok=not violations, violations=violations)
