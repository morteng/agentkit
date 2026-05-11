"""Canonical finalize_response tool description + parameter schema.

Consumers register this as a tool via their own decorator; this module
exports only the constants so the contract the model sees stays
identical across consumers.

The handler is NOT registered as a default builtin — consumers wire
their own (typically a no-op that lets the loop intercept the call).

See ``agentkit.envelope`` for the typed model these params parse to,
and ``agentkit.finalize_validator.validate_envelope`` for the rules.
"""

from __future__ import annotations

from typing import Any

FINALIZE_RESPONSE_DESCRIPTION = (
    "Call this tool exactly once per turn, when you are finished.\n\n"
    "intent_kind — what the user's last message asked you to do:\n"
    '  • "action": Do work. Editing, creating, deleting, publishing, anything that writes.\n'
    '  • "answer": Answer a question or analyze. No writes.\n'
    '  • "clarify": You don\'t have enough info to act safely.\n\n'
    'If intent_kind="action" you MUST have called write tools and listed '
    'them in actions_performed. Promising future action ("I\'ll start", '
    '"Setter i gang") without calling write tools in this same turn is invalid.\n\n'
    'If intent_kind="answer" you MUST NOT list actions_performed.\n\n'
    'If intent_kind="clarify" you MUST set status="blocked" with pending_confirmation.\n\n'
    'expected_count — when the user named a number ("3 articles", "all five", '
    '"kjør alt på alle"), set this to that number so the system can verify you '
    "finished. Use null when no count was implied.\n"
)


FINALIZE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["done", "partial", "blocked"],
            "description": "Overall turn outcome.",
        },
        "intent_kind": {
            "type": "string",
            "enum": ["action", "answer", "clarify"],
            "description": (
                "Self-classification of what the user's last message asked you to do. "
                "See the tool description for semantics."
            ),
        },
        "summary": {
            "type": "string",
            "description": "Optional short prose summary of the turn.",
        },
        "actions_performed": {
            "type": "array",
            "description": (
                "One entry per WRITE action you took this turn. NEVER list read tools. "
                "Empty [] when intent_kind='answer' or no writes happened."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string"},
                    "target": {"type": ["string", "null"]},
                    "description": {"type": "string"},
                },
                "required": ["tool", "description"],
            },
        },
        "pending_confirmation": {
            "type": ["object", "null"],
            "description": "Question awaiting the editor when status='partial' or 'blocked'.",
            "properties": {
                "question": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": ["confirm", "choose", "free_text"],
                },
            },
            "required": ["question"],
        },
        "expected_count": {
            "type": ["integer", "null"],
            "minimum": 1,
            "description": (
                "When the user named a count, set this to that number. "
                "Use null when no count was implied. 0 is invalid."
            ),
        },
        "proposed_autonomous_scope": {
            "type": ["object", "null"],
            "description": (
                "Optional scope declaration for autonomous bulk turns; opaque to the validator."
            ),
        },
    },
    "required": ["status", "intent_kind", "actions_performed"],
}


__all__ = ["FINALIZE_RESPONSE_DESCRIPTION", "FINALIZE_RESPONSE_SCHEMA"]
