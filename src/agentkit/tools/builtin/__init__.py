"""Built-in tool exports + DEFAULT_BUILTINS for convenient registration."""

from agentkit.tools.builtin.approval import (
    REQUEST_APPROVAL_SPEC,
    PendingApproval,
    request_approval_handler,
)
from agentkit.tools.builtin.finalize import FINALIZE_SPEC, finalize_handler
from agentkit.tools.builtin.memory import (
    MEMORY_RECALL_SPEC,
    MEMORY_SAVE_SPEC,
    memory_recall_handler,
    memory_save_handler,
)
from agentkit.tools.builtin.note import NOTE_SPEC, note_handler
from agentkit.tools.builtin.subagent import SUBAGENT_SPAWN_SPEC, subagent_spawn_handler
from agentkit.tools.builtin.time import CURRENT_TIME_SPEC, current_time_handler

DEFAULT_BUILTINS = [
    (FINALIZE_SPEC, finalize_handler),
    (CURRENT_TIME_SPEC, current_time_handler),
    (MEMORY_SAVE_SPEC, memory_save_handler),
    (MEMORY_RECALL_SPEC, memory_recall_handler),
    (SUBAGENT_SPAWN_SPEC, subagent_spawn_handler),
    # NOTE_SPEC is opt-in; not in DEFAULT_BUILTINS.
    # REQUEST_APPROVAL_SPEC is exported but not registered by default — the
    # current handler appends to ctx.pending_approvals which no orchestrator
    # phase currently surfaces, so the user is never actually prompted.
    # Consumers who want agent-initiated approvals can register it manually
    # and supply their own surfacing path. See docs/recipes.md.
]


__all__ = [
    "CURRENT_TIME_SPEC",
    "DEFAULT_BUILTINS",
    "FINALIZE_SPEC",
    "MEMORY_RECALL_SPEC",
    "MEMORY_SAVE_SPEC",
    "NOTE_SPEC",
    "REQUEST_APPROVAL_SPEC",
    "SUBAGENT_SPAWN_SPEC",
    "PendingApproval",
    "current_time_handler",
    "finalize_handler",
    "memory_recall_handler",
    "memory_save_handler",
    "note_handler",
    "request_approval_handler",
    "subagent_spawn_handler",
]
