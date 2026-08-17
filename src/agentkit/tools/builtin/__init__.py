"""Built-in tool exports + DEFAULT_BUILTINS for convenient registration."""

from agentkit.tools.builtin.approval import (
    REQUEST_APPROVAL_SPEC,
    PendingApproval,
    request_approval_handler,
)
from agentkit.tools.builtin.finalize import FINALIZE_SPEC, finalize_handler
from agentkit.tools.builtin.finalize_response import (
    FINALIZE_RESPONSE_DESCRIPTION,
    FINALIZE_RESPONSE_SCHEMA,
)
from agentkit.tools.builtin.memory import (
    MEMORY_FORGET_SPEC,
    MEMORY_LIST_SPEC,
    MEMORY_RECALL_SPEC,
    MEMORY_SAVE_SPEC,
    MEMORY_SEARCH_SPEC,
    memory_forget_handler,
    memory_list_handler,
    memory_recall_handler,
    memory_save_handler,
    memory_search_handler,
)
from agentkit.tools.builtin.note import NOTE_SPEC, note_handler
from agentkit.tools.builtin.subagent import SUBAGENT_SPAWN_SPEC, subagent_spawn_handler
from agentkit.tools.builtin.time import CURRENT_TIME_SPEC, current_time_handler

DEFAULT_BUILTINS = [
    (FINALIZE_SPEC, finalize_handler),
    (CURRENT_TIME_SPEC, current_time_handler),
    (MEMORY_SAVE_SPEC, memory_save_handler),
    (MEMORY_RECALL_SPEC, memory_recall_handler),
    # search/list/forget are defaults for the same reason save/recall are: a
    # host that attaches a MemoryStore has asked for memory, and memory without
    # a way to find or remove things is a drawer that only opens inward. Hosts
    # that want a narrower surface filter DEFAULT_BUILTINS, which is already how
    # role ceilings are applied downstream (search and list are READ, forget is
    # LOW_WRITE, so a read-only role keeps the two that answer questions).
    (MEMORY_SEARCH_SPEC, memory_search_handler),
    (MEMORY_LIST_SPEC, memory_list_handler),
    (MEMORY_FORGET_SPEC, memory_forget_handler),
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
    "FINALIZE_RESPONSE_DESCRIPTION",
    "FINALIZE_RESPONSE_SCHEMA",
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
