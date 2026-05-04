"""Provider-agnostic cache breakpoint policy.

Yields breakpoint hints — adapters honour them in their SDK's format.
"""

from dataclasses import dataclass

from agentkit._messages import Message
from agentkit.providers.base import SystemBlock, ToolDefinition


@dataclass(frozen=True)
class CacheBreakpoints:
    """Where to insert cache_control markers in a request.

    - ``cache_system``: cache the entire system block(s) as one breakpoint.
    - ``cache_tools``: cache the tool definitions block.
    - ``history_cache_index``: index in ``messages`` such that messages
      ``[0:history_cache_index]`` are cacheable; the rest are fresh.
      0 means no history cache (history is too short to bother).
    """

    cache_system: bool
    cache_tools: bool
    history_cache_index: int


def compute_breakpoints(
    *,
    system: list[SystemBlock],
    tools: list[ToolDefinition],
    messages: list[Message],
    fresh_tail_messages: int = 2,
) -> CacheBreakpoints:
    """Default policy:
    - System and tools always cacheable when present.
    - History cacheable except the last ``fresh_tail_messages`` items.
    - History cache disabled when fewer than 2*fresh_tail messages exist.
    """
    cache_system = bool(system)
    cache_tools = bool(tools)
    if len(messages) < fresh_tail_messages * 2:
        idx = 0
    else:
        idx = max(0, len(messages) - fresh_tail_messages)
    return CacheBreakpoints(
        cache_system=cache_system,
        cache_tools=cache_tools,
        history_cache_index=idx,
    )
