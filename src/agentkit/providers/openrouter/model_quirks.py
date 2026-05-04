"""Per-model behaviour switches for OpenRouter.

OpenRouter speaks the OpenAI Chat Completions protocol. Underlying models
behave differently re: prompt caching:

- Anthropic + Gemini: must receive content blocks with explicit ``cache_control``.
- OpenAI + DeepSeek: caching is automatic; sending cache_control is harmless
  but the format must remain plain strings to avoid validation errors.

This module is the single source of truth for those distinctions.
"""

from typing import Literal

# Models / model-family prefixes that need explicit cache_control content blocks.
_NEEDS_CACHE_BLOCKS_PREFIXES: frozenset[str] = frozenset(
    {
        "anthropic/",
        "google/gemini",
        "google/google-gemini",
    }
)


def requires_cache_blocks(model: str) -> bool:
    """True if this model needs content-block format with cache_control."""
    m = model.lower()
    return any(m.startswith(prefix) for prefix in _NEEDS_CACHE_BLOCKS_PREFIXES)


_FINISH_REASON_MAP: dict[str, Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"]] = {
    "stop": "end_turn",
    "end_turn": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "tool_use": "tool_use",
    "length": "max_tokens",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
}


def parse_finish_reason(
    raw: str | None,
) -> Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"]:
    if raw is None:
        return "end_turn"
    return _FINISH_REASON_MAP.get(raw, "end_turn")
