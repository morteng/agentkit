"""Translate agentkit ToolDefinition into OpenAI Chat-Completions tool spec.

Also hosts ``parse_tool_args_with_repair`` — a fallback parser for malformed
tool-call JSON arguments emitted by DeepSeek V4 Flash and other open models.
"""

import json
import logging
from typing import Any

import json_repair

from agentkit.providers.base import ToolDefinition

logger = logging.getLogger(__name__)


def to_openai_tool(td: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": td.name,
            "description": td.description,
            "parameters": td.parameters,
        },
    }


def parse_tool_args_with_repair(args_str: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse tool-call argument JSON, falling back to json_repair on failure.

    Returns ``(parsed_dict, None)`` on success or ``(None, error_message)``
    when even the repair pass cannot recover a non-trivial dict. Empty input
    is treated as an empty argument set.

    Why: DeepSeek V4 Flash (and other open models) periodically emit JSON with
    unquoted values, e.g. ``{"facts": Drammens Teater ble bygd i 1869, ...}``.
    Without recovery the loop bounces the call back to the model with a vague
    "JSON parse error" several times before the model self-corrects, burning
    3-5 iterations of read budget per occurrence.

    ``json_repair.loads`` returns ``{}`` on total parse failure rather than
    raising, so we treat empty-result + non-trivial input as still-failed and
    surface the original error so the caller's retry path still fires.
    """
    if not args_str:
        return {}, None
    try:
        return json.loads(args_str), None
    except json.JSONDecodeError as exc:
        original_error = str(exc)
        try:
            repaired = json_repair.loads(args_str)
        except Exception as repair_exc:  # pragma: no cover - defensive
            return None, f"{original_error} (repair also failed: {repair_exc})"
        if isinstance(repaired, dict) and repaired:
            logger.warning(
                "openrouter.tool_args_repaired",
                extra={
                    "original_error": original_error,
                    "recovered_keys": list(repaired.keys()),
                },
            )
            return repaired, None
        return None, original_error
