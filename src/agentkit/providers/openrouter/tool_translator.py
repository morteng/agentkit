"""Translate agentkit ToolDefinition into OpenAI Chat-Completions tool spec."""

from typing import Any

from agentkit.providers.base import ToolDefinition


def to_openai_tool(td: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": td.name,
            "description": td.description,
            "parameters": td.parameters,
        },
    }
