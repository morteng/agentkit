"""Translate agentkit ToolDefinition into Anthropic SDK's tool spec."""

from typing import Any

from agentkit.providers.base import ToolDefinition


def to_anthropic_tool(td: ToolDefinition) -> dict[str, Any]:
    """Anthropic's API expects ``input_schema`` (not ``parameters``)."""
    return {
        "name": td.name,
        "description": td.description,
        "input_schema": td.parameters,
    }
