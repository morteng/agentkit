from agentkit.providers.anthropic.tool_translator import to_anthropic_tool
from agentkit.providers.base import ToolDefinition


def test_basic_translation():
    td = ToolDefinition(name="add", description="adds", parameters={"type": "object"})
    out = to_anthropic_tool(td)
    assert out == {"name": "add", "description": "adds", "input_schema": {"type": "object"}}
