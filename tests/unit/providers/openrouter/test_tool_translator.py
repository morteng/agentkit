from agentkit.providers.base import ToolDefinition
from agentkit.providers.openrouter.tool_translator import to_openai_tool


def test_translation_uses_function_envelope():
    td = ToolDefinition(name="add", description="adds", parameters={"type": "object"})
    out = to_openai_tool(td)
    assert out == {
        "type": "function",
        "function": {"name": "add", "description": "adds", "parameters": {"type": "object"}},
    }
