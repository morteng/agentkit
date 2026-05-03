from pydantic import TypeAdapter

from agentkit._content import ContentBlock, TextBlock, ToolUseBlock


def test_text_block_serialises_with_discriminator():
    b = TextBlock(text="hello")
    dumped = b.model_dump()
    assert dumped == {"type": "text", "text": "hello"}


def test_content_block_union_discriminates():
    adapter = TypeAdapter(ContentBlock)
    parsed = adapter.validate_python(
        {"type": "tool_use", "id": "call_1", "name": "x", "arguments": {"a": 1}}
    )
    assert isinstance(parsed, ToolUseBlock)
    assert parsed.id == "call_1"
