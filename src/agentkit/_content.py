"""Provider-agnostic content blocks.

Mirrors the union shape used by Anthropic and OpenAI's Responses APIs so
the loop carries a single canonical message representation regardless of
which provider produced it.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    text: str
    signature: str | None = None  # Anthropic-only; ignored elsewhere


class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    media_type: str  # "image/png", etc.
    data: str  # base64
    source: Literal["base64", "url"] = "base64"
    url: str | None = None


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str  # provider's tool_use_id
    name: str
    arguments: dict[str, Any]


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: list["ContentBlock"]  # forward-ref; tool results may include text/images
    is_error: bool = False


ContentBlock = Annotated[
    TextBlock | ThinkingBlock | ImageBlock | ToolUseBlock | ToolResultBlock,
    Field(discriminator="type"),
]

# Resolve ToolResultBlock's forward-reference to ContentBlock now that the union exists.
ToolResultBlock.model_rebuild()
