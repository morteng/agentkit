"""Provider-agnostic content blocks.

Mirrors the union shape used by Anthropic and OpenAI's Responses APIs so
the loop carries a single canonical message representation regardless of
which provider produced it.
"""

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class Provenance(StrEnum):
    """Where the bytes in a piece of content came from.

    The trust axis the taint guard reasons over (see ``agentkit.guards.taint``):

    * ``SYSTEM`` — produced by the runtime or a tool the operator controls.
      The default: existing callers keep the trust level they had before
      provenance existed.
    * ``PRINCIPAL`` — authored by the human principal of this session. Trusted
      to the same degree as the user's own turn.
    * ``UNTRUSTED`` — third-party content the principal did not author: web
      pages, inbound email, scraped documents, another tenant's records. Text
      that reaches the model from here may be an instruction-injection attempt,
      so a turn that ingests any of it is *tainted* for the rest of the turn.
    """

    SYSTEM = "system"
    PRINCIPAL = "principal"
    UNTRUSTED = "untrusted"


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
    provenance: Provenance = Provenance.SYSTEM
    """Trust level of the content this block carries.

    Defaults to ``SYSTEM`` so pre-provenance callers are unaffected. A tool
    that returns third-party text (web fetch, inbox read, scraped document)
    should mark its result ``UNTRUSTED``; the dispatcher then taints the turn
    and the taint guard disables write actions for the remainder of it.
    """


ContentBlock = Annotated[
    TextBlock | ThinkingBlock | ImageBlock | ToolUseBlock | ToolResultBlock,
    Field(discriminator="type"),
]

# Resolve ToolResultBlock's forward-reference to ContentBlock now that the union exists.
ToolResultBlock.model_rebuild()
