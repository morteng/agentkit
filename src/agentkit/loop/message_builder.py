"""MessageBuilder — assemble a ProviderRequest from a TurnContext + tools.

Pure transformation: takes inputs and returns a request. No I/O, no state.
"""

from collections.abc import Sequence

from agentkit._messages import Message
from agentkit.providers.base import (
    ProviderRequest,
    SystemBlock,
    ThinkingConfig,
    ToolDefinition,
)
from agentkit.tools.spec import ToolSpec


class MessageBuilder:
    def __init__(
        self,
        *,
        model: str,
        max_tokens: int,
        temperature: float | None = None,
        thinking: ThinkingConfig | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._thinking = thinking
        self._metadata = metadata or {}

    def build(
        self,
        *,
        system_blocks: Sequence[SystemBlock],
        history: Sequence[Message],
        tool_specs: Sequence[ToolSpec],
    ) -> ProviderRequest:
        return ProviderRequest(
            model=self._model,
            system=list(system_blocks),
            messages=list(history),
            tools=[self._spec_to_def(s) for s in tool_specs],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            thinking=self._thinking,
            metadata=self._metadata,
        )

    @staticmethod
    def _spec_to_def(spec: ToolSpec) -> ToolDefinition:
        return ToolDefinition(
            name=spec.name,
            description=spec.description,
            parameters=spec.parameters,
        )
