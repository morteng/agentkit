"""FinalizeValidator — gate the agent's "I'm done" claim."""

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agentkit._content import TextBlock, ToolUseBlock
from agentkit._messages import Message, MessageRole
from agentkit.loop.context import TurnContext
from agentkit.tools.spec import ToolCall


@dataclass(frozen=True)
class FinalizeVerdict:
    accept: bool
    feedback: str | None = None


@runtime_checkable
class FinalizeValidator(Protocol):
    async def validate(self, finalize_call: ToolCall, ctx: TurnContext) -> FinalizeVerdict: ...


# Rough action-intent heuristic. Keep simple: imperative-style verbs.
_ACTION_VERBS = re.compile(
    r"\b(turn|switch|set|create|delete|remove|cancel|schedule|enable|disable|"
    r"send|update|change|start|stop|run|restart|configure|add|"
    r"slå|skru|sett|opprett|slett|fjern|kanseller|aktiver|deaktiver|send|"
    r"oppdater|endre|start|stopp|kjør|konfigurer|legg)\b",
    flags=re.IGNORECASE,
)


class RuleBasedFinalizeValidator(FinalizeValidator):
    """Rules:
    - If the latest user message was action-oriented AND no tool call (other
      than ``kit.*``) appears in the assistant's turn-history, reject.
    - Otherwise accept.
    """

    def __init__(self, *, kit_namespace: str = "kit.") -> None:
        self._kit = kit_namespace

    async def validate(self, finalize_call: ToolCall, ctx: TurnContext) -> FinalizeVerdict:
        latest_user = self._latest_user_message(ctx)
        if latest_user is None:
            return FinalizeVerdict(accept=True)
        is_action = self._is_action_request(latest_user)
        if not is_action:
            return FinalizeVerdict(accept=True)
        if self._has_non_kit_tool_call(ctx):
            return FinalizeVerdict(accept=True)
        return FinalizeVerdict(
            accept=False,
            feedback=(
                "You called kit.finalize but the user asked for an action and you "
                "did not invoke any non-kit tool. Call the appropriate tool to "
                "actually carry out the request, then finalize."
            ),
        )

    def _latest_user_message(self, ctx: TurnContext) -> Message | None:
        for msg in reversed(ctx.history):
            if msg.role is MessageRole.USER:
                return msg
        return None

    def _is_action_request(self, msg: Message) -> bool:
        text = "\n".join(b.text for b in msg.content if isinstance(b, TextBlock))
        return bool(_ACTION_VERBS.search(text))

    def _has_non_kit_tool_call(self, ctx: TurnContext) -> bool:
        for msg in ctx.history:
            if msg.role is not MessageRole.ASSISTANT:
                continue
            for block in msg.content:
                if isinstance(block, ToolUseBlock) and not block.name.startswith(self._kit):
                    return True
        return False
