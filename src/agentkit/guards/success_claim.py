"""SuccessClaimGuard — flag streaming text claiming success without a write tool."""

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agentkit._content import ToolUseBlock
from agentkit._messages import MessageRole
from agentkit.loop.context import TurnContext


@dataclass(frozen=True)
class ClaimVerdict:
    flag: bool
    suggested_correction: str | None = None


@runtime_checkable
class SuccessClaimGuard(Protocol):
    async def check(self, text_so_far: str, ctx: TurnContext) -> ClaimVerdict: ...


_CLAIM_PATTERNS = re.compile(
    r"\b(i(?:'ve| have)?\s+(created|updated|deleted|set up|turned (?:on|off)|"
    r"sent|scheduled|enabled|disabled|configured|saved)\b|done!|all set|"
    r"jeg har\s+(opprettet|oppdatert|slettet|satt opp|sendt|aktivert|"
    r"deaktivert|konfigurert|lagret))",
    flags=re.IGNORECASE,
)


class RegexSuccessClaimGuard(SuccessClaimGuard):
    """Off by default in AgentConfig; consumers opt in.

    A claim of success is flagged when no non-``kit`` tool has been invoked
    in the current turn's history. The Loop's streaming handler may then
    cancel the stream, inject a correction, and retry.
    """

    def __init__(self, *, kit_namespace: str = "kit.") -> None:
        self._kit = kit_namespace

    async def check(self, text_so_far: str, ctx: TurnContext) -> ClaimVerdict:
        if not _CLAIM_PATTERNS.search(text_so_far):
            return ClaimVerdict(flag=False)
        if self._has_non_kit_tool_call(ctx):
            return ClaimVerdict(flag=False)
        return ClaimVerdict(
            flag=True,
            suggested_correction=(
                "You claimed to have completed the task but have not yet invoked "
                "the tool that actually changes state. Call the appropriate tool first."
            ),
        )

    def _has_non_kit_tool_call(self, ctx: TurnContext) -> bool:
        for msg in ctx.history:
            if msg.role is not MessageRole.ASSISTANT:
                continue
            for block in msg.content:
                if isinstance(block, ToolUseBlock) and not block.name.startswith(self._kit):
                    return True
        return False
