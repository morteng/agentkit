"""TaintPolicy — disable write actions once untrusted content enters a turn.

The anti-prompt-injection control. A tool result carries a
:class:`~agentkit._content.Provenance`; when a result marked
``Provenance.UNTRUSTED`` enters a turn, the turn is *tainted*
(:attr:`~agentkit.loop.context.TurnContext.tainted`). From that moment the
model has read third-party text that may contain instructions the principal
never wrote, so every tool above :attr:`~agentkit.tools.spec.RiskLevel.READ`
is denied for the remainder of that turn.

Denial is a normal ``ToolResult`` with ``status="denied"`` and a message the
model can narrate back to the user — never a silent no-op and never an
exception. The user then restates the action in a fresh turn, which starts
untainted, so the write happens because the *principal* asked for it and not
because a web page did.

Enforced at execution time in :meth:`agentkit.tools.registry.ToolRegistry.invoke`,
on by default, fail-closed: an unrecognised risk level counts as a write, and
a policy that raises denies the call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

from agentkit._content import Provenance
from agentkit.tools.spec import ContentBlockOut, RiskLevel, ToolResult
from agentkit.tools.spec import ToolError as ToolErrorModel

if TYPE_CHECKING:
    from agentkit.tools.spec import ToolSpec

RISK_RANK: dict[RiskLevel, int] = {
    RiskLevel.READ: 0,
    RiskLevel.LOW_WRITE: 1,
    RiskLevel.HIGH_WRITE: 2,
    RiskLevel.DESTRUCTIVE: 3,
}
"""Total order over risk levels. Anything ranked above the policy ceiling is a
"write" for taint purposes."""

TAINT_DENIAL_MESSAGE = (
    "denied: this turn has ingested untrusted external content, so write "
    "actions are disabled. Ask the user to restate the action in a new turn."
)

TAINT_DENIAL_CODE = "tainted_turn"


class TaintSource(BaseModel):
    """One untrusted-content tool result a turn has ingested.

    The boolean :attr:`~agentkit.loop.context.TurnContext.tainted` says *that*
    the turn is tainted; this says *what* tainted it, so an approval card can
    name the tool whose output the model read before it proposed the write.
    """

    call_id: str
    tool_name: str
    kind: str
    """The :class:`~agentkit._content.Provenance` value that caused the taint.
    A string, not the enum, so a future provenance level does not break a
    consumer parsing older events."""

    model_config = {"frozen": True}


def is_tainted(ctx: Any) -> bool:
    """True when ``ctx`` has ingested untrusted content this turn.

    Tolerates any context object: a context without the attribute (a subagent
    stub, a test double) is simply untainted.
    """
    return bool(getattr(ctx, "tainted", False))


def taint_sources(ctx: Any) -> list[TaintSource]:
    """The untrusted results ``ctx`` has ingested, in ingestion order.

    Tolerates any context object, same as :func:`is_tainted`: one without the
    attribute simply has no sources. A tainted context with an empty list is
    possible (taint restored from a checkpoint written before sources were
    recorded), so never infer taint from this being empty — use
    :func:`is_tainted`.
    """
    sources: Any = getattr(ctx, "taint_sources", None)
    if not sources:
        return []
    return list(sources)


def mark_taint(ctx: Any, result: ToolResult, *, tool_name: str = "") -> bool:
    """Taint ``ctx`` if ``result`` carries untrusted content.

    Records the result in ``ctx.taint_sources`` — every untrusted result, not
    just the first — so consumers can render what tainted the turn.

    Returns ``True`` only on the transition — the first untrusted result of the
    turn — so callers can log the event once. Idempotent, and a no-op for a
    context object that does not accept the attributes.
    """
    if result.provenance is not Provenance.UNTRUSTED:
        return False
    _record_source(ctx, result, tool_name)
    if is_tainted(ctx):
        return False
    try:
        ctx.tainted = True
    except AttributeError:  # frozen/slotted context object — nothing to mark
        return False
    return True


def _record_source(ctx: Any, result: ToolResult, tool_name: str) -> None:
    sources: Any = getattr(ctx, "taint_sources", None)
    if sources is None:
        return
    source = TaintSource(
        call_id=result.call_id,
        tool_name=tool_name,
        kind=result.provenance.value,
    )
    # A retried call can produce the same result twice; the card should list
    # the tool once. TaintSource is frozen, so equality is by value.
    if source not in sources:
        sources.append(source)


@runtime_checkable
class TaintPolicy(Protocol):
    """Decides whether a tool may run given the turn's taint state."""

    def denial_reason(self, spec: ToolSpec, ctx: Any) -> str | None:
        """Return why the call is denied, or ``None`` to allow it.

        ``ctx`` is the ``TurnContext`` — typed loosely to avoid a circular
        import, same as :class:`agentkit.guards.approval.ApprovalGate`.
        """
        ...


class RiskBasedTaintPolicy(TaintPolicy):
    """Default policy: after taint, allow nothing above ``max_risk_when_tainted``.

    ``max_risk_when_tainted`` defaults to ``READ``, i.e. a tainted turn may keep
    reading (so the model can finish answering the question that pulled the
    untrusted content in) but may not write anything.
    """

    def __init__(
        self,
        *,
        max_risk_when_tainted: RiskLevel = RiskLevel.READ,
        message: str = TAINT_DENIAL_MESSAGE,
    ) -> None:
        self._ceiling = RISK_RANK.get(max_risk_when_tainted, 0)
        self._message = message

    def denial_reason(self, spec: ToolSpec, ctx: Any) -> str | None:
        if not is_tainted(ctx):
            return None
        # Fail closed: a risk level this table does not know is treated as the
        # most dangerous one rather than waved through.
        rank = RISK_RANK.get(spec.risk, max(RISK_RANK.values()))
        if rank <= self._ceiling:
            return None
        return f"{self._message} (blocked tool: {spec.name}, risk={spec.risk})"


class NullTaintPolicy(TaintPolicy):
    """Opt-out policy: never denies. Pass explicitly to disable the guard."""

    def denial_reason(self, spec: ToolSpec, ctx: Any) -> str | None:
        return None


def denied_result(call_id: str, reason: str, *, code: str = TAINT_DENIAL_CODE) -> ToolResult:
    """Build the ``status="denied"`` result a gate returns to the model.

    ``retryable=False``: retrying inside this turn will be denied again — the
    user has to restate the action in a new turn.
    """
    return ToolResult(
        call_id=call_id,
        status="denied",
        content=[ContentBlockOut(type="text", text=reason)],
        error=ToolErrorModel(code=code, message=reason, retryable=False),
        duration_ms=0,
        cached=False,
    )
