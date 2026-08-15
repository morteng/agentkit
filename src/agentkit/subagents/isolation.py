"""Isolation primitives for subagents: fresh context, restricted tools, no approvals.

A child subagent gets its own session_id, turn_id, history, and event queue —
its operations don't leak into the parent's view. This module also holds the
three things that make that isolation a *security* boundary rather than a
bookkeeping one:

* :data:`RESERVED_CONTEXT_KEYS` — metadata a spawn request may never carry,
  because the model composes the spawn request.
* :class:`RestrictedToolRegistry` — a default-deny registry *view* so the
  ``allowed_tools`` list is enforced instead of merely recorded, and a child
  can never see or invoke a tool its parent could not.
* :class:`NestedApprovalGate` — a subagent cannot suspend for human approval
  (nothing would ever resume it), so a call that needs one is denied loudly
  instead of vanishing.
"""

from collections.abc import Collection, Sequence
from typing import Any, cast

from agentkit._content import TextBlock
from agentkit._ids import MessageId, SessionId, TurnId, new_id
from agentkit._logging import get_logger
from agentkit._messages import Message, MessageRole
from agentkit.errors import ToolError as ToolErr
from agentkit.guards.approval import ApprovalDecision, ApprovalGate, RiskBasedApprovalGate
from agentkit.guards.taint import denied_result
from agentkit.loop.context import TurnContext
from agentkit.tools.registry import ToolAuthorizer, ToolRegistry
from agentkit.tools.spec import ToolCall, ToolResult, ToolSpec

log = get_logger(__name__)


#: Metadata keys a spawn request must never be able to set. Every one of them
#: is a security decision the *runtime* makes about the child: how deep the
#: recursion has gone, who the child acts as, which tools it may reach, which
#: capability tiers apply. ``kit.subagent.spawn``'s ``context`` argument is
#: model-authored, so these are hard-rejected rather than merely overwritten
#: after the merge — an ordering bug is one refactor away, and a rejection is
#: visible where a silent overwrite is not.
RESERVED_CONTEXT_KEYS = frozenset(
    {
        "subagent_depth",
        "owner",
        "allowed_tools",
        "tier_overrides",
        "capabilities",
        "discovered_tools",
    }
)

#: Bare tool-name suffixes that are loop control rather than capability — the
#: same convention ``loop.handlers.streaming`` and the finalize validator use.
#: A child always keeps these regardless of its allowlist: without a finalize
#: tool the child cannot end its turn cleanly, and the model would burn its
#: reprompt budget on a tool it was never given.
_LOOP_CONTROL_BARE_NAMES = frozenset({"finalize_response", "finalize"})

_NOT_PERMITTED_TEMPLATE = (
    "denied: {name} is not in this subagent's allowed_tools. A subagent may "
    "only use the tools it was spawned with. Report what you need back to the "
    "parent agent instead of retrying."
)

#: Recorded on the child context (``subagent_denied_approvals``) for every call
#: :class:`NestedApprovalGate` turned into an auto-deny.
NESTED_APPROVAL_DENIAL_REASON = (
    "subagent required approval; not permitted in nested context — "
    "there is no user attached to a subagent turn to answer it"
)


def _metadata_of(ctx: Any) -> dict[str, Any] | None:
    """``ctx.metadata`` when it is a dict, else ``None``.

    ``ctx`` is typed loosely across the guard/authorizer protocols (to avoid a
    circular import), so both call sites need the same defensive read.
    """
    metadata = getattr(ctx, "metadata", None)
    return cast("dict[str, Any]", metadata) if isinstance(metadata, dict) else None


def is_loop_control_tool(name: str) -> bool:
    """True for the finalize tool, whatever namespace it is registered under."""
    return name.split(".", 1)[-1] in _LOOP_CONTROL_BARE_NAMES


def fresh_child_context(parent: TurnContext, *, prompt: str) -> TurnContext:
    # Local import for readability; no cycle risk.
    from datetime import UTC, datetime  # noqa: PLC0415

    child_session = new_id(SessionId)
    user_msg = Message(
        id=new_id(MessageId),
        session_id=child_session,
        role=MessageRole.USER,
        content=[TextBlock(text=prompt)],
        created_at=datetime.now(UTC),
    )
    child = TurnContext(
        session_id=child_session,
        turn_id=new_id(TurnId),
        call_id="",
        history=[user_msg],
        clock=parent.clock,
        memory_store=parent.memory_store,
        memory_scope=parent.memory_scope,
    )
    # Taint latches downward: a parent turn that already ingested untrusted
    # content must not be able to launder it through a fresh child whose
    # ``tainted`` flag starts clean. The provenance list has to travel with
    # it — a child that knows it is tainted but not why defeats provenance
    # reporting — copied rather than shared so the child mutating its own
    # list (e.g. ingesting more untrusted content) cannot corrupt the
    # parent's record.
    child.tainted = parent.tainted
    child.taint_sources = list(parent.taint_sources)
    return child


def reserved_keys_in(context: dict[str, Any]) -> list[str]:
    """The :data:`RESERVED_CONTEXT_KEYS` present in ``context``, sorted."""
    return sorted(RESERVED_CONTEXT_KEYS & context.keys())


def effective_allowlist(parent_registry: ToolRegistry, requested: Sequence[str]) -> frozenset[str]:
    """Resolve a spawn's ``tools`` argument against what the parent actually has.

    The result is an intersection, never a union: a child's effective tool set
    can only shrink relative to its parent's. When the parent is itself a
    subagent its registry is already a :class:`RestrictedToolRegistry`, so
    nesting composes — a grandchild cannot name a tool its parent lost.

    Names the parent does not have are dropped (and logged) rather than
    raising: the model authored this list, and a hallucinated tool name should
    narrow the child, not kill the turn.
    """
    available = {spec.name for spec in parent_registry.list_specs()}
    allowed = {name for name in requested if name in available}
    unknown = [name for name in requested if name not in available]
    if unknown:
        log.warning(
            "subagent_allowlist_dropped_unknown_tools",
            requested=list(requested),
            dropped=unknown,
            hint="names not registered on the parent registry cannot be granted to a child",
        )
    allowed |= {name for name in available if is_loop_control_tool(name)}
    return frozenset(allowed)


class RestrictedToolRegistry(ToolRegistry):
    """Default-deny *view* of another registry, limited to ``allowed``.

    Subclasses :class:`ToolRegistry` for type compatibility with the handlers,
    but owns no tools of its own: every read and every invocation is delegated
    to ``inner`` after the allowlist check. Three properties matter:

    * **Not advertised.** :meth:`list_specs` returns only allowed specs, so a
      disallowed tool never reaches the provider request.
    * **Not resolvable.** :meth:`spec_for` returns ``None`` for a disallowed
      name, so a model that names one out of thin air gets the ordinary
      unknown-tool result and no signal that the tool exists elsewhere.
    * **Not invocable.** :meth:`invoke` denies before delegating, so a handler
      or dispatcher that reaches the registry directly is still gated. This is
      the layer that makes the allowlist real rather than decorative.

    Lifecycle is deliberately inert: the MCP clients belong to the parent
    session, so :meth:`initialize_mcp_servers` and :meth:`shutdown` are no-ops
    here. A child tearing down its parent's stdio subprocesses on the way out
    would break every subsequent parent turn.
    """

    def __init__(self, *, inner: ToolRegistry, allowed: Collection[str]) -> None:
        super().__init__()
        self._inner = inner
        self._allowed = frozenset(allowed)

    @property
    def inner(self) -> ToolRegistry:
        """The registry this view restricts."""
        return self._inner

    @property
    def allowed_tools(self) -> frozenset[str]:
        """The names this view permits. Everything else is denied."""
        return self._allowed

    def restrict(self, allowed: Collection[str]) -> "RestrictedToolRegistry":
        """A further-restricted view — the intersection, never a widening."""
        return RestrictedToolRegistry(inner=self._inner, allowed=self._allowed & frozenset(allowed))

    # ---- Listing (delegated + filtered) ------------------------------------

    def list_specs(self, *, available_to_provider: bool = True) -> list[ToolSpec]:
        return [
            spec
            for spec in self._inner.list_specs(available_to_provider=available_to_provider)
            if spec.name in self._allowed
        ]

    def spec_for(self, name: str) -> ToolSpec | None:
        if name not in self._allowed:
            return None
        return self._inner.spec_for(name)

    @property
    def failed_servers(self) -> dict[str, str]:
        return self._inner.failed_servers

    # ---- Invocation --------------------------------------------------------

    async def invoke(self, call: ToolCall, ctx: TurnContext) -> ToolResult:
        if call.name not in self._allowed:
            log.warning(
                "subagent_tool_not_permitted",
                tool=call.name,
                session_id=str(ctx.session_id),
                allowed=sorted(self._allowed),
            )
            return denied_result(
                call.id, _NOT_PERMITTED_TEMPLATE.format(name=call.name), code="not_authorized"
            )
        return await self._inner.invoke(call, ctx)

    # ---- Lifecycle / mutation (owned by the parent) ------------------------

    async def initialize_mcp_servers(self) -> None:
        """No-op: the parent session owns MCP client lifecycle."""
        return None

    async def shutdown(self) -> None:
        """No-op: a child must not tear down its parent's MCP clients."""
        return None

    def register_builtin(self, spec: ToolSpec, handler: Any) -> None:
        raise ToolErr("RestrictedToolRegistry is a read-only view; register on the parent registry")

    def register_mcp_server(self, name: str, client: Any) -> None:
        raise ToolErr("RestrictedToolRegistry is a read-only view; register on the parent registry")

    def set_authorizer(self, authorizer: ToolAuthorizer | None) -> None:
        raise ToolErr(
            "RestrictedToolRegistry is a read-only view; set the authorizer on the "
            "parent registry so it applies to parent and child alike"
        )


class SubagentToolAuthorizer:
    """:class:`~agentkit.tools.registry.ToolAuthorizer` for ``allowed_tools``.

    :class:`RestrictedToolRegistry` is the primary enforcement point and is
    wired up automatically by :class:`~agentkit.subagents.dispatcher.SubagentDispatcher`.
    This authorizer is the same rule expressed at the registry's own
    execution-time gate, for deployments that share one registry instance and
    prefer a single choke point:

    .. code-block:: python

        registry.set_authorizer(chain_authorizers(plane_authorizer, SubagentToolAuthorizer()))

    It reads ``ctx.metadata["allowed_tools"]``, which only the dispatcher
    stamps (a spawn's ``context`` argument cannot set it — see
    :data:`RESERVED_CONTEXT_KEYS`). A context without the key is not a
    subagent context and is permitted.
    """

    def authorize(self, spec: ToolSpec, ctx: Any) -> str | None:
        metadata = _metadata_of(ctx)
        if metadata is None:
            return None
        allowed: Collection[str] | None = metadata.get("allowed_tools")
        if allowed is None:
            return None
        if spec.name in allowed or is_loop_control_tool(spec.name):
            return None
        return _NOT_PERMITTED_TEMPLATE.format(name=spec.name)


def chain_authorizers(*authorizers: ToolAuthorizer | None) -> ToolAuthorizer:
    """Combine authorizers into one that denies on the first denial.

    :meth:`ToolRegistry.set_authorizer` takes a single authorizer, so a
    deployment that wants both a ToolPlane visibility gate and
    :class:`SubagentToolAuthorizer` needs them composed. ``None`` entries are
    skipped so callers can pass an optional existing authorizer straight
    through.
    """
    active: list[ToolAuthorizer] = [a for a in authorizers if a is not None]

    class _Chained:
        def authorize(self, spec: ToolSpec, ctx: Any) -> str | None:
            for authorizer in active:
                reason = authorizer.authorize(spec, ctx)
                if reason is not None:
                    return reason
            return None

    return _Chained()


class NestedApprovalGate:
    """Approval gate for a subagent turn: ``NEEDS_USER`` becomes ``AUTO_DENY``.

    A subagent turn has no user attached and no resume path — its checkpoint
    is keyed by a turn id the parent session will never resume, so suspending
    means the call disappears and the child returns whatever text it had
    already produced. Denying instead keeps the outcome visible: the child
    model gets a ``denied`` ToolResult it can report on, and the parent gets
    the list on ``ctx.metadata["subagent_denied_approvals"]``.

    Every other decision (``AUTO_APPROVE``, ``AUTO_DENY``) is passed through
    from ``inner`` unchanged, so a subagent is never *more* permitted than the
    session's own policy allows.
    """

    def __init__(self, inner: ApprovalGate | None = None) -> None:
        self._inner: ApprovalGate = inner if inner is not None else RiskBasedApprovalGate()

    async def decide(self, call: ToolCall, spec: ToolSpec, ctx: Any) -> ApprovalDecision:
        decision = await self._inner.decide(call, spec, ctx)
        if decision is not ApprovalDecision.NEEDS_USER:
            return decision
        log.warning(
            "subagent_approval_auto_denied",
            tool=call.name,
            call_id=call.id,
            risk=str(spec.risk),
            reason=NESTED_APPROVAL_DENIAL_REASON,
        )
        metadata = _metadata_of(ctx)
        if metadata is not None:
            denied: list[dict[str, Any]] = metadata.setdefault("subagent_denied_approvals", [])
            denied.append(
                {"id": call.id, "name": call.name, "reason": NESTED_APPROVAL_DENIAL_REASON}
            )
        return ApprovalDecision.AUTO_DENY
