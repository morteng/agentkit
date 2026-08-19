"""A non-idempotent tool runs at most once per turn, however often it is called.

``ToolSpec.idempotent`` was declared from the start and read in exactly one
place — whether a batch is safe to dispatch in parallel. Nothing stopped the
same non-idempotent tool executing twice inside one turn, and the loop produces
that situation on its own: a rejected finalize envelope returns to
``Phase.CONTEXT_BUILD`` with the full tool catalog and a history in which the
write already succeeded, and so does every tool result. One user request became
two allocated external resources, the second silently invalidating the first,
with no error raised anywhere — re-entry is a normal phase transition, not an
error path, so nothing had cause to complain.

Every assertion here is on the boundary: how many times the real handler ran,
and what came back to the caller. None of them read the guard's bookkeeping,
because a ledger with perfect contents proves nothing if the dispatcher never
consults it — which was the whole shape of the original defect.
"""

from typing import Any

from agentkit.audit import AuditRecord, AuditSink
from agentkit.loop.context import TurnContext, from_checkpoint_payload, to_checkpoint_payload
from agentkit.loop.tool_dispatcher import DispatchPolicy, ToolDispatcher
from agentkit.tools.registry import ToolRegistry
from agentkit.tools.spec import (
    ApprovalPolicy,
    ContentBlockOut,
    RiskLevel,
    SideEffects,
    ToolCall,
    ToolResult,
    ToolSpec,
)

PROVISION = "acme.provision_device"


def _spec(
    name: str,
    *,
    idempotent: bool,
    risk: RiskLevel = RiskLevel.HIGH_WRITE,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="d",
        parameters={"type": "object"},
        returns=None,
        risk=risk,
        idempotent=idempotent,
        side_effects=SideEffects.EXTERNAL_IRREVERSIBLE,
        requires_approval=ApprovalPolicy.NEVER,
        cache_ttl_seconds=None,
        timeout_seconds=10.0,
    )


class _CountingHandler:
    """A tool handler that records every execution and numbers its output.

    Numbering matters as much as counting: it is what lets a test tell "the
    second call returned the FIRST result" from "the second call ran and
    happened to return something that looks the same".

    ``__call__`` is bound against the real handler signature the registry
    invokes — positional ``(arguments, ctx)`` — so a dispatcher that called it
    any other way would fail here exactly as it would against a real tool.
    """

    def __init__(self, *, statuses: list[str] | None = None) -> None:
        self.runs: list[dict[str, Any]] = []
        # Per-run status, consumed in order; anything past the end is "ok".
        self._statuses = list(statuses or [])

    async def __call__(self, arguments: dict[str, Any], ctx: TurnContext) -> ToolResult:
        self.runs.append(dict(arguments))
        n = len(self.runs)
        status = self._statuses.pop(0) if self._statuses else "ok"
        return ToolResult(
            call_id=ctx.call_id,
            status=status,  # type: ignore[arg-type]
            content=[ContentBlockOut(type="text", text=f"run-{n}")],
            duration_ms=1,
        )

    @property
    def call_count(self) -> int:
        return len(self.runs)


class _CountingRegistry(ToolRegistry):
    """Counts dispatches that reach the registry at all.

    The observable for "this call was NOT suppressed" when the call has no
    handler to count — an unknown tool name, where the point is that the
    registry still gets to produce its own unknown-tool error.
    """

    def __init__(self) -> None:
        super().__init__()
        self.invocations: list[str] = []

    async def invoke(self, call: ToolCall, ctx: Any) -> ToolResult:
        self.invocations.append(call.name)
        return await super().invoke(call, ctx)


class _Recorder(AuditSink):
    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    async def record(self, record: AuditRecord) -> None:
        self.records.append(record)


def _dispatcher(
    reg: ToolRegistry,
    *,
    guard: bool = True,
    audit: AuditSink | None = None,
) -> ToolDispatcher:
    return ToolDispatcher(
        registry=reg,
        policy=DispatchPolicy(max_parallel=8, guard_nonidempotent_replay=guard),
        audit=audit,
    )


def _text(result: ToolResult) -> str:
    return "\n".join(b.text or "" for b in result.content)


async def _call(
    disp: ToolDispatcher,
    ctx: TurnContext,
    *,
    call_id: str,
    name: str = PROVISION,
    arguments: dict[str, Any] | None = None,
) -> ToolResult:
    """One dispatch, one result — the shape a loop iteration actually produces."""
    results = await disp.run(
        [ToolCall(id=call_id, name=name, arguments=arguments or {})],
        ctx,
    )
    assert len(results) == 1
    return results[0]


# ---------------------------------------------------------------------------
# The defect itself
# ---------------------------------------------------------------------------


async def test_repeat_of_a_non_idempotent_call_runs_the_handler_once():
    """The reported bug, reduced: two dispatches, one real effect.

    The two dispatches stand in for the two sides of a CONTEXT_BUILD re-entry —
    a fresh model call issuing the same tool call against the same turn.
    """
    reg = ToolRegistry()
    handler = _CountingHandler()
    reg.register_builtin(_spec(PROVISION, idempotent=False), handler)
    disp = _dispatcher(reg)
    ctx = TurnContext.empty()
    args = {"user": "u-1", "label": "laptop"}

    first = await _call(disp, ctx, call_id="c1", arguments=args)
    second = await _call(disp, ctx, call_id="c2", arguments=args)

    assert handler.call_count == 1, "the second dispatch must not reach the handler"
    assert _text(first) == "run-1"
    # The model gets the effect it already earned, not a bare refusal.
    assert "run-1" in _text(second)
    # ...and it is told plainly that the effect already happened, so it does not
    # read a missing result as a failure and go looking for a third route.
    assert "already ran" in _text(second)
    # The result must answer the call the model just made, not the earlier one:
    # a stale id is one the provider never issued for this message.
    assert second.call_id == "c2"


async def test_argument_key_order_does_not_defeat_the_guard():
    """Same effect, different JSON key order — still one execution.

    A model re-emitting a call after a correction routinely reorders its
    arguments. An identity sensitive to key order would miss precisely the
    repeats this guard exists to catch.
    """
    reg = ToolRegistry()
    handler = _CountingHandler()
    reg.register_builtin(_spec(PROVISION, idempotent=False), handler)
    disp = _dispatcher(reg)
    ctx = TurnContext.empty()

    await _call(disp, ctx, call_id="c1", arguments={"user": "u-1", "label": "laptop"})
    second = await _call(disp, ctx, call_id="c2", arguments={"label": "laptop", "user": "u-1"})

    assert handler.call_count == 1
    assert "run-1" in _text(second)


# ---------------------------------------------------------------------------
# What the guard must NOT do
# ---------------------------------------------------------------------------


async def test_different_arguments_are_a_different_effect_and_execute():
    reg = ToolRegistry()
    handler = _CountingHandler()
    reg.register_builtin(_spec(PROVISION, idempotent=False), handler)
    disp = _dispatcher(reg)
    ctx = TurnContext.empty()

    await _call(disp, ctx, call_id="c1", arguments={"user": "u-1"})
    second = await _call(disp, ctx, call_id="c2", arguments={"user": "u-2"})

    assert handler.call_count == 2
    assert handler.runs == [{"user": "u-1"}, {"user": "u-2"}]
    assert _text(second) == "run-2"


async def test_idempotent_tool_is_never_guarded():
    """Repeating an idempotent call is safe by declaration — and often the point.

    A model re-reads a resource to see whether its own write landed. Answering
    that second read from a store would hide exactly the change it re-read to
    find.
    """
    reg = ToolRegistry()
    handler = _CountingHandler()
    reg.register_builtin(_spec("acme.list_devices", idempotent=True, risk=RiskLevel.READ), handler)
    disp = _dispatcher(reg)
    ctx = TurnContext.empty()

    await _call(disp, ctx, call_id="c1", name="acme.list_devices", arguments={"user": "u-1"})
    second = await _call(
        disp, ctx, call_id="c2", name="acme.list_devices", arguments={"user": "u-1"}
    )

    assert handler.call_count == 2
    assert _text(second) == "run-2"


async def test_a_failed_call_is_not_recorded_so_the_retry_executes():
    """A call that failed consumed nothing, and the model may retry it.

    Recording failures would convert one transient error into a permanent one
    for the rest of the turn: every retry would come back as the original
    failure, with the tool never touched again.
    """
    reg = ToolRegistry()
    handler = _CountingHandler(statuses=["error"])
    reg.register_builtin(_spec(PROVISION, idempotent=False), handler)
    disp = _dispatcher(reg)
    ctx = TurnContext.empty()
    args = {"user": "u-1"}

    first = await _call(disp, ctx, call_id="c1", arguments=args)
    second = await _call(disp, ctx, call_id="c2", arguments=args)

    assert first.status == "error"
    assert handler.call_count == 2, "a failed call must not be recorded"
    assert second.status == "ok"
    assert _text(second) == "run-2"


async def test_a_denied_call_is_not_recorded_either():
    """Same reasoning as a failure: denial consumed nothing.

    Worth its own case because ``denied`` is not ``error`` — a status test
    written as ``!= "error"`` would pass the failure test above and still
    record this one, permanently locking out a call the user is about to
    approve.
    """
    reg = ToolRegistry()
    handler = _CountingHandler(statuses=["denied"])
    reg.register_builtin(_spec(PROVISION, idempotent=False), handler)
    disp = _dispatcher(reg)
    ctx = TurnContext.empty()
    args = {"user": "u-1"}

    await _call(disp, ctx, call_id="c1", arguments=args)
    second = await _call(disp, ctx, call_id="c2", arguments=args)

    assert handler.call_count == 2
    assert second.status == "ok"


async def test_an_unknown_tool_is_not_guarded():
    """No spec means no declared semantics to guard by.

    Suppressing here would swallow ``registry.invoke``'s unknown-tool error,
    which is the only thing that tells the model the name does not exist — and
    the guard would be deciding policy for a tool nobody described.
    """
    reg = _CountingRegistry()  # 'ghost' is registered nowhere
    disp = _dispatcher(reg)
    ctx = TurnContext.empty()

    first = await _call(disp, ctx, call_id="c1", name="ghost", arguments={"x": 1})
    second = await _call(disp, ctx, call_id="c2", name="ghost", arguments={"x": 1})

    assert reg.invocations == ["ghost", "ghost"], "both calls must reach the registry"
    assert first.status == "error"
    assert second.status == "error"
    assert second.error is not None and second.error.code == "unknown_tool"
    assert second.call_id == "c2"


# ---------------------------------------------------------------------------
# Scope: exactly one turn
# ---------------------------------------------------------------------------


async def test_the_guard_does_not_leak_into_a_later_turn():
    """A second turn is the principal asking again, and that must run.

    This is the property the whole storage choice rests on. A ledger that
    outlived its turn would refuse a legitimate repeat next time the user asked
    for the same thing — a far worse failure than the one being fixed, because
    it is silent and permanent.
    """
    reg = ToolRegistry()
    handler = _CountingHandler()
    reg.register_builtin(_spec(PROVISION, idempotent=False), handler)
    disp = _dispatcher(reg)
    args = {"user": "u-1"}

    turn_one = TurnContext.empty()
    await _call(disp, turn_one, call_id="c1", arguments=args)
    await _call(disp, turn_one, call_id="c2", arguments=args)

    turn_two = TurnContext.empty()
    result = await _call(disp, turn_two, call_id="c3", arguments=args)

    assert handler.call_count == 2, "once in turn one, once in turn two"
    assert _text(result) == "run-2"


async def test_the_guard_survives_an_approval_suspend_and_resume():
    """A resumed turn is the SAME turn, so the ledger has to come back with it.

    ``to_checkpoint_payload`` serialises metadata with
    ``json.dumps(..., default=str)``, which flattens anything it cannot encode
    into a repr string. Storing the ledger as JSON-safe dicts is what makes the
    round-trip lossless; this test is what would notice if that changed.
    """
    reg = ToolRegistry()
    handler = _CountingHandler()
    reg.register_builtin(_spec(PROVISION, idempotent=False), handler)
    disp = _dispatcher(reg)
    args = {"user": "u-1"}

    before = TurnContext.empty()
    await _call(disp, before, call_id="c1", arguments=args)

    data = from_checkpoint_payload(to_checkpoint_payload(before))
    # Rebuilt the way AgentSession.resume_approval does it: same turn_id, and
    # the persisted metadata merged in.
    after = TurnContext(
        session_id=before.session_id,
        turn_id=before.turn_id,
        call_id="",
    )
    after.metadata.update(data.get("metadata", {}))

    resumed = await _call(disp, after, call_id="c2", arguments=args)

    assert handler.call_count == 1
    assert "run-1" in _text(resumed)
    assert resumed.call_id == "c2"


# ---------------------------------------------------------------------------
# Observability and the off switch
# ---------------------------------------------------------------------------


async def test_a_suppressed_repeat_is_audited_and_flagged():
    """The request was still made, so a row is still owed — marked as suppressed.

    Without the flag the row is indistinguishable from a real second execution,
    which is the exact question an auditor of a non-idempotent tool is asking.
    """
    reg = ToolRegistry()
    reg.register_builtin(_spec(PROVISION, idempotent=False), _CountingHandler())
    audit = _Recorder()
    disp = _dispatcher(reg, audit=audit)
    ctx = TurnContext.empty()
    args = {"user": "u-1"}

    await _call(disp, ctx, call_id="c1", arguments=args)
    await _call(disp, ctx, call_id="c2", arguments=args)

    assert [r.call_id for r in audit.records] == ["c1", "c2"]
    assert audit.records[0].detail["replayed"] is False
    assert audit.records[1].detail["replayed"] is True


async def test_disabling_the_guard_restores_the_old_behaviour():
    reg = ToolRegistry()
    handler = _CountingHandler()
    reg.register_builtin(_spec(PROVISION, idempotent=False), handler)
    disp = _dispatcher(reg, guard=False)
    ctx = TurnContext.empty()
    args = {"user": "u-1"}

    await _call(disp, ctx, call_id="c1", arguments=args)
    second = await _call(disp, ctx, call_id="c2", arguments=args)

    assert handler.call_count == 2
    assert _text(second) == "run-2"


async def test_a_context_without_metadata_still_dispatches():
    """A context with nowhere to keep a ledger must execute, not fail.

    Subagent stubs and other minimal contexts reach the dispatcher; the rest of
    the module already tolerates them (``_audit_record`` reads metadata through
    ``getattr``). Cannot-guard means execute, never raise.
    """

    class _NoMetadataCtx:
        call_id = ""
        tainted = False

    reg = ToolRegistry()
    handler = _CountingHandler()
    reg.register_builtin(_spec(PROVISION, idempotent=False), handler)
    disp = _dispatcher(reg)
    ctx = _NoMetadataCtx()
    args = {"user": "u-1"}

    await disp.run([ToolCall(id="c1", name=PROVISION, arguments=args)], ctx)
    results = await disp.run([ToolCall(id="c2", name=PROVISION, arguments=args)], ctx)

    assert handler.call_count == 2
    assert results[0].status == "ok"


async def test_a_duplicate_inside_one_batch_runs_once():
    """The model can also emit the same call twice in a single message.

    Non-idempotent calls never take the parallel path (``_safe_for_parallel``
    requires READ + idempotent of every call in the batch), so the two arrive
    at the funnel in order and the second sees the first's record.
    """
    reg = ToolRegistry()
    handler = _CountingHandler()
    reg.register_builtin(_spec(PROVISION, idempotent=False), handler)
    disp = _dispatcher(reg)
    ctx = TurnContext.empty()
    args = {"user": "u-1"}

    results = await disp.run(
        [
            ToolCall(id="c1", name=PROVISION, arguments=args),
            ToolCall(id="c2", name=PROVISION, arguments=args),
        ],
        ctx,
    )

    assert handler.call_count == 1
    assert [r.call_id for r in results] == ["c1", "c2"]
    assert "run-1" in _text(results[1])
