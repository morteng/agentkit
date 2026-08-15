"""Every dispatched tool call writes exactly one audit record.

The failure this guards against is not "auditing is broken" but "auditing is
partial": a runtime where the torrent tool writes rows and the vault-deletion
tool does not, because the audit callback was a per-client constructor argument
someone forgot on one client.
"""

import pytest

from agentkit.audit import ACTION_TOOL_CALL, SOURCE_AGENT, AuditRecord, AuditSink
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


class _Recorder(AuditSink):
    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    async def record(self, record: AuditRecord) -> None:
        self.records.append(record)


class _Exploding(AuditSink):
    async def record(self, record: AuditRecord) -> None:
        raise RuntimeError("ledger unavailable")


class _Ctx:
    call_id = "c1"
    session_id = "REDACTEDmorten"
    turn_id = "turn-1"
    tainted = False

    def __init__(self) -> None:
        self.metadata: dict[str, object] = {"owner": "u:morten"}


def _spec(
    name: str,
    *,
    risk: RiskLevel = RiskLevel.DESTRUCTIVE,
    side_effects: SideEffects = SideEffects.EXTERNAL_IRREVERSIBLE,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="d",
        parameters={"type": "object"},
        returns=None,
        risk=risk,
        idempotent=False,
        side_effects=side_effects,
        requires_approval=ApprovalPolicy.NEVER,
        cache_ttl_seconds=None,
        timeout_seconds=10.0,
    )


def _registry(name: str, handler=None, **spec_kwargs) -> ToolRegistry:
    reg = ToolRegistry()

    async def _ok(args, ctx):
        return ToolResult(
            call_id=ctx.call_id,
            status="ok",
            content=[ContentBlockOut(type="text", text="done")],
            duration_ms=3,
        )

    reg.register_builtin(_spec(name, **spec_kwargs), handler or _ok)
    return reg


def _dispatcher(reg: ToolRegistry, audit: AuditSink | None) -> ToolDispatcher:
    return ToolDispatcher(registry=reg, policy=DispatchPolicy(max_parallel=8), audit=audit)


@pytest.mark.asyncio
async def test_the_destructive_call_writes_a_record():
    audit = _Recorder()
    reg = _registry("vaultwarden_user_delete")
    await _dispatcher(reg, audit).run(
        [ToolCall(id="c1", name="vaultwarden_user_delete", arguments={"email": "m@REDACTED"})],
        ctx=_Ctx(),
    )

    assert len(audit.records) == 1
    record = audit.records[0]
    assert record.action == ACTION_TOOL_CALL
    assert record.target == "vaultwarden_user_delete"
    assert record.actor == "u:morten"
    assert record.source == SOURCE_AGENT
    assert record.session_id == "REDACTEDmorten"
    assert record.turn_id == "turn-1"
    assert record.call_id == "c1"
    assert record.detail["status"] == "ok"
    assert record.detail["arguments"] == {"email": "m@REDACTED"}


@pytest.mark.asyncio
async def test_the_record_carries_both_classification_axes():
    """A reader must be able to tell a rename from a destruction without
    keeping their own tool table."""
    audit = _Recorder()
    reg = _registry("jellyfin_rename", risk=RiskLevel.HIGH_WRITE, side_effects=SideEffects.LOCAL)
    await _dispatcher(reg, audit).run(
        [ToolCall(id="c1", name="jellyfin_rename", arguments={})], ctx=_Ctx()
    )

    assert audit.records[0].detail["risk"] == "high_write"
    assert audit.records[0].detail["side_effects"] == "local"


@pytest.mark.asyncio
async def test_an_unregistered_tool_is_still_audited_as_unknown():
    """The call that reached no spec is the one an operator most wants to see."""
    audit = _Recorder()
    await _dispatcher(ToolRegistry(), audit).run(
        [ToolCall(id="c1", name="ghost", arguments={})], ctx=_Ctx()
    )

    assert len(audit.records) == 1
    assert audit.records[0].detail["risk"] == "unknown"
    assert audit.records[0].detail["side_effects"] == "unknown"
    assert audit.records[0].detail["status"] == "error"


@pytest.mark.asyncio
async def test_a_handler_that_raises_is_audited_too():
    async def _boom(args, ctx):
        raise RuntimeError("upstream 500")

    audit = _Recorder()
    reg = _registry("torrent_add", _boom)
    await _dispatcher(reg, audit).run(
        [ToolCall(id="c1", name="torrent_add", arguments={})], ctx=_Ctx()
    )

    assert len(audit.records) == 1
    assert audit.records[0].detail["status"] == "error"
    assert audit.records[0].detail["error"]["code"]


@pytest.mark.asyncio
async def test_secret_arguments_never_reach_the_ledger():
    audit = _Recorder()
    reg = _registry("provision")
    await _dispatcher(reg, audit).run(
        [
            ToolCall(
                id="c1",
                name="provision",
                arguments={
                    "user": "REDACTED",
                    "password": "hunter2",
                    "api_token": "sk-live-1234",
                    "options": {"deep": {"nested": True}},
                },
            )
        ],
        ctx=_Ctx(),
    )

    arguments = audit.records[0].detail["arguments"]
    assert arguments["password"] == "***"
    assert arguments["api_token"] == "***"
    assert arguments["user"] == "REDACTED"
    # Nesting is flattened, so a payload cannot ride into the ledger inside an
    # argument value.
    assert arguments["options"] == "[object]"


@pytest.mark.asyncio
async def test_every_call_in_a_batch_gets_its_own_record():
    audit = _Recorder()
    reg = ToolRegistry()

    async def _ok(args, ctx):
        return ToolResult(call_id=ctx.call_id, status="ok", duration_ms=1)

    reg.register_builtin(_spec("a", risk=RiskLevel.READ), _ok)
    reg.register_builtin(_spec("b", risk=RiskLevel.READ), _ok)
    await _dispatcher(reg, audit).run(
        [ToolCall(id="c1", name="a", arguments={}), ToolCall(id="c2", name="b", arguments={})],
        ctx=_Ctx(),
    )

    assert sorted(r.call_id or "" for r in audit.records) == ["c1", "c2"]


@pytest.mark.asyncio
async def test_a_broken_sink_does_not_break_the_tool_call():
    reg = _registry("torrent_add")
    results = await _dispatcher(reg, _Exploding()).run(
        [ToolCall(id="c1", name="torrent_add", arguments={})], ctx=_Ctx()
    )
    assert results[0].status == "ok"


@pytest.mark.asyncio
async def test_no_sink_configured_is_still_a_working_dispatcher():
    reg = _registry("torrent_add")
    results = await _dispatcher(reg, None).run(
        [ToolCall(id="c1", name="torrent_add", arguments={})], ctx=_Ctx()
    )
    assert results[0].status == "ok"
