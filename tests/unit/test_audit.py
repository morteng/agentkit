"""The audit sink itself: defaults, fan-out, and failure containment."""

from datetime import UTC, datetime

import pytest

from agentkit.audit import (
    ACTION_TOOL_CALL,
    SOURCE_AGENT,
    AuditRecord,
    AuditSink,
    LoggingAuditSink,
    MultiAuditSink,
    NullAuditSink,
    record_audit,
)


def _record(action: str = ACTION_TOOL_CALL) -> AuditRecord:
    return AuditRecord(
        ts=datetime.now(UTC),
        actor="u:alice",
        action=action,
        target="vault_user_delete",
    )


class _Recorder(AuditSink):
    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    async def record(self, record: AuditRecord) -> None:
        self.records.append(record)


class _Exploding(AuditSink):
    async def record(self, record: AuditRecord) -> None:
        raise RuntimeError("sink is down")


def test_record_defaults_are_the_ledger_row_shape():
    """The first six fields are what a consumer's audit table already has."""
    record = _record()
    dumped = record.model_dump(mode="json")
    assert set(dumped) == {
        "ts",
        "actor",
        "action",
        "target",
        "detail",
        "source",
        "session_id",
        "turn_id",
        "call_id",
    }
    assert dumped["detail"] == {}
    assert dumped["source"] == SOURCE_AGENT


@pytest.mark.asyncio
async def test_null_sink_writes_nothing_and_is_a_real_sink():
    """A no-op object, not None: the call site is unconditional."""
    assert isinstance(NullAuditSink(), AuditSink)
    await record_audit(NullAuditSink(), _record())


@pytest.mark.asyncio
async def test_multi_sink_fans_out_to_every_sink():
    store, mirror = _Recorder(), _Recorder()
    await MultiAuditSink([store, mirror]).record(_record())
    assert len(store.records) == 1
    assert len(mirror.records) == 1


@pytest.mark.asyncio
async def test_a_failing_mirror_does_not_lose_the_durable_row():
    """The flaky sink is the chat mirror; the store write must still land."""
    store = _Recorder()
    await MultiAuditSink([_Exploding(), store]).record(_record())
    assert len(store.records) == 1


@pytest.mark.asyncio
async def test_record_audit_never_raises():
    """The audited action already happened. Raising here would lose the record
    AND break a turn that otherwise succeeded."""
    await record_audit(_Exploding(), _record())


@pytest.mark.asyncio
async def test_record_audit_tolerates_no_sink_at_all():
    await record_audit(None, _record())


@pytest.mark.asyncio
async def test_logging_sink_accepts_a_full_record():
    await LoggingAuditSink().record(
        AuditRecord(
            ts=datetime.now(UTC),
            actor="u:alice",
            action=ACTION_TOOL_CALL,
            target="t",
            detail={"nested": {"a": 1}},
            session_id="s",
            turn_id="t1",
            call_id="c1",
        )
    )
