"""Outbound-payload audit (egress surface #8 / §8).

Lightweight structured record emitted per scrubbed request: a hash of the
scrubbed form plus per-kind substitution counts. No PII is recorded — only the
scrubbed hash and counts. Consumers can install their own sink via
``set_audit_sink`` (e.g. to write to a ledger); the default logs via structlog.
"""

from collections.abc import Callable

from pydantic import BaseModel, Field

from agentkit._logging import get_logger

_log = get_logger(__name__)


class OutboundAudit(BaseModel):
    """One outbound-scrub audit record."""

    scrubbed_hash: str
    """SHA-256 hex of the concatenated scrubbed text that will leave the boundary."""
    substitutions: dict[str, int] = Field(default_factory=dict)  # type: ignore[reportUnknownVariableType]
    """Per-kind substitution counts (e.g. ``{"EMAIL": 2, "PHONE": 1}``)."""
    never_send_hits: int = 0
    """Count of NEVER_SEND spans dropped (never tokenized/stored)."""
    model: str | None = None


AuditSink = Callable[[OutboundAudit], None]

# One-element holder so ``set_audit_sink`` can rebind without a ``global``.
_sink_holder: list[AuditSink | None] = [None]


def set_audit_sink(sink: AuditSink | None) -> None:
    """Install (or clear, with ``None``) the outbound-audit sink."""
    _sink_holder[0] = sink


def emit_audit(record: OutboundAudit) -> None:
    """Emit an audit record to the installed sink, else to the default logger."""
    sink = _sink_holder[0]
    if sink is not None:
        sink(record)
        return
    _log.info(
        "pii.outbound_audit",
        scrubbed_hash=record.scrubbed_hash,
        substitutions=record.substitutions,
        never_send_hits=record.never_send_hits,
        model=record.model,
    )
