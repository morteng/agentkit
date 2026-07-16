"""Outbound audit: emitted on the scrubbing path with counts + hash, no PII."""

from typing import TYPE_CHECKING

from agentkit._content import TextBlock
from agentkit._messages import MessageRole
from agentkit.pii import set_audit_sink
from agentkit.pii.firewall import Firewall
from agentkit.pii.policy import PiiPolicy
from agentkit.pii.provider import wrap_provider
from agentkit.providers.base import ProviderRequest
from agentkit.providers.fakes import FakeProvider

from .conftest import FakeDetector, FakeTokenMap, make_msg

if TYPE_CHECKING:
    from agentkit.pii.audit import OutboundAudit


async def test_audit_records_counts_and_no_pii(detector: FakeDetector, tmap: FakeTokenMap):
    records: list[OutboundAudit] = []
    set_audit_sink(records.append)
    try:
        inner = FakeProvider().script(FakeProvider.text("ok"))
        fw = Firewall(detector, PiiPolicy(require_zdr=False))
        wrapped = wrap_provider(inner, fw, tmap_resolver=lambda req: tmap)
        req = ProviderRequest(
            model="openai/gpt-5.5",
            messages=[
                make_msg(
                    MessageRole.USER,
                    TextBlock(text="Kari Nordmann kari@example.no fnr 12345678901"),
                )
            ],
        )
        _ = [ev async for ev in wrapped.stream(req)]
    finally:
        set_audit_sink(None)

    assert len(records) == 1
    rec = records[0]
    assert rec.substitutions.get("NAME") == 1
    assert rec.substitutions.get("EMAIL") == 1
    assert rec.never_send_hits == 1
    assert rec.model == "openai/gpt-5.5"
    # Hash only — no raw PII anywhere in the record.
    assert "kari@example.no" not in repr(rec)
    assert "12345678901" not in repr(rec)
    assert len(rec.scrubbed_hash) == 64
