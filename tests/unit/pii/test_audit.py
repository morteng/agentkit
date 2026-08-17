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


async def test_audit_counts_replacements_not_placeholder_shaped_text(
    detector: FakeDetector, tmap: FakeTokenMap
):
    """``substitutions`` records what the scrubber replaced on THIS request.

    The record used to be built by scanning the outgoing text for anything
    placeholder-shaped, which is a different quantity with two failure modes,
    both observed live: bracketed-capitals literals (torrent scene tags like
    ``[FLAC]``/``[PMEDIA]``) were reported as substitutions of detector kinds
    that do not exist, and placeholders the model emitted on earlier turns —
    replayed verbatim in history — were re-counted on every request, inflating
    a PHONE count to 81. The audit is the record of what left the house;
    driven here through the same wrap_provider seam the live line came from.
    """
    records: list[OutboundAudit] = []
    set_audit_sink(records.append)
    try:
        inner = FakeProvider().script(FakeProvider.text("ok"))
        fw = Firewall(detector, PiiPolicy(require_zdr=False))
        wrapped = wrap_provider(inner, fw, tmap_resolver=lambda req: tmap)
        req = ProviderRequest(
            model="openai/gpt-5.5",
            messages=[
                # Earlier turn, already placeholder-bearing: the model's own
                # output replayed as history, plus scene tags from a search
                # result. Nothing here is scrubbed on this request.
                make_msg(
                    MessageRole.ASSISTANT,
                    TextBlock(text="Fant Album (1987) [FLAC] [PMEDIA] for [PHONE_1] og [EMAIL_1]"),
                ),
                # This turn's new PII: exactly one name replacement happens.
                make_msg(MessageRole.USER, TextBlock(text="Send den til Kari Nordmann")),
            ],
        )
        _ = [ev async for ev in wrapped.stream(req)]
    finally:
        set_audit_sink(None)

    assert len(records) == 1
    rec = records[0]
    assert rec.substitutions == {"NAME": 1}
    assert rec.never_send_hits == 0
