"""ResourceNamespace.__getattr__ — declared non-CRUD verb dispatch."""

import pytest

from agentkit.resources import OpRegistry, OpSpec, ResourceNamespace, Reversibility


def _ns(registry, records):
    async def recorder(spec, kwargs, before, after, inverse):
        records.append((spec.name, kwargs, before, after, inverse))

    return ResourceNamespace(
        "kb", registry, ctx=object(), recorder=recorder, op_charge=lambda: None
    )


def test_unknown_verb_raises_attribute_error():
    ns = _ns(OpRegistry(), [])
    with pytest.raises(AttributeError):
        _ = ns.nonexistent
    with pytest.raises(AttributeError):
        _ = ns._private


async def test_declared_read_verb_calls_apply_no_record():
    async def geocode(ctx, *, address):
        return {"lat": 1.0, "lng": 2.0, "address": address}

    reg = OpRegistry()
    reg.register(OpSpec(name="kb.geocode", apply=geocode, is_read=True))
    records: list = []
    ns = _ns(reg, records)
    out = await ns.geocode(address="Oslo")
    assert out == {"lat": 1.0, "lng": 2.0, "address": "Oslo"}
    assert records == []  # reads do not record


async def test_declared_write_verb_records_with_inverse():
    async def cite(ctx, *, content_id, fact_id):
        return {"citation_id": "c1", "content_id": content_id, "fact_id": fact_id}

    def inverse(kwargs, before, after):
        return {"op": "remove_citation", "citation_id": after["citation_id"]}

    reg = OpRegistry()
    reg.register(
        OpSpec(
            name="kb.cite",
            apply=cite,
            subject_type="content",
            inverse=inverse,
            classify=lambda s, d: Reversibility.REVERSIBLE,
        )
    )
    records: list = []
    ns = _ns(reg, records)
    out = await ns.cite(content_id="x", fact_id="y")
    assert out["citation_id"] == "c1"
    assert len(records) == 1
    name, kwargs, _before, _after, inv = records[0]
    assert name == "kb.cite"
    assert kwargs == {"content_id": "x", "fact_id": "y"}
    assert inv == {"op": "remove_citation", "citation_id": "c1"}


async def test_declared_write_skips_patch_field_whitelist():
    # A declared verb carries a structured payload that is NOT in any patchable
    # set; it must pass through without the patch/create whitelist rejecting it.
    async def link(ctx, *, source_id, target_id, link_type):
        return {"link_id": "L1"}

    reg = OpRegistry()
    reg.register(
        OpSpec(
            name="kb.link",
            apply=link,
            classify=lambda s, d: Reversibility.REVERSIBLE,
        )
    )
    ns = _ns(reg, [])
    out = await ns.link(source_id="a", target_id="b", link_type="describes")
    assert out == {"link_id": "L1"}


async def test_charge_called_on_declared_write():
    async def w(ctx, **kw):
        return {"ok": True}

    reg = OpRegistry()
    reg.register(OpSpec(name="kb.w", apply=w, classify=lambda s, d: Reversibility.REVERSIBLE))
    charges = []
    ns = ResourceNamespace(
        "kb",
        reg,
        ctx=object(),
        recorder=_noop_recorder,
        op_charge=lambda: charges.append(1),
    )
    await ns.w(x=1)
    assert charges == [1]


async def _noop_recorder(spec, kwargs, before, after, inverse):
    pass
