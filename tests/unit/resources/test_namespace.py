import pytest

from agentkit.resources import OpRegistry, OpSpec, ResourceNamespace, Reversibility


def _classify(static_kwargs, dynamic_args):
    return Reversibility.REVERSIBLE


def _build(records, charges):
    async def _patch_apply(ctx, *, id, **fields):
        return {"id": id, **fields}

    async def _get_apply(ctx, *, id):
        return {"id": id, "title": "T"}

    async def _snapshot(ctx, *, id, **fields):
        return {"id": id, "old": True}

    def _inverse(kwargs, before, after):
        return {"op": "patch", "id": kwargs["id"], "restore": before}

    reg = OpRegistry()
    reg.register(OpSpec(name="content.get", apply=_get_apply, is_read=True))
    reg.register(
        OpSpec(
            name="content.patch",
            apply=_patch_apply,
            subject_type="content",
            patchable=frozenset({"title", "tags"}),
            snapshot=_snapshot,
            inverse=_inverse,
            classify=_classify,
        )
    )

    async def recorder(spec, kwargs, before, after, inverse):
        records.append((spec.name, kwargs, before, after, inverse))

    def charge():
        charges.append(1)

    return ResourceNamespace("content", reg, ctx=object(), recorder=recorder, op_charge=charge)


async def test_read_skips_recorder_and_charge():
    records, charges = [], []
    ns = _build(records, charges)
    out = await ns.get("c1")
    assert out == {"id": "c1", "title": "T"}
    assert records == []
    assert charges == []


async def test_patch_records_with_inverse_and_charges():
    records, charges = [], []
    ns = _build(records, charges)
    out = await ns.patch("c1", title="New")
    assert out == {"id": "c1", "title": "New"}
    assert charges == [1]
    assert len(records) == 1
    name, _kwargs, before, after, inverse = records[0]
    assert name == "content.patch"
    assert before == {"id": "c1", "old": True}
    assert after == {"id": "c1", "title": "New"}
    assert inverse == {"op": "patch", "id": "c1", "restore": {"id": "c1", "old": True}}


async def test_patch_rejects_unknown_field():
    records, charges = [], []
    ns = _build(records, charges)
    with pytest.raises(ValueError, match="not patchable"):
        await ns.patch("c1", bogus=1)
    assert records == []


async def test_unknown_verb_raises():
    records, charges = [], []
    ns = _build(records, charges)
    with pytest.raises(KeyError):
        await ns.create(title="x")  # no content.create spec registered
