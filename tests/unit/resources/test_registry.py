import pytest

from agentkit.resources import OpRegistry, OpSpec, Reversibility


def _classify_status(static_kwargs, dynamic_args):
    if "status" in dynamic_args:
        return Reversibility.GATED  # value unknown statically -> conservative
    if static_kwargs.get("status") in {"published", "archived"}:
        return Reversibility.GATED
    return Reversibility.REVERSIBLE


def _reg():
    async def _apply(ctx, **kw):
        return {"ok": True}

    reg = OpRegistry()
    reg.register(OpSpec(name="content.get", apply=_apply, is_read=True))
    reg.register(
        OpSpec(
            name="content.patch",
            apply=_apply,
            subject_type="content",
            patchable=frozenset({"title", "tags", "status"}),
            classify=_classify_status,
        )
    )
    return reg


def test_get_returns_spec():
    assert _reg().get("content.patch").subject_type == "content"


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        _reg().get("content.nope")


def test_classify_read_is_reversible():
    assert _reg().classify("content.get", {}, frozenset()) is Reversibility.REVERSIBLE


def test_classify_static_published_is_gated():
    assert (
        _reg().classify("content.patch", {"status": "published"}, frozenset())
        is Reversibility.GATED
    )


def test_classify_reversible_field():
    assert (
        _reg().classify("content.patch", {"tags": ["x"]}, frozenset()) is Reversibility.REVERSIBLE
    )


def test_classify_dynamic_status_is_gated():
    assert _reg().classify("content.patch", {}, frozenset({"status"})) is Reversibility.GATED


def test_classify_unknown_op_is_gated():
    # An unrecognised mutating call is treated conservatively, never silent.
    assert _reg().classify("content.frobnicate", {}, frozenset()) is Reversibility.GATED
