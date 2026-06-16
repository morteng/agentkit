from agentkit.resources import OpSpec, Reversibility


def test_reversibility_ordering_worst_wins():
    tiers = [Reversibility.REVERSIBLE, Reversibility.GATED, Reversibility.IRREVERSIBLE]
    assert max(tiers, key=lambda t: t.severity) is Reversibility.IRREVERSIBLE
    assert Reversibility.GATED.severity > Reversibility.REVERSIBLE.severity


def test_opspec_is_read_defaults_false():
    async def _apply(ctx, **kw):
        return {"ok": True}

    spec = OpSpec(name="content.patch", apply=_apply)
    assert spec.is_read is False
    assert spec.patchable == frozenset()
