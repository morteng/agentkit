from agentkit.resources import OpSpec, Param, Reversibility


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


async def _noop(ctx):  # apply stub
    return {}


def test_param_defaults():
    p = Param(type="string", description="Content ID", required=True, alias="content_id")
    assert p.type == "string"
    assert p.alias == "content_id"
    assert p.required is True
    assert p.enum is None


def test_opspec_carries_flat_alias_and_params():
    spec = OpSpec(
        name="content.get",
        apply=_noop,
        is_read=True,
        description="Get a content item.",
        flat_alias="get_content",
        params={"id": Param(description="Content ID", required=True, alias="content_id")},
    )
    assert spec.flat_alias == "get_content"
    assert spec.params["id"].alias == "content_id"
    assert spec.description == "Get a content item."


def test_opspec_defaults_unchanged():
    spec = OpSpec(name="content.search", apply=_noop, is_read=True)
    assert spec.flat_alias is None
    assert spec.params == {}
    assert spec.description == ""
