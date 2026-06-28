from agentkit.resources.namespace import ResourceNamespace
from agentkit.resources.registry import OpRegistry
from agentkit.resources.types import OpSpec, Param


class _Ctx:
    pass


async def _noop_recorder(spec, kwargs, before, after, inverse):
    return None


def _registry() -> OpRegistry:
    reg = OpRegistry()

    async def _get(ctx, *, id):
        return {"id": id}

    async def _link(ctx, *, content_id, location_id):
        return {"content_id": content_id, "location_id": location_id}

    reg.register(
        OpSpec(
            name="content.get",
            apply=_get,
            is_read=True,
            params={"id": Param(alias="content_id", required=True)},
        )
    )
    reg.register(
        OpSpec(
            name="content.link_location",
            apply=_link,
            subject_type="content",
            params={"content_id": Param(required=True), "location_id": Param(required=True)},
        )
    )
    return reg


def _ns() -> ResourceNamespace:
    return ResourceNamespace(
        "content",
        _registry(),
        ctx=_Ctx(),
        recorder=_noop_recorder,
        op_charge=lambda: None,
    )


async def test_declared_verb_accepts_positional_args():
    ns = _ns()
    out = await ns.link_location("c1", "l1")  # positional, natural Python
    assert out == {"content_id": "c1", "location_id": "l1"}


async def test_get_accepts_id_alias():
    ns = _ns()
    out = await ns.get(content_id="c1")  # alias instead of id
    assert out == {"id": "c1"}


async def test_get_still_accepts_plain_id():
    ns = _ns()
    out = await ns.get("c1")
    assert out == {"id": "c1"}
