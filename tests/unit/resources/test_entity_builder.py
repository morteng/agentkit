"""build_crud_specs — generic CRUD OpSpec builder + snapshot/inverse round-trip."""

from agentkit.resources import EntitySpec, Reversibility, build_crud_specs


class _View:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Ctx:
    class _DB:
        def expire(self, _row):
            pass

    db = _DB()


def _spec(rows=None):
    rows = rows or {"1": _View(id="1", name="Old", city="A")}

    async def load(ctx, id):
        if str(id) not in rows:
            raise ValueError("not found")
        return rows[str(id)]

    async def list_(ctx, query="", **filters):
        return list(rows.values())

    async def patch_adapter(ctx, row, fields):
        for k, v in fields.items():
            setattr(row, k, v)

    async def soft_delete(ctx, row):
        row.deleted = True

    return EntitySpec(
        resource="loc",
        subject_type="location",
        patchable=frozenset({"name", "city"}),
        view=lambda r: _View(id=r.id, name=r.name, city=r.city),
        load=load,
        list_=list_,
        patch_adapter=patch_adapter,
        soft_delete=soft_delete,
        snapshot_fields=frozenset({"name", "city"}),
        classify=lambda s, d: Reversibility.REVERSIBLE,
        delete_classify=lambda s, d: Reversibility.GATED,
    )


def test_emits_four_named_crud_specs():
    specs = {s.name: s for s in build_crud_specs(_spec())}
    assert set(specs) == {"loc.get", "loc.search", "loc.patch", "loc.delete"}
    assert specs["loc.get"].is_read and specs["loc.search"].is_read
    assert not specs["loc.patch"].is_read
    patch_cls, delete_cls = specs["loc.patch"].classify, specs["loc.delete"].classify
    assert patch_cls and delete_cls
    assert patch_cls({}, frozenset()) is Reversibility.REVERSIBLE
    assert delete_cls({}, frozenset()) is Reversibility.GATED
    assert specs["loc.delete"].action_kind == "soft_delete"
    assert specs["loc.patch"].patchable == frozenset({"name", "city"})


async def test_get_and_search_views():
    specs = {s.name: s for s in build_crud_specs(_spec())}
    got = await specs["loc.get"].apply(_Ctx(), id="1")
    assert got == {"id": "1", "name": "Old", "city": "A"}
    found = await specs["loc.search"].apply(_Ctx(), query="")
    assert found == [{"id": "1", "name": "Old", "city": "A"}]


async def test_patch_reloads_view_and_snapshot_inverse_round_trip():
    specs = {s.name: s for s in build_crud_specs(_spec())}
    patch = specs["loc.patch"]
    assert patch.snapshot and patch.inverse
    before = await patch.snapshot(_Ctx(), id="1", name="New")
    assert before == {"name": "Old", "city": "A"}
    after = await patch.apply(_Ctx(), id="1", name="New")
    assert after["name"] == "New"
    # inverse restores only the touched field that was snapshotted
    inv = patch.inverse({"id": "1", "name": "New"}, before, after)
    assert inv == {"op": "loc.patch", "id": "1", "fields": {"name": "Old"}}


def test_inverse_patch_none_when_no_before():
    patch = {s.name: s for s in build_crud_specs(_spec())}["loc.patch"]
    assert patch.inverse
    assert patch.inverse({"id": "1", "name": "New"}, None, {}) is None


async def test_snapshot_returns_none_for_missing_row():
    patch = {s.name: s for s in build_crud_specs(_spec())}["loc.patch"]
    assert patch.snapshot
    assert await patch.snapshot(_Ctx(), id="missing") is None


async def test_delete_inverse_shape():
    specs = {s.name: s for s in build_crud_specs(_spec())}
    delete = specs["loc.delete"]
    assert delete.inverse
    out = await delete.apply(_Ctx(), id="1")
    assert out == {"id": "1", "deleted": True}
    inv = delete.inverse({"id": "1"}, {"name": "Old"}, out)
    assert inv == {"op": "loc.restore", "id": "1"}


def test_search_view_override_used_for_list():
    spec = _spec()
    spec.search_view = lambda r: _View(id=r.id, lean=True)
    specs = {s.name: s for s in build_crud_specs(spec)}
    assert specs["loc.search"].name == "loc.search"
    # exercised via apply in test below; here assert the spec wired distinctly
    assert spec.search_view is not spec.view


async def test_search_view_override_projects_list_rows():
    spec = _spec()
    spec.search_view = lambda r: _View(id=r.id, lean=True)
    search = {s.name: s for s in build_crud_specs(spec)}["loc.search"]
    found = await search.apply(_Ctx(), query="")
    assert found == [{"id": "1", "lean": True}]
