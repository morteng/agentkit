from agentkit.resources.entity import EntitySpec, build_crud_specs
from agentkit.resources.types import Param


async def _load(ctx, id):
    return {"id": id}


async def _list(ctx, *, query="", **f):
    return []


async def _patch(ctx, row, fields):
    return None


async def _del(ctx, row):
    return None


async def _create(ctx, **f):
    return {"id": "new"}


def _spec() -> EntitySpec:
    return EntitySpec(
        resource="content",
        subject_type="content",
        patchable=frozenset({"title", "body"}),
        creatable=frozenset({"title", "body"}),
        view=lambda r: r,
        load=_load,
        list_=_list,
        patch_adapter=_patch,
        soft_delete=_del,
        snapshot_fields=frozenset({"title"}),
        create_adapter=_create,
        id_param=Param(description="Content ID", required=True, alias="content_id"),
        field_params={
            "title": Param(description="Title", required=True),
            "body": Param(description="Body"),
        },
        flat_aliases={"get": "get_content", "create": "create_content", "patch": "update_draft"},
        descriptions={"get": "Get a content item.", "create": "Create a draft."},
    )


def _by_name(specs):
    return {s.name: s for s in specs}


def test_crud_specs_carry_flat_alias_and_params():
    specs = _by_name(build_crud_specs(_spec()))
    get = specs["content.get"]
    assert get.flat_alias == "get_content"
    assert get.description == "Get a content item."
    assert get.params["id"].alias == "content_id"

    create = specs["content.create"]
    assert create.flat_alias == "create_content"
    assert set(create.params) == {"title", "body"}
    assert create.params["title"].required is True

    patch = specs["content.patch"]
    assert patch.flat_alias == "update_draft"
    assert set(patch.params) == {"id", "title", "body"}  # id + patchable fields


def test_verbs_without_alias_stay_script_only():
    specs = _by_name(build_crud_specs(_spec()))
    assert specs["content.delete"].flat_alias is None  # no entry in flat_aliases
    assert specs["content.search"].flat_alias is None
