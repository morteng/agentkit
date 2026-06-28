from agentkit.resources.toolgen import op_to_toolspec
from agentkit.resources.types import OpSpec, Param
from agentkit.tools.spec import ApprovalPolicy, RiskLevel, SideEffects


async def _noop(ctx, **kw):
    return {}


def test_script_only_op_emits_no_flat_tool():
    spec = OpSpec(name="content.search", apply=_noop, is_read=True)  # no flat_alias
    assert op_to_toolspec(spec) is None


def test_read_op_generates_read_toolspec_with_alias_property():
    spec = OpSpec(
        name="content.get",
        apply=_noop,
        is_read=True,
        description="Get a content item.",
        flat_alias="get_content",
        params={"id": Param(description="Content ID", required=True, alias="content_id")},
    )
    ts = op_to_toolspec(spec)
    assert ts is not None
    assert ts.name == "get_content"
    assert ts.description == "Get a content item."
    assert ts.risk == RiskLevel.READ
    assert ts.idempotent is True
    assert ts.side_effects == SideEffects.NONE
    assert ts.requires_approval == ApprovalPolicy.NEVER
    # alias is the LLM-facing property name, not the apply key
    assert "content_id" in ts.parameters["properties"]
    assert "id" not in ts.parameters["properties"]
    assert ts.parameters["required"] == ["content_id"]
    assert ts.parameters["properties"]["content_id"]["type"] == "string"


def test_write_op_generates_low_write_toolspec_with_enum_and_array():
    spec = OpSpec(
        name="content.create",
        apply=_noop,
        description="Create a draft.",
        flat_alias="create_content",
        params={
            "title": Param(description="Title", required=True),
            "status": Param(description="Status", enum=["draft", "review"]),
            "tags": Param(type="array", items_type="string", description="Tags"),
        },
    )
    ts = op_to_toolspec(spec)
    assert ts is not None
    assert ts.name == "create_content"
    assert ts.risk == RiskLevel.LOW_WRITE
    assert ts.idempotent is False
    assert ts.requires_approval == ApprovalPolicy.BY_RISK
    assert ts.parameters["properties"]["status"]["enum"] == ["draft", "review"]
    assert ts.parameters["properties"]["tags"] == {
        "type": "array",
        "description": "Tags",
        "items": {"type": "string"},
    }
    assert ts.parameters["required"] == ["title"]
