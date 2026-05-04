from agentkit._content import TextBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.loop.message_builder import MessageBuilder
from agentkit.providers.base import SystemBlock, ToolDefinition
from agentkit.tools.spec import (
    ApprovalPolicy,
    RiskLevel,
    SideEffects,
    ToolSpec,
)


def _spec(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="d",
        parameters={"type": "object"},
        returns=None,
        risk=RiskLevel.READ,
        idempotent=True,
        side_effects=SideEffects.NONE,
        requires_approval=ApprovalPolicy.BY_RISK,
        cache_ttl_seconds=None,
        timeout_seconds=10.0,
    )


def _msg(role: MessageRole, text: str) -> Message:
    from datetime import UTC, datetime

    return Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=role,
        content=[TextBlock(text=text)],
        created_at=datetime.now(UTC),
    )


def test_builds_request_with_system_history_and_tools():
    builder = MessageBuilder(model="claude-sonnet-4-6", max_tokens=4096)
    req = builder.build(
        system_blocks=[SystemBlock(text="You are helpful.")],
        history=[_msg(MessageRole.USER, "hi")],
        tool_specs=[_spec("kit.x")],
    )
    assert req.model == "claude-sonnet-4-6"
    assert len(req.system) == 1
    assert len(req.messages) == 1
    assert [t.name for t in req.tools] == ["kit.x"]


def test_translates_tool_specs_to_tool_definitions():
    builder = MessageBuilder(model="m", max_tokens=128)
    req = builder.build(system_blocks=[], history=[], tool_specs=[_spec("a")])
    assert isinstance(req.tools[0], ToolDefinition)
    assert req.tools[0].name == "a"
