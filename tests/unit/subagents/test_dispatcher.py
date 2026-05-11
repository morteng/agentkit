import pytest

from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.guards.finalize import StructuralFinalizeValidator
from agentkit.loop.context import TurnContext
from agentkit.loop.message_builder import MessageBuilder
from agentkit.loop.tool_dispatcher import DispatchPolicy, ToolDispatcher
from agentkit.providers.fakes import FakeProvider
from agentkit.subagents.dispatcher import SubagentDispatcher
from agentkit.tools.builtin import DEFAULT_BUILTINS
from agentkit.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_subagent_dispatcher_returns_summary():
    """Dispatcher runs a child Loop and returns the assistant's final text."""
    child_provider = FakeProvider().script(FakeProvider.text("the answer is 42"))

    registry = ToolRegistry()
    for spec, handler in DEFAULT_BUILTINS:
        registry.register_builtin(spec, handler)

    deps = {
        "provider": child_provider,
        "message_builder": MessageBuilder(model="m", max_tokens=128),
        "registry": registry,
        "system_blocks": [],
        "intent_gate": None,
        "approval_gate": RiskBasedApprovalGate(),
        "dispatcher": ToolDispatcher(registry=registry, policy=DispatchPolicy()),
        "finalize_validator": StructuralFinalizeValidator(),
        "max_finalize_retries": 2,
        "max_iterations": 5,
    }

    sd = SubagentDispatcher(deps=deps, max_depth=3)
    parent_ctx = TurnContext.empty()
    summary = await sd.spawn(
        parent_ctx, prompt="research: what is 42?", tools=["kit.finalize"], extra_context={}
    )
    assert "42" in summary
