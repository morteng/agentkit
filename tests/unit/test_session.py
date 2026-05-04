import pytest

from agentkit import AgentConfig, AgentSession
from agentkit._ids import OwnerId
from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.guards.finalize import RuleBasedFinalizeValidator
from agentkit.providers.fakes import FakeProvider
from agentkit.store.fakes import FakeCheckpointStore, FakeMemoryStore, FakeSessionStore
from agentkit.tools.builtin import DEFAULT_BUILTINS
from agentkit.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_agent_session_runs_text_turn():
    config = AgentConfig()
    config.guards.approval = RiskBasedApprovalGate()
    config.guards.finalize = RuleBasedFinalizeValidator()
    config.stores.session = FakeSessionStore()
    config.stores.memory = FakeMemoryStore()
    config.stores.checkpoint = FakeCheckpointStore()

    registry = ToolRegistry()
    for spec, handler in DEFAULT_BUILTINS:
        registry.register_builtin(spec, handler)

    session = AgentSession(
        owner=OwnerId("u:1"),
        config=config,
        provider=FakeProvider().script(FakeProvider.text("hi")),
        registry=registry,
        model="m",
    )

    events = []
    async with session.run("hello") as stream:
        async for ev in stream:
            events.append(ev)

    types = [type(e).__name__ for e in events]
    assert "TurnStarted" in types
    assert "TurnEnded" in types
