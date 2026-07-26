"""AgentSession must thread the configured MemoryScope onto every TurnContext.

Regression: ``run()`` and ``_load_resume_context()`` both set ``memory_store``
but never ``memory_scope``, and there was no config field to carry one. Since
``kit.memory.save``/``kit.memory.recall`` require BOTH, every session answered
``memory_not_configured`` no matter how the stores were wired.
"""

import pytest

from agentkit import AgentConfig, AgentSession
from agentkit._ids import CheckpointId, OwnerId, TurnId, new_id
from agentkit.events import ToolCallResult
from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.loop.context import TurnContext, to_checkpoint_payload
from agentkit.providers.fakes import FakeProvider
from agentkit.store.fakes import FakeCheckpointStore, FakeMemoryStore, FakeSessionStore
from agentkit.store.memory import MemoryScope
from agentkit.tools.registry import ToolRegistry

SCOPE = MemoryScope(namespace="agentkit.test", tenant_id="t1", user_id="u1")

_FINALIZE_ARGS = {
    "status": "done",
    "intent_kind": "answer",
    "summary": "Done.",
    "answer_evidence": "general_knowledge",
}


def _make_config(*, scope: MemoryScope | None) -> tuple[AgentConfig, FakeMemoryStore]:
    memory = FakeMemoryStore()
    config = AgentConfig()
    config.guards.approval = RiskBasedApprovalGate()
    config.stores.session = FakeSessionStore()
    config.stores.memory = memory
    config.stores.memory_scope = scope
    config.stores.checkpoint = FakeCheckpointStore()
    return config, memory


def _make_session(config: AgentConfig, provider: FakeProvider) -> AgentSession:
    registry = ToolRegistry()
    registry.register_default_builtins()
    return AgentSession(
        owner=OwnerId("u:1"),
        config=config,
        provider=provider,
        registry=registry,
        model="m",
    )


async def _run_capturing_context(session: AgentSession, text: str) -> TurnContext:
    """Run one turn, capturing the TurnContext via the on_iteration_start hook."""
    seen: list[TurnContext] = []

    async def hook(ctx: TurnContext) -> None:
        seen.append(ctx)

    session.config.on_iteration_start = hook
    async with session.run(text) as stream:
        async for _ev in stream:
            pass
    assert seen, "on_iteration_start never fired — no TurnContext captured"
    return seen[0]


@pytest.mark.asyncio
async def test_run_passes_configured_memory_scope_to_turn_context():
    config, _ = _make_config(scope=SCOPE)
    session = _make_session(config, FakeProvider().script(FakeProvider.text("hi")))

    ctx = await _run_capturing_context(session, "hello")

    assert ctx.memory_scope == SCOPE
    assert ctx.memory_store is config.stores.memory
    assert ctx.metadata["owner"] == session.owner  # the pre-existing wiring survives


@pytest.mark.asyncio
async def test_run_leaves_memory_scope_none_when_unconfigured():
    config, _ = _make_config(scope=None)
    session = _make_session(config, FakeProvider().script(FakeProvider.text("hi")))

    ctx = await _run_capturing_context(session, "hello")

    assert ctx.memory_scope is None


async def _seed_checkpoint(session: AgentSession) -> TurnId:
    """Persist a minimal approval checkpoint so _load_resume_context can load it."""
    turn_id = new_id(TurnId)
    ckpt_ctx = TurnContext.empty()
    payload = to_checkpoint_payload(ckpt_ctx)
    assert session.config.stores.checkpoint is not None
    await session.config.stores.checkpoint.save(CheckpointId(f"approval:{turn_id}"), payload)
    return turn_id


@pytest.mark.asyncio
async def test_resume_context_passes_configured_memory_scope():
    """The approval-resume path builds its own TurnContext — it needs the scope too."""
    config, _ = _make_config(scope=SCOPE)
    session = _make_session(config, FakeProvider())
    turn_id = await _seed_checkpoint(session)

    ctx, _queue = await session._load_resume_context(turn_id)

    assert ctx.memory_scope == SCOPE
    assert ctx.memory_store is config.stores.memory
    assert ctx.metadata["owner"] == session.owner


@pytest.mark.asyncio
async def test_resume_context_leaves_memory_scope_none_when_unconfigured():
    config, _ = _make_config(scope=None)
    session = _make_session(config, FakeProvider())
    turn_id = await _seed_checkpoint(session)

    ctx, _queue = await session._load_resume_context(turn_id)

    assert ctx.memory_scope is None


@pytest.mark.asyncio
async def test_memory_save_then_recall_round_trip_through_session():
    """End-to-end proof of the fix: the model saves a fact, then recalls it."""
    config, memory = _make_config(scope=SCOPE)
    provider = FakeProvider().script(
        FakeProvider.tool_call(
            "kit.memory.save", {"key": "home_city", "text": "user lives in Oslo"}
        ),
        FakeProvider.tool_call("kit.memory.recall", {"key": "home_city"}),
        FakeProvider.tool_call("kit.finalize", _FINALIZE_ARGS),
    )
    session = _make_session(config, provider)

    results: list[ToolCallResult] = []
    async with session.run("remember where I live, then tell me") as stream:
        async for ev in stream:
            if isinstance(ev, ToolCallResult):
                results.append(ev)

    by_status = {r.status for r in results}
    assert "error" not in by_status, [r.error for r in results if r.error]

    # The recall came back with the saved text (results[1] is the recall call).
    recall = results[1]
    assert recall.status == "ok"
    assert "Oslo" in (recall.content[0].text or "")

    # ...and it landed under the CONFIGURED scope, not some default.
    stored = await memory.recall(SCOPE, "home_city")
    assert stored is not None
    assert stored.text == "user lives in Oslo"
    assert await memory.list_keys(MemoryScope(namespace="other")) == []


@pytest.mark.asyncio
async def test_memory_tools_error_when_no_scope_configured():
    """Contract preserved: no scope -> explicit memory_not_configured, no silent no-op."""
    config, memory = _make_config(scope=None)
    provider = FakeProvider().script(
        FakeProvider.tool_call("kit.memory.save", {"key": "k", "text": "v"}),
        FakeProvider.tool_call("kit.finalize", _FINALIZE_ARGS),
    )
    session = _make_session(config, provider)

    results: list[ToolCallResult] = []
    async with session.run("remember this") as stream:
        async for ev in stream:
            if isinstance(ev, ToolCallResult):
                results.append(ev)

    save = results[0]
    assert save.status == "error"
    assert save.error is not None
    assert save.error.code == "memory_not_configured"
    assert memory._data == {}  # nothing written under any scope
