import pytest
from pydantic import ValidationError

from agentkit.config import AgentConfig, GuardConfig, LoopConfig, StoreBundle
from agentkit.store.memory import MemoryScope


def test_default_config_constructs():
    cfg = AgentConfig()
    assert cfg.loop.max_iterations == 10
    assert cfg.tool_dispatch.max_parallel == 8
    assert cfg.events.queue_size == 256
    assert cfg.guards.success_claim_enabled is False


def test_overrides_apply():
    cfg = AgentConfig(loop=LoopConfig(max_iterations=20))
    assert cfg.loop.max_iterations == 20


def test_guard_config_holds_components_loosely():
    """GuardConfig accepts callable objects implementing the protocols.

    We don't enforce isinstance — duck-typed Protocol structural matching applies
    at use-site only.
    """
    cfg = GuardConfig()
    assert cfg.success_claim_enabled is False


def test_store_bundle_memory_scope_defaults_to_none():
    """No scope by default — the memory tools then report memory_not_configured."""
    assert AgentConfig().stores.memory_scope is None


def test_store_bundle_carries_memory_scope():
    scope = MemoryScope(namespace="ns", tenant_id="t1", user_id="u1")
    bundle = StoreBundle(memory_scope=scope)
    assert bundle.memory_scope == scope
    # AgentConfig accepts it through the nested-model constructor too.
    assert AgentConfig(stores=bundle).stores.memory_scope == scope


def test_store_bundle_validates_memory_scope_type():
    """Typed concretely (not Any), so a bad value fails at construction."""
    with pytest.raises(ValidationError):
        StoreBundle(memory_scope="not-a-scope")  # type: ignore[arg-type]


def test_published_loop_knobs_exist_with_documented_defaults():
    lc = LoopConfig()
    assert lc.max_claim_corrections == 1
    assert lc.streaming_chunk_timeout_seconds == 60.0


def test_dead_knobs_are_gone():
    """A published knob that does nothing is worse than no knob.

    ``builtin_tool_note_enabled`` never registered anything: the consumer owns
    the ToolRegistry and registers ``NOTE_SPEC`` itself.
    """
    assert "builtin_tool_note_enabled" not in LoopConfig.model_fields


def test_rate_limit_is_on_by_default_and_permissive():
    from agentkit.guards.intent import DEFAULT_TURNS_PER_MINUTE

    assert GuardConfig().rate_limit_turns_per_minute == DEFAULT_TURNS_PER_MINUTE


def test_effective_intent_gate_wires_the_rate_limiter():
    from agentkit.guards.intent import DefaultIntentGate

    gate = GuardConfig().effective_intent_gate()
    assert isinstance(gate, DefaultIntentGate)


def test_effective_intent_gate_is_cached():
    """The limiter owns the sliding window, so it must outlive one turn —
    AgentSession rebuilds its deps every turn."""
    guards = GuardConfig()
    assert guards.effective_intent_gate() is guards.effective_intent_gate()


def test_effective_intent_gate_is_none_when_everything_is_off():
    assert GuardConfig(rate_limit_turns_per_minute=None).effective_intent_gate() is None


@pytest.mark.asyncio
async def test_effective_intent_gate_composes_a_custom_gate():
    """Injecting ``intent`` adds checks; it does not silently drop the limit."""
    from agentkit.guards.intent import IntentDecision
    from agentkit.loop.context import TurnContext

    class _Reject:
        async def evaluate(self, ctx):
            return IntentDecision(allow=False, reason="nope")

    guards = GuardConfig(intent=_Reject())
    gate = guards.effective_intent_gate()
    decision = await gate.evaluate(TurnContext.empty())
    assert decision.allow is False
    assert decision.reason == "nope"


@pytest.mark.asyncio
async def test_effective_intent_gate_rate_limits_before_the_custom_gate():
    from agentkit.guards.intent import IntentDecision
    from agentkit.loop.context import TurnContext

    seen: list[str] = []

    class _Recording:
        async def evaluate(self, ctx):
            seen.append("custom")
            return IntentDecision(allow=True)

    guards = GuardConfig(intent=_Recording(), rate_limit_turns_per_minute=1)
    gate = guards.effective_intent_gate()
    ctx = TurnContext.empty()
    ctx.metadata["owner"] = "u1"

    assert (await gate.evaluate(ctx)).allow
    assert not (await gate.evaluate(ctx)).allow
    assert seen == ["custom"], "the over-quota turn must be rejected before custom checks run"
