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
