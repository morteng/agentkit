from agentkit.config import AgentConfig, GuardConfig, LoopConfig


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
