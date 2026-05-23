"""AgentConfig.provider_selector — optional per-call provider override hook.

When set, the streaming handler resolves the provider via selector(ctx)
per iteration instead of the static AgentSession.provider attribute.
This task adds the field; later tasks wire it into the session and
streaming handler.
"""

from agentkit.config import AgentConfig


def test_provider_selector_defaults_to_none():
    """Default value is None so consumers that don't set it preserve
    today's static-provider behavior."""
    cfg = AgentConfig()
    assert cfg.provider_selector is None


def test_provider_selector_accepts_callable():
    """The field accepts any callable — typed Any to avoid circular
    imports with the Provider protocol (same pattern GuardConfig uses)."""

    def dummy_selector(ctx):
        return None

    cfg = AgentConfig(provider_selector=dummy_selector)
    assert callable(cfg.provider_selector)
