"""AgentConfig.model_selector — optional per-iteration model override hook.

When set, the streaming handler resolves ``model = selector(ctx)`` per
iteration and threads it into MessageBuilder as a per-build override.
Orthogonal to ``provider_selector``: a consumer may swap provider AND
swap model in lockstep using a shared tracker.
"""

from agentkit.config import AgentConfig


def test_model_selector_defaults_to_none():
    """Default is None so consumers that don't set it preserve today's
    constructor-time-model behavior."""
    cfg = AgentConfig()
    assert cfg.model_selector is None


def test_model_selector_accepts_callable():
    """Typed Any to avoid circular imports with TurnContext — same
    pattern provider_selector / GuardConfig fields use."""

    def dummy(ctx):
        return "fake/test-quality"

    cfg = AgentConfig(model_selector=dummy)
    assert callable(cfg.model_selector)


def test_model_and_provider_selectors_are_independent():
    """The two selectors are independent fields — setting one doesn't
    affect the other. Pikkolo's tier router sets both."""

    def prov(ctx):
        return None

    def mod(ctx):
        return "fake/test"

    cfg = AgentConfig(provider_selector=prov, model_selector=mod)
    assert cfg.provider_selector is prov
    assert cfg.model_selector is mod
