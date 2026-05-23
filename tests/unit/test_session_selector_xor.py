"""AgentSession must accept either a static provider or a
config.provider_selector — exactly one, never both, never neither.

Threads the selector into deps['provider_selector'] so the streaming
handler (Task 1.9) can resolve provider per iteration.
"""

import pytest

from agentkit._ids import OwnerId
from agentkit.config import AgentConfig
from agentkit.providers.fakes import FakeProvider
from agentkit.session import AgentSession
from agentkit.tools.registry import ToolRegistry


def test_session_rejects_neither_provider_nor_selector():
    cfg = AgentConfig()
    cfg.provider_selector = None
    with pytest.raises(ValueError, match="provider"):
        AgentSession(
            owner=OwnerId("t1"),
            config=cfg,
            provider=None,
            registry=ToolRegistry(),
            model="fake/test",
        )


def test_session_rejects_both_provider_and_selector():
    cfg = AgentConfig()
    cfg.provider_selector = lambda ctx: FakeProvider()
    with pytest.raises(ValueError, match="provider"):
        AgentSession(
            owner=OwnerId("t1"),
            config=cfg,
            provider=FakeProvider(),
            registry=ToolRegistry(),
            model="fake/test",
        )


def test_session_accepts_static_provider():
    cfg = AgentConfig()
    sess = AgentSession(
        owner=OwnerId("t1"),
        config=cfg,
        provider=FakeProvider(),
        registry=ToolRegistry(),
        model="fake/test",
    )
    assert sess.provider is not None
    assert sess.config.provider_selector is None


def test_session_accepts_selector_only():
    cfg = AgentConfig()
    cfg.provider_selector = lambda ctx: FakeProvider()
    sess = AgentSession(
        owner=OwnerId("t1"),
        config=cfg,
        provider=None,
        registry=ToolRegistry(),
        model="fake/test",
    )
    assert sess.provider is None
    assert sess.config.provider_selector is not None


def test_build_deps_threads_selector():
    """The selector must be threaded into deps so the streaming handler
    (Task 1.9) can find it under deps['provider_selector']."""
    cfg = AgentConfig()
    selector = lambda ctx: FakeProvider()  # noqa: E731
    cfg.provider_selector = selector
    sess = AgentSession(
        owner=OwnerId("t1"),
        config=cfg,
        provider=None,
        registry=ToolRegistry(),
        model="fake/test",
    )
    deps = sess._build_deps()
    assert deps["provider_selector"] is selector
    # When selector is set, provider may be None — handler will resolve via selector
    assert deps["provider"] is None
