"""The tool_selector hook filters the spec list the model sees."""

import asyncio
import contextlib
from typing import ClassVar

from agentkit.config import AgentConfig
from agentkit.tools.spec import ApprovalPolicy, RiskLevel, SideEffects, ToolSpec


def test_config_has_tool_selector_default_none():
    cfg = AgentConfig()
    assert cfg.tool_selector is None


def _spec(name):
    return ToolSpec(
        name=name,
        description=name,
        parameters={},
        returns=None,
        risk=RiskLevel.READ,
        idempotent=True,
        side_effects=SideEffects.NONE,
        requires_approval=ApprovalPolicy.NEVER,
        cache_ttl_seconds=None,
        timeout_seconds=30.0,
    )


def test_selector_applied_to_list_specs():
    """Verify the selector narrows registry.list_specs() before build()."""
    from agentkit.loop.handlers import streaming as mod

    captured = {}

    class _StopBuild(Exception):
        pass

    class FakeBuilder:
        def build(self, *, system_blocks, history, tool_specs, model_override):
            captured["specs"] = [s.name for s in tool_specs]
            raise _StopBuild()

    class FakeRegistry:
        def list_specs(self):
            return [_spec("a"), _spec("b"), _spec("c")]

    class FakeProvider:
        def stream(self, request):  # pragma: no cover - not reached
            raise AssertionError

    class FakeCtx:
        event_queue: ClassVar[None] = None
        history: ClassVar[list] = []
        metadata: ClassVar[dict] = {}

    deps = {
        "provider": FakeProvider(),
        "message_builder": FakeBuilder(),
        "registry": FakeRegistry(),
        "tool_selector": lambda ctx, specs: [s for s in specs if s.name in {"a", "c"}],
    }
    with contextlib.suppress(_StopBuild):
        asyncio.run(mod.handle_streaming(FakeCtx(), deps))  # pyright: ignore[reportArgumentType]
    assert captured["specs"] == ["a", "c"]
