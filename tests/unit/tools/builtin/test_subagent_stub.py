import pytest

from agentkit.loop.context import TurnContext
from agentkit.tools.builtin.subagent import subagent_spawn_handler


@pytest.mark.asyncio
async def test_subagent_spawn_handler_calls_injected_callable():
    spawned = {}

    async def fake_spawn(prompt: str, tools: list[str], extra_context: dict) -> str:
        spawned["prompt"] = prompt
        spawned["tools"] = tools
        spawned["extra_context"] = extra_context
        return "subagent finished: result"

    ctx = TurnContext.empty(call_id="c1")
    ctx.spawn_subagent = fake_spawn

    res = await subagent_spawn_handler(
        {"prompt": "research nordpool", "tools": ["web.search"], "context": {"x": 1}},
        ctx,
    )
    assert res.status == "ok"
    assert spawned["prompt"] == "research nordpool"
    assert spawned["tools"] == ["web.search"]
    assert "subagent finished" in (res.content[0].text or "")


@pytest.mark.asyncio
async def test_subagent_spawn_without_injection_errors():
    ctx = TurnContext.empty(call_id="c1")
    res = await subagent_spawn_handler({"prompt": "x", "tools": []}, ctx)
    assert res.status == "error"
