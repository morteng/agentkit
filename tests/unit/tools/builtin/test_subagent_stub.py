import pytest

from agentkit._content import Provenance
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


@pytest.mark.asyncio
async def test_a_tainting_child_marks_the_spawn_results_own_provenance():
    """The upward half of anti-laundering, at the handler seam.

    ``SubagentDispatcher.spawn`` (the real ``ctx.spawn_subagent``) latches
    child taint onto ``ctx`` directly before returning — this stub only
    fakes that one side effect, since it is the handler's *reaction* to it
    under test here, not the dispatcher's own propagation (see
    ``tests/unit/subagents/test_dispatcher_security.py`` for that). A
    ``ctx`` that comes back tainted must produce a ``ToolResult`` whose own
    provenance says so, not the ``SYSTEM`` default — otherwise the persisted
    transcript message for this call misreports where its content came from.
    """

    async def tainting_spawn(prompt: str, tools: list[str], extra_context: dict) -> str:
        ctx.tainted = True  # what SubagentDispatcher.spawn() would have done
        return "summarised the page"

    ctx = TurnContext.empty(call_id="c1")
    ctx.spawn_subagent = tainting_spawn

    res = await subagent_spawn_handler({"prompt": "read it", "tools": ["web.fetch"]}, ctx)

    assert res.status == "ok"
    assert res.provenance is Provenance.UNTRUSTED


@pytest.mark.asyncio
async def test_a_clean_child_leaves_the_spawn_results_provenance_at_system():
    """Control for the test above: an untainted child must not flip this."""

    async def clean_spawn(prompt: str, tools: list[str], extra_context: dict) -> str:
        return "nothing untrusted here"

    ctx = TurnContext.empty(call_id="c1")
    ctx.spawn_subagent = clean_spawn

    res = await subagent_spawn_handler({"prompt": "compute", "tools": []}, ctx)

    assert res.status == "ok"
    assert res.provenance is Provenance.SYSTEM
