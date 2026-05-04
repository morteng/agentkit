import pytest

from agentkit.loop.context import TurnContext
from agentkit.tools.builtin.note import note_handler


@pytest.mark.asyncio
async def test_note_appends_to_scratchpad():
    ctx = TurnContext.empty(call_id="c1")
    await note_handler({"text": "first"}, ctx)
    await note_handler({"text": "second"}, ctx)
    assert ctx.scratchpad == ["first", "second"]
