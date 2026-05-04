from datetime import UTC, datetime

import pytest

from agentkit.loop.context import FixedClock, TurnContext
from agentkit.tools.builtin.time import current_time_handler


@pytest.mark.asyncio
async def test_current_time_uses_context_clock():
    fixed = datetime(2026, 5, 3, 12, 0, tzinfo=UTC)
    ctx = TurnContext.empty(call_id="c1", clock=FixedClock(fixed))
    res = await current_time_handler({}, ctx)
    assert res.status == "ok"
    assert res.content[0].text is not None
    assert "2026-05-03T12:00:00" in res.content[0].text
