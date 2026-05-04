import pytest

from agentkit.loop.context import TurnContext
from agentkit.tools.builtin.approval import request_approval_handler


@pytest.mark.asyncio
async def test_request_approval_appends_to_pending():
    ctx = TurnContext.empty(call_id="c1")
    res = await request_approval_handler(
        {"prompt": "Confirm device shutdown?", "options": ["yes", "no"]},
        ctx,
    )
    assert res.status == "ok"
    assert ctx.pending_approvals
    assert ctx.pending_approvals[0].prompt == "Confirm device shutdown?"
