import pytest

from agentkit.loop.context import TurnContext
from agentkit.tools.builtin.finalize import FINALIZE_SPEC, finalize_handler


@pytest.mark.asyncio
async def test_finalize_sets_flag_on_context():
    ctx = TurnContext.empty(call_id="c1")
    res = await finalize_handler({"reason": "task complete"}, ctx)
    assert res.status == "ok"
    assert ctx.finalize_called is True
    assert ctx.finalize_reason == "task complete"


def test_finalize_spec_is_destructive_intent_marker():
    assert FINALIZE_SPEC.name == "kit.finalize"
    # Finalize itself doesn't write anything; it signals.
    assert FINALIZE_SPEC.cache_ttl_seconds is None
