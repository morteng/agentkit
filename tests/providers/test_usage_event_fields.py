"""UsageEvent carries model + provider_name so consumers can write
per-call ledger rows without guessing which provider produced the call."""

from datetime import UTC, datetime

from agentkit._ids import EventId, SessionId, TurnId, new_id
from agentkit._messages import Usage
from agentkit.providers.base import UsageEvent


def _mk_kwargs():
    return dict(
        event_id=new_id(EventId),
        session_id=new_id(SessionId),
        turn_id=new_id(TurnId),
        ts=datetime.now(UTC),
        sequence=1,
    )


def test_usage_event_carries_model_and_provider_name():
    ev = UsageEvent(
        **_mk_kwargs(),
        usage=Usage(input_tokens=100, output_tokens=50),
        model="openai/gpt-5",
        provider_name="openrouter",
    )
    assert ev.model == "openai/gpt-5"
    assert ev.provider_name == "openrouter"
    assert ev.usage.input_tokens == 100
    assert ev.type == "usage"


def test_usage_event_serializes_round_trip():
    ev = UsageEvent(
        **_mk_kwargs(),
        usage=Usage(input_tokens=10, output_tokens=20),
        model="anthropic/claude-opus-4",
        provider_name="anthropic",
    )
    dumped = ev.model_dump(mode="json")
    assert dumped["model"] == "anthropic/claude-opus-4"
    assert dumped["provider_name"] == "anthropic"
