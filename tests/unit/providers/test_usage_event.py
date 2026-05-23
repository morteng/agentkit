"""UsageEvent carries model + provider_name so consumers can write
per-call ledger rows without guessing which provider produced the call."""

from datetime import UTC, datetime

import pytest

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


def test_usage_event_round_trips_through_provider_event_union():
    """Dumping then re-validating through the ProviderEvent discriminated
    union must preserve all fields. This is the contract consumers depend
    on when they serialize and re-load events from a queue/store."""
    from pydantic import TypeAdapter

    from agentkit.providers.base import ProviderEvent

    ev = UsageEvent(
        **_mk_kwargs(),
        usage=Usage(input_tokens=10, output_tokens=20),
        model="anthropic/claude-opus-4",
        provider_name="anthropic",
    )
    dumped = ev.model_dump(mode="json")
    assert dumped["type"] == "usage"
    assert dumped["model"] == "anthropic/claude-opus-4"
    assert dumped["provider_name"] == "anthropic"

    adapter = TypeAdapter(ProviderEvent)
    parsed = adapter.validate_python(dumped)
    assert parsed.model == "anthropic/claude-opus-4"
    assert parsed.provider_name == "anthropic"
    assert parsed.usage.input_tokens == 10


def test_usage_event_requires_model_and_provider_name():
    """Both fields are required — the optional defaults from the
    transitional period (Tasks 1.1-1.4) have been tightened."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="model"):
        UsageEvent(**_mk_kwargs(), usage=Usage(input_tokens=1, output_tokens=1))
