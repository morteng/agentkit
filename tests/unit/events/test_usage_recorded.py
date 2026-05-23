"""UsageRecorded — public per-LLM-call usage event.

Yielded by stream_mux alongside MessageCompleted so consumers can persist
per-call ledger rows. Internal ctx.metadata['usages'] capture stays for
backward compatibility."""

from datetime import UTC, datetime

from agentkit._ids import EventId, MessageId, SessionId, TurnId, new_id
from agentkit._messages import Usage
from agentkit.events import UsageRecorded


def test_usage_recorded_schema_round_trip():
    """Construct, dump to JSON, assert all fields preserve correctly."""
    ev = UsageRecorded(
        event_id=new_id(EventId),
        session_id=new_id(SessionId),
        turn_id=new_id(TurnId),
        ts=datetime.now(UTC),
        sequence=42,
        message_id=new_id(MessageId),
        model="openai/gpt-5",
        usage=Usage(input_tokens=100, output_tokens=50, cached_input_tokens=20),
        provider_name="openrouter",
    )
    assert ev.type == "usage_recorded"
    dumped = ev.model_dump(mode="json")
    assert dumped["type"] == "usage_recorded"
    assert dumped["model"] == "openai/gpt-5"
    assert dumped["usage"]["input_tokens"] == 100
    assert dumped["usage"]["cached_input_tokens"] == 20
    assert dumped["provider_name"] == "openrouter"


def test_usage_recorded_round_trips_through_event_union():
    """Dumping and re-validating through the public Event discriminated
    union must preserve all UsageRecorded fields. Confirms the event
    is properly registered in the union."""
    from pydantic import TypeAdapter

    from agentkit.events import Event

    ev = UsageRecorded(
        event_id=new_id(EventId),
        session_id=new_id(SessionId),
        turn_id=new_id(TurnId),
        ts=datetime.now(UTC),
        sequence=1,
        message_id=new_id(MessageId),
        model="anthropic/claude-opus-4",
        usage=Usage(input_tokens=10, output_tokens=20),
        provider_name="anthropic",
    )
    dumped = ev.model_dump(mode="json")
    adapter = TypeAdapter(Event)
    parsed = adapter.validate_python(dumped)
    assert isinstance(parsed, UsageRecorded)
    assert parsed.model == "anthropic/claude-opus-4"
    assert parsed.usage.input_tokens == 10
