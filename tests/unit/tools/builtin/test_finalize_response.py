"""Snapshot test for the finalize_response tool description + schema.

These constants are the contract the model sees on every turn — drift
shows up in eval failures, so trip a unit test on intentional changes.
"""

from agentkit.tools.builtin.finalize_response import (
    FINALIZE_RESPONSE_DESCRIPTION,
    FINALIZE_RESPONSE_SCHEMA,
)


def test_description_mentions_intent_kind_and_three_values():
    desc = FINALIZE_RESPONSE_DESCRIPTION
    assert "intent_kind" in desc
    assert '"action"' in desc
    assert '"answer"' in desc
    assert '"clarify"' in desc


def test_description_mentions_expected_count():
    assert "expected_count" in FINALIZE_RESPONSE_DESCRIPTION


def test_description_forbids_promise_without_action():
    desc = FINALIZE_RESPONSE_DESCRIPTION.lower()
    assert "i'll start" in desc or "i will start" in desc


def test_schema_has_intent_kind_required():
    schema = FINALIZE_RESPONSE_SCHEMA
    assert "intent_kind" in schema["properties"]
    assert "intent_kind" in schema["required"]
    assert schema["properties"]["intent_kind"]["enum"] == ["action", "answer", "clarify"]


def test_schema_has_expected_count_optional():
    schema = FINALIZE_RESPONSE_SCHEMA
    assert "expected_count" in schema["properties"]
    assert "expected_count" not in schema["required"]


def test_schema_status_enum():
    assert FINALIZE_RESPONSE_SCHEMA["properties"]["status"]["enum"] == [
        "done",
        "partial",
        "blocked",
    ]


def test_schema_actions_performed_is_array():
    actions = FINALIZE_RESPONSE_SCHEMA["properties"]["actions_performed"]
    assert actions["type"] == "array"
    assert actions["items"]["type"] == "object"
    assert "tool" in actions["items"]["required"]
    assert "description" in actions["items"]["required"]
