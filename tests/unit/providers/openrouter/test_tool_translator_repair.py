"""Tests for parse_tool_args_with_repair — fallback for malformed tool-call JSON."""

from agentkit.providers.openrouter.tool_translator import parse_tool_args_with_repair


def test_well_formed_json_parses_directly():
    parsed, err = parse_tool_args_with_repair('{"a": 1, "b": "x"}')
    assert parsed == {"a": 1, "b": "x"}
    assert err is None


def test_empty_string_yields_empty_dict():
    parsed, err = parse_tool_args_with_repair("")
    assert parsed == {}
    assert err is None


def test_unquoted_value_recovered_via_json_repair():
    """DeepSeek V4 Flash sometimes emits values un-quoted. Repair should recover."""
    bad = '{"facts": Drammens Teater ble bygd i 1869, "category": "history"}'
    parsed, err = parse_tool_args_with_repair(bad)
    assert err is None
    assert parsed is not None
    assert "facts" in parsed
    assert "category" in parsed


def test_total_garbage_returns_error():
    """Repair returns {} on total failure — surface the original error so callers retry."""
    parsed, err = parse_tool_args_with_repair("not json at all !!!")
    assert parsed is None
    assert err is not None
    assert "Expecting value" in err or "JSON" in err
