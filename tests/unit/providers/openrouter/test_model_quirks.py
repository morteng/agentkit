from agentkit.providers.openrouter.model_quirks import parse_finish_reason, requires_cache_blocks


def test_anthropic_models_require_cache_blocks():
    assert requires_cache_blocks("anthropic/claude-opus-4-7") is True
    assert requires_cache_blocks("anthropic/claude-sonnet-4-6") is True


def test_gemini_models_require_cache_blocks():
    assert requires_cache_blocks("google/gemini-2.5-pro") is True


def test_openai_models_auto_cache():
    assert requires_cache_blocks("openai/gpt-5") is False
    assert requires_cache_blocks("openai/gpt-4o-mini") is False


def test_deepseek_models_auto_cache():
    assert requires_cache_blocks("deepseek/deepseek-chat") is False


def test_unknown_model_defaults_to_no_cache_blocks():
    # Conservative — don't send cache_control we're not sure the model accepts.
    assert requires_cache_blocks("totally/unknown") is False


def test_finish_reason_translation():
    assert parse_finish_reason("stop") == "end_turn"
    assert parse_finish_reason("tool_calls") == "tool_use"
    assert parse_finish_reason("length") == "max_tokens"
    assert parse_finish_reason("unknown") == "end_turn"
