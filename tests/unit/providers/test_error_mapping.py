"""F22: provider adapters map SDK exceptions to ErrorEvent."""

import httpx
import pytest

from agentkit.providers.base import ErrorEvent


def _httpx_response(status: int, body: dict | None = None) -> httpx.Response:
    """Build a minimal httpx.Response usable by SDK exception constructors."""
    request = httpx.Request("POST", "https://example.com")
    return httpx.Response(status, request=request, json=body or {"error": {"message": "x"}})


class TestAnthropicErrorMapping:
    def test_authentication_error_maps_to_auth_failed(self):
        import anthropic

        from agentkit.providers.anthropic.adapter import _map_anthropic_error

        exc = anthropic.AuthenticationError("bad key", response=_httpx_response(401), body=None)
        ev = _map_anthropic_error(exc)
        assert isinstance(ev, ErrorEvent)
        assert ev.code == "auth_failed"
        assert ev.recoverable is False

    def test_rate_limit_error_maps_to_rate_limited_recoverable(self):
        import anthropic

        from agentkit.providers.anthropic.adapter import _map_anthropic_error

        exc = anthropic.RateLimitError("slow down", response=_httpx_response(429), body=None)
        ev = _map_anthropic_error(exc)
        assert ev.code == "rate_limited"
        assert ev.recoverable is True

    def test_not_found_maps_to_not_found(self):
        import anthropic

        from agentkit.providers.anthropic.adapter import _map_anthropic_error

        exc = anthropic.NotFoundError("no such model", response=_httpx_response(404), body=None)
        ev = _map_anthropic_error(exc)
        assert ev.code == "not_found"

    def test_unknown_exception_falls_through_to_provider_error(self):
        from agentkit.providers.anthropic.adapter import _map_anthropic_error

        ev = _map_anthropic_error(ValueError("totally unrelated"))
        assert ev.code == "provider_error"
        assert "ValueError" in ev.message
        assert "totally unrelated" in ev.message


class TestOpenAIErrorMapping:
    def test_authentication_error_maps_to_auth_failed(self):
        import openai

        from agentkit.providers.openrouter.adapter import _map_openai_error

        exc = openai.AuthenticationError("bad key", response=_httpx_response(401), body=None)
        ev = _map_openai_error(exc)
        assert ev.code == "auth_failed"
        assert ev.recoverable is False

    def test_not_found_maps_to_not_found(self):
        import openai

        from agentkit.providers.openrouter.adapter import _map_openai_error

        exc = openai.NotFoundError("no such model", response=_httpx_response(404), body=None)
        ev = _map_openai_error(exc)
        assert ev.code == "not_found"

    def test_rate_limit_error_maps_to_rate_limited_recoverable(self):
        import openai

        from agentkit.providers.openrouter.adapter import _map_openai_error

        exc = openai.RateLimitError("slow down", response=_httpx_response(429), body=None)
        ev = _map_openai_error(exc)
        assert ev.code == "rate_limited"
        assert ev.recoverable is True


@pytest.mark.asyncio
async def test_anthropic_provider_yields_error_event_on_sdk_exception():
    """End-to-end: an exception inside .stream() becomes a yielded ErrorEvent, not a raise."""
    import anthropic

    from agentkit.providers.anthropic.adapter import AnthropicProvider
    from agentkit.providers.base import ProviderRequest

    class FailingClient:
        class messages:
            @staticmethod
            def stream(**_kwargs):
                raise anthropic.AuthenticationError(
                    "bad creds", response=_httpx_response(401), body=None
                )

    provider = AnthropicProvider(client=FailingClient())  # type: ignore[arg-type]
    request = ProviderRequest(model="claude-haiku-4-5-20251001")

    events = [ev async for ev in provider.stream(request)]
    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    assert events[0].code == "auth_failed"


@pytest.mark.asyncio
async def test_openrouter_provider_yields_error_event_on_sdk_exception():
    """End-to-end: OpenRouterProvider returns an ErrorEvent instead of raising."""
    import openai

    from agentkit.providers.base import ProviderRequest
    from agentkit.providers.openrouter.adapter import OpenRouterProvider

    class FailingChat:
        class completions:
            @staticmethod
            async def create(**_kwargs):
                raise openai.AuthenticationError(
                    "bad creds", response=_httpx_response(401), body=None
                )

    class FailingClient:
        chat = FailingChat()

    provider = OpenRouterProvider(client=FailingClient())  # type: ignore[arg-type]
    request = ProviderRequest(model="anything")

    events = [ev async for ev in provider.stream(request)]
    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    assert events[0].code == "auth_failed"
