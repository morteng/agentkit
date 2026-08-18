"""`scrub_free_text`, and the seam it exists to protect.

WHY THIS EXISTS
---------------
`_redaction.is_secret_key` masks by KEY NAME. That works only when the value
arrives in a field somebody already labelled. An exception message is not that:
it is free text with the credential inside it, and there is no key to match.

`InProcessMCPClient._call` used to put a bare `str(exc)` into
`ToolError.message` — a field an HTTP surface hands to a browser. httpx embeds
the full request URL in its exception text, so that path shipped an internal
host, a port and an `api_key` query parameter to a user's screen. Observed in
a downstream consumer on 2026-08-18, by running the real client rather than reading it.

The tests are split deliberately: the *function* tests are unit-level, but the
one that matters drives `InProcessMCPClient._call` itself, because a correct
scrubber that nobody calls is the exact failure this repo family keeps
producing.
"""

from __future__ import annotations

import pytest

from agentkit._redaction import REDACTED, scrub_free_text

LEAKY_URL = (
    "Client error '404 Not Found' for url 'http://jellyfin:8096/Items?api_key=super-secret-abc123'"
)
SECRET = "super-secret-abc123"


class TestTheFixtureCouldActuallyLeak:
    """Controls. Every assertion below is "the secret is gone"; that is
    evidence only once the same check has been shown to find it."""

    def test_the_secret_is_in_the_fixture(self) -> None:
        assert SECRET in LEAKY_URL

    def test_an_untouched_string_is_returned_unchanged(self) -> None:
        # Proves the function is not simply blanking everything, which would
        # make every "secret is absent" assertion pass for the wrong reason.
        plain = "Jellyfin did not answer in time."
        assert scrub_free_text(plain) == plain


class TestQueryStrings:
    def test_a_url_query_string_is_dropped_wholesale(self) -> None:
        out = scrub_free_text(LEAKY_URL)
        assert SECRET not in out
        assert REDACTED in out

    def test_the_host_and_path_survive_because_they_are_not_credentials(self) -> None:
        # Named explicitly so nobody "hardens" this into hiding hostnames and
        # breaks diagnosis while believing they improved it.
        out = scrub_free_text(LEAKY_URL)
        assert "jellyfin:8096" in out
        assert "/Items" in out

    @pytest.mark.parametrize(
        "param",
        ["api_key", "apikey", "t", "s", "token", "X-Api-Key", "sig"],
    )
    def test_it_does_not_matter_what_the_parameter_is_called(self, param: str) -> None:
        # Enumerating secret-looking parameter names is the mistake: Subsonic
        # uses `t`/`s`, which no key-name heuristic would ever flag.
        out = scrub_free_text(f"GET https://navidrome:4533/rest/ping?{param}={SECRET} failed")
        assert SECRET not in out


class TestOtherShapesRealExceptionsCarry:
    def test_a_bearer_token_is_masked(self) -> None:
        out = scrub_free_text(f"headers: {{'authorization': 'Bearer {SECRET}'}}")
        assert SECRET not in out
        assert REDACTED in out

    def test_a_loose_key_value_pair_is_masked(self) -> None:
        out = scrub_free_text(f"connect failed (password={SECRET})")
        assert SECRET not in out

    def test_a_basic_auth_header_is_masked(self) -> None:
        out = scrub_free_text("authorization: Basic YWRtaW46aHVudGVyMg==")
        assert "YWRtaW46aHVudGVyMg==" not in out


class TestTheSeam:
    """The scrubber is only worth anything if the failing path calls it."""

    @pytest.mark.asyncio
    async def test_a_handler_exception_reaches_the_caller_scrubbed(self) -> None:
        # Driving the REAL client. A test of scrub_free_text alone would pass
        # against an inprocess.py that never imports it.
        from agentkit.mcp_client.inprocess import InProcessMCPClient
        from agentkit.tools.spec import (
            ApprovalPolicy,
            RiskLevel,
            SideEffects,
            ToolResult,
            ToolSpec,
        )

        async def explode(_arguments: dict[str, object]) -> ToolResult:
            # The shape httpx raises: the failure text carries the request URL.
            raise RuntimeError(LEAKY_URL)

        spec = ToolSpec(
            name="jellyfin.search",
            description="search",
            parameters={"type": "object"},
            returns=None,
            risk=RiskLevel.READ,
            idempotent=True,
            side_effects=SideEffects.NONE,
            requires_approval=ApprovalPolicy.BY_RISK,
            cache_ttl_seconds=None,
            timeout_seconds=10.0,
        )
        client = InProcessMCPClient("test")
        client.register_tool(spec, explode)
        await client.initialize()
        result = await client.call_tool("jellyfin.search", {})

        assert result.status == "error"
        assert result.error is not None
        assert SECRET not in result.error.message, "an API key reached ToolError.message"
        assert REDACTED in result.error.message, "the scrubber did not run on this path"
        # The diagnostic must still be useful, or the next person removes it.
        assert "404" in result.error.message
