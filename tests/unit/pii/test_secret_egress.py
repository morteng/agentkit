"""End-to-end egress coverage for credentials — the incident this all exists for.

REDACTED's user-provisioning tool mints a household password with
``secrets.token_urlsafe(16)`` and returns it inside its tool result. That
result goes back to the model, out to a third-party inference provider on the
same turn, and into the transcript forever. The identity-shaped detectors saw
nothing. These tests assert the value does not leave the boundary — through the
tool result (leaking *in* to the model) and through tool arguments (leaking
*out* to a tool).
"""

import json
import secrets as pysecrets

from agentkit._content import TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from agentkit._messages import MessageRole
from agentkit.pii.composite import CompositeDetector
from agentkit.pii.firewall import Firewall
from agentkit.pii.policy import PiiPolicy
from agentkit.pii.provider import wrap_provider
from agentkit.providers.base import ErrorEvent, ProviderRequest, SystemBlock
from agentkit.providers.fakes import FakeProvider

from .conftest import FakeDetector, FakeTokenMap, make_msg


def _firewall(**policy_kw: object) -> Firewall:
    """The firewall a consumer gets when it composes instead of replacing."""
    return Firewall(
        CompositeDetector.with_defaults(FakeDetector()),
        PiiPolicy(**policy_kw),  # type: ignore[arg-type]
    )


def _provisioning_result(password: str) -> ToolResultBlock:
    """What ``create_household_user`` actually returns."""
    return ToolResultBlock(
        tool_use_id="t1",
        content=[
            TextBlock(
                text=json.dumps(
                    {
                        "status": "created",
                        "username": "kari",
                        "password": password,
                        "note": "share this with the household member",
                    }
                )
            )
        ],
    )


# ---- The regression ---------------------------------------------------------


def test_generated_password_in_tool_result_is_redacted(tmap: FakeTokenMap):
    for _ in range(50):
        password = pysecrets.token_urlsafe(16)
        req = ProviderRequest(
            model="openai/gpt-5.5",
            messages=[make_msg(MessageRole.TOOL, _provisioning_result(password))],
        )
        out = _firewall().scrub_request(req, tmap)
        block = out.messages[0].content[0]
        assert isinstance(block, ToolResultBlock)
        inner = block.content[0]
        assert isinstance(inner, TextBlock)
        assert password not in inner.text, password
        assert "[REDACTED]" in inner.text
        # Still valid JSON the model can read — only the value is gone.
        assert json.loads(inner.text)["username"] == "kari"


async def test_generated_password_never_reaches_the_provider(tmap: FakeTokenMap):
    """The full egress path: wrap_provider is the choke point."""
    captured: dict[str, ProviderRequest] = {}

    class CapturingFake(FakeProvider):
        async def stream(self, request):  # type: ignore[no-untyped-def, override]
            captured["req"] = request
            async for ev in super().stream(request):
                yield ev

    password = pysecrets.token_urlsafe(16)
    inner = CapturingFake().script(FakeProvider.text("Done — I created the account."))
    wrapped = wrap_provider(inner, _firewall(), tmap_resolver=lambda req: tmap)
    req = ProviderRequest(
        model="openai/gpt-5.5",
        messages=[make_msg(MessageRole.TOOL, _provisioning_result(password))],
    )
    _ = [ev async for ev in wrapped.stream(req)]

    sent = captured["req"].model_dump_json()
    assert password not in sent
    # And the caller's own request object was not mutated.
    assert password in req.model_dump_json()


def test_never_send_secret_is_not_written_to_the_token_map(tmap: FakeTokenMap):
    password = pysecrets.token_urlsafe(16)
    req = ProviderRequest(
        model="openai/gpt-5.5",
        messages=[make_msg(MessageRole.TOOL, _provisioning_result(password))],
    )
    _ = _firewall().scrub_request(req, tmap)
    assert all(tmap.value_for(t) != password for t in tmap.all_tokens())


def test_scrubbing_is_idempotent(tmap: FakeTokenMap):
    password = pysecrets.token_urlsafe(16)
    req = ProviderRequest(
        model="openai/gpt-5.5",
        messages=[make_msg(MessageRole.TOOL, _provisioning_result(password))],
    )
    firewall = _firewall()
    once = firewall.scrub_request(req, tmap)
    twice = firewall.scrub_request(once, tmap)
    assert once.model_dump_json() == twice.model_dump_json()


# ---- Coverage of every text-bearing surface --------------------------------


def test_secret_in_tool_arguments_is_redacted(tmap: FakeTokenMap):
    """An argument is how a secret leaks OUT to a tool."""
    password = pysecrets.token_urlsafe(16)
    req = ProviderRequest(
        model="openai/gpt-5.5",
        messages=[
            make_msg(
                MessageRole.ASSISTANT,
                ToolUseBlock(
                    id="t1",
                    name="send_email",
                    arguments={
                        "subject": "Your account",
                        "body": f"your password is {password}",
                        "attempts": 3,
                    },
                ),
            )
        ],
    )
    out = _firewall().scrub_request(req, tmap)
    block = out.messages[0].content[0]
    assert isinstance(block, ToolUseBlock)
    assert password not in block.arguments["body"]
    assert block.arguments["attempts"] == 3


def test_secret_named_argument_field_is_redacted_whatever_its_value(tmap: FakeTokenMap):
    req = ProviderRequest(
        model="openai/gpt-5.5",
        messages=[
            make_msg(
                MessageRole.ASSISTANT,
                ToolUseBlock(
                    id="t1",
                    name="set_credentials",
                    arguments={"user": "kari", "password": "sommer2026"},
                ),
            )
        ],
    )
    out = _firewall().scrub_request(req, tmap)
    block = out.messages[0].content[0]
    assert isinstance(block, ToolUseBlock)
    assert block.arguments["password"] == "[REDACTED]"
    assert block.arguments["user"] == "kari"


def test_field_context_reaches_nested_argument_structures(tmap: FakeTokenMap):
    req = ProviderRequest(
        model="openai/gpt-5.5",
        messages=[
            make_msg(
                MessageRole.ASSISTANT,
                ToolUseBlock(
                    id="t1",
                    name="provision",
                    arguments={
                        "accounts": [
                            {"user": "kari", "api_key": "abc123"},
                            {"user": "ola", "api_key": "def456"},
                        ]
                    },
                ),
            )
        ],
    )
    out = _firewall().scrub_request(req, tmap)
    block = out.messages[0].content[0]
    assert isinstance(block, ToolUseBlock)
    assert [a["api_key"] for a in block.arguments["accounts"]] == ["[REDACTED]", "[REDACTED]"]
    assert [a["user"] for a in block.arguments["accounts"]] == ["kari", "ola"]


def test_detector_without_the_field_extension_is_unaffected(tmap: FakeTokenMap):
    """``FieldContextDetector`` is additive: a plain detector still gets called.

    Consumer detectors written before the extension existed implement
    ``detect`` only. The firewall must walk arguments with field names for the
    detectors that want them and fall back to ``detect`` for the ones that do
    not — never crash, never skip.
    """
    firewall = Firewall(FakeDetector(), PiiPolicy())
    req = ProviderRequest(
        model="openai/gpt-5.5",
        messages=[
            make_msg(
                MessageRole.ASSISTANT,
                ToolUseBlock(
                    id="t1",
                    name="notify",
                    arguments={"password": "kari@example.no", "count": 2},
                ),
            )
        ],
    )
    out = firewall.scrub_request(req, tmap)
    block = out.messages[0].content[0]
    assert isinstance(block, ToolUseBlock)
    # The plain detector's own recognizer fired; the field name changed nothing.
    assert block.arguments["password"] == "[EMAIL_1]"
    assert block.arguments["count"] == 2


def test_secret_in_system_block_and_user_text_is_redacted(tmap: FakeTokenMap):
    password = pysecrets.token_urlsafe(16)
    req = ProviderRequest(
        model="openai/gpt-5.5",
        system=[SystemBlock(text=f"The admin password is {password}.")],
        messages=[make_msg(MessageRole.USER, TextBlock(text=f"is {password} still valid?"))],
    )
    out = _firewall().scrub_request(req, tmap)
    assert password not in out.system[0].text
    user_block = out.messages[0].content[0]
    assert isinstance(user_block, TextBlock)
    assert password not in user_block.text


def test_unsigned_thinking_block_is_scrubbed(tmap: FakeTokenMap):
    password = pysecrets.token_urlsafe(16)
    req = ProviderRequest(
        model="openai/gpt-5.5",
        messages=[
            make_msg(
                MessageRole.ASSISTANT,
                ThinkingBlock(text=f"I should tell them {password} and kari@example.no"),
            )
        ],
    )
    out = _firewall().scrub_request(req, tmap)
    block = out.messages[0].content[0]
    assert isinstance(block, ThinkingBlock)
    assert password not in block.text
    assert "kari@example.no" not in block.text


def test_signed_thinking_block_is_left_byte_exact(tmap: FakeTokenMap):
    """Rewriting signed thinking invalidates the signature and the provider
    rejects the turn — and those bytes are that provider's own output coming
    home, so scrubbing them protects nothing."""
    original = "I should tell them kari@example.no"
    req = ProviderRequest(
        model="anthropic/claude-sonnet-5",
        messages=[
            make_msg(MessageRole.ASSISTANT, ThinkingBlock(text=original, signature="sig-abc"))
        ],
    )
    out = _firewall().scrub_request(req, tmap)
    block = out.messages[0].content[0]
    assert isinstance(block, ThinkingBlock)
    assert block.text == original


def test_tool_result_nested_blocks_are_scrubbed(tmap: FakeTokenMap):
    password = pysecrets.token_urlsafe(16)
    req = ProviderRequest(
        model="openai/gpt-5.5",
        messages=[
            make_msg(
                MessageRole.TOOL,
                ToolResultBlock(
                    tool_use_id="outer",
                    content=[
                        ToolResultBlock(
                            tool_use_id="inner",
                            content=[TextBlock(text=f"password={password}")],
                        ),
                        ToolUseBlock(id="t2", name="retry", arguments={"password": password}),
                    ],
                ),
            )
        ],
    )
    out = _firewall().scrub_request(req, tmap)
    assert password not in out.model_dump_json()


def test_tool_schemas_are_still_left_alone(tmap: FakeTokenMap):
    from agentkit.providers.base import ToolDefinition

    req = ProviderRequest(
        model="openai/gpt-5.5",
        tools=[
            ToolDefinition(
                name="set_password",
                description="Set the household password",
                parameters={"type": "object", "properties": {"password": {"type": "string"}}},
            )
        ],
    )
    out = _firewall().scrub_request(req, tmap)
    assert out.tools[0].description == "Set the household password"
    assert out.tools[0].parameters["properties"]["password"] == {"type": "string"}


async def test_error_events_are_scrubbed_of_secrets(tmap: FakeTokenMap):
    password = pysecrets.token_urlsafe(16)
    inner = FakeProvider().script(
        FakeProvider.error("server_error", f"upstream rejected password {password}")
    )
    # require_zdr off so a generic error is not read as a route failure.
    wrapped = wrap_provider(inner, _firewall(require_zdr=False), tmap_resolver=lambda req: tmap)
    events = [ev async for ev in wrapped.stream(ProviderRequest(model="openai/gpt-5.5"))]
    err = next(ev for ev in events if isinstance(ev, ErrorEvent))
    assert password not in err.message
    assert "[REDACTED]" in err.message
