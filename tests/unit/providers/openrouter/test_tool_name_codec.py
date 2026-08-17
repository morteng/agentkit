from datetime import UTC, datetime

import pytest

from agentkit._content import ToolUseBlock
from agentkit._ids import MessageId, SessionId, new_id
from agentkit._messages import Message, MessageRole
from agentkit.providers.base import NamedToolChoice, ProviderRequest, ToolDefinition
from agentkit.providers.openrouter.tool_name_codec import (
    UNKNOWN_INVALID_TOOL_NAME,
    ToolNameCodec,
    is_wire_safe_name,
)


def test_qualified_names_encode_and_round_trip():
    codec = ToolNameCodec.from_names(["acme.search", "kit.current_time"])

    assert codec.encode("acme.search") == "acme__search"
    assert codec.encode("kit.current_time") == "kit__current_time"
    assert codec.decode("acme__search") == "acme.search"
    assert codec.decode("kit__current_time") == "kit.current_time"
    assert all(is_wire_safe_name(name) for name in codec.wire_to_canonical)


def test_dotted_name_wins_its_own_encoding_from_a_safe_sibling():
    """``a.b`` claims ``a__b``; the flat ``a__b`` gets the suffix.

    The flat name used to win this collision, and that is how one decode miss
    became permanent: a preserved wire name stored into history as if
    canonical held the real dotted tool's natural alias on every later
    request, so the real tool was advertised under ``_2`` while the model kept
    calling the name it had seen work — which decoded to the poisoned entry
    and failed "unknown tool" forever (observed live with
    ``torrent__torrent_add``). Dotted-name priority makes the same call decode
    to the registered tool instead, and both spellings stay dispatchable.
    """
    codec = ToolNameCodec.from_names(["a.b", "a__b"])

    assert codec.encode("a.b") == "a__b"
    assert codec.encode("a__b") == "a__b_2"
    assert codec.decode("a__b") == "a.b"
    assert codec.decode("a__b_2") == "a__b"


def test_long_or_unicode_name_uses_safe_digest():
    canonical = "søketøy." + ("x" * 100)
    codec = ToolNameCodec.from_names([canonical])
    wire = codec.encode(canonical)

    assert is_wire_safe_name(wire)
    assert codec.decode(wire) == canonical


def test_malformed_unknown_name_fails_closed_to_safe_sentinel():
    codec = ToolNameCodec.from_names(["acme.search"])

    assert codec.decode("no.such.tool") == UNKNOWN_INVALID_TOOL_NAME
    assert is_wire_safe_name(codec.decode("no.such.tool"))


def test_echoed_canonical_name_decodes_to_itself():
    """A dotted name of this request is accepted as canonical on decode.

    Models echo the dotted names they see in history prose and in "Did you
    mean acme.search?" diagnostics. The name identifies exactly one tool of
    this request, so sentinelling it (the old behaviour) turned correct intent
    into ``__invalid_tool_name__``.
    """
    codec = ToolNameCodec.from_names(["acme.search"])

    assert codec.decode("acme.search") == "acme.search"


def test_poisoned_history_name_no_longer_shadows_the_registered_tool():
    """Regression for the live self-perpetuating unknown-tool failure.

    Seed: an earlier request was built without ``torrent.torrent_add`` (its
    codec could not decode ``torrent__torrent_add``), so the preserved wire
    name was stored into session history as if it were canonical. From then
    on every ``from_request`` saw both names — and the poisoned flat name used
    to keep the natural alias, locking the registered tool out of it.
    """
    poisoned = Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=MessageRole.ASSISTANT,
        content=[ToolUseBlock(id="call_1", name="torrent__torrent_add", arguments={})],
        created_at=datetime.now(UTC),
    )
    request = ProviderRequest(
        model="openai/gpt-5.6-luna",
        messages=[poisoned],
        tools=[ToolDefinition(name="torrent.torrent_add", description="", parameters={})],
    )
    codec = ToolNameCodec.from_request(request)

    # The registered tool is advertised under its natural alias again …
    assert codec.encode("torrent.torrent_add") == "torrent__torrent_add"
    # … so the model calling the name it has seen work reaches the real tool:
    assert codec.decode("torrent__torrent_add") == "torrent.torrent_add"
    # The poisoned history entry keeps a distinct alias — replay stays valid
    # and the mapping stays bijective.
    assert codec.encode("torrent__torrent_add") == "torrent__torrent_add_2"
    assert codec.decode("torrent__torrent_add_2") == "torrent__torrent_add"


def test_request_codec_includes_tool_choice_and_replayed_history():
    history = Message(
        id=new_id(MessageId),
        session_id=new_id(SessionId),
        role=MessageRole.ASSISTANT,
        content=[ToolUseBlock(id="call_1", name="kit.current_time", arguments={})],
        created_at=datetime.now(UTC),
    )
    request = ProviderRequest(
        model="openai/gpt-5.6-luna",
        messages=[history],
        tools=[ToolDefinition(name="acme.search", description="søk", parameters={})],
        tool_choice=NamedToolChoice(name="acme.search"),
    )
    codec = ToolNameCodec.from_request(request)

    assert codec.encode("kit.current_time") == "kit__current_time"
    assert codec.encode("acme.search") == "acme__search"


def test_empty_name_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        ToolNameCodec.from_names([""])
