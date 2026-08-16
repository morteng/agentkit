"""The rehydration half of the PII firewall has no consumer path.

CHARACTERIZATION, not aspiration. Every assertion here describes what agentkit
does *today*. If one starts failing, the gap has been closed and this file
should be rewritten as a round-trip test rather than deleted.

Why this file exists
--------------------
``Firewall.scrub_request`` replaces a name with ``[NAME_1]`` on the way to the
provider. Nothing puts it back. ``ScrubbingProvider.stream`` says so directly::

    else:
        # Text deltas and tool-call args keep placeholders by design.
        yield event

That is a defensible design — a consumer may want to persist the tokenised form
— but it makes rehydration the consumer's job, and agentkit ships:

* ``Firewall.rehydrate_output``, whose only caller in the package is
  ``Firewall._rehydrate_json`` (i.e. itself);
* ``Firewall.assert_no_residual_tokens``, the guard written for exactly this
  miss, with **no callers at all**;
* no streaming helper, and no test anywhere that a consumer *can* do the job.

Both functions are individually correct and jointly unreachable. This is not
hypothetical: a downstream consumer wired the scrubbing half, never wired the
other, and put a raw placeholder in front of an end user as an example
username. Nothing in this package would have told them.

The two things a consumer actually needs, neither of which exists here:

1. **A streaming rehydrator.** ``rehydrate_output`` takes a whole string. A
   provider emits deltas, and a token splits across them (``[NAME_`` then
   ``1]``), so the obvious per-delta call silently misses every token that
   straddles a chunk boundary. ``test_a_token_split_across_deltas_defeats_the_
   only_available_helper`` below is that failure, executable.
2. **A place the guard is called.** ``assert_no_residual_tokens`` can only fire
   if a consumer remembers to call it, which is the same class of mistake it
   exists to catch.
"""

import pytest

from agentkit._content import TextBlock
from agentkit._messages import MessageRole
from agentkit.pii.firewall import Firewall, ResidualTokenError
from agentkit.pii.policy import PiiPolicy
from agentkit.pii.provider import wrap_provider
from agentkit.providers.base import ProviderRequest, TextDelta
from agentkit.providers.fakes import FakeProvider

from .conftest import FakeDetector, FakeTokenMap, make_msg


def _fw(detector: FakeDetector) -> Firewall:
    return Firewall(detector=detector, policy=PiiPolicy())


def _req(text: str) -> ProviderRequest:
    return ProviderRequest(
        model="openai/gpt-5.5",
        messages=[make_msg(MessageRole.USER, TextBlock(text=text))],
    )


async def _text_of(agen) -> str:  # type: ignore[no-untyped-def]
    parts = [e.delta async for e in agen if isinstance(e, TextDelta)]
    return "".join(parts)


async def test_a_placeholder_in_the_reply_reaches_the_consumer_verbatim(
    detector: FakeDetector, tmap: FakeTokenMap
):
    """The headline gap: what the model says is what the user sees.

    The model has just been handed a request containing ``[NAME_1]``, so it
    naturally uses that string in its reply. Nothing between the provider and
    the consumer turns it back into a person's name.
    """
    token = tmap.token_for("Kari Nordmann", "NAME")
    inner = FakeProvider().script(FakeProvider.text(f"Skal jeg sende den til {token}?"))
    wrapped = wrap_provider(inner, _fw(detector), tmap_resolver=lambda _req: tmap)

    out = await _text_of(wrapped.stream(_req("Kari Nordmann vil ha en konfigurasjon")))

    assert token in out
    assert "Kari Nordmann" not in out


async def test_the_guard_would_catch_it_and_is_never_invoked(
    detector: FakeDetector, tmap: FakeTokenMap
):
    """``assert_no_residual_tokens`` works. Nothing calls it.

    Both halves are asserted, because the point is the join: the escape happens
    on a path where the detector of that escape is not installed.
    """
    token = tmap.token_for("Kari Nordmann", "NAME")
    inner = FakeProvider().script(FakeProvider.text(f"Hei {token}!"))
    fw = _fw(detector)
    wrapped = wrap_provider(inner, fw, tmap_resolver=lambda _req: tmap)

    out = await _text_of(wrapped.stream(_req("Kari Nordmann")))

    # It escaped...
    assert token in out
    # ...and the guard, called by hand, proves it should not have.
    with pytest.raises(ResidualTokenError):
        fw.assert_no_residual_tokens(out)


def test_the_only_rehydration_helper_requires_a_complete_string(
    detector: FakeDetector, tmap: FakeTokenMap
):
    """``rehydrate_output`` is whole-string. That is fine, and insufficient."""
    token = tmap.token_for("Kari Nordmann", "NAME")
    fw = _fw(detector)

    assert fw.rehydrate_output(f"Hei {token}!", tmap) == "Hei Kari Nordmann!"


@pytest.mark.parametrize("split_at", range(1, len("[NAME_1]")))
def test_a_token_split_across_deltas_defeats_the_only_available_helper(
    detector: FakeDetector, tmap: FakeTokenMap, split_at: int
):
    """Per-delta rehydration — the obvious consumer implementation — is wrong.

    Parametrized over every interior split point, because the bug's signature
    is that it depends on where the provider happens to chunk. A consumer that
    tests one split sees it pass and ships an intermittent leak.
    """
    token = tmap.token_for("Kari Nordmann", "NAME")
    assert token == "[NAME_1]", "fixture drifted; the split offsets assume this token"
    fw = _fw(detector)

    deltas = [token[:split_at], token[split_at:]]
    naive = "".join(fw.rehydrate_output(d, tmap) for d in deltas)

    # Neither half contains the whole token, so neither is replaced.
    assert naive == token
    assert "Kari Nordmann" not in naive
    # The same bytes, rehydrated whole, are correct — so the data was always
    # sufficient. Only the chunking defeated it.
    assert fw.rehydrate_output("".join(deltas), tmap) == "Kari Nordmann"


def test_package_ships_no_streaming_rehydrator():
    """Guard against this file going stale.

    If agentkit grows a streaming rehydrator, this fails and the whole file
    should be rewritten as a round-trip test.
    """
    from agentkit import pii

    # These two exist today and are about tool ARGUMENTS, not text output, so
    # they are not what this file is waiting for. Named explicitly rather than
    # loosening the match, so a genuinely new export still trips the wire.
    known = {"RehydratePolicy", "RehydrationRefused"}
    candidates = {n for n in dir(pii) if "rehydr" in n.lower() or "stream" in n.lower()} - known
    assert candidates == set(), (
        f"agentkit.pii now exports {sorted(candidates)} — if one of these is a "
        "streaming rehydrator, close this gap and rewrite this module."
    )
