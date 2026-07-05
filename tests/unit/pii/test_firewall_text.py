"""scrub_text / rehydrate_output / assert_no_residual_tokens."""

import pytest

from agentkit.pii.firewall import Firewall, ResidualTokenError
from agentkit.pii.policy import PiiPolicy

from .conftest import FakeDetector, FakeTokenMap


def _fw(detector: FakeDetector) -> Firewall:
    return Firewall(detector=detector, policy=PiiPolicy())


def test_scrub_text_tokenizes(detector: FakeDetector, tmap: FakeTokenMap):
    fw = _fw(detector)
    out = fw.scrub_text("Contact Kari Nordmann at kari@example.no", tmap)
    assert "Kari Nordmann" not in out
    assert "kari@example.no" not in out
    assert "[NAME_1]" in out
    assert "[EMAIL_1]" in out


def test_scrub_text_never_send_is_dropped_not_tokenized(detector: FakeDetector, tmap: FakeTokenMap):
    fw = _fw(detector)
    out = fw.scrub_text("fnr 12345678901 here", tmap)
    assert "12345678901" not in out
    assert "[REDACTED]" in out
    # NEVER_SEND is never stored in the map.
    assert "12345678901" not in tmap.all_tokens()
    assert all(tmap.value_for(t) != "12345678901" for t in tmap.all_tokens())


def test_right_to_left_offsets_multiple_spans(detector: FakeDetector, tmap: FakeTokenMap):
    # Two emails + a name; ensure every one is replaced correctly.
    text = "a@b.no and Kari Nordmann and c@d.no"
    fw = _fw(detector)
    out = fw.scrub_text(text, tmap)
    assert "a@b.no" not in out
    assert "c@d.no" not in out
    assert "Kari Nordmann" not in out
    assert out.count("[EMAIL_") == 2
    assert "[NAME_1]" in out


def test_scrub_text_deterministic_same_value_same_token(detector: FakeDetector, tmap: FakeTokenMap):
    fw = _fw(detector)
    out1 = fw.scrub_text("kari@example.no", tmap)
    out2 = fw.scrub_text("again kari@example.no", tmap)
    assert out1.strip() == out2.replace("again ", "").strip()


def test_scrub_text_empty(detector: FakeDetector, tmap: FakeTokenMap):
    assert _fw(detector).scrub_text("", tmap) == ""


def test_rehydrate_output_round_trip(detector: FakeDetector, tmap: FakeTokenMap):
    fw = _fw(detector)
    scrubbed = fw.scrub_text("Kari Nordmann kari@example.no", tmap)
    back = fw.rehydrate_output(scrubbed, tmap)
    assert back == "Kari Nordmann kari@example.no"


def test_rehydrate_longer_tokens_first(detector: FakeDetector, tmap: FakeTokenMap):
    fw = _fw(detector)
    # Force 11 email tokens so [EMAIL_1] and [EMAIL_11] coexist.
    for i in range(11):
        tmap.token_for(f"user{i}@x.no", "EMAIL")
    text = "[EMAIL_11] and [EMAIL_1]"
    out = fw.rehydrate_output(text, tmap)
    assert out == "user10@x.no and user0@x.no"


def test_assert_no_residual_tokens_passes_clean(detector: FakeDetector):
    _fw(detector).assert_no_residual_tokens("all clean text here")


@pytest.mark.parametrize("bad", ["[EMAIL_1]", "hi [CANDIDATE_NAME] there", "x [REDACTED] y"])
def test_assert_no_residual_tokens_raises(detector: FakeDetector, bad: str):
    with pytest.raises(ResidualTokenError):
        _fw(detector).assert_no_residual_tokens(bad)
