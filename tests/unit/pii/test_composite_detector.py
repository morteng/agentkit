"""CompositeDetector: adding domain patterns must not subtract the built-ins.

The bug this file guards against shipped once already — a consumer wrote a
five-pattern Norwegian detector, passed it to ``Firewall`` as *the* detector,
and silently turned off everything else the firewall could have caught.
"""

from itertools import pairwise

import pytest

from agentkit.pii.composite import CompositeDetector, default_detector, default_detectors
from agentkit.pii.protocols import Detector, FieldContextDetector
from agentkit.pii.secrets import SecretDetector, SecretPolicy
from agentkit.pii.types import Action, Span

from .conftest import FakeDetector

TOKEN = "nZ8Kq2vL9pXwT3aBcDeF"


class _AlwaysRaises:
    def detect(self, text: str) -> list[Span]:
        raise RuntimeError("detector exploded")


class _FieldAware:
    """Flags any value under a field literally named ``nickname``."""

    def detect(self, text: str) -> list[Span]:
        return []

    def detect_in_field(self, text: str, field: str | None) -> list[Span]:
        if field == "nickname" and text:
            return [Span(start=0, end=len(text), kind="NICK", action=Action.TOKENIZE)]
        return []


def test_composite_satisfies_the_detector_protocol():
    composite = CompositeDetector.with_defaults()
    assert isinstance(composite, Detector)
    assert isinstance(composite, FieldContextDetector)


def test_with_defaults_keeps_the_secret_detector():
    composite = CompositeDetector.with_defaults(FakeDetector())
    assert any(isinstance(d, SecretDetector) for d in composite.detectors)


def test_defaults_and_domain_patterns_both_fire():
    composite = CompositeDetector.with_defaults(FakeDetector())
    text = f"Kari Nordmann's password is {TOKEN}"
    kinds = {s.kind for s in composite.detect(text)}
    assert "NAME" in kinds  # the consumer's recognizer
    assert "SECRET" in kinds  # the built-in, not lost


def test_domain_detector_alone_misses_the_secret():
    """The regression itself: a bare domain detector sees nothing here."""
    assert FakeDetector().detect(f"password is {TOKEN}") == []


def test_default_detector_shorthand_matches_with_defaults():
    assert type(default_detector()) is CompositeDetector
    assert len(default_detector(FakeDetector()).detectors) == 2
    assert len(default_detectors()) == 1


def test_secret_policy_is_threaded_through():
    composite = CompositeDetector.with_defaults(secret_policy=SecretPolicy(action=Action.TOKENIZE))
    spans = composite.detect(f"password is {TOKEN}")
    assert spans and all(s.action is Action.TOKENIZE for s in spans)


def test_order_does_not_change_the_result():
    text = f"Kari Nordmann kari@example.no {TOKEN}"
    a = CompositeDetector([SecretDetector(), FakeDetector()]).detect(text)
    b = CompositeDetector([FakeDetector(), SecretDetector()]).detect(text)
    assert a == b


def test_overlapping_members_produce_disjoint_spans():
    # Two members that both match the whole string.
    composite = CompositeDetector([SecretDetector(), SecretDetector()])
    spans = composite.detect(f"password is {TOKEN}")
    assert len(spans) == 1
    for x, y in pairwise(spans):
        assert x.end <= y.start


def test_field_context_is_passed_to_members_that_want_it():
    composite = CompositeDetector([_FieldAware(), SecretDetector()])
    assert [s.kind for s in composite.detect_in_field("Kaia", "nickname")] == ["NICK"]
    # A member without the extension is still called through plain detect.
    assert composite.detect_in_field("AKIAIOSFODNN7EXAMPLE", "nickname")


def test_plain_detect_does_not_use_field_context():
    composite = CompositeDetector([_FieldAware()])
    assert composite.detect("Kaia") == []


def test_a_failing_member_is_not_swallowed():
    composite = CompositeDetector([SecretDetector(), _AlwaysRaises()])
    with pytest.raises(RuntimeError):
        composite.detect("anything")


def test_empty_composite_detects_nothing():
    assert CompositeDetector([]).detect(f"password is {TOKEN}") == []
