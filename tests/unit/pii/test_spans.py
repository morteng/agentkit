"""merge_spans: the overlap resolution both the composite and the firewall use."""

from itertools import pairwise

from agentkit.pii.spans import merge_spans
from agentkit.pii.types import Action, Span


def _span(start: int, end: int, kind: str = "K", action: Action = Action.TOKENIZE) -> Span:
    return Span(start=start, end=end, kind=kind, action=action)


def test_disjoint_spans_are_all_kept_and_sorted():
    out = merge_spans([_span(10, 15), _span(0, 5)])
    assert [(s.start, s.end) for s in out] == [(0, 5), (10, 15)]


def test_never_send_wins_over_tokenize_on_overlap():
    tokenize = _span(0, 20, "NAME", Action.TOKENIZE)
    never = _span(5, 10, "FNR", Action.NEVER_SEND)
    out = merge_spans([tokenize, never])
    assert [(s.kind, s.start, s.end) for s in out] == [("FNR", 5, 10)]


def test_longer_span_wins_between_equal_actions():
    out = merge_spans([_span(0, 5, "SHORT"), _span(0, 20, "LONG")])
    assert [s.kind for s in out] == ["LONG"]


def test_identical_spans_are_deduplicated():
    out = merge_spans([_span(3, 9, "A"), _span(3, 9, "B")])
    assert len(out) == 1


def test_degenerate_spans_are_dropped():
    assert merge_spans([_span(5, 5), _span(9, 3), _span(-1, 2)]) == []


def test_touching_spans_are_not_overlapping():
    out = merge_spans([_span(0, 5), _span(5, 10)])
    assert len(out) == 2


def test_output_is_pairwise_disjoint():
    spans = [_span(0, 10), _span(4, 6), _span(8, 20), _span(19, 25), _span(30, 31)]
    out = merge_spans(spans)
    for a, b in pairwise(out):
        assert a.end <= b.start


def test_empty_input():
    assert merge_spans([]) == []
