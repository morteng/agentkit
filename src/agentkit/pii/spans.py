"""Span arithmetic shared by the detectors and the firewall.

``Firewall.scrub_text`` replaces spans right-to-left, which is only correct
when the spans are disjoint. A single detector can return overlapping spans
(two patterns matching the same run), and a :class:`~agentkit.pii.composite.
CompositeDetector` makes overlap the normal case. ``merge_spans`` is the one
place that resolves them, so both the composite and the firewall agree on the
answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentkit.pii.types import Action

if TYPE_CHECKING:
    from collections.abc import Iterable

    from agentkit.pii.types import Span


def _priority(span: Span) -> tuple[int, int, int]:
    """Sort key: NEVER_SEND first, then longest, then leftmost.

    Fail-closed ordering — when a TOKENIZE span and a NEVER_SEND span cover the
    same characters the value is dropped rather than stored in the token map.
    """
    never_send = 0 if span.action is Action.NEVER_SEND else 1
    return (never_send, -(span.end - span.start), span.start)


def merge_spans(spans: Iterable[Span]) -> list[Span]:
    """Drop overlapping spans, keeping the highest-priority one of each cluster.

    Returns spans sorted by ``start``, guaranteed pairwise disjoint and
    well-formed (``0 <= start < end``). Degenerate spans are discarded.
    """
    ordered = sorted(
        (s for s in spans if s.end > s.start >= 0),
        key=_priority,
    )
    kept: list[Span] = []
    for span in ordered:
        if any(span.start < k.end and k.start < span.end for k in kept):
            continue
        kept.append(span)
    kept.sort(key=lambda s: s.start)
    return kept
