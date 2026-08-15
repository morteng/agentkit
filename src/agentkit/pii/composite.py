"""``CompositeDetector`` — add domain recognizers without dropping the built-ins.

The failure mode this exists to prevent is real and was found in the wild: a
consumer needed Norwegian national-ID and bank-account patterns, wrote a
five-pattern detector, passed it to :class:`~agentkit.pii.firewall.Firewall`,
and thereby replaced *everything else the firewall could see*. Credentials
walked straight through, because the ``Detector`` slot is singular and
"register my patterns" and "use only my patterns" were the same call.

With a composite, they are not::

    from agentkit.pii import CompositeDetector, Firewall, PiiPolicy

    # Built-in secret detection plus the consumer's own recognizers.
    detector = CompositeDetector.with_defaults(NorwegianPiiDetector())
    firewall = Firewall(detector, PiiPolicy())

Order does not matter: overlapping spans from different members are resolved
by :func:`~agentkit.pii.spans.merge_spans`, which prefers ``NEVER_SEND`` over
``TOKENIZE`` and the longer span over the shorter one. A member that raises is
*not* swallowed — a detector that cannot run is a fail-closed condition, and
silently scrubbing less than you think you are is the whole bug.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentkit.pii.protocols import FieldContextDetector
from agentkit.pii.secrets import SecretDetector
from agentkit.pii.spans import merge_spans

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from agentkit.pii.protocols import Detector
    from agentkit.pii.secrets import SecretPolicy
    from agentkit.pii.types import Span


class CompositeDetector:
    """Run every member detector and merge their spans.

    Implements both :class:`~agentkit.pii.protocols.Detector` and
    :class:`~agentkit.pii.protocols.FieldContextDetector`; members that
    implement the field-aware extension get the field name, members that do not
    are called through the plain ``detect``.
    """

    def __init__(self, detectors: Iterable[Detector]) -> None:
        self.detectors: tuple[Detector, ...] = tuple(detectors)

    @classmethod
    def with_defaults(
        cls,
        *extra: Detector,
        secret_policy: SecretPolicy | None = None,
    ) -> CompositeDetector:
        """The library's default detectors, plus ``extra``.

        Today the defaults are :class:`~agentkit.pii.secrets.SecretDetector`
        alone — agentkit ships no identity recognizers, those are the
        consumer's domain. Future built-ins land here, and consumers that
        composed this way pick them up for free.
        """
        return cls((SecretDetector(secret_policy), *extra))

    def detect(self, text: str) -> list[Span]:
        return self._merged(text, field=None, use_field=False)

    def detect_in_field(self, text: str, field: str | None) -> list[Span]:
        return self._merged(text, field=field, use_field=True)

    def _merged(self, text: str, *, field: str | None, use_field: bool) -> list[Span]:
        spans: list[Span] = []
        for detector in self.detectors:
            if use_field and isinstance(detector, FieldContextDetector):
                spans.extend(detector.detect_in_field(text, field))
            else:
                spans.extend(detector.detect(text))
        return merge_spans(spans)


def default_detector(
    *extra: Detector,
    secret_policy: SecretPolicy | None = None,
) -> CompositeDetector:
    """Shorthand for :meth:`CompositeDetector.with_defaults`."""
    return CompositeDetector.with_defaults(*extra, secret_policy=secret_policy)


def default_detectors(secret_policy: SecretPolicy | None = None) -> Sequence[Detector]:
    """The default detector set as a sequence, for callers assembling their own."""
    return (SecretDetector(secret_policy),)
