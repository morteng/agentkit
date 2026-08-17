"""The pure ``Firewall`` — scrub / rehydrate primitives + routing derivation.

Every method is pure (no I/O, no mutation of inputs). ``wrap_provider`` (in
``provider.py``) composes these at the Provider boundary.

Egress coverage
---------------
``scrub_request`` walks every text-bearing field of a ``ProviderRequest``:

============================  =========================================
surface                       scrubbed
============================  =========================================
``system[].text``             yes
user / assistant ``TextBlock``  yes, at any nesting depth
``ThinkingBlock.text``        yes, unless the block carries a provider
                              ``signature`` (rewriting signed thinking
                              invalidates it and the provider rejects
                              the turn — and those bytes are that
                              provider's own output coming home)
``ToolUseBlock.arguments``    yes — every string leaf of the JSON, with
                              the field name passed to the detector
``ToolResultBlock.content``   yes — recursively, including tool-use and
                              nested tool-result blocks
``tools[]`` (schemas)         no — static, operator-authored
``ImageBlock.data`` / ``url`` no — binary payload and fetch URL; a
                              redacted URL is a broken request
``metadata``                  no — provider routing/telemetry keys
============================  =========================================

Arguments and results are the two directions of the same hole: an argument is
how a secret leaks *out* to a tool, a result is how it leaks *in* to the model
and onward to the inference provider. Both are covered, and both are walked
with field names so a ``{"password": …}`` leaf is recognisable as one.
"""

import hashlib
import re
from collections import Counter
from typing import TYPE_CHECKING, Any

from agentkit._content import TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from agentkit.pii.audit import OutboundAudit
from agentkit.pii.policy import PiiPolicy
from agentkit.pii.protocols import Detector, FieldContextDetector, TokenMap
from agentkit.pii.spans import merge_spans
from agentkit.pii.types import Action, RehydratePolicy
from agentkit.providers.base import ProviderRequest, RoutingPreferences

if TYPE_CHECKING:  # avoid a runtime import cycle (spec imports pii.types)
    from agentkit.tools.spec import ToolSpec

#: Placeholder pattern for the fail-closed finalize gate. Matches ``[EMAIL]``,
#: ``[CANDIDATE_NAME]``, ``[PHONE_1]`` etc.
#:
#: Deliberately broad, and it must stay that way. The two errors this gate can
#: make are not symmetric: a false positive (a bracketed-capitals literal like
#: a ``[FLAC]`` scene tag tripping it) refuses one finalize, loudly, and the
#: caller can see and route around it; a false negative lets a real PII
#: placeholder through an *irreversible* send, silently. Consumers mint their
#: own token shapes — the docstring above promises suffix-less ``[EMAIL]`` and
#: ``[CANDIDATE_NAME]`` are caught — and a model can invent or mangle a
#: placeholder into any nearby shape, so no narrower pattern is safe here.
#: (The token map cannot replace the pattern either: a mangled or invented
#: placeholder is by definition not in the map, and those are exactly the
#: "map miss" cases the gate exists for.) Do NOT reuse this pattern where the
#: risk runs the other way — counting or classifying, where breadth
#: fabricates matches; ``build_audit`` learned that the hard way and now
#: counts what the scrubber actually replaced instead of scanning.
_RESIDUAL_TOKEN_RE = re.compile(r"\[[A-Z_]+(_\d+)?\]")

#: Arg-field name substrings that a rehydrate=ALLOW tool must NOT accept — a
#: model-supplied destination is an exfiltration channel (§6, C3 finding).
_DESTINATION_FIELD_MARKERS = (
    "url",
    "uri",
    "href",
    "endpoint",
    "callback",
    "webhook",
    "redirect",
    "destination",
    "dest",
    "recipient",
)


class ResidualTokenError(ValueError):
    """A placeholder survived to the fail-closed finalize gate."""


class RehydrationRefused(ValueError):
    """A ``rehydrate=ALLOW`` tool declared a model-supplied destination field."""


class Firewall:
    """Pure PII scrub/rehydrate operations parameterised by a detector + policy.

    The ``TokenMap`` is passed per-call (it is per-candidate and durable, owned
    by the consumer), never held on the firewall.
    """

    def __init__(self, detector: Detector, policy: PiiPolicy) -> None:
        self.detector = detector
        self.policy = policy
        #: ``detector`` again if it implements the optional field-name
        #: extension, else ``None``. Resolved once — ``isinstance`` against a
        #: runtime-checkable Protocol is not free and ``scrub_text`` runs once
        #: per string leaf of every request.
        self._field_detector: FieldContextDetector | None = (
            detector if isinstance(detector, FieldContextDetector) else None
        )

    # ---- Primitive ---------------------------------------------------------

    def scrub_text(
        self,
        text: str,
        tmap: TokenMap,
        *,
        field: str | None = None,
        counts: Counter[str] | None = None,
    ) -> str:
        """Run the detector; replace spans right-to-left so offsets stay valid.

        NEVER_SEND → ``[REDACTED]`` (never tokenized, never stored). TOKENIZE →
        a stable placeholder from ``tmap``.

        ``field`` is the name of the JSON field ``text`` was found under, when
        there is one. Detectors implementing
        :class:`~agentkit.pii.protocols.FieldContextDetector` receive it;
        others are called through plain ``detect`` exactly as before.

        ``counts``, when supplied, is incremented once per replacement actually
        made — by span kind for TOKENIZE, under ``"REDACTED"`` for NEVER_SEND.
        This is the ground truth :meth:`build_audit` reports as
        ``substitutions``: counting the replacements as they happen is the only
        definition that cannot be fooled by placeholder-shaped literals already
        in the text or by placeholders replayed from earlier turns' history.

        Overlapping spans are resolved by
        :func:`~agentkit.pii.spans.merge_spans` before substitution — two
        detectors matching the same characters would otherwise corrupt the
        output rather than scrub it.
        """
        if not text:
            return text
        if field is not None and self._field_detector is not None:
            spans = self._field_detector.detect_in_field(text, field)
        else:
            spans = self.detector.detect(text)
        if not spans:
            return text
        # Right-to-left by start offset: later replacements never shift the
        # offsets of earlier (leftward) spans.
        out = text
        for span in sorted(merge_spans(spans), key=lambda s: s.start, reverse=True):
            value = text[span.start : span.end]
            if span.action is Action.NEVER_SEND:
                replacement = "[REDACTED]"
                if counts is not None:
                    counts["REDACTED"] += 1
            else:
                replacement = tmap.token_for(value, span.kind)
                if counts is not None:
                    counts[span.kind] += 1
            out = out[: span.start] + replacement + out[span.end :]
        return out

    # ---- Request scrubbing (egress) ----------------------------------------

    def scrub_request(
        self,
        req: ProviderRequest,
        tmap: TokenMap,
        *,
        counts: Counter[str] | None = None,
    ) -> ProviderRequest:
        """Deep-copy ``req`` and scrub every text-bearing field. Never mutates input.

        ``counts``, when supplied, accumulates the replacements actually made
        (see :meth:`scrub_text`) — pass it on to :meth:`build_audit` so the
        audit reports this request's real scrubbing work.

        See the module docstring for the surface-by-surface coverage table.
        """
        scrubbed: ProviderRequest = req.model_copy(deep=True)
        for block in scrubbed.system:
            block.text = self.scrub_text(block.text, tmap, counts=counts)
        for msg in scrubbed.messages:
            for content in msg.content:
                self._scrub_content_block(content, tmap, counts)
        return scrubbed

    def _scrub_content_block(
        self, block: Any, tmap: TokenMap, counts: Counter[str] | None = None
    ) -> None:
        if isinstance(block, TextBlock):
            block.text = self.scrub_text(block.text, tmap, counts=counts)
        elif isinstance(block, ThinkingBlock):
            # A signed thinking block must go back to the provider byte-exact
            # or the signature fails — and it is that provider's own output
            # returning, so scrubbing it protects nothing. Unsigned thinking
            # (other providers, replayed transcripts) is scrubbed.
            if block.signature is None:
                block.text = self.scrub_text(block.text, tmap, counts=counts)
        elif isinstance(block, ToolUseBlock):
            block.arguments = self._scrub_json(block.arguments, tmap, counts=counts)
        elif isinstance(block, ToolResultBlock):
            for inner in block.content:
                self._scrub_content_block(inner, tmap, counts)

    def _scrub_json(
        self,
        value: Any,
        tmap: TokenMap,
        field: str | None = None,
        *,
        counts: Counter[str] | None = None,
    ) -> Any:
        """Recursively scrub string leaves of a JSON-ish structure.

        ``field`` carries the enclosing key down to the leaf, so a detector can
        tell ``{"password": "sommer2026"}`` from ``{"city": "sommer2026"}``.
        List elements inherit the key of the list itself.
        """
        if isinstance(value, str):
            return self.scrub_text(value, tmap, field=field, counts=counts)
        if isinstance(value, dict):
            return {k: self._scrub_json(v, tmap, str(k), counts=counts) for k, v in value.items()}  # type: ignore[reportUnknownVariableType]
        if isinstance(value, list):
            return [self._scrub_json(v, tmap, field, counts=counts) for v in value]  # type: ignore[reportUnknownVariableType]
        return value

    # ---- Rehydration -------------------------------------------------------

    def rehydrate_output(self, text: str, tmap: TokenMap) -> str:
        """Replace every known token back to its real value (server-side only).

        Longest tokens first so ``[EMAIL_11]`` is not clobbered by ``[EMAIL_1]``.
        """
        if not text:
            return text
        out = text
        for token in sorted(tmap.all_tokens(), key=len, reverse=True):
            if token in out:
                value = tmap.value_for(token)
                if value is not None:
                    out = out.replace(token, value)
        return out

    def rehydrate_tool_args(
        self, tool: "ToolSpec", args: dict[str, Any], tmap: TokenMap
    ) -> dict[str, Any]:
        """Per-tool rehydration of model-produced tool arguments.

        DENY (default): return args unchanged — tokens stay tokens, closing the
        exfiltration channel. ALLOW: refuse if the tool declares a model-supplied
        destination field; otherwise rehydrate only exact known tokens within
        arg fields.
        """
        if tool.rehydrate is RehydratePolicy.DENY:
            return args
        # ALLOW — validate no model-supplied destination fields.
        for field in args:
            lowered = field.lower()
            if any(marker in lowered for marker in _DESTINATION_FIELD_MARKERS):
                raise RehydrationRefused(
                    f"tool {tool.name!r} has rehydrate=ALLOW but arg {field!r} looks "
                    "like a model-supplied destination — refusing to rehydrate real "
                    "PII into an exfiltration channel"
                )
        return {k: self._rehydrate_json(v, tmap) for k, v in args.items()}

    def _rehydrate_json(self, value: Any, tmap: TokenMap) -> Any:
        if isinstance(value, str):
            return self.rehydrate_output(value, tmap)
        if isinstance(value, dict):
            return {k: self._rehydrate_json(v, tmap) for k, v in value.items()}  # type: ignore[reportUnknownVariableType]
        if isinstance(value, list):
            return [self._rehydrate_json(v, tmap) for v in value]  # type: ignore[reportUnknownVariableType]
        return value

    # ---- Fail-closed finalize gate -----------------------------------------

    def assert_no_residual_tokens(self, text: str) -> None:
        """Raise if any placeholder survived to an irreversible send (§8)."""
        match = _RESIDUAL_TOKEN_RE.search(text)
        if match is not None:
            raise ResidualTokenError(
                f"residual PII placeholder {match.group(0)!r} survived to finalize — "
                "refusing to export/submit (map miss or mangled token)"
            )

    # ---- Routing derivation ------------------------------------------------

    def routing_prefs(self) -> RoutingPreferences | None:
        """Derive request routing preferences from the policy.

        Only ever called for a request the firewall is *active* on — i.e. one
        carrying PII. Saying nothing about data collection on such a request
        means taking the provider's default, and provider defaults are "may
        retain, may train". So the firewall states a preference explicitly
        rather than letting silence pick one:

        * ``require_zdr`` → ``zdr``, ``data_collection="deny"`` and
          ``allow_fallbacks=False``, the fail-closed route.
        * otherwise → :attr:`PiiPolicy.default_data_collection`, which is
          ``"deny"`` out of the box: no zero-retention *requirement*, but an
          explicit request not to have the payload collected, with fallbacks
          still allowed so routing never hard-fails.

        Returns ``None`` only when the operator has opted out with
        ``default_data_collection="unset"`` and imposes neither ZDR nor EU
        residency — the pre-0.22 behaviour, byte-identical payloads.
        """
        policy = self.policy
        if policy.require_zdr:
            return RoutingPreferences(
                zdr=True,
                data_collection="deny",
                # Fail closed: with allow_fallbacks False, OpenRouter errors
                # rather than routing to a non-compliant provider.
                allow_fallbacks=False,
                eu_only=policy.eu_only,
            )
        if policy.default_data_collection == "unset":
            if not policy.eu_only:
                return None
            return RoutingPreferences(
                zdr=False, data_collection="allow", allow_fallbacks=True, eu_only=True
            )
        return RoutingPreferences(
            zdr=False,
            data_collection=policy.default_data_collection,
            allow_fallbacks=True,
            eu_only=policy.eu_only,
        )

    # ---- Audit helper (used by the decorator) ------------------------------

    def build_audit(
        self,
        scrubbed: ProviderRequest,
        *,
        substitutions: Counter[str] | None = None,
    ) -> OutboundAudit:
        """Build an outbound-audit record from an already-scrubbed request.

        ``substitutions`` is the counter :meth:`scrub_request` filled while
        scrubbing this request — the replacements the scrubber actually made.
        Pass it whenever you have it: it is the only faithful source for the
        record. Kinds are upper-cased for the record so it matches the
        placeholder casing token maps mint (``[PERSON_NAME_3]`` → key
        ``PERSON_NAME``); the ``REDACTED`` key becomes ``never_send_hits``.

        Without it, the legacy fallback *scans* the scrubbed text with the gate
        pattern and reports every bracketed-capitals match as a substitution.
        That is an approximation with two known distortions, kept only for
        backward compatibility: any ``[BRACKETED_CAPITALS]`` literal in the
        payload is reported as a substitution of a detector kind that does not
        exist (observed live: torrent scene tags produced
        ``{'FLAC': 13, 'PMEDIA': 4, …}`` in a record meant to prove what left
        the house), and placeholders replayed from earlier turns' history are
        re-counted on every request, so counts inflate monotonically over a
        session.
        """
        texts: list[str] = [b.text for b in scrubbed.system]
        for msg in scrubbed.messages:
            texts.extend(self._collect_texts(msg.content))
        blob = "\n".join(texts)
        counts: Counter[str] = Counter()
        if substitutions is not None:
            for kind, n in substitutions.items():
                counts[kind.upper()] += n
        else:
            for token in _RESIDUAL_TOKEN_RE.finditer(blob):
                kind = re.sub(r"_\d+$", "", token.group(0)[1:-1])
                counts[kind] += 1
        never_send = counts.pop("REDACTED", 0)
        return OutboundAudit(
            scrubbed_hash=hashlib.sha256(blob.encode("utf-8")).hexdigest(),
            substitutions=dict(counts),
            never_send_hits=never_send,
            model=scrubbed.model,
        )

    def _collect_texts(self, blocks: list[Any]) -> list[str]:
        out: list[str] = []
        for b in blocks:
            if isinstance(b, TextBlock | ThinkingBlock):
                out.append(b.text)
            elif isinstance(b, ToolUseBlock):
                out.append(str(b.arguments))
            elif isinstance(b, ToolResultBlock):
                out.extend(self._collect_texts(b.content))
        return out
