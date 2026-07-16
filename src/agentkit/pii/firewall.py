"""The pure ``Firewall`` — scrub / rehydrate primitives + routing derivation.

Every method is pure (no I/O, no mutation of inputs). ``wrap_provider`` (in
``provider.py``) composes these at the Provider boundary.
"""

import hashlib
import re
from collections import Counter
from typing import TYPE_CHECKING, Any

from agentkit._content import TextBlock, ToolResultBlock, ToolUseBlock
from agentkit.pii.audit import OutboundAudit
from agentkit.pii.policy import PiiPolicy
from agentkit.pii.protocols import Detector, TokenMap
from agentkit.pii.types import Action, RehydratePolicy
from agentkit.providers.base import ProviderRequest, RoutingPreferences

if TYPE_CHECKING:  # avoid a runtime import cycle (spec imports pii.types)
    from agentkit.tools.spec import ToolSpec

#: Placeholder pattern for the fail-closed finalize gate. Matches ``[EMAIL]``,
#: ``[CANDIDATE_NAME]``, ``[PHONE_1]`` etc.
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

    # ---- Primitive ---------------------------------------------------------

    def scrub_text(self, text: str, tmap: TokenMap) -> str:
        """Run the detector; replace spans right-to-left so offsets stay valid.

        NEVER_SEND → ``[REDACTED]`` (never tokenized, never stored). TOKENIZE →
        a stable placeholder from ``tmap``.
        """
        if not text:
            return text
        spans = self.detector.detect(text)
        if not spans:
            return text
        # Right-to-left by start offset: later replacements never shift the
        # offsets of earlier (leftward) spans.
        out = text
        for span in sorted(spans, key=lambda s: s.start, reverse=True):
            value = text[span.start : span.end]
            if span.action is Action.NEVER_SEND:
                replacement = "[REDACTED]"
            else:
                replacement = tmap.token_for(value, span.kind)
            out = out[: span.start] + replacement + out[span.end :]
        return out

    # ---- Request scrubbing (egress) ----------------------------------------

    def scrub_request(self, req: ProviderRequest, tmap: TokenMap) -> ProviderRequest:
        """Deep-copy ``req`` and scrub every text-bearing field. Never mutates input.

        Scrubs: system blocks, message text blocks, tool-result inner text,
        tool-use argument string values. Does NOT touch ``tools`` (static
        schemas) or non-text content.
        """
        scrubbed: ProviderRequest = req.model_copy(deep=True)
        for block in scrubbed.system:
            block.text = self.scrub_text(block.text, tmap)
        for msg in scrubbed.messages:
            for content in msg.content:
                self._scrub_content_block(content, tmap)
        return scrubbed

    def _scrub_content_block(self, block: Any, tmap: TokenMap) -> None:
        if isinstance(block, TextBlock):
            block.text = self.scrub_text(block.text, tmap)
        elif isinstance(block, ToolUseBlock):
            block.arguments = self._scrub_json(block.arguments, tmap)
        elif isinstance(block, ToolResultBlock):
            for inner in block.content:
                self._scrub_content_block(inner, tmap)

    def _scrub_json(self, value: Any, tmap: TokenMap) -> Any:
        """Recursively scrub string leaves of a JSON-ish structure."""
        if isinstance(value, str):
            return self.scrub_text(value, tmap)
        if isinstance(value, dict):
            return {k: self._scrub_json(v, tmap) for k, v in value.items()}  # type: ignore[reportUnknownVariableType]
        if isinstance(value, list):
            return [self._scrub_json(v, tmap) for v in value]  # type: ignore[reportUnknownVariableType]
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

        ``None`` when the policy imposes nothing (no ZDR, no EU) so the payload
        is unchanged.
        """
        if not self.policy.require_zdr and not self.policy.eu_only:
            return None
        return RoutingPreferences(
            zdr=self.policy.require_zdr,
            data_collection="deny" if self.policy.require_zdr else "allow",
            # Fail closed: with allow_fallbacks False, OpenRouter errors rather
            # than routing to a non-compliant provider.
            allow_fallbacks=not self.policy.require_zdr,
            eu_only=self.policy.eu_only,
        )

    # ---- Audit helper (used by the decorator) ------------------------------

    def build_audit(self, scrubbed: ProviderRequest) -> OutboundAudit:
        """Build an outbound-audit record from an already-scrubbed request."""
        texts: list[str] = [b.text for b in scrubbed.system]
        for msg in scrubbed.messages:
            texts.extend(self._collect_texts(msg.content))
        blob = "\n".join(texts)
        counts: Counter[str] = Counter()
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
            if isinstance(b, TextBlock):
                out.append(b.text)
            elif isinstance(b, ToolUseBlock):
                out.append(str(b.arguments))
            elif isinstance(b, ToolResultBlock):
                out.extend(self._collect_texts(b.content))
        return out
