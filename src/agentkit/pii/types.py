"""Import-safe PII value types.

This module deliberately imports NOTHING from the rest of agentkit so it can be
imported from ``tools/spec.py`` (for the ``rehydrate`` field) without creating a
circular import back through the ``pii`` package's heavier modules (which import
``providers.base``). Keep it dependency-light.
"""

from enum import StrEnum

from pydantic import BaseModel


class Action(StrEnum):
    """What the firewall does with a detected span.

    - ``NEVER_SEND``: strip and drop — never tokenized, never stored, never sent
      (fødselsnummer, kontonummer, IBAN, DOB, passport, …).
    - ``TOKENIZE``: replace with a stable placeholder via the ``TokenMap``.
    """

    NEVER_SEND = "never_send"
    TOKENIZE = "tokenize"


class RehydratePolicy(StrEnum):
    """Per-tool rehydration policy for model-produced tool arguments.

    Default ``DENY`` closes the exfiltration channel: tokens stay tokens in
    model-controlled tool args. ``ALLOW`` is reserved for artifact-render /
    form-fill tools with no model-supplied destination fields.
    """

    DENY = "deny"
    ALLOW = "allow"


class Span(BaseModel):
    """A detected region of text.

    Half-open ``[start, end)`` character offsets into the string handed to
    ``Detector.detect``.
    """

    start: int
    end: int
    kind: str
    action: Action

    model_config = {"frozen": True}
