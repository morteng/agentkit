"""Wire-level error codes.

:class:`ErrorCode` is re-exported from :mod:`agentkit.events`, which is still
where a consumer imports it from and where its place in the event schema is
documented. It is *defined* here, in a module that imports nothing from
agentkit, because two things need it and they sit on opposite sides of an
import cycle: :mod:`agentkit.events.lifecycle` puts it on the ``Errored``
event, and :mod:`agentkit.errors` lets every exception declare the code it ends
a turn with. Importing it from the events package inside ``errors`` closes the
loop — ``events/__init__`` pulls in the approval events, which reach into
``guards``, which reach into the tool registry, which imports ``errors``. A
leaf cannot close a cycle, so it lives in one.
"""

from enum import StrEnum


class ErrorCode(StrEnum):
    PROVIDER_FAULT = "provider_fault"
    TOOL_FAULT = "tool_fault"
    RATE_LIMITED = "rate_limited"
    INTENT_REJECTED = "intent_rejected"
    APPROVAL_TIMEOUT = "approval_timeout"
    #: A policy this runtime enforces refused the call before it was made —
    #: today the PII firewall's two fail-closed paths (no zero-data-retention
    #: route for a PII-carrying request; a model in ``blocked_models``).
    #:
    #: Distinct from INTERNAL because nothing malfunctioned, and distinct from
    #: INTENT_REJECTED because that is the intent gate judging the *request*
    #: while this is the egress boundary judging the *route*. The difference is
    #: load-bearing for a consumer: a refusal has a remedy the caller can act on
    #: (choose a different model, or relax the policy), and retrying unchanged
    #: is guaranteed to fail — neither of which is true of a crash.
    POLICY_REFUSED = "policy_refused"
    INTERNAL = "internal"
