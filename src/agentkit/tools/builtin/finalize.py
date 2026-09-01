"""kit.finalize — signal that the turn is complete.

The handler sets a flag on the TurnContext; the loop's finalize_check phase
then validates the claim before terminating.

**The argument shape is the ``finalize_response`` envelope** (``status``,
``intent_kind``, ``summary``, ``actions_performed``, …). The handler stores
whatever it is given on ``ctx.finalize_args``, and
``loop.handlers.finalize_check`` hands that straight to the
:class:`~agentkit.guards.finalize.FinalizeValidator`, synthesizing
``{"reason": ...}`` only when nothing was recorded. ``reason`` survives as an
optional alias for ``summary`` so callers written against the pre-0.22 schema
keep working, but it satisfies no validator rule on its own and a call carrying
only ``reason`` is no longer schema-legal.

Before 0.22 the schema advertised ``reason`` as the only property *and* as
required, which was never true of either the handler (it uses ``args.get``) or
of any caller in this repo. That drift was invisible until the registry started
validating arguments before dispatch.

**``status`` and ``intent_kind`` are required.** They were not until 0.22.1, and
the omission made this tool unsatisfiable in the default configuration.
``AgentSession`` installs :class:`StructuralFinalizeValidator` whenever
``config.guards.finalize`` is unset, and that validator parses these arguments
as an :class:`~agentkit.envelope.Envelope`, which requires both fields. A model
reading the old schema saw ``reason`` listed first, described as the summary to
write, and nothing marked required — so it sent ``{"reason": "..."}``, and the
validator rejected it, every time. With ``max_finalize_retries=2`` that is three
rejected finalize calls per turn: attempt, retry, retry, then
``finalize_exhausted``. Each one costs a provider round-trip, and each one
reaches a chat UI as an assistant turn that produces no text — the model appears
to keep starting to speak and then saying nothing.

The earlier reasoning for requiring nothing was that ``actions_performed`` is
forbidden on an ``intent_kind="answer"`` envelope, so no single required set
covers both shapes. That is true of ``actions_performed`` specifically, and the
conclusion drawn from it was too broad: ``status`` and ``intent_kind`` are
unconditional in ``Envelope`` and safe to require. ``actions_performed`` stays
optional here for exactly the stated reason — which is why this list is written
out rather than reusing ``FINALIZE_RESPONSE_SCHEMA["required"]``.

The genuinely conditional rules — ``answer_evidence`` required given
``intent_kind="answer"``, ``pending_confirmation`` given ``"clarify"`` — a flat
JSON Schema ``required`` list cannot express. They live in the validator and are
stated in the description below, which is the only channel the model reads.
"""

from typing import Any

from agentkit.loop.context import TurnContext
from agentkit.tools.builtin.finalize_response import (
    FINALIZE_RESPONSE_DESCRIPTION,
    FINALIZE_RESPONSE_SCHEMA,
)
from agentkit.tools.spec import (
    ApprovalPolicy,
    ContentBlockOut,
    RiskLevel,
    SideEffects,
    ToolResult,
    ToolSpec,
)

# The bare (namespace-stripped) names a finalize tool may register under.
# Lives here, beside the tool, because two independent places need to recognise
# a finalize tool among arbitrary specs: `loop.handlers.streaming` when it
# constrains a re-prompt turn to it, and `loop.handlers.finalize_check` when it
# reads this spec's `required` list to tell the model what to send. A second
# copy of this tuple would be a third thing to keep in step with the schema.
FINALIZE_BARE_NAMES = ("finalize_response", "finalize")

_PREAMBLE = (
    "Signal that you have completed the user's request. "
    "Only call this once you have actually invoked the tools needed to "
    "carry out the user's request — do NOT call it just because you have "
    "produced a written response.\n\n"
)

FINALIZE_SPEC = ToolSpec(
    name="kit.finalize",
    # The envelope semantics are the same contract `finalize_response` states,
    # and the validator judging this call is the same one. Concatenating that
    # description rather than paraphrasing it keeps the two from drifting apart
    # again — the drift is what made this tool unsatisfiable.
    description=_PREAMBLE + FINALIZE_RESPONSE_DESCRIPTION,
    parameters={
        "type": "object",
        "properties": {
            **FINALIZE_RESPONSE_SCHEMA["properties"],
            "reason": {
                "type": "string",
                "description": (
                    "Optional alias for 'summary', kept for callers written "
                    "against the pre-0.22 schema. Prefer 'summary'. Listed "
                    "last deliberately: when it came first, models filled it "
                    "and omitted the envelope fields the validator requires."
                ),
            },
        },
        # Not FINALIZE_RESPONSE_SCHEMA["required"] — see the module docstring
        # for why `actions_performed` is excluded from this one.
        "required": ["status", "intent_kind"],
    },
    returns=None,
    risk=RiskLevel.READ,
    idempotent=True,
    side_effects=SideEffects.LOCAL,
    requires_approval=ApprovalPolicy.NEVER,
    cache_ttl_seconds=None,
    timeout_seconds=5.0,
)


async def finalize_handler(args: dict[str, Any], ctx: TurnContext) -> ToolResult:
    ctx.finalize_called = True
    # Fall back to the envelope's ``summary``: an envelope-shaped call carries
    # the human-readable sentence there, and without this fallback every such
    # call ended the turn with ``TurnEnded.summary=None`` — the field the tool
    # description promises to fill.
    reason = str(args.get("reason") or args.get("summary") or "").strip()
    ctx.finalize_reason = reason or None
    ctx.finalize_args = dict(args)  # store full args for StructuralFinalizeValidator
    return ToolResult(
        call_id=ctx.call_id,
        status="ok",
        content=[ContentBlockOut(type="text", text="Finalize acknowledged.")],
        error=None,
        duration_ms=0,
        cached=False,
    )
