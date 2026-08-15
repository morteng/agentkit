"""The finalize schema and the default validator must agree.

``kit.finalize`` advertises a JSON Schema to the model. ``AgentSession``
installs :class:`StructuralFinalizeValidator` whenever ``config.guards.finalize``
is unset, and that validator judges the resulting call. Nothing linked the two,
and they drifted apart in both directions at once:

* ``kit.finalize`` marked nothing required and listed ``reason`` first, so a
  model sent ``{"reason": "..."}`` — which the validator cannot parse as an
  ``Envelope`` at all, because ``status`` and ``intent_kind`` are mandatory
  there.
* ``answer_evidence`` was absent from the advertised properties entirely, while
  ``validate_envelope`` rejects any ``intent_kind="answer"`` envelope that
  omits it. A model obeying the schema to the letter still could not finalize
  an answer turn.

Either one alone produces the same visible failure: every finalize call is
rejected, ``max_finalize_retries`` is spent, and the turn ends with
``finalize_exhausted`` after three wasted provider round-trips. In a streaming
chat UI each rejected attempt is an assistant turn that emits no text, so the
agent appears to start speaking three times and say nothing.

These tests assert the agreement directly rather than testing each side against
its own idea of the contract, which is how the drift stayed invisible.
"""

from typing import Any

import pytest

from agentkit.guards.finalize import StructuralFinalizeValidator
from agentkit.loop.context import TurnContext
from agentkit.tools.builtin.finalize import FINALIZE_SPEC
from agentkit.tools.builtin.finalize_response import FINALIZE_RESPONSE_SCHEMA
from agentkit.tools.spec import ToolCall

# One minimal, schema-legal call per intent kind, each valid against an empty
# turn log. `validate_envelope` also cross-checks the envelope against what
# actually ran — `answer_evidence="tool_results"` requires a successful read
# this turn, and `intent_kind="action"` requires a write — so both cases here
# use the claims that stand on their own. Those cross-checks are the validator's
# own subject and are covered in tests/unit/test_finalize_validator.py; what is
# under test here is only whether a schema-obedient call *can* be accepted.
_CASES: dict[str, dict[str, Any]] = {
    "answer": {
        "status": "done",
        "intent_kind": "answer",
        "summary": "Explained how a systemd unit file is structured.",
        "answer_evidence": "general_knowledge",
    },
    "clarify": {
        "status": "blocked",
        "intent_kind": "clarify",
        "summary": "Asked which of the two hosts to restart.",
        "pending_confirmation": {"question": "Which host?", "kind": "choose"},
    },
}


def _properties() -> dict[str, Any]:
    props = FINALIZE_SPEC.parameters["properties"]
    assert isinstance(props, dict)
    return props


@pytest.mark.parametrize("intent_kind", sorted(_CASES))
@pytest.mark.asyncio
async def test_a_schema_obedient_call_is_accepted(intent_kind: str) -> None:
    """The headline contract: obeying the advertised schema must be enough.

    A model cannot send a field it was never told about, so any field the
    validator requires and the schema omits is a guaranteed rejection loop.
    """
    args = _CASES[intent_kind]
    unadvertised = set(args) - set(_properties())
    assert not unadvertised, (
        f"the {intent_kind} case uses fields kit.finalize does not advertise: "
        f"{sorted(unadvertised)} — a real model could not have sent them"
    )

    verdict = await StructuralFinalizeValidator().validate(
        ToolCall(id="finalize", name="kit.finalize", arguments=dict(args)),
        TurnContext.empty(),
    )
    assert verdict.accept, verdict.feedback


def test_every_unconditionally_required_envelope_field_is_required_here() -> None:
    """``status`` and ``intent_kind`` are mandatory in ``Envelope``.

    Advertising them as optional is what let a model omit both and send only
    ``reason``. ``actions_performed`` is excluded on purpose — an
    ``intent_kind="answer"`` envelope is forbidden from carrying it, so it
    cannot be unconditionally required even though ``finalize_response`` lists
    it. That asymmetry is the reason this test names the fields instead of
    comparing the two ``required`` lists.
    """
    required = FINALIZE_SPEC.parameters.get("required")
    assert required is not None, "kit.finalize must declare a required list"
    assert set(required) == {"status", "intent_kind"}


def test_conditionally_required_fields_are_at_least_advertised() -> None:
    """A conditional rule still needs a field the model can fill.

    ``required`` cannot express "mandatory given intent_kind", so these are
    optional in the schema and enforced by the validator. That split only works
    if the property exists — ``answer_evidence`` did not, which made every
    answer turn unsatisfiable no matter how carefully the model complied.
    """
    props = _properties()
    for field in ("answer_evidence", "pending_confirmation", "actions_performed"):
        assert field in props, f"validate_envelope can require {field!r}; the model must see it"


def test_reason_is_not_the_first_property_the_model_reads() -> None:
    """Ordering is part of the prompt.

    ``reason`` is a legacy alias that satisfies no validator rule on its own.
    While it was listed first, with a description reading like the summary
    field to fill, models filled it and omitted the envelope. Keeping it last
    is the fix; this test exists so a later tidy-up does not undo it.
    """
    keys = list(_properties())
    assert keys[0] != "reason"
    assert keys.index("reason") > keys.index("intent_kind")


def test_finalize_advertises_the_full_envelope_vocabulary() -> None:
    """kit.finalize and finalize_response are judged by the same validator, so
    a field one accepts and the other hides is a trap for whichever consumer
    picked the smaller tool."""
    missing = set(FINALIZE_RESPONSE_SCHEMA["properties"]) - set(_properties())
    assert not missing, f"kit.finalize hides envelope fields: {sorted(missing)}"
