"""The shared contract for a tool call a provider stream could not deliver.

Both stream parsers face the same two failures — argument JSON that will not
parse, and a call the stream stopped emitting halfway through — and both used
to resolve them differently: OpenRouter dropped the call silently, Anthropic
dispatched it with ``arguments={}``. Neither is acceptable, and having *two*
answers is worse than either, because a consumer's error handling then depends
on which provider happened to serve the turn.

One answer, defined here and used by both: never guess arguments, never drop a
call in silence. Emit an :class:`~agentkit.providers.base.ErrorEvent` carrying
one of these codes and a message written for the *model* to read, so the turn
can recover by re-issuing the call rather than acting on a corrupted intent.

``arguments={}`` deserves its own note, since it is the tempting shortcut: for
a tool like "delete everything matching filter X" an empty filter is the single
most destructive reading of a corrupted call, and the model never finds out its
JSON was broken. An empty *buffer* is different and still valid — a zero-arg
tool legitimately streams no arguments. That is a parse success, not a guess.
"""

INVALID_TOOL_ARGUMENTS_CODE = "tool_arguments_unparseable"
"""``ErrorEvent.code`` for a tool call whose streamed argument JSON is garbage.

The call is NOT dispatched. Consumers that want the model to self-correct
should feed the event's ``message`` back as the failed call's result: it is
phrased for the model, not for a log line.
"""

INCOMPLETE_TOOL_CALL_CODE = "tool_call_incomplete"
"""``ErrorEvent.code`` for a tool call the stream never finished emitting.

Covers both halves of "the provider stopped mid-call": a stream that ended
without a tool-call finish reason, and a slot whose function name never
arrived. Neither is dispatchable; both used to be silent.
"""

UNNAMED_TOOL = "<unnamed>"
"""Stand-in name for a slot whose function-name delta never arrived."""


def incomplete_call_message(*, tool_name: str, call_id: str, finish_reason: str | None) -> str:
    """Model-facing text for a tool call the stream never finished."""
    return (
        f"Tool call {tool_name!r} (id {call_id}) never completed: the response "
        f"stream ended with finish_reason={finish_reason!r} while its arguments "
        "were still streaming. The tool was NOT executed and had no effect. "
        "Re-issue the call if you still need it."
    )


def invalid_arguments_message(*, tool_name: str, call_id: str) -> str:
    """Model-facing text for a tool call whose argument JSON did not parse."""
    return (
        f"Tool call {tool_name!r} (id {call_id}) was rejected: its arguments were "
        "not valid JSON and could not be repaired. The tool was NOT executed and "
        "had no effect. Re-issue the call with well-formed JSON arguments."
    )


__all__ = [
    "INCOMPLETE_TOOL_CALL_CODE",
    "INVALID_TOOL_ARGUMENTS_CODE",
    "UNNAMED_TOOL",
    "incomplete_call_message",
    "invalid_arguments_message",
]
