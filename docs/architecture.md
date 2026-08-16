# Architecture

agentkit is a domain-blind agent runtime. Consumers wire it to:

- **A Provider** (`AnthropicProvider`, `OpenRouterProvider`, or fake)
- **A ToolRegistry** populated with built-ins + MCP servers (in-process or stdio)
- **Stores** (`SessionStore`, `MemoryStore`, `CheckpointStore`) backed by Redis
- **Guards** (intent, approval, finalize, success-claim)

## Phase machine

A turn moves through 11 phases:

```
IDLE -> INTENT_GATE -> CONTEXT_BUILD -> STREAMING -> { TOOL_PHASE | FINALIZE_CHECK }
                                                              |
                                                  APPROVAL_WAIT (suspend)
                                                  TOOL_EXECUTING -> TOOL_RESULTS
                                                  TOOL_RESULTS -> CONTEXT_BUILD (iterate)
                                                                 -> FINALIZE_CHECK (done)
                                                  FINALIZE_CHECK -> MEMORY_EXTRACT -> TURN_ENDED
                                                                 -> CONTEXT_BUILD (retry)
```

Every transition is validated against a transition table and emits a
`PhaseChanged` event for observability.

## Suspend / resume

When approval is required, the loop persists context to `CheckpointStore` and
emits `TurnEnded(reason=AWAITING_APPROVAL)`. The consumer's UI shows approval
cards; on user response, `session.resume_with_approval(...)` rehydrates the
checkpoint into a new turn.

Nobody has to respond, though, and an approval that is never answered is the
case the safety property depends on: silence is not consent. Three things can
end a suspended approval, and all of them produce the same `ApprovalResolved`
— the one event a card should close on — plus an audit row:

| Path | Who drives it | When |
|---|---|---|
| `resume_with_approval` / `..._batch` | the user | a verdict arrives |
| `check_approval_expiry(turn_id)` | the host, for a turn it can name | any access path that already knows the turn |
| `expire_due()` | the host, periodically | sweeps every overdue approval of that session |

`expire_due()` covers the approval nobody ever comes back for. It is
caller-driven — agentkit owns no scheduler — and needs a `CheckpointStore` that
also implements `EnumerableCheckpointStore`; it raises rather than quietly
reporting an all-clear it could not verify. All three paths funnel into one
expiry implementation, so an approval resolves exactly once however it was
discovered.

## Multi-replica

Orchestrators publish events to Redis pub/sub via `RedisEventBus`; bridges
subscribe and forward to clients. Any replica can run the agent for any session.
