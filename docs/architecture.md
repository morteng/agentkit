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

## Multi-replica

Orchestrators publish events to Redis pub/sub via `RedisEventBus`; bridges
subscribe and forward to clients. Any replica can run the agent for any session.
