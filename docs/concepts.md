# Concepts

## AgentSession

One conversation, owned by an `OwnerId`. Holds history (in `SessionStore`),
config, registry, and provider.

## TurnContext

Per-turn mutable state passed to every handler and built-in tool. Carries
history, scratchpad, finalize flag, pending approvals, event queue, memory
store, and clock.

## ToolSpec

Provider-agnostic tool definition. Adapters translate to each SDK's format.
Includes `risk`, `idempotent`, `side_effects`, `requires_approval`,
`cache_ttl_seconds`, `timeout_seconds`.

## RiskLevel

`READ | LOW_WRITE | HIGH_WRITE | DESTRUCTIVE`. Drives the default approval
gate's policy. Per-tool overrides via `RiskBasedApprovalGate(policy_overrides=...)`.

## MCPClient

Either `InProcessMCPClient` (Python handlers, sub-millisecond dispatch) or
`StdioMCPClient` (subprocess speaking JSON-RPC). Same interface; consumer
picks per server.

## Events

18 typed events forming a discriminated union. Consumers `match` on type or
filter to events they care about.
