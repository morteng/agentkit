# Recipes

## Wiring an in-process MCP server

```python
from agentkit.mcp_client import InProcessMCPClient
from agentkit.tools.spec import ToolSpec, ToolResult, RiskLevel, SideEffects, ApprovalPolicy

client = InProcessMCPClient(name="myapp")

async def list_devices(args):
    return ToolResult(call_id="", status="ok", content=[...], error=None, duration_ms=0, cached=False)

client.register_tool(
    ToolSpec(
        name="list_devices",
        description="List devices on the site.",
        parameters={"type": "object"},
        returns=None,
        risk=RiskLevel.READ,
        idempotent=True,
        side_effects=SideEffects.NONE,
        requires_approval=ApprovalPolicy.BY_RISK,
        cache_ttl_seconds=300,
        timeout_seconds=10.0,
    ),
    list_devices,
)

registry.register_mcp_server("myapp", client)
```

## Custom approval policy

```python
from agentkit.guards.approval import RiskBasedApprovalGate, ApprovalDecision

config.guards.approval = RiskBasedApprovalGate(policy_overrides={
    "myapp.delete_everything": ApprovalDecision.AUTO_DENY,
    "myapp.send_email":         ApprovalDecision.NEEDS_USER,
})
```

The default table auto-approves `LOW_WRITE` — reversible, small blast radius —
because prompting on every low write makes an interactive assistant unusable
and trains users to click through. If your deployment would rather ask, that is
one line:

```python
config.guards.approval = RiskBasedApprovalGate.strict()
```

Strict auto-approves only `READ`, and it stops honouring a *third-party* tool's
own `ApprovalPolicy.NEVER` declaration — a server asserting "no approval needed"
about itself is not evidence. Only `kit.*` tools keep that privilege; pass
`spec_never_prefixes=None` to keep honouring `NEVER` everywhere while still
tightening the risk table.

## Multi-replica deployment

Configure orchestrator and bridge replicas to use `RedisEventBus`:

```python
from agentkit.transports.redis_bus import RedisEventBus
bus = RedisEventBus(client=redis_client, buffer_ttl_seconds=30 * 24 * 3600)
# Bridge subscribes; orchestrator publishes.
```

A reconnecting bridge replays what it missed. The cursor is
`(turn_id, sequence)` — a turn id alone is ambiguous once a second turn starts
its own sequence at 0:

```python
missed = await bus.replay_buffer(
    session_id,
    since_turn_id=last_seen_turn_id,
    since_sequence=last_seen_sequence,
)
```

Defaults (`since_turn_id=None, since_sequence=0`) replay the whole buffer. If
the cursor's turn has already been evicted the buffer is replayed in full
rather than silently skipped — over-delivery is recoverable by the consumer's
own dedupe, a gap is not.

## Building a chat UI with the WebSocket bridge

`mount_websocket_route` exposes one async endpoint that translates inbound
JSON commands into `AgentSession` calls and outbound events into JSON frames.
Cancel works mid-turn — the server runs the agent stream concurrently with a
`receive_json` watcher and aborts on `{"type":"cancel"}`.

**`auth=` is required.** It has no default: a socket that anyone can open is a
socket that runs tools as whoever it feels like, and the old implicit
allow-all made that the path of least resistance. Implement `WSAuth` against
your own session/token store; `InsecureAllowAllAuth()` exists for local
development only and warns on construction (promote it to an error in CI with
`warnings.simplefilter("error", InsecureTransportWarning)`).

```python
from fastapi import FastAPI, WebSocket
from agentkit import AgentConfig, AgentSession
from agentkit._ids import OwnerId
from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.providers.openrouter import OpenRouterProvider
from agentkit.store.fakes import FakeCheckpointStore, FakeMemoryStore, FakeSessionStore
from agentkit.store.memory import MemoryScope
from agentkit.tools.registry import ToolRegistry
from agentkit.transports import WSAuth, mount_websocket_route

app = FastAPI()

class HeaderTokenAuth:
    """Minimal WSAuth: admit the socket, or close it with 4001."""

    async def authenticate(self, ws: WebSocket) -> bool:
        token = ws.headers.get("authorization", "").removeprefix("Bearer ")
        return await my_token_store.is_valid(token)


async def session_factory(ws: WebSocket) -> AgentSession:
    user_id = ws.headers.get("x-user-id", "anon")
    config = AgentConfig()
    config.guards.approval = RiskBasedApprovalGate()
    config.stores.session = FakeSessionStore()
    config.stores.memory = FakeMemoryStore()
    # A memory store on its own is inert: kit.memory.save/recall need a scope
    # as well, and answer memory_not_configured without one. session_id stays
    # None so the memories are persistent — visible in future sessions too.
    config.stores.memory_scope = MemoryScope(
        namespace="prefs", tenant_id="acme", user_id=user_id
    )
    config.stores.checkpoint = FakeCheckpointStore()
    registry = ToolRegistry()
    registry.register_default_builtins()
    return AgentSession(
        owner=OwnerId(user_id),
        config=config,
        provider=OpenRouterProvider(api_key=...),
        registry=registry,
        model="openrouter/owl-alpha",
    )

mount_websocket_route(
    app,
    path="/ws/agent",
    session_factory=session_factory,
    origin_allowlist=["https://your.site"],
    auth=HeaderTokenAuth(),
)
```

Two more things this endpoint will not decide for you:

**Origins.** `origin_allowlist` is matched exactly against the handshake's
`Origin` header. `["*"]` raises `ValueError` unless you also pass
`dev_mode=True`. Non-browser clients send no `Origin` at all — allowlist `""`
for those rather than reaching for the wildcard.

**Who may approve.** By default the socket that started the turn may also
approve the calls that turn produced (`SameSocketApprovalAuthority`). That
means the approval gate protects you from a confused model, not from a
compromised client — an injected instruction that reaches the model over this
socket can propose a `DESTRUCTIVE` call *and* the client can rubber-stamp it.
To bind approvals to a second principal, pass a `WSApprovalAuthority`; it
receives the raw frame, so it can require a one-time token minted by a separate
approver UI and bound to `call_id`:

```python
class TokenBoundApproval:
    async def authorize_approval(self, ws: WebSocket, command: dict) -> bool:
        return await approvals.consume(command.get("call_id"), command.get("approval_token"))

mount_websocket_route(..., auth=HeaderTokenAuth(), approval_authority=TokenBoundApproval())
```

### Wire protocol

Inbound commands (client → server):

| `type`                   | Fields                                                     | Notes |
| ------------------------ | ---------------------------------------------------------- | ----- |
| `send_message`           | `text`                                                     | Starts a turn. |
| `respond_to_approval`    | `turn_id`, `call_id`, `decision` ("approve"/"deny"), optional `edited_args`, `reason` | After receiving an `approval_needed` event. |
| `cancel`                 | optional `reason`                                          | Aborts an active turn or no-ops between turns. |

Outbound events (server → client) are JSON dumps of every `agentkit.events.Event`
plus a `cancelled` ack frame. UIs typically render: `text_delta` (typewriter),
`tool_call_started` (indicator), `approval_needed` (modal), `tool_call_result`
(replace indicator), `turn_ended` (re-enable input).

Close the approval modal on `approval_resolved`, not on `approval_granted` /
`approval_denied`: it is the only event that also fires when the approval
expires, and it carries `resolved_by` and `expired` so the card can report who
ruled instead of assuming it was the person looking at it.

## Showing reasoning ("thinking…") affordances

Reasoning models (DeepSeek V4, etc.) often think for several seconds before
the first visible character. agentkit forwards their chain-of-thought as
`ThinkingDelta` events:

```python
async with session.run("Plan a refactor.") as stream:
    async for event in stream:
        if isinstance(event, ThinkingDelta):
            ui.show_thinking_indicator(event.delta)
        elif isinstance(event, TextDelta):
            ui.append_visible_text(event.delta)
        elif isinstance(event, TurnEnded):
            ui.hide_thinking_indicator()
```

If you don't render `ThinkingDelta` your UI will appear frozen for 1–3s on
the first prompt. The `OpenRouterProvider` translates both `reasoning_content`
and `reasoning` field shapes that different upstreams use.

## Handling tool denial in your system prompt

When the user denies a tool call, the model receives a `denied` `ToolResult`
in its next iteration. Without explicit guidance models often improvise —
calling alternate tools, asking the user to reconsider, or hallucinating a
workaround. Prompt explicitly:

```text
If a tool returns status "denied", acknowledge the denial in plain language
and call kit.finalize without further attempts. Do not propose alternate
tools, do not retry, do not negotiate.
```

This works well across Claude, DeepSeek, and OpenAI-compatible models.

## Resilient MCP server registration

A subprocess that fails to start no longer aborts session initialization —
the failed server is recorded and its tools are skipped. Inspect after init:

```python
await session.initialize()
if session.failed_mcp_servers:
    log.warning("MCP servers down: %s", session.failed_mcp_servers)
    # Still safe to run turns — failed servers' tools simply aren't exposed.
```

## Per-model capabilities

`OpenRouterProvider` ships a small built-in table mapping known model IDs to
accurate capabilities (e.g. DeepSeek V4 Flash's 1M context). For models not
in the table, `capabilities_for(model)` falls back to a conservative default;
register your own via the constructor:

```python
from agentkit.providers.base import ProviderCapabilities

provider = OpenRouterProvider(
    api_key=...,
    model_capabilities={
        "vendor/exotic": ProviderCapabilities(
            supports_tool_use=True, supports_parallel_tools=True,
            supports_prompt_caching=False, supports_vision=False,
            supports_thinking=False, max_context_tokens=4_000_000,
            max_output_tokens=16_384,
        ),
    },
)
```
