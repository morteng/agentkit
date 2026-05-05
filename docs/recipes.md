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

## Multi-replica deployment

Configure orchestrator and bridge replicas to use `RedisEventBus`:

```python
from agentkit.transports.redis_bus import RedisEventBus
bus = RedisEventBus(client=redis_client)
# Bridge subscribes; orchestrator publishes.
```

## Building a chat UI with the WebSocket bridge

`mount_websocket_route` exposes one async endpoint that translates inbound
JSON commands into `AgentSession` calls and outbound events into JSON frames.
Cancel works mid-turn — the server runs the agent stream concurrently with a
`receive_json` watcher and aborts on `{"type":"cancel"}`.

```python
from fastapi import FastAPI, WebSocket
from agentkit import AgentConfig, AgentSession
from agentkit._ids import OwnerId
from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.providers.openrouter import OpenRouterProvider
from agentkit.store.fakes import FakeCheckpointStore, FakeMemoryStore, FakeSessionStore
from agentkit.tools.registry import ToolRegistry
from agentkit.transports.websocket import mount_websocket_route

app = FastAPI()

async def session_factory(ws: WebSocket) -> AgentSession:
    config = AgentConfig()
    config.guards.approval = RiskBasedApprovalGate()
    config.stores.session = FakeSessionStore()
    config.stores.memory = FakeMemoryStore()
    config.stores.checkpoint = FakeCheckpointStore()
    registry = ToolRegistry()
    registry.register_default_builtins()
    return AgentSession(
        owner=OwnerId(ws.headers.get("x-user-id", "anon")),
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
)
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
