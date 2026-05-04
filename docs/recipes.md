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
