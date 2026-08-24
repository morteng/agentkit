# Concepts

## AgentSession

One conversation, owned by an `OwnerId`. Holds history (in `SessionStore`),
config, registry, and provider.

## TurnContext

Per-turn mutable state passed to every handler and built-in tool. Carries
history, scratchpad, finalize flag, pending approvals, event queue, memory
store, clock — and the turn's `tainted` flag (see [Provenance and
taint](#provenance-and-taint)).

## ToolSpec

Provider-agnostic tool definition. Adapters translate to each SDK's format.
Includes `risk`, `idempotent`, `side_effects`, `requires_approval`,
`cache_ttl_seconds`, `timeout_seconds`.

`idempotent` is load-bearing in two places. It decides whether a call may join
a concurrent batch, and — for `idempotent=False` — it makes the dispatcher
answer an identical repeat within the same turn with the first call's result
instead of executing again. The loop can re-open a finished question through
ordinary control flow (a rejected `finalize_response` is a fresh model call
with the full catalogue attached, against a history where the write already
succeeded), so a tool that must not run twice needs the flag rather than a
sentence in the prompt. Declare `idempotent=True` only when a second execution
is harmless on the far side, not merely in the arguments you send: a
create-or-update endpoint is idempotent in its request and destructive in the
resource it replaces. Disable with
`ToolDispatchConfig(guard_nonidempotent_replay=False)`.

## RiskLevel

`READ | LOW_WRITE | HIGH_WRITE | DESTRUCTIVE`. Drives the default approval
gate's policy. Per-tool overrides via `RiskBasedApprovalGate(policy_overrides=...)`,
or `RiskBasedApprovalGate.strict()` for a preset that only auto-approves
`READ`. It is also the axis the taint guard measures: a tainted turn may keep
reading and may not write.

## Provenance and taint

The anti-prompt-injection control. Every `ToolResult` carries a `Provenance`:

| value | meaning |
| --- | --- |
| `SYSTEM` | produced by the runtime or a tool the operator controls (**the default**) |
| `PRINCIPAL` | authored by the human principal of this session |
| `UNTRUSTED` | third-party content the principal did not author — web pages, inbound email, scraped documents, another tenant's records |

When a result marked `UNTRUSTED` enters a turn, the turn is *tainted* and every
tool above `READ` is denied for the rest of that turn. The reasoning: from that
point the model has read text that may contain instructions the user never
wrote, so a write it proposes can no longer be attributed to the principal. The
user restates the action in a fresh turn — which starts clean — and the write
then happens because a human asked for it.

Denial is a normal `ToolResult` with `status="denied"` and
`error.code="tainted_turn"`, phrased for the model to relay:

> denied: this turn has ingested untrusted external content, so write actions
> are disabled. Ask the user to restate the action in a new turn.

Nothing infers provenance — **a tool must opt in**. If your web-fetch or inbox
tool does not mark its results, the guard never fires:

```python
from agentkit import Provenance

async def fetch_page(args, ctx):
    body = await http_get(args["url"])
    return ToolResult(..., provenance=Provenance.UNTRUSTED)
```

Enforcement lives in `ToolRegistry.invoke`, is on by default, and fails closed
(an unrecognised risk level counts as a write; a policy that raises denies).
Tune or disable it explicitly:

```python
from agentkit.guards import NullTaintPolicy, RiskBasedTaintPolicy

ToolRegistry(taint_policy=RiskBasedTaintPolicy(max_risk_when_tainted=RiskLevel.LOW_WRITE))
ToolRegistry(taint_policy=NullTaintPolicy())  # opt out, explicitly
```

The flag latches for the turn, survives an approval suspend/resume through the
checkpoint, and propagates across `kit.subagent.spawn` in **both** directions:
a tainted parent cannot launder untrusted content through a fresh child
(`fresh_child_context` copies `tainted`/`taint_sources` down when the child is
created), and a child cannot launder it back up either — `SubagentDispatcher.spawn`
propagates the child's taint state onto the parent's `TurnContext` before
returning, from a `finally` so it still happens if the child errored out or
ended suspended on an approval it had no way to resolve. `ApprovalNeeded.taint`
lists what tainted the turn — including a tool a subagent called several
levels down, not just `kit.subagent.spawn` itself — so an approval card can
name the tool whose output the model read before it proposed the action. Note
that the guard sits *below* approval: a human "approve" does not lift it, and
the call is still refused.

### Provenance and memory

Taint is per-turn, but memory is not: recalled facts are injected as context at
the start of a later turn, which makes a memory store a *persistence layer for
context*. So `MemoryValue` carries a `provenance` too, and `MemoryStore.save`
takes it as a keyword:

```python
await store.save(scope, key, value, provenance=Provenance.UNTRUSTED)
```

The keyword defaults to `None`, meaning *keep the label already on the value* —
not "trusted". A literal `SYSTEM` default would overwrite a classification the
caller had already made, which is the same laundering one step further in.
Values and stored rows written before the field existed read back as `SYSTEM`,
so nothing needs migrating.

The builtin memory tools never take the default. `kit.memory.save` records
`UNTRUSTED` when the turn is tainted and `PRINCIPAL` otherwise — never
`SYSTEM`, which would assert the runtime authored text a model composed — and
`kit.memory.recall` / `kit.memory.search` put the stored label back on the
`ToolResult`, so recalling an untrusted fact taints the turn that recalls it.
`kit.memory.list` and `kit.memory.forget` return keys rather than remembered
text and are not labelled; drop `MEMORY_LIST_SPEC` from your registration if
you treat a model-chosen key string as a carrier.

If you implement `MemoryStore` yourself, route the write through
`agentkit.store.stamp_provenance` so the keyword means the same thing in your
backend as in the shipped ones. A store that accepts the argument and discards
it leaves the gap exactly where it was.

## Execution-time tool gates

Filtering the advertised catalog is advisory — a model can name a tool it was
never shown. `ToolRegistry.invoke` therefore re-checks, in order: authorization
→ taint → argument validation → timeout-bounded execution. Each gate returns a
`ToolResult` the model can read rather than raising, because a raised exception
ends the turn with no result and the model never learns what went wrong.

`AgentSession` wires the authorizer for you: a `ToolPlane` configured as
`config.tool_selector` becomes a `ToolPlaneAuthorizer` (so `min_role` and
capability gates are enforced, not merely advertised), chained with
`SubagentToolAuthorizer` (inert outside a subagent). Anything you installed
yourself runs first and is preserved. To compose your own:

```python
from agentkit.subagents import SubagentToolAuthorizer, chain_authorizers

registry.set_authorizer(chain_authorizers(my_gate, SubagentToolAuthorizer()))
```

`ToolSpec.timeout_seconds` is enforced (default 60s when a spec declares none):
expiry cancels the handler and returns `status="timeout"`.

## MCPClient

Either `InProcessMCPClient` (Python handlers, sub-millisecond dispatch) or
`StdioMCPClient` (subprocess speaking JSON-RPC). Same interface; consumer
picks per server.

MCP carries a name, a description and a JSON Schema — nothing that maps onto
agentkit's risk model. An **unclassified external tool is therefore treated as
`HIGH_WRITE` with `ApprovalPolicy.ALWAYS`**: the most dangerous thing it could
be, until a consumer says otherwise. (`ALWAYS` rather than `BY_RISK` on
purpose: `BY_RISK` would be re-opened by any deployment that remaps
`HIGH_WRITE` to auto-approve.)

Vouch for tools you actually know with the `classifier` hook — it receives the
fail-closed default spec and returns the real one, or `None` for "unknown":

```python
def classify(spec: ToolSpec) -> ToolSpec | None:
    if spec.name == "echo.reverse":
        return spec.model_copy(update={"risk": RiskLevel.READ,
                                       "requires_approval": ApprovalPolicy.NEVER})
    return None

StdioMCPClient(name="echo", command=[...], classifier=classify)
```

Pass `require_classification=True` to go further: anything the classifier
declines is dropped from `list_tools` and refused by `call_tool`.

## PII firewall

`agentkit.pii` scrubs outbound requests via `wrap_provider`. Detectors are
consumer-supplied, and composing them is the normal case — registering your own
patterns must not silently drop the built-in ones:

```python
from agentkit.pii import CompositeDetector, SecretDetector, default_detector

detector = CompositeDetector.with_defaults(MyDomainDetector())  # defaults + yours
detector = default_detector()                                   # just the defaults
```

`SecretDetector` is the high-entropy credential net: PEM private-key blocks,
JWTs, vendor-prefixed keys (`sk-`, `ghp_`, `AKIA`, …), plus an entropy path over
base64url and hex runs. It also carries a **field-name rule** — a value under
`password` / `api_key` / `refresh_token` is redacted regardless of entropy,
which is what catches human-chosen passwords that entropy alone cannot. Its
default action is `NEVER_SEND`: never tokenized, never stored, never sent.

Tuning is a real tradeoff and every threshold is a `SecretPolicy` field. The
notable default: digest-shaped hex (32/40/64/… chars) is *exempt* unless a
credential keyword sits next to it, because a 40-char hex string is a git SHA
far more often than an API key. Flip it with
`SecretPolicy(exempt_digest_shaped_hex=False)`.

Detectors that implement the optional `FieldContextDetector` extension
(`detect_in_field(text, field)`) also see the enclosing JSON key, which is how
tool arguments and tool results get the field-name rule. A plain `detect`-only
detector still works unchanged.

## Events

Typed events forming a discriminated union. Consumers `match` on type or
filter to events they care about.
