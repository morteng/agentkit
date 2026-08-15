# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
from v1.0.0 onward. Pre-1.0 minor versions may include breaking changes.

## [0.22.0] - 2026-08-15

Security-hardening release. Several defaults changed from "convenient" to
"fail closed", two subsystems that did not do what their names promised were
deleted rather than patched, and the audit's headline control — provenance and
taint tracking — landed. **Read the BREAKING section before upgrading**: this
release will refuse things your deployment currently allows, and that is the
point.

### Added

- **Provenance and taint tracking** — the anti-prompt-injection control.
  `Provenance` (`SYSTEM` | `PRINCIPAL` | `UNTRUSTED`) on `ToolResult` and
  `ToolResultBlock`, defaulting to `SYSTEM` so existing tools are unaffected. A
  result marked `UNTRUSTED` taints the turn (`TurnContext.tainted`, latching),
  and from that point every tool above `READ` is denied for the rest of the
  turn: the model has read third-party text, so a write it proposes can no
  longer be attributed to the principal. Denial is a normal
  `ToolResult(status="denied", error.code="tainted_turn")` phrased for the
  model to relay — never a silent no-op, never an exception. The flag survives
  an approval suspend/resume through the checkpoint and propagates to subagent
  children. Enforced in `ToolRegistry.invoke`, on by default, fail-closed
  (unknown risk counts as a write; a policy that raises denies). Configure via
  `ToolRegistry(taint_policy=RiskBasedTaintPolicy(max_risk_when_tainted=...))`
  or opt out with `NullTaintPolicy()`. Nothing infers provenance: a tool must
  mark its own results, and no in-tree builtin ingests third-party content
  yet, so the guard is inert until your tools opt in. `ApprovalNeeded.taint`
  lists what tainted the turn so an approval card can name it. New public
  surface: `agentkit.Provenance`, `agentkit.guards.{TaintPolicy,
  RiskBasedTaintPolicy, NullTaintPolicy, TaintSource, is_tainted, mark_taint,
  taint_sources, TAINT_DENIAL_MESSAGE, TAINT_DENIAL_CODE}`.
- **Execution-time authorization gate.** `ToolRegistry` accepts an
  `authorizer` (constructor or `set_authorizer`) consulted before the handler
  runs; denial returns `status="denied"`, `error.code="not_authorized"`.
  `ToolPlaneAuthorizer` re-runs the ToolPlane's own `_decide` at invoke time,
  so `min_role` / `mcp_clients` / capability gates are *enforced* rather than
  merely un-advertised. `SubagentToolAuthorizer` + `chain_authorizers` express
  the subagent allowlist at the same choke point. `AgentSession` wires both
  automatically (see BREAKING #6). New `ToolRegistry.authorizer` property so a
  caller can compose with an existing gate instead of replacing it.
- **`RiskBasedApprovalGate.strict()`** — one-line security-sensitive preset
  (`config.guards.approval = RiskBasedApprovalGate.strict()`). Only `READ`
  auto-approves, and a third-party spec's `ApprovalPolicy.NEVER` is honoured
  only for `kit.*` tools (`spec_never_prefixes`), because a server declaring
  "no approval needed" about itself is not evidence. New
  `STRICT_APPROVAL_POLICY`, `KIT_NAMESPACE_PREFIXES`.
- **`SecretDetector` / `SecretPolicy`** — high-entropy credential detection for
  the PII firewall: PEM private-key blocks, JWTs, vendor-prefixed keys (`sk-`,
  `ghp_`, `AKIA`, `AIza`, …), plus an entropy path over base64url and hex runs.
  Includes a field-name rule (a value under `password` / `api_key` /
  `refresh_token` is redacted regardless of entropy — what catches
  human-chosen passwords). Default action `NEVER_SEND`. Every threshold is a
  `SecretPolicy` field; digest-shaped hex is exempt by default (a 40-char hex
  string is a git SHA far more often than a key) — flip with
  `exempt_digest_shaped_hex=False`.
- **`CompositeDetector`** (`with_defaults()`, `default_detector()`) closes the
  "register my patterns == silently drop the built-in ones" hole, and
  `FieldContextDetector` is a runtime-checkable optional extension
  (`detect_in_field(text, field)`) so detectors can see the enclosing JSON key.
  `merge_spans` gives one deterministic overlap resolution.
- **`ApprovalResolved` event** — one terminal event for every approval outcome,
  including the **timeout path**, which previously emitted nothing at all and
  left a client unable to distinguish an expired approval from a hung turn.
  Carries `decision`, `resolved_by` (owner, or `"system"`), `expired`.
  `ApprovalNeeded` also gained `side_effects`, because reversibility is a
  separate axis from risk and every consumer was rebuilding that map by hand.
- **Per-principal turn rate limiting**, on by default at 60 turns/minute
  (`GuardConfig.rate_limit_turns_per_minute`, `None` to disable). Keyed on the
  runtime-stamped owner via the new `verified_principal()`, not on anything the
  model can influence, so a burst funnelled through spawned subagents counts
  against the principal that started it.
- `ToolSpec.timeout_seconds` is now **enforced** (`DEFAULT_TOOL_TIMEOUT_SECONDS
  = 60.0` when a spec declares none); expiry cancels the handler and returns
  `status="timeout"`. Structural argument validation before dispatch
  (`validate_tool_arguments`) returns `error.code="invalid_arguments"` rather
  than dispatching a call the schema already rejects; disable with
  `ToolRegistry(validate_arguments=False)`.
- `AgentSession(history_limit=...)` + `DEFAULT_HISTORY_LIMIT`, plus
  `SYNTHETIC_TOOL_RESULT_ANNOTATION` for filtering synthesized tool results.
- `Loop(publish_phase_changed=...)` (wired from
  `AgentConfig.events.publish_phase_changed`), `LoopConfig.max_claim_corrections`
  and `LoopConfig.streaming_chunk_timeout_seconds` are now actually threaded
  into the loop by `AgentSession`.
- `StdioMCPClient(classifier=…, require_classification=…)` for vouching for
  external MCP tools; `RedisEventBus(buffer_ttl_seconds=…)`;
  `WSApprovalAuthority` / `SameSocketApprovalAuthority`; `InsecureAllowAllAuth`
  / `InsecureTransportWarning`; `validate_memory_key` / `MAX_MEMORY_KEY_LENGTH`.

### Changed

- `TurnEnded.metrics` is populated for real (tokens from `ctx.metadata["usages"]`,
  cost from `Provider.estimate_cost`, duration, tool-call count, iterations)
  instead of always being an empty `TurnMetrics()`.
- Every path into `Phase.ERRORED` now logs with context, stashes
  `ctx.metadata["turn_error"]`, and emits an `Errored` event. Three failure
  modes previously ended a turn in silence.
- `agentkit.providers.tool_call_errors` gives both stream parsers one answer for
  an undeliverable tool call. The OpenRouter constants
  (`INVALID_TOOL_ARGUMENTS_CODE`, `INCOMPLETE_TOOL_CALL_CODE`) keep working from
  their old location.
- `kit.finalize`'s declared schema now matches its two real argument shapes
  (bare `{"reason"}` or the full `finalize_response` envelope) and requires
  nothing. It previously advertised `reason` as required, which was true of
  neither the handler nor any caller — invisible until argument validation
  started enforcing declarations. `TurnEnded.summary` now falls back to the
  envelope's `summary`, so envelope-shaped calls stop ending turns with a null
  summary.
- The `examples/with_mcp_tools` demo passes a `classifier`; without one it would
  now stop and wait for an approval nobody is there to grant.

### Fixed

- **Resumed turns were never persisted.** Tool results and the final reply
  produced after `resume_with_approval` were dropped on the floor. Also: a
  cancelled or abandoned turn now closes its dangling `tool_use` blocks with a
  synthetic `cancelled` result, so turn N+1 does not load a transcript that
  providers hard-reject; legacy transcripts are repaired in memory on load.
- **A bad `call_id` no longer destroys the checkpoint.** Call ids are validated
  (duplicates in a batch included) *before* the delete, so a rejected resume
  stays resumable. Approvals left unresolved by a partial resume are explicitly
  auto-denied with a real `ApprovalDenied` event instead of vanishing.
- **Silent history truncation.** `list_messages` was called with an implicit
  limit; the cut is now explicit, tool-pair-safe, logged, and surfaced on
  `ctx.metadata["history_load"]`.
- **`ToolPlane` leaked resolution state across concurrent sessions** via
  instance attributes. `resolve_detailed()` is pure; `plane.rationale` /
  `.last_discoverable` are now per-execution-context.
- **Subagent context injection.** A spawn's `context` could overwrite
  `subagent_depth`, `owner`, `allowed_tools` and other runtime-stamped keys —
  i.e. a model could deepen its own recursion budget or widen its own tool set.
  Reserved keys now hard-reject with `SubagentContextRejected`, and trusted
  values are stamped after the merge.
- **A subagent's tool allowlist is now enforced**, not advertised: children get
  a default-deny `RestrictedToolRegistry` view, a dispatcher rebuilt against it,
  and their own nested dispatcher so a grandchild cannot regain what the child
  lost. Approval inside a subagent is auto-denied (it could never be resolved)
  rather than silently dropped, and no orphan checkpoint is writable.
- **Malformed streamed tool calls are no longer executed on guessed arguments.**
  Both parsers now emit a recoverable `ErrorEvent` instead — OpenRouter used to
  drop the call silently, Anthropic used to dispatch it with `arguments={}`. For
  a tool like "delete everything matching filter X", an empty filter is the most
  destructive possible reading of a corrupted call.
- **Redis key injection.** Every model- or request-derived key segment is
  percent-encoded; the memory key index moved to the reserved `%index` suffix,
  which no encoded user key can produce. `RedisSessionStore.append_message` is a
  single MULTI (a message could exist in the list while the document had never
  heard of it), and `message_count` derives from `LLEN` instead of a
  read-modify-write that lost concurrent increments. `touch()` refreshes the
  messages TTL, the owner index carries a TTL, and `list_for_owner` prunes
  entries whose document is gone.
- **`PiiPolicy.blocked_models` was read by nothing.** Now enforced on
  PII-carrying requests (`BlockedModelError`). `ThinkingBlock.text` is scrubbed
  (unsigned blocks only — rewriting a signed one invalidates it). Overlapping
  detector spans resolve deterministically instead of corrupting the output.
- The success-claim guard could loop forever; corrections are now counted
  against `max_claim_corrections`, delivered through the one path the
  MessageBuilder reads, and the guard stands down when the budget is spent.
  Each chunk await is bounded by `streaming_chunk_timeout_seconds`, and streams
  are closed on every exit path.
- A tool handler that raises, hangs, or is cancelled can no longer kill the turn
  or its sibling calls; each failure maps to the correct positional result.
- `RedisEventBus` buffer keys had no TTL — a replay buffer for a dead session
  lived forever.
- Model-authored memory keys are validated at the tool boundary (a readable
  `invalid_memory_key` error) and again in the store.

### BREAKING

Each item says what to change. Where the old behaviour is still available, the
opt-out is named — read why before reaching for it.

1. **`mount_websocket_route(auth=…)` is now required** (no default). A missing,
   `None`, or malformed authenticator raises `TypeError` at *mount* time.
   Migration: implement `WSAuth` (`async def authenticate(ws) -> bool`) against
   your own token store. `InsecureAllowAllAuth()` restores the old behaviour
   and warns on construction — but the old behaviour was "anyone who can reach
   the port gets the session's full tool surface, privileged tools included",
   so use it for local development only. Promote the warning to an error in CI
   with `warnings.simplefilter("error", InsecureTransportWarning)`. The private
   `_AllowAllAuth` is gone.
2. **`origin_allowlist=["*"]` raises `ValueError`** unless you also pass
   `dev_mode=True`. Migration: list real origins. Non-browser clients send no
   `Origin` header at all — allowlist `""` for those instead of the wildcard.
3. **External stdio MCP tools now require user approval by default.** An
   unclassified tool is treated as `HIGH_WRITE` + `ApprovalPolicy.ALWAYS` +
   `EXTERNAL_IRREVERSIBLE`, because MCP carries nothing that maps onto a risk
   model and the old default (`LOW_WRITE`, silently auto-approved) meant any
   MCP server could hand you a destructive tool that never prompted. Migration:
   pass `StdioMCPClient(classifier=…)` to vouch for the tools you actually know
   (return `None` for the rest and they stay fail-closed), or add
   `RiskBasedApprovalGate(policy_overrides={"srv.tool": AUTO_APPROVE})`.
   `ALWAYS` is deliberate — `BY_RISK` would be re-opened by any deployment that
   remaps `HIGH_WRITE`.
4. **`ToolRegistry.invoke` no longer executes unconditionally.** A registered
   tool can come back `denied` (taint or authorizer), `error`
   (`invalid_arguments`), or `timeout`. Migration: handle those statuses — they
   are ordinary `ToolResult`s, not exceptions. Per-gate opt-outs:
   `taint_policy=NullTaintPolicy()`, `validate_arguments=False`,
   `default_timeout_seconds=…`.
5. **A subagent's tool set is genuinely restricted.** `tools=[]` now means
   "finalize only", and a child that reaches for a parent tool it was not
   granted gets an unknown/denied result. A spawn `context` containing a
   reserved key (`subagent_depth`, `owner`, `allowed_tools`, …) raises
   `SubagentContextRejected` where it previously merged. Migration: list the
   tools each subagent actually needs.
6. **`AgentSession` installs an execution-time authorizer on your registry.**
   If `config.tool_selector` is a `ToolPlane`, a tool that resolves to `hidden`
   is now *refused at invoke*, not merely left out of the advertised catalog —
   a model that names it from an earlier turn or an injected instruction no
   longer reaches the handler. `SubagentToolAuthorizer` is chained in and is
   inert outside subagent contexts. Any authorizer you installed yourself is
   preserved and runs first. Migration: if a tool must stay callable while
   hidden, adjust `deny_tiers` or the plane's decision; note that the registry
   is mutated, so sharing one registry across sessions with different planes
   means the newest session's plane wins.
7. **Rate limiting is on by default** at 60 turns/minute per principal. Under
   the previous release the knob existed but no wiring consulted it. Migration:
   `AgentConfig().guards.rate_limit_turns_per_minute = None` to disable, or
   raise the ceiling. In-process state — N workers admit up to N× the limit;
   swap in a Redis-backed `IntentCheck` for a real distributed limit. A
   consumer-supplied `guards.intent` gate now runs *after* the limiter instead
   of replacing it.
8. **`agentkit.codeexec` is removed** (`execute`, `SAFE_MODULES`, `ExecLimits`,
   `ExecutionResult`, `Code*Error`). Its AST allowlist did not sandbox: the
   escape `"{0.ping.__globals__[API_TOKEN]}".format(client)` returns the host
   module's secret, and `str.format` traversal happens at runtime inside a
   string where an AST allowlist cannot see it. Removing `format` would not
   close the class of bug (`format_map`, `%`-formatting, any method on any
   injected object). Migration: run model-authored code in a real isolate
   (subprocess with seccomp, container, WASM) or not at all. There is no
   in-library replacement, deliberately.
9. **`agentkit.tools.cache` is removed** (`ToolResultCache`, `cache_key`). It
   had no consumer, and its key was a global `sha256(name, args)` with no
   session or tenant component — wiring it as-is would have served one user's
   tool results to another. `ToolSpec.cache_ttl_seconds` remains as declarative
   metadata with no in-library consumer.
10. **`AgentConfig.loop.builtin_tool_note_enabled` is removed.** It could never
    work: the consumer owns the `ToolRegistry`, so a loop-config flag cannot
    register a tool. Migration:
    `registry.register_builtin(NOTE_SPEC, note_handler)`. Pydantic
    `extra="ignore"` means an existing env var or kwarg will not raise — it
    will simply be ignored, so check for it rather than relying on a crash.
11. **Redis key format changed** and existing data is not migrated. All
    model/request-derived segments are percent-encoded, so owner index keys move
    whenever an owner id contains a reserved character — and `OwnerId("u:1")` is
    the house style, meaning `list_for_owner` returns empty for those owners
    until sessions are re-indexed. Session and message documents remain
    reachable by id (ULIDs escape to themselves). Memory scope indexes move from
    `…:_index` to `…:%index`, and any memory key containing `:` `%` `/` moves.
    Migration: run a key-migration script, or accept a cold start.
12. **`RedisEventBus.replay_buffer` semantics changed.** The default
    (`since_sequence=0`) now returns the whole buffer instead of dropping every
    turn's `sequence == 0` event. The cursor is now `(since_turn_id,
    since_sequence)` — a sequence alone is ambiguous once a second turn restarts
    numbering. An evicted cursor replays everything rather than silently
    gapping. Signature is source-compatible.
13. **`Firewall.routing_prefs()` no longer returns `None`** when
    `require_zdr=False` and `eu_only=False`; it returns
    `RoutingPreferences(zdr=False, data_collection="deny", allow_fallbacks=True)`.
    Sending no preference does not mean "no preference" on the wire — it takes
    the provider's default, and provider defaults permit retention and training.
    OpenRouter payloads that previously carried no `provider` block now carry
    one. Migration: `PiiPolicy(default_data_collection="unset")` restores the
    silent payload.
14. **`PiiPolicy.blocked_models` is enforced** and raises `BlockedModelError`.
    Anyone who populated it was previously unprotected; anyone who populated it
    *and* depended on it not firing will now get an exception.
15. **Exiting `async with session.run(...)` early cancels the turn.** It
    previously kept running detached until GC. Related: cancelled, errored and
    abandoned turns now append synthetic `MessageRole.TOOL` messages to the
    store — filter on `SYNTHETIC_TOOL_RESULT_ANNOTATION` if your UI renders every
    stored message. `resume_with_approval` on a multi-pending turn emits extra
    `ApprovalDenied` events for the calls it did not rule on, so consumers that
    count events will see more.
16. **Wire shape changed for `approval_needed`**: two additive fields
    (`side_effects`, `taint`). `ToolResult` / `ToolResultBlock` gained
    `provenance`, which appears in `model_dump()`. A new `approval_resolved`
    event joins the `Event` union — exhaustive `match` statements over it need a
    new arm.
17. Private-but-referenced: `AgentSession._resumed_loop_stream` is now
    `_resumed_turn_runner` and returns a `_TurnRunner` rather than an
    `AsyncIterator[Event]`. The 0.14.0 entry below still names the old helper;
    it is left as written, since a changelog records what shipped then.

## [0.21.1] - 2026-08-11

### Fixed
- Streamed tool calls that agentkit discards or empties are no longer silent. Four sites gained a `WARNING`; no behaviour anywhere changed, so event sequences, arguments, finish reasons, and phase transitions are byte-identical to 0.21.0. The OpenRouter parser logs `openrouter.pending_tool_calls_dropped` when a non-`tool_calls` finish reason discards accumulated calls (carrying `finish_reason`, `dropped_count`, and per-slot `name`/`has_id`/`args_buf_len`), `openrouter.nameless_tool_slot_skipped` when arguments arrive for a slot whose function-name delta never did, and `openrouter.tool_args_unparseable_defaulted_empty` when arguments survive neither `json.loads` nor `json_repair` and are coerced to `{}`. The Anthropic parser logs the same coercion as `anthropic.tool_args_unparseable_defaulted_empty`. `StreamMux` logs `tool_call_start_without_complete` at end of stream — provider-agnostic, and the only seam that catches the symptom for a provider whose parser has no end-of-stream handling, since the mux forwards only `tool_call_complete` and a dropped call therefore reaches no consumer at all. Argument buffers are never logged, only their lengths: they carry user text and a raw-args log would sidestep the PII firewall. **Known divergence, deliberately unchanged:** on truncated tool JSON, OpenRouter **drops** the call while Anthropic **emits** it with `arguments={}`. Both are now visible; which one is right is a contract question that waits on the data these logs produce. Consumers that page on `WARNING` should expect these names to be the first real occurrence, and the names themselves are now grep and alert targets — treat renames as breaking.

## [0.21.0] - 2026-07-31

### Fixed
- OpenRouter now encodes agentkit's canonical qualified tool names (for example,
  `REDACTED.search` and `kit.current_time`) into OpenAI-compatible function names
  at the provider boundary, and decodes streamed calls back before dispatch.
  Canonical names remain unchanged in routing, history, audit records, and UI
  events, including replayed tool-call history and named tool choices.

## [0.20.0] - 2026-07-26

### Fixed
- `kit.memory.save` / `kit.memory.recall` now work. Their guard requires both `TurnContext.memory_store` and `TurnContext.memory_scope`, but `AgentSession` only ever set `memory_store`, and no config field could carry a scope — so `memory_scope` was structurally always `None` and every session answered `memory_not_configured` regardless of how the stores were wired. New `StoreBundle.memory_scope: MemoryScope | None = None`, threaded into `TurnContext` at both construction sites: `run()` and `_load_resume_context()` (the approval-resume path builds its own context, so memory would otherwise vanish across a suspend/resume). The field sits on `StoreBundle` next to `memory` because a store and its scope are only useful together, and is typed concretely rather than `Any` — `MemoryScope` is a leaf pydantic model, so importing it into `config.py` adds no cycle, and pydantic validates it. Left unset the tools still return the explicit `memory_not_configured` error; that contract is unchanged and now covered by tests. `subagents/isolation.py` already inherited `parent.memory_scope`, so subagent memory starts working too. Additive and backward compatible. The WebSocket recipe in `docs/recipes.md` now configures a scope alongside the store.

## [0.19.0] - 2026-07-21

### Added
- Conversation-history compaction: `agentkit.compaction.compact_history(store, session_id, summarizer, *, keep_recent=8, min_messages=12)` — the first trimming/summarization primitive in agentkit. `summarizer` is a host-supplied `Callable[[list[Message]], Awaitable[str]]`; agentkit never calls an LLM itself, so model choice, prompting, and cost stay with the host. Loads the full transcript, no-ops (`CompactionResult(compacted=False)`) below `min_messages`, otherwise keeps the trailing `keep_recent` messages and walks the cut point earlier to the nearest safe boundary — a genuine USER turn (skipping `INJECTED_CORRECTION_ANNOTATION` messages) with no `ToolUseBlock`/`ToolResultBlock` pair split across it, since providers like Gemini and Anthropic hard-reject an orphaned tool result. If the only safe boundary is index 0, also a no-op. On a real cut, awaits `summarizer` with exactly the dropped prefix and writes `[summary_message, *kept_tail]` back via the new `SessionStore.replace(session_id, messages)` — added to the protocol plus both `FakeSessionStore` and `RedisSessionStore` (atomic MULTI/EXEC swap, preserves the messages-key TTL). The summary message is a USER message tagged `metadata.annotations[COMPACTION_SUMMARY_ANNOTATION]` with a bilingual "[Samtalesammendrag / conversation summary — earlier messages were condensed]" prefix. No auto-trigger is wired into the loop — a host invokes `compact_history` from its own seam (e.g. `AgentConfig.on_iteration_start`); auto-compaction policy ships separately, if at all.

## [0.18.0] - 2026-07-16

### Added
- PII firewall subsystem reconciled onto the 0.17.x trunk (previously lived only on `feat/pii-firewall`). New `agentkit.pii` package: `Firewall` (`scrub_request`/`scrub_text`/`rehydrate_output`/`rehydrate_tool_args`/`assert_no_residual_tokens`/`routing_prefs`), consumer-supplied `Detector` & `TokenMap` protocols, `PiiPolicy`, `RehydratePolicy`, `Span`, `ZdrRouteUnavailable`, and `wrap_provider` — a decorating `Provider` that scrubs the full request on egress, attaches ZDR-fail-closed routing prefs, scrubs error messages, and emits an outbound-payload audit. Inert (zero cost) when the tmap resolver returns `None`. Additive edits alongside it: `RoutingPreferences` + optional `ProviderRequest.routing`; the OpenRouter request builder emits provider prefs (fail-closed `allow_fallbacks: false`) only when routing is set; `ToolSpec.rehydrate` defaults to `DENY`; "organization has been disabled" is treated as a ZDR route failure.

## [0.17.1] - 2026-07-16

### Fixed
- `search_tools` now advertises (and records) fully-qualified tool names, matching what `ToolRegistry.invoke` actually routes, so a model copying the advertised name no longer hits a fatal `unknown_tool`. A bare unknown name whose suffix matches exactly one registered qualified name gets a "did you mean …?" suggestion at both invoke boundaries (suggestion only, no transparent rerouting).

## [0.17.0] - 2026-07-16

### Added
- `AgentConfig.on_iteration_start` — an optional async hook awaited at the top of every `CONTEXT_BUILD`, i.e. before each LLM call within a turn, not only at turn boundaries. It receives the live `TurnContext`, so a consumer can append a message to `ctx.history` (or otherwise mutate context) and have it seen by the very next provider request. Mirrors the existing `model_selector`/`tool_selector` per-iteration hooks (carried on `AgentConfig`, threaded through `AgentSession._build_deps()`, typed `Any` to avoid a circular import). This is the supported seam for cooperative mid-run injection — steering a running turn without driving the loop by hand or overriding private handlers. Fully additive: `on_iteration_start=None` (the default) preserves the prior pass-through behavior exactly.

## [0.16.1] - 2026-06-29

### Changed
- Unknown-tool errors now hint at scripting-namespace methods. A call to a dotted name (`content.patch`, `tasks.patch`) that matches no registered tool returns a message naming it as a scripting-namespace method to call inside the scripting tool (or via the matching flat tool), so a model that reached for a namespace verb as a standalone tool self-corrects in one hop instead of ping-ponging on a bare "unknown tool". Shared `unknown_tool_message()` covers both the registry and tool-executing handler paths. Plain (non-dotted) names are unchanged. Additive; substring `"unknown tool: <name>"` is preserved.

## [0.16.0] - 2026-06-28

### Added
- Flat-surface generation from one `OpSpec`. `OpSpec` now carries `flat_alias`, `params` (a dict of `Param` with type/description/enum/required/alias/items_type), and `description`. A new `op_to_toolspec()` emits the provider `ToolSpec` (the flat chat tool a model sees) from an `OpSpec`, so the flat tool and the script-namespace method are derived from the same declaration and their names/params cannot drift. `ResourceNamespace` now binds positional args from the declared `params` order and accepts the id `alias` on `get`/`patch`/`delete`, folding the consumer-side "forgiving namespace" recovery shims into the framework. `EntitySpec` gained `id_param`/`field_params`/`flat_aliases`/`descriptions` so `build_crud_specs` attaches the flat metadata to every emitted CRUD op. Fully additive: an `OpSpec` without `flat_alias`/`params` behaves exactly as before (signature-introspection path unchanged), and `op_to_toolspec` returns `None` for it.

## [0.15.1] - 2026-06-17

### Added
- `codeexec` exposes the builtin exception classes (`Exception`, `ValueError`, `TypeError`, `KeyError`, `IndexError`, `AttributeError`, `RuntimeError`, `LookupError`, `ArithmeticError`, `ZeroDivisionError`, `OverflowError`, `StopIteration`, `StopAsyncIteration`, `AssertionError`, `NotImplementedError`, `BaseException`) in the safe-builtins namespace. Generated code can now write defensive `try/except ValueError` and `raise ValueError(...)` instead of crashing with `NameError: name 'ValueError' is not defined` the moment it references them. Same safety reasoning as `type`/`next`: the AST validator already rejects dunder access, so an exception class cannot be walked to `__subclasses__`/`__globals__` — exposing the names adds no sandbox-escape surface.

## [0.15.0] - 2026-06-16

### Added
- Recoverable-stream retry. A streaming attempt that fails with a *recoverable* provider error (rate limit, timeout, transient connection drop) **before any output has reached the consumer** is now retried — the loop re-enters `CONTEXT_BUILD` after an exponential backoff instead of ending the turn in `ERRORED`. This keeps a long multi-step / bulk turn alive across a brief provider blip rather than aborting the whole worklist mid-flight. Governed by `LoopConfig.max_stream_retries` (default `2`) and `LoopConfig.stream_retry_base_delay_seconds` (default `0.5`); set retries to `0` to restore surface-every-error behavior. The retry fires only on a clean early failure (nothing streamed yet), so it can never duplicate output the consumer already saw, and the held error is forwarded normally once the budget is spent. The budget is per stream attempt — a clean stream resets it — so each iteration of a multi-step turn gets a fresh allowance.
- `FakeProvider.error(code, message, *, recoverable=False)` gained the `recoverable` flag so tests can script recoverable vs terminal provider failures.

### Fixed
- `STREAMING -> CONTEXT_BUILD` is now a declared-legal phase transition. The success-claim correction path already returned `CONTEXT_BUILD` from streaming, but the transition table omitted it — so that retry (and the new recoverable-stream retry) would have been rejected as an illegal transition and forced the turn to `ERRORED`.

## [0.14.3] - 2026-06-16

### Changed
- `agentkit.__version__` is now read from the installed package metadata (`importlib.metadata`) instead of a hand-edited string, so it can no longer drift from `pyproject.toml` on a release. It had drifted badly — the literal still read `0.1.0` at 0.14.x.

### Docs
- Backfilled the CHANGELOG entries that were skipped during fast iteration: `0.9.0`, `0.12.0`, `0.13.0`, `0.14.1`, `0.14.2`. The file now documents every shipped version.
- Updated the install snippets in `README.md` and `docs/index.md` (they still pinned the long-obsolete `v0.1.0`).

## [0.14.2] - 2026-06-16

### Added
- `agentkit.codeexec` now exposes a wider set of escape-safe builtins to model-authored scripts: `next`, `iter`, `type`, `bytes`, and the pure numeric/formatting family `divmod`, `pow`, `chr`, `ord`, `hex`, `oct`, `bin`, `format`, `hash`. None of these reach IO, imports, or the process, so the sandbox boundary is unchanged; they let a script iterate explicitly, do byte/number work, and introspect a value's type without a manual workaround.

### Fixed
- Removed `type` from the validator's `FORBIDDEN_NAMES` so it no longer rejects a builtin the namespace now allows — the namespace allowlist and validator denylist had drifted apart. Added `test_denylist_and_namespace_allowlist_are_disjoint` so the two lists can never silently contradict each other again.

## [0.14.1] - 2026-06-14

### Fixed
- The finalize validator's Rule 1 (every claimed `action.tool` must correspond to a real tool call this turn) now normalizes server-qualified tool names before matching. A model that echoes a qualified name like `REDACTED.save_memory` in `actions_performed`, while the call log records the bare `save_memory`, no longer trips a false `fabricated_tool` violation and a needless finalize re-prompt on a legitimate action turn.

## [0.14.0] - 2026-06-09

### Added
- `AgentSession.resume_with_approval_batch(turn_id, decisions)` — resume a suspended turn after applying a batch of approval verdicts in one call. Each entry is `{"call_id", "decision", "edited_args"?, "reason"?}`; verdicts are applied (and their `ApprovalGranted`/`ApprovalDenied` events emitted) in list order before the Loop restarts once at `TOOL_EXECUTING`. This is required for correctness when a turn suspends on multiple pending tool calls: `handle_tool_executing` runs only the approved/denied/unknown buckets, so any call left in `pending_user_approvals` after a single-call `resume_with_approval` is silently dropped. A UI that presents one approval card for N calls must use the batch method to resume them all on a single verdict.
- `FakeProvider.tool_calls([(name, args), ...])` — script several tool calls in a single assistant message (parallel calls), so tests can exercise multi-pending-approval turns.

### Changed
- Refactored `resume_with_approval` internals into shared `_approval_timeout_stream`, `_build_verdict_event`, and `_resumed_loop_stream` helpers (behavior-preserving) now reused by the batch variant.

## [0.13.0] - 2026-06-02

### Added
- Tool Plane capability hard-gate. A tool may declare a `capability`, and `ToolPlane` will keep it out of the per-turn catalog until the turn's `ToolContext` reports that capability satisfied — so a consumer can gate a whole family of tools behind tenant entitlement, page context, or feature flag without per-tool branching. The `tool_capability_satisfied(tool, context)` predicate is exported for reuse, and `ToolPlane.hot_set` lets a host pin a tool visible for the current turn.
- `agentkit.resources` — a domain-free scriptable-resource framework the consuming app populates with `OpSpec`s. `OpRegistry` classifies operations conservatively (read / reversible-write / irreversible-write via the `Reversibility` enum); `ResourceNamespace` exposes uniform CRUD verbs (`create`/`replace`/`restore`/…) with a per-field whitelist; `EntitySpec` + `build_crud_specs` generate the specs for an entity; and `ApprovalScanner` walks an agent-authored script's AST, constant-propagates literal bindings, and classifies each call so the host can decide what needs approval before anything runs.

## [0.12.0] - 2026-05-31

### Added
- `AgentConfig.tool_selector` hook — a per-iteration filter over the tool catalog, so the visible tool set can shrink or grow turn-by-turn (progressive disclosure) instead of being fixed for the whole session.
- Generic per-turn tool resolver plus a built-in BM25 `search_tools` tool (`make_search_tools_builtin`): when the full catalog is too large to expose at once, the model can search for the tool it needs and the resolver promotes the match into the live set for that turn.

## [0.11.0] - 2026-05-31

### Added
- `LoopConfig.force_finalize_on_missing_reprompt` (default `False`) — when a turn ends without calling the finalize tool and `handle_finalize_check` re-prompts the model to finalize, this constrains that re-prompt turn to the finalize tool via `tool_choice`. Without it, a model that already answered inline can spend a whole additional free-form turn (thinking, re-narrating) before — or instead of — finalizing, holding the consumer in a streaming state for minutes even though the answer is already on screen. The re-prompt now resolves to a fast, guaranteed finalize call that yields a real envelope. Opt-in because it requires provider support for named `tool_choice`; the finalize tool is resolved from the registry by the same bare-name convention the validator uses (`finalize_response` / `finalize`), and the handler falls back to an unconstrained re-prompt when no finalize tool is registered. The flag is one-shot per re-prompt (consumed in `handle_streaming`), so only the recovery turn is constrained.

## [0.10.0] - 2026-05-31

### Added
- `agentkit.codeexec.SAFE_MODULES` — a curated mapping of pure-compute stdlib modules (`math`, `statistics`, `datetime`, `json`, `decimal`, `itertools`, `collections`, `re`) a host MAY merge into a script's namespace so model-authored scripts can do real math/date/parsing work **without** an `import` statement. Imports stay banned by the validator; the modules are handed in as objects exactly like any other injected name, and dunder attribute access on them is still rejected at parse time, so this does not reopen the sandbox escape. Modules with IO / process / import reach (`os`, `sys`, `subprocess`, `importlib`, `pathlib`, `socket`, `builtins`) are deliberately excluded, as is `random` (mutable global state). Opt-in per call: `execute({**SAFE_MODULES, "client": client}, source)`.

### Changed
- The validator's import-rejection message now guides the model toward pre-bound modules ("…are already available by name — use them directly without import") instead of the bare "import statements are not allowed", so a failed first attempt self-corrects rather than falling back to manual computation.

## [0.9.0] - 2026-05-23

### Added
- `AgentConfig.provider_selector` hook — pick the provider per streaming iteration (e.g. route a quality turn to a stronger model, a cheap turn to a fast one) instead of binding one provider for the whole session. Validated as selector-XOR-provider so a session configures exactly one of the two.
- `UsageRecorded` public event — token usage now surfaces on the event stream alongside the existing `ctx.metadata` usages, so consumers can meter spend live without reaching into loop internals.
- `UsageEvent` widened with required `model` and `provider_name` fields, stamped by every provider (Anthropic, OpenRouter, and the fakes), so usage records are attributable to a specific model/provider.

## [0.8.0] - 2026-05-22

### Changed
- A turn that ends **without** a `finalize_response` call is no longer silently accepted as "the conversation naturally ended". When a finalize validator is configured, `handle_finalize_check` now re-prompts the model once (bounded by `LoopConfig.max_missing_finalize_reprompts`, default 1) to emit a real envelope, then lets the turn end if the model still won't finalize. This fixes turns that stop mid-thought — typically by asking the user a question — settling with no envelope: the model now gets an explicit chance to classify them (e.g. `intent_kind="clarify"`). Consumers with no finalize validator are unaffected (pass-through).
- `FINALIZE_RESPONSE_DESCRIPTION` reworded: "call at the END of EVERY turn, including turns where you stop to ask the user a question", and the `clarify` bullet broadened to cover "asking a question / offering a choice / needing a decision", including turns that already did some work.

### Fixed
- Finalize-retry corrections now actually reach the model. `finalize_correction` was stashed in `ctx.metadata` but never surfaced — `MessageBuilder` reads `ctx.history`, so the rejected-finalize retry re-ran blind. The correction (and the new missing-finalize re-prompt) is now appended to `ctx.history` as a user-role message before re-streaming.
- Injected correction messages are tagged `metadata.annotations[INJECTED_CORRECTION_ANNOTATION]` so `_summaries_since_last_user_turn` (Rule 9 scoping) does not mistake a finalize re-prompt for a fresh human prompt — which would otherwise drop the turn's reads and false-fail `answer_evidence="tool_results"`.

### Added
- `LoopConfig.max_missing_finalize_reprompts: int = 1` — how many times a missing `finalize_response` is re-prompted before the turn is allowed to end.
- `INJECTED_CORRECTION_ANNOTATION` constant (`agentkit._messages`) — the `Message.metadata.annotations` key marking a loop-injected correction. Consumer code that infers turn boundaries from the most recent USER message should skip messages carrying it.

## [0.7.2] - 2026-05-21

### Fixed
- Unknown tool names no longer silently kill a turn. When the model calls an
  unregistered tool, `handle_tool_phase` now files it under a new
  `unknown_tool_calls` bucket and routes to `TOOL_EXECUTING` (previously it
  fell straight to `TOOL_RESULTS`, skipping result construction entirely — the
  model got no ToolResult, no error, just silence, and could never
  self-correct). `handle_tool_executing` builds a `status="error"` ToolResult
  naming the bad tool so the model can retry with a registered name.
- `handle_tool_results` counts unknown-tool errors toward the consecutive
  error abort (F20), so a model that keeps hallucinating the same tool name
  trips the abort instead of looping.

### Changed
- Defense-in-depth: `ToolRegistry.invoke` and `ToolDispatcher` no longer raise
  on an unknown tool name. `invoke` returns a `status="error"` ToolResult;
  `ToolDispatcher._safe_for_parallel` treats a spec-less call as
  not-parallel-safe instead of raising. A raised exception there bubbles to
  the orchestrator and ends the turn with no result for the model.

## [0.7.0] - 2026-05-15

### Added
- `Envelope.answer_evidence: Literal["tool_results", "context", "general_knowledge"] | None` field. Required when `intent_kind="answer"` (enforced by `validate_envelope`), ignored otherwise. Lets the model self-attest what evidence its answer rests on so the structural validator can check claim ↔ tool-log consistency.
- Validator Rule 8 (`answer_evidence_required`): rejects `intent_kind="answer"` envelopes missing `answer_evidence`.
- Validator Rule 9 (`answer_evidence_consistent`): rejects `answer_evidence="tool_results"` claims when the current turn has no successful read tool call. Uses a new `_summaries_since_last_user_turn` helper to scope reads to this turn only.
- `validate_envelope` accepts an optional `turn_summaries=` kwarg; when omitted, falls back to the full `tool_calls` list (backwards-compatible for existing callers).

### Fixed
- `recall_*` tools (e.g. `recall_memories`) now classify as reads in `_DEFAULT_READ_PREFIXES`. Previously misclassified as writes by the conservative default, which would have invalidated Rule 9 for memory-recall turns.

## [0.1.0] — 2026-05-04

### Added
- Provider abstraction: `Provider` protocol, `AnthropicProvider`, `OpenRouterProvider`
  with model-quirks-driven prompt caching.
- 11-phase agent loop with explicit transition table and `PhaseChanged` events.
- Tool registry with `kit.*` built-ins (finalize, current_time, memory, approval,
  subagent, note) plus MCP transports: `InProcessMCPClient` and `StdioMCPClient`.
- Storage protocols: `SessionStore`, `MemoryStore`, `CheckpointStore` with
  Redis-backed default implementations and in-memory fakes.
- Guards: `RiskBasedApprovalGate`, `DefaultIntentGate`, `RuleBasedFinalizeValidator`,
  `RegexSuccessClaimGuard`.
- 18 Pydantic event types with discriminated union.
- Optional FastAPI WebSocket bridge under `agentkit[fastapi]`.
- Multi-replica fan-out via `RedisEventBus`.
- `AgentSession` high-level entry point with `resume_with_approval` for
  suspend/resume flows.
- Subagent dispatch with isolated child contexts.
- Examples: `minimal/` and `with_mcp_tools/`.
- MkDocs documentation site.
