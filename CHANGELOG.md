# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
from v1.0.0 onward. Pre-1.0 minor versions may include breaking changes.

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
- The finalize validator's Rule 1 (every claimed `action.tool` must correspond to a real tool call this turn) now normalizes server-qualified tool names before matching. A model that echoes a qualified name like `pikkolo.save_memory` in `actions_performed`, while the call log records the bare `save_memory`, no longer trips a false `fabricated_tool` violation and a needless finalize re-prompt on a legitimate action turn.

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
