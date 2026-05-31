# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
from v1.0.0 onward. Pre-1.0 minor versions may include breaking changes.

## [0.10.0] - 2026-05-31

### Added
- `agentkit.codeexec.SAFE_MODULES` — a curated mapping of pure-compute stdlib modules (`math`, `statistics`, `datetime`, `json`, `decimal`, `itertools`, `collections`, `re`) a host MAY merge into a script's namespace so model-authored scripts can do real math/date/parsing work **without** an `import` statement. Imports stay banned by the validator; the modules are handed in as objects exactly like any other injected name, and dunder attribute access on them is still rejected at parse time, so this does not reopen the sandbox escape. Modules with IO / process / import reach (`os`, `sys`, `subprocess`, `importlib`, `pathlib`, `socket`, `builtins`) are deliberately excluded, as is `random` (mutable global state). Opt-in per call: `execute({**SAFE_MODULES, "client": client}, source)`.

### Changed
- The validator's import-rejection message now guides the model toward pre-bound modules ("…are already available by name — use them directly without import") instead of the bare "import statements are not allowed", so a failed first attempt self-corrects rather than falling back to manual computation.

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
